from typing import Callable, Optional, Union, Tuple
from numpy.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp

from nemos.observation_models import Observations
from nemos.typing import DESIGN_INPUT_TYPE

from . import utils

class MonteCarloApproximation(Observations):
    def __init__(
            self,
            n_basis_funcs,
            n_batches_scan,
            history_window,
            eval_function,
            mc_n_samples=None,
            inverse_link_function=jnp.exp,
    ):
        super().__init__(inverse_link_function=inverse_link_function)
        self.scale = 1.0
        self.recording_time = None
        self.T = None
        self.recording_time = None
        self.max_window = None
        self.ll_function = None
        self.ll_function_MC = None
        self.intensity_function = None

        self.n_basis_funcs = n_basis_funcs
        self.n_batches_scan = n_batches_scan
        self.history_window = history_window
        self.eval_function = eval_function
        # model specific
        self.M = mc_n_samples

    def _set_ll_function(self, population=False):
        self.ll_function = self.compute_summed_ll_vec if population else self.compute_summed_ll
        self.ll_function_MC = self.compute_summed_ll_vec_MC if population else self.compute_summed_ll
        self.intensity_function = self.linear_non_linear_vec if population else self.linear_non_linear

    def _initialize_data_params(
            self,
            recording_time,
            max_window: int,
    ):
        if self.recording_time is None:
            self.recording_time = recording_time
            self.T = self.recording_time.tot_length()
        if self.max_window is None:
            self.max_window = max_window

    @partial(jax.jit, static_argnames=("self",))
    def sum_basis_and_dot(self, weights, dts):
        """compilable linear-non-linear transform"""
        fx = self.eval_function(dts)
        return jnp.sum(fx * weights)

    def linear_non_linear(self, dts, weights, bias):
        """compute estmiated firing rates"""
        ll = self.inverse_link_function(self.sum_basis_and_dot(weights, dts) + bias)
        return ll

    def draw_mc_sample(self, X, M, random_key):
        """draw sample for a Monte-Carlo estimate of /int_0^T(lambda(t))dt"""
        dt = self.T/M
        start, end = self.recording_time.start[0], self.recording_time.end[-1]
        s_m = jnp.linspace(start+dt, end, M) - dt/2
        epsilon_m = jax.random.uniform(random_key, shape=(M,), minval=0.0, maxval=dt)
        tau_m = s_m + epsilon_m
        tau_m_idx = jnp.searchsorted(X[0], tau_m)
        mc_spikes = jnp.vstack((tau_m, tau_m_idx))

        return mc_spikes

    def compute_summed_ll(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.array,
            params: Tuple[jnp.array, jnp.array],
            log=True
    ):
        r"""
        computes summed lambda or log(lambda):

        \Phi(\sum_t \sum_{substack{t_s \in \\ \cX(t, W)}} \mathbf{w}_{n}^\top \mbphi(t - t_s))

        This function performs a parallelized scan over all reference spike times. For each one of them,
         it selects all presynaptic spike that may fall within the history window, evaluates basis functions
         at the spike time differences, computes their dot product with the weights and applies nonlinearuty

        Parameters:
            X: (2, S) the spike time data
            y: (2, S) postsynaptic spike times and insertion indices
            params: current model parameters
            log: (bool) whether or not to apply log to computed lambda
        Returns:
            summed (log) lambda, shape (1,)
        """
        optional_log = jnp.log if log else lambda x: x

        weights, bias = params
        weights = weights.reshape(-1, self.n_basis_funcs)

        shifted_spikes_array, padding = utils.reshape_for_vmap(y, self.n_batches_scan)

        # body of the scan function
        def scan_fn(lam_s, i):
            spk_in_window = utils.slice_array(
                X, i[1].astype(int), self.max_window
            )

            dts = spk_in_window[0] - i[0]

            ll = optional_log(
                self.intensity_function(dts, weights[spk_in_window[1].astype(int)], bias)
            )
            lam_s += ll.sum()
            return lam_s, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=1)
        out, _ = scan_vmap(shifted_spikes_array)
        sub, _ = scan_vmap(padding[None,:])
        return jnp.sum(out) - jnp.sum(sub)

    @partial(jax.jit, static_argnames=("self",))
    def sum_basis_and_dot_vec(self, weights, dts):
        """compilable linear-non-linear transform (population)"""
        fx = self.eval_function(dts)
        return jnp.sum(fx[:,:,None] * weights, axis=(0,1))

    def linear_non_linear_vec(self, dts, weights, bias):
        """compute estmiated firing rates (population)"""
        ll = self.inverse_link_function(self.sum_basis_and_dot_vec(weights, dts) + bias)
        return ll

    def compute_summed_ll_vec_MC(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.array,
            params: Tuple[jnp.array, jnp.array],
            log=True
    ):
        optional_log = jnp.log if log else lambda x: x
        weights, bias = params
        n_neurons = weights.shape[1]
        weights = weights.reshape(-1, self.n_basis_funcs, n_neurons).squeeze()
        n_neurons = weights.shape[2]

        # body of the scan function
        def scan_fn(lam_s, i):
            spk_in_window = utils.slice_array(
                X, i[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - i[0]
            ll = optional_log(
                self.linear_non_linear_vec(dts, weights[spk_in_window[1].astype(int)], bias)
            )

            lam_s += ll
            return lam_s, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.zeros(n_neurons), idxs), in_axes=1)
        shifted_spikes_array, padding = utils.reshape_for_vmap(y, self.n_batches_scan)
        out, _ = scan_vmap(shifted_spikes_array)
        sub, _ = scan_vmap(padding[None, :])
        ll_term = jnp.sum(out, axis=0) - jnp.sum(sub, axis=0)
        return ll_term.sum()

    def compute_summed_ll_vec(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.array,
            params: Tuple[jnp.array, jnp.array],
            log=True
    ):
        optional_log = jnp.log if log else lambda x: x
        weights, bias = params
        n_neurons = weights.shape[1]
        weights = weights.reshape(-1, self.n_basis_funcs, n_neurons)
        n_neurons = weights.shape[2]

        # body of the scan function
        def scan_fn(lam_s, i):
            spk_in_window = utils.slice_array(
                X, i[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - i[0]
            ll = optional_log(
                self.linear_non_linear(dts, weights[spk_in_window[1].astype(int),:,i[1].astype(int)], bias[i[1].astype(int)])
            )
            lam_s = lam_s.at[i[1].astype(int)].add(ll)

            return lam_s, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.zeros(n_neurons), idxs), in_axes=1)
        shifted_spikes_array, _ = utils.reshape_for_vmap(y, self.n_batches_scan)
        out, _ = scan_vmap(shifted_spikes_array)
        # sub, _ = scan_vmap(padding[None, :])
        ll_term = jnp.sum(out, axis=0) #- jnp.sum(sub, axis=0)
        return ll_term.sum()

    def _negative_log_likelihood(
            self,
            X,
            y,
            params: Optional=None,
            random_key: Optional=None,
    ):
        r"""
        computes Poisson point process negative log-likelihood with MC estimate of the CIF

        $\sum_{k=1}^K \log \lambda(y_k) - \frac{T}{M} \sum_{m=1}^M \lambda(\tau_m)$

        Parameters:
            X: (2, S) the spike time data
            y: (2, S) postsynaptic spike times and insertion indices
            params: current model parameters
            random_key: JAX random key for drawing MC samples
        Returns:
            negative log-likelihood, shape (1,)
        """


        log_lam_y = self.ll_function(
            X,
            y,
            params,
            log=True
        )

        mc_samples = self.draw_mc_sample(X, self.M, random_key)
        mc_estimate = self.ll_function_MC(
            X,
            mc_samples,
            params,
            log=False
        )

        estimated_rate = (self.T/self.M) * mc_estimate

        return estimated_rate - log_lam_y

    def _predict(
            self,
            X: Union[DESIGN_INPUT_TYPE, ArrayLike],
            params: Tuple[jnp.array, jnp.array],
            bin_size: Optional = 0.001,
            time_int: Optional = None,
            n_batches_scan=1,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        def int_f_scan(carry, t):
            spk_in_window = utils.slice_array(
                X, t[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - t[0]
            pred_rate = self.intensity_function(dts, weights[spk_in_window[1].astype(int)], bias)
            return None, pred_rate

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(int_f_scan, None, idxs), in_axes=1, out_axes=1)

        start, end = time_int
        duration = end - start

        weights, bias = params
        if len(weights.shape) == 1:
            n_neurons = 1
            weights = weights.reshape(-1, self.n_basis_funcs)
        else:
            n_neurons = weights.shape[1]
            weights = weights.reshape(-1, self.n_basis_funcs, n_neurons)
        X = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window)

        bin_times = jnp.linspace(start+bin_size, end, int(duration / bin_size)) - bin_size / 2
        bin_idx = jnp.searchsorted(X[0], bin_times, "right")
        bin_array, _ = utils.reshape_for_vmap(jnp.vstack((bin_times, bin_idx)), int(n_batches_scan))
        _, out = scan_vmap(bin_array)
        return out.reshape(-1, n_neurons).squeeze()[:bin_times.size]

    def log_likelihood(
            self,
            X,
            y,
            scale: Union[float, jnp.ndarray] = 1.0,
            aggregate_sample_scores: Callable = jnp.mean,
            params: Optional = None,
            random_key: Optional = None,
    ):
        r"""Compute the observation model log-likelihood.

        This computes the log-likelihood of the predicted rates
        for the observed neural activity including the normalization constant

        Parameters:
            X: (2, S) the spike time data
            y: (2, S) postsynaptic spike times and insertion indices

        Returns:
            The log-likehood. Shape (1,).
        """
        nll = self._negative_log_likelihood(X, y, params, random_key)
        return -nll

    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """
        Sample from the estimated distribution.

        This method generates random numbers from the desired distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate of the distribution. Shape (n_time_bins, ), or (n_time_bins, n_neurons)..
        scale:
            Scale parameter for the distribution.

        Returns
        -------
        :
            Random numbers generated from the observation model with `predicted_rate`.
        """
        pass

    def deviance(
            self,
            spike_counts: jnp.ndarray,
            predicted_rate: jnp.ndarray,
            scale: Union[float, jnp.ndarray] = 1.0,
    ):
        r"""Compute the residual deviance for the observation model.

        Parameters
        ----------
        spike_counts:
            The spike counts. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_neurons)`` for population models.
        predicted_rate:
            The predicted firing rates. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_neurons)`` for population models.
        scale:
            Scale parameter of the model.

        Returns
        -------
        :
            The residual deviance of the model.
        """
        pass

    def estimate_scale(
            self,
            y: jnp.ndarray,
            predicted_rate: jnp.ndarray,
            dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""Estimate the scale parameter for the model.

        This method estimates the scale parameter, often denoted as :math:`\phi`, which determines the dispersion
        of an exponential family distribution. The probability density function (pdf) for such a distribution
        is generally expressed as
        :math:`f(x; \theta, \phi) \propto \exp \left(a(\phi)\left(  y\theta - \mathcal{k}(\theta) \right)\right)`.

        The relationship between variance and the scale parameter is given by:

        .. math::
           \text{var}(Y) = \frac{V(\mu)}{a(\phi)}.

        The scale parameter, :math:`\phi`, is necessary for capturing the variance of the data accurately.

        Parameters
        ----------
        y :
            Observed activity.
        predicted_rate :
            The predicted rate values.
        dof_resid :
            The DOF of the residual.
        """
        pass

class PolynomialApproximation(MonteCarloApproximation):

    def __init__(
            self,
            n_basis_funcs,
            n_batches_scan,
            history_window,
            eval_function,
            int_function,
            prod_int_function,
            inverse_link_function=jnp.exp,
    ):
        super().__init__(
            inverse_link_function=inverse_link_function,
            n_basis_funcs = n_basis_funcs,
            n_batches_scan = n_batches_scan,
            history_window = history_window,
            eval_function = eval_function,
        )
        self.scale = 1.0
        self.T = None
        self.max_window = None
        self.suff = None

        self.n_basis_funcs = n_basis_funcs
        self.n_batches_scan = n_batches_scan
        self.history_window = history_window
        self.eval_function = eval_function
        self.prod_int_function = prod_int_function
        self.int_function = int_function

    def _set_ll_function(self, population=False):
        self.ll_function = self.compute_summed_ll_vec if population else self.compute_summed_ll
        self.ll_function_MC = self.compute_summed_ll_vec_MC if population else self.compute_summed_ll
        self.intensity_function = self.linear_non_linear_vec if population else self.linear_non_linear

    def _initialize_data_params(
            self,
            recording_time,
            max_window: int,
    ):
        if self.recording_time is None:
            self.recording_time = recording_time
            self.T = self.recording_time.tot_length()
        if self.max_window is None:
            self.max_window = max_window

    def _check_suff(self, X):
        if self.suff is None:
            self.suff = self.suff_stats(X)

    def compute_m(self, spikes_n):
        r"""
        computed the linear sufficient statistic m of scaled single basis function integrals
        $\mathbf{m} =|S_{1}|\bm{\varphi}, |S_{2}|\bm{\varphi} \dots |S_{N}|\bm{\varphi}$

        Parameters:
            spikes_n: (n_neurons,) total number of spikes per neuron
        Returns:
            (NJ,) linear sufficient statistic
        """
        phi_int = self.int_function(-self.history_window)
        return (spikes_n[:, None] * phi_int).ravel()

    def scatter_block_add(self,
                          operand: jnp.ndarray,
                          updates: jnp.ndarray,
                          row_idx: jnp.ndarray,
                          col_idx: jnp.ndarray) -> jnp.ndarray:
        S, J, _ = updates.shape

        row_base = (row_idx * J)[:, None, None]
        col_base = (col_idx * J)[:, None, None]
        row_offsets = jnp.arange(J)[None, :, None]
        col_offsets = jnp.arange(J)[None, None, :]

        row_indices = jnp.broadcast_to(row_base + row_offsets, (S, J, J))
        col_indices = jnp.broadcast_to(col_base + col_offsets, (S, J, J))

        scatter_indices = jnp.stack([row_indices, col_indices], axis=-1).reshape(-1, 2)
        scatter_updates = updates.reshape(-1)
        dnums = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1)
        )
        return jax.lax.scatter_add(operand, scatter_indices, scatter_updates, dnums)

    def compute_M_full(self, X):
        """
        computes the full (NJ,NJ) interaction matrix by running one pass over the entire dataset
        and performing lax.scatter_add() updates
        Parameters:
            X: (2, S) the spike time data
        Returns:
            M_int: (NJ, NJ) a matrix of between spike interactions
        """
        n_features = self.n_basis_funcs * len(jnp.unique(X[1, self.max_window:]))
        def scan_fn(M_scan, idx):
            spk_in_window = utils.slice_array(
                X, idx[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - idx[0]
            cross_prod_sums = self.prod_int_function(dts)
            # M_scan = M_scan.at[spk_in_window[2].astype(int)*n_basis_funcs, idx[2].astype(int)*n_basis_funcs].add(cross_prod_sums)
            M_scan = self.scatter_block_add(M_scan, cross_prod_sums, spk_in_window[1].astype(int), jnp.tile(idx[1].astype(int), self.max_window))
            M_scan = self.scatter_block_add(M_scan, cross_prod_sums.transpose(0,2,1), jnp.tile(idx[1].astype(int), self.max_window), spk_in_window[1].astype(int))
            return M_scan, None
        scan_vmap = jax.vmap(
            lambda idxs: jax.lax.scan(scan_fn, jnp.zeros((n_features, n_features)), idxs),
            in_axes=1
        )
        post_idx_array, padding = utils.reshape_for_vmap(X[:, self.max_window:], self.n_batches_scan)
        out, _ = scan_vmap(post_idx_array)
        sub, _ = scan_vmap(padding[None, :])
        M_int = jnp.sum(out, 0) - jnp.sum(sub, 0)
        return M_int

    def compute_M(
            self,
            X,
            spikes_n,
            neuron_ids,
    ):
        r"""
        computes the quadratic sufficient statistic of between- and self-spike interactions

        $\int_{0}^{T}a_2 \left(\sum\limits_{t_{s} \in S}\mathbf{w}^{\top}\bm{\phi}(t-t_{s})\right)^2 dt
        = a_2 \sum_{t_{s},t_{s'}\in S}\mathbf{w}^{\top} \mathbf{M}_{t_s, t_{s'}} \mathbf{w} $

        Parameters:
            X: (2, S) the spike time data
            spikes_n: (N,) the total number of spikes per neuron
            neuron_ids: (N,) an array of neuron ID labels
        Returns:
            M_full: (NJ, NJ) the quadratic sufficient statistic
        """

        self_products = jnp.einsum("i,jk->ijk", spikes_n,
                                   self.prod_int_function(jnp.array([0]))[0])

        X = jnp.vstack((X, jnp.arange(X.shape[1])))
        M_full = self.compute_M_full(X)

        def insert_block(matrix, idx, block):
            return jax.lax.dynamic_update_slice(matrix, block, idx)

        diag_starts = (neuron_ids * self.n_basis_funcs).astype(int)
        diag = jax.vmap(insert_block, in_axes=(None, 0, 0))(jnp.zeros_like(M_full), (diag_starts, diag_starts),
                                                            self_products).sum(0)
        M_full += diag

        return M_full

    def suff_stats(
            self,
            X,
    ):
        """
        precomputes sufficient statics m, M for PA-c glm
        Parameters:
            X: (2, S) the spike time data
        Returns:
            list of sufficient statistics
        """
        neuron_ids, spikes_n = jnp.unique(X[1], return_counts=True)

        m = self.compute_m(spikes_n)
        M = self.compute_M(
            X,
            spikes_n,
            neuron_ids
        )

        m = jnp.append(m, jnp.array([self.T]))
        M = jnp.hstack([M, m[:-1, None]])
        M = jnp.vstack([M, m[None, :]])
        M = M.at[-1, -1].set(self.T)

        return [m, M]

    def integral_approximation(self, params, approx_interval):
        """
        computes a quadratic approximation to the Poisson point process CIF
        Parameters:
            params: model parameters
            approx_interval: interval of inverse firing rates to approximate
        Returns:
           approximate CIF evaluation
        """
        w = jnp.hstack((params[0], params[1]))
        coefs = utils.compute_chebyshev(self.inverse_link_function, approx_interval).squeeze()
        linear_term = jnp.dot(self.suff[0], w)
        quadratic_term = jnp.dot(w, jnp.dot(self.suff[1], w))
        return self.T * coefs[0] + coefs[1] * linear_term + coefs[2] * quadratic_term

    def compute_event_logl(self, X, y):
        r"""
        computes the linear event log-likelihood for PA-c closed-form solution
        $\mathbf{k}=[\bm{\psi}_{1}^\top, \bm{\psi}_{2}^\top, \dots, \bm{\psi}_{N}^\top]$

        Parameters:
            X: (2, S) the spike time data
            y: (2, S) postsynaptic spikes and insertion indices
        Returns:
           k: (N,J) first term sufficient statistic
        """
        def scan_k(k_sum, i):
            spk_in_window = utils.slice_array(
                X, i[1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - i[0]
            eval = self.eval_function(dts)

            k_sum = k_sum.at[spk_in_window[1].astype(int)].add(eval)

            return k_sum, None

        k_init = jnp.zeros((len(jnp.unique(X[1, self.max_window:])),self.n_basis_funcs))
        # X, y = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window, y)
        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_k, k_init, idxs), in_axes=1)
        target_idx_array, padding = utils.reshape_for_vmap(y, self.n_batches_scan)
        out, _ = scan_vmap(target_idx_array)
        sub, _ = scan_vmap(padding[:, None])

        k = jnp.sum(out, 0).ravel() - jnp.sum(sub, 0).ravel()

        return jnp.append(k, jnp.array([y.shape[1]]))

    def _closed_form_solution(
            self,
            X,
            y,
            approx_interval,
            reg_strength: Optional=0.
    ):
        k = self.compute_event_logl(X, y)

        if reg_strength is None:
            Cinv = jnp.zeros_like(self.suff[1])
        else:
            Cinv = reg_strength*jnp.eye(self.suff[1].shape[0])
            Cinv = Cinv.at[-1, -1].set(0.0)

        coefs = utils.compute_chebyshev(self.inverse_link_function, approx_interval).squeeze()
        b = (k - self.suff[0] * coefs[1])
        weights_cf = jnp.linalg.solve(2 * coefs[2] * self.suff[1] + Cinv, b)

        return (weights_cf[:-1], weights_cf[-1])

    def _negative_log_likelihood(
            self,
            X,
            y,
            params: Optional = None,
            random_key: Optional = None,
            suff: Optional = None,
            approx_interval: Optional=None,
    ):

        log_lam_y = self.compute_summed_ll(
            X,
            y,
            params,
            log=True
        )

        estimated_rate = self.integral_approximation(params, approx_interval)

        return estimated_rate - log_lam_y

    def log_likelihood(
            self,
            X,
            y,
            scale: Union[float, jnp.ndarray] = 1.0,
            aggregate_sample_scores: Callable = jnp.mean,
            params: Optional = None,
            random_key: Optional = None,
    ):
        r"""Compute the observation model log-likelihood.

        This computes the log-likelihood of the predicted rates
        for the observed neural activity including the normalization constant

        Parameters
        ----------

        Returns
        -------
        :
            The log-likehood. Shape (1,).
        """
        nll = self._negative_log_likelihood(X, y, params)
        return -nll

    def _predict(
            self,
            X: Union[DESIGN_INPUT_TYPE, ArrayLike],
            params: Tuple[jnp.array, jnp.array],
            approx_interval: Optional = None,
            bin_size: Optional = 0.001,
            time_int: Optional = None,
            n_batches_scan: Optional = 1,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        def int_f_scan(carry, t):
            spk_in_window = utils.slice_array(
                X, t[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - t[0]
            dot_prod = self.sum_basis_and_dot(weights[spk_in_window[1].astype(int)], dts) + bias
            pred_rate = utils.quadratic(dot_prod, self.inverse_link_function, approx_interval)
            return None, pred_rate

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(int_f_scan, None, idxs), in_axes=1)

        start, end = time_int
        duration = end - start

        weights, bias = params
        weights = weights.reshape(-1, self.n_basis_funcs)
        X = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window)

        bin_times = jnp.linspace(start+bin_size, end, int(duration / bin_size)) - bin_size / 2
        bin_idx = jnp.searchsorted(X[0], bin_times, "right")
        bin_array, _ = utils.reshape_for_vmap(jnp.vstack((bin_times, bin_idx)), int(n_batches_scan))
        _, out = scan_vmap(bin_array)
        return out.transpose(1,0,2).ravel()[:bin_times.size]


class PopulationPolynomialApproximation(PolynomialApproximation):
    """
    The implementation of PA-c model to fit multiple postsynaptic neurons.
    """
    def __init__(
            self,
            n_basis_funcs,
            n_batches_scan,
            history_window,
            eval_function,
            int_function,
            prod_int_function,
            inverse_link_function=jnp.exp,
    ):
        super().__init__(
            inverse_link_function=inverse_link_function,
            n_basis_funcs = n_basis_funcs,
            n_batches_scan = n_batches_scan,
            history_window = history_window,
            eval_function = eval_function,
            int_function=int_function,
            prod_int_function=prod_int_function,
        )
        self.scale = 1.0
        self.T = None
        self.recording_time = None
        self.max_window = None
        self.suff = None

        self.n_basis_funcs = n_basis_funcs
        self.n_batches_scan = n_batches_scan
        self.history_window = history_window
        self.eval_function = eval_function
        self.prod_int_function = prod_int_function
        self.int_function = int_function

    def _set_ll_function(self, population=True):
        self.ll_function = self.compute_summed_ll_vec if population else self.compute_summed_ll
        self.ll_function_MC = self.compute_summed_ll_vec_MC if population else self.compute_summed_ll
        self.intensity_function = self.linear_non_linear_vec if population else self.linear_non_linear

    def integral_approximation(self, params, approx_interval):
        w = jnp.vstack((params[0], params[1]))
        coefs = utils.compute_chebyshev(self.inverse_link_function, approx_interval)
        linear_term = jnp.dot(self.suff[0], w)
        quadratic_term = jnp.einsum('in,ij,jn->n', w, self.suff[1], w)
        return self.T * coefs[0] + coefs[1] * linear_term + coefs[2] * quadratic_term

    def compute_event_logl(self, X, y):
        def scan_k(k_sum, i):
            spk_in_window = utils.slice_array(
                X, i[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - i[0]
            eval = self.eval_function(dts)

            k_sum = k_sum.at[spk_in_window[1].astype(int),:,i[1].astype(int)].add(eval)

            return k_sum, None

        n_target, spikes_n = jnp.unique(y[1, self.max_window:], return_counts=True)
        n_target = len(n_target)
        n_neurons = len(jnp.unique(X[1, self.max_window:]))
        k_init = jnp.zeros((n_neurons, self.n_basis_funcs, n_target))
        # X, y = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window, y)
        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_k, k_init, idxs), in_axes=1)
        target_idx_array, padding = utils.reshape_for_vmap(y, self.n_batches_scan)
        out, _ = scan_vmap(target_idx_array)
        sub, _ = scan_vmap(padding[:, None])

        k = jnp.sum(out, 0).reshape(-1, n_target) - jnp.sum(sub, 0).reshape(-1, n_target)

        return jnp.vstack((k, spikes_n[None, :]))

    def _closed_form_solution(
            self,
            X,
            y,
            approx_interval,
            reg_strength: Optional = 0.,
    ):
        k = self.compute_event_logl(X, y)

        if reg_strength is None:
            Cinv = jnp.zeros_like(self.suff[1])
        else:
            Cinv = reg_strength * jnp.eye(self.suff[1].shape[0])
            Cinv = Cinv.at[-1, -1].set(0.0)

        weights_cf_single = lambda a_mat, b_vec: jnp.linalg.solve(a_mat, b_vec)
        @jax.jit
        def weights_cf_batch(ab, bb):
            return jax.vmap(weights_cf_single, in_axes=(2, 1), out_axes=1)(ab, bb)

        coefs = utils.compute_chebyshev(self.inverse_link_function, approx_interval)
        if coefs.shape[1] > 300:
            batch_size = 300
            # n_batches = int(jnp.ceil(b.shape[1] / batch_size))
            results = []
            for start in range(0, coefs.shape[1], batch_size):
                end = min(start + batch_size, coefs.shape[1])
                ab = 2 * coefs[2, start:end][None, None, :] * self.suff[1][:, :, None] + Cinv[:, :, None]
                bb = (k[:,start:end]- self.suff[0][:, None] * coefs[1, start:end][None, :])
                res = weights_cf_batch(ab, bb)
                results.append(res)
            weights_cf = jnp.concatenate(results, axis=1)

        else:
            a = 2 * coefs[2][None, None, :] * self.suff[1][:, :, None] + Cinv[:, :, None]
            b = (k - self.suff[0][:, None] * coefs[1][None, :])
            weights_cf = jax.vmap(weights_cf_single, in_axes=(2, 1), out_axes=1)(a, b)

        return (weights_cf[:-1], weights_cf[-1])

    def _negative_log_likelihood(
            self,
            X,
            y,
            params: Optional = None,
            random_key: Optional = None,
            suff: Optional = None,
            approx_interval: Optional = None,
    ):
        log_lam_y = self.compute_summed_ll_vec(
            X,
            y,
            params,
            log=True
        )

        estimated_rate = self.integral_approximation(params, approx_interval)

        return estimated_rate.sum() - log_lam_y

    def _predict(
            self,
            X: Union[DESIGN_INPUT_TYPE, ArrayLike],
            params: Tuple[jnp.array, jnp.array],
            approx_interval: Optional = None,
            bin_size: Optional = 0.001,
            time_int: Optional = None,
            n_batches_scan=1,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        def int_f_scan(carry, t):
            spk_in_window = utils.slice_array(
                X, t[-1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - t[0]
            dot_prod = self.sum_basis_and_dot_vec(weights[spk_in_window[1].astype(int)], dts) + bias
            pred_rate = utils.quadratic(dot_prod, self.inverse_link_function, approx_interval)
            return None, pred_rate

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(int_f_scan, None, idxs), in_axes=1, out_axes=1)

        start, end = time_int
        duration = end - start

        weights, bias = params
        n_neurons = weights.shape[1]
        weights = weights.reshape(-1, self.n_basis_funcs, n_neurons)
        X = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window)

        bin_times = jnp.linspace(start+bin_size, end, int(duration / bin_size)) - bin_size / 2
        bin_idx = jnp.searchsorted(X[0], bin_times, "right")
        bin_array, _ = utils.reshape_for_vmap(jnp.vstack((bin_times, bin_idx)), int(n_batches_scan))
        _, out = scan_vmap(bin_array)
        return out.reshape(-1, n_neurons)[:bin_times.size]
