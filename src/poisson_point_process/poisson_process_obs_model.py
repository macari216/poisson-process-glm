from typing import Callable, Optional, Union, Tuple
from numpy.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp
from scipy.integrate import simpson
from itertools import combinations

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
        self.T = None
        self.max_window = None
        self.ll_function = None
        self.intensity_function = None

        self.n_basis_funcs = n_basis_funcs
        self.n_batches_scan = n_batches_scan
        self.history_window = history_window
        self.eval_function = eval_function
        # model specific
        self.M = mc_n_samples

    def _set_ll_function(self, population=False):
        self.ll_function = self.compute_summed_ll_vec if population else self.compute_summed_ll
        self.intensity_function = self.linear_non_linear_vec if population else self.linear_non_linear

    def _initialize_data_params(
            self,
            T: Union[float, int],
            max_window: int,
    ):
        if self.T is None:
            self.T = T
        if self.max_window is None:
            self.max_window = max_window

    @partial(jax.jit, static_argnames=("self",))
    def sum_basis_and_dot(self, weights, dts):
        """compilable linear-non-linear transform"""
        fx = self.eval_function(dts)
        return jnp.sum(fx * weights)

    def linear_non_linear(self, dts, weights, bias):
        ll = self.inverse_link_function(self.sum_basis_and_dot(weights, dts) + bias)
        return ll

    def draw_mc_sample(self, X, M, random_key):
        """draw sample for a Monte-Carlo estimate of /int_0^T(lambda(t))dt"""

        s_m, dt = jnp.linspace(0, self.T, M, retstep=True)
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
            lam_s += jnp.sum(ll)
            return lam_s, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=0)
        out, _ = scan_vmap(shifted_spikes_array)
        sub, _ = scan_vmap(padding[None,:])
        return jnp.sum(out) - jnp.sum(sub)

    @partial(jax.jit, static_argnames=("self",))
    def sum_basis_and_dot_vec(self, weights, dts):
        """compilable linear-non-linear transform"""
        fx = self.eval_function(dts)
        return jnp.sum(fx[:,:,None] * weights, axis=(0,1))

    def linear_non_linear_vec(self, dts, weights, bias):
        ll = self.inverse_link_function(self.sum_basis_and_dot_vec(weights, dts) + bias)
        return ll

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
                self.linear_non_linear_vec(dts, weights[spk_in_window[1].astype(int)], bias)
            )
            lam_s = jax.lax.cond(
                log,
                lambda lam_s: lam_s.at[i[1].astype(int)].add(ll[i[1].astype(int)]),
                lambda lam_s: lam_s + ll,
                lam_s
            )
            return lam_s, None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.zeros(n_neurons), idxs), in_axes=0)
        shifted_spikes_array, padding = utils.reshape_for_vmap(y, self.n_batches_scan)
        out, _ = scan_vmap(shifted_spikes_array)
        sub, _ = scan_vmap(padding[None, :])
        ll_term = jnp.sum(out, axis=0) - jnp.sum(sub, axis=0)
        return ll_term.sum()

    def _negative_log_likelihood(
            self,
            X,
            y,
            params: Optional=None,
            random_key: Optional=None,
    ):


        log_lam_y = self.ll_function(
            X,
            y,
            params,
            log=True
        )

        mc_samples = self.draw_mc_sample(X, self.M, random_key)
        mc_estimate = self.ll_function(
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
            bin_size: Optional=0.001,
            time_sec: Optional=None,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        @jax.jit
        def intensity_function(X, t):
            spk_in_window = utils.slice_array(
                X, t[1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - t[0]

            return self.intensity_function(dts, weights[spk_in_window[1].astype(int)], bias)
        intensity_function_vmap = jax.vmap(lambda idx: intensity_function(X, idx), in_axes=(1,))

        if time_sec is None:
            time_sec = self.T

        weights, bias = params
        if (weights.shape) == 1:
            weights = weights.reshape(-1, self.n_basis_funcs)
        else:
            n_neurons = weights.shape[1]
            weights = weights.reshape(-1, self.n_basis_funcs, n_neurons)
        X = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window)

        bin_times = jnp.linspace(bin_size, time_sec, int(time_sec / bin_size)) - bin_size/2
        bin_idx = jnp.searchsorted(X[0], bin_times, "right")
        bin_array =  jnp.vstack((bin_times, bin_idx))

        predicted_rate = intensity_function_vmap(bin_array).squeeze()

        return predicted_rate

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
            approx_interval=None,
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
        # model specific
        self.approx_interval = approx_interval

    def _initialize_data_params(
            self,
            T: Union[float, int],
            max_window: int,
    ):
        if self.T is None:
            self.T = T
        if self.max_window is None:
            self.max_window = max_window

    def _check_suff(self, X):
        if self.suff is None:
            self.suff = self.suff_stats(X)

    def compute_m(self, spikes_n):
        phi_int = self.int_function(-self.history_window)
        return (spikes_n[:, None] * phi_int).ravel()

    def compute_M_block(self, tot_spikes, post_indices, pair):
        def compute_Mn_half(pair):
            def valid_idx(M_scan, idx):
                spk_in_window = utils.slice_array(
                    tot_spikes, idx[1].astype(int), self.max_window
                )
                dts = spk_in_window[0] - idx[0]
                dts_valid = jnp.where(spk_in_window[1] == pair[0], dts, -self.history_window - 1)
                cross_prod_sum = self.prod_int_function(dts_valid).sum(0)
                M_scan = M_scan + cross_prod_sum
                return M_scan

            def invalid_idx(M_scan, idx):
                M_scan += jnp.zeros((self.n_basis_funcs, self.n_basis_funcs))
                return M_scan

            def scan_fn(M_scan, idx):
                M_scan = jax.lax.cond(idx[1].astype(int) > tot_spikes.shape[1], invalid_idx, valid_idx, M_scan, idx)
                return M_scan, None

            scan_vmap = jax.vmap(
                lambda idxs: jax.lax.scan(scan_fn, jnp.zeros((self.n_basis_funcs, self.n_basis_funcs)), idxs),
                in_axes=0
            )

            post_idx_array, padding = utils.reshape_for_vmap(
                jnp.vstack(
                    (tot_spikes[0, post_indices[pair[1].astype(int)]],
                     post_indices[pair[1].astype(int)])
                ),
                self.n_batches_scan
            )
            out, _ = scan_vmap(post_idx_array)
            sub, _ = scan_vmap(padding[None, :])

            return jnp.sum(out, 0) - jnp.sum(sub, 0)

        pair_double = jnp.stack((pair, pair[::-1]))
        pair_M = jax.vmap(compute_Mn_half)(pair_double)

        return pair_M[0] + pair_M[1].T

    def construct_M(self, M_self, M_cross):
        N, J = M_self.shape[0], M_self.shape[1]
        M_empty = jnp.zeros((N * J, N * J), dtype=(M_self.dtype))
        diag_starts = jnp.arange(N) * J

        @jax.jit
        def insert_block(matrix, idx, block):
            return jax.lax.dynamic_update_slice(matrix, block, idx)

        insert_block_vmap = jax.vmap(lambda idxs, blocks: insert_block(M_empty, idxs, blocks), in_axes=(0, 0))
        M_full = insert_block_vmap((diag_starts, diag_starts), M_self).sum(0)

        if M_cross is not None:
            rows = jnp.arange((N * (N - 1)) // 2) // (N - 1)
            cols = jnp.arange((N * (N - 1)) // 2) % (N - 1) + 1 + rows
            i_starts, j_starts = rows * J, cols * J
            M_full += insert_block_vmap((i_starts, j_starts), M_cross).sum(0)
            M_full += insert_block_vmap((j_starts, i_starts), jnp.transpose(M_cross, (0, 2, 1))).sum(0)

        return M_full

    def compute_M(
            self,
            X,
            spikes_n,
            neuron_ids,
    ):
        self_pairs = jnp.array(list(zip(neuron_ids, neuron_ids)))
        cross_pairs = jnp.array(list(utils.jax_pairs(neuron_ids)))

        self_products = jnp.einsum("i,jk->ijk", spikes_n,
                                   self.prod_int_function(jnp.array([0]))[0])

        post_indices = utils.precompute_spike_indices(X[1], neuron_ids, int(spikes_n.max()))
        M_block_pair = lambda p: self.compute_M_block(X, post_indices, p)

        def batched_scan(pairs, batch_size):
            results = []
            for i in range(0, pairs.shape[0], batch_size):
                batch = pairs[i:i + batch_size]
                processed = jax.vmap(M_block_pair)(batch)
                results.append(processed)
            return jnp.concatenate(results, axis=0)


        # M_self = jax.vmap(M_block_pair)(self_pairs)
        M_self = batched_scan(self_pairs, 10)
        M_self += self_products
        # M_cross = jax.vmap(M_block_pair)(cross_pairs) if len(cross_pairs) != 0 else None
        M_cross = batched_scan(cross_pairs, 100) if len(cross_pairs) != 0 else None

        M_full = self.construct_M(M_self, M_cross)
        return M_full

    def suff_stats(
            self,
            X,
    ):
        """
        precompute sufficient statics m, M for pac glm
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

    def integral_approximation(self, params):
        w = jnp.hstack((params[0], params[1]))
        coefs = utils.compute_chebyshev(self.inverse_link_function, self.approx_interval)
        linear_term = jnp.dot(self.suff[0], w)
        quadratic_term = jnp.dot(w, jnp.dot(self.suff[1], w))
        return self.T * coefs[0] + coefs[1] * linear_term + coefs[2] * quadratic_term

    def compute_event_logl(self, X, y):
        def scan_k(k_sum, i):
            spk_in_window = utils.slice_array(
                X, i[1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - i[0]
            eval = self.eval_function(dts)

            k_sum = k_sum.at[spk_in_window[1].astype(int)].add(eval)

            return k_sum, None

        k_init = jnp.zeros((len(jnp.unique(X[1, self.max_window:])),self.n_basis_funcs))
        X, y = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window, y)
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
    ):
        k = self.compute_event_logl(X, y)

        coefs = utils.compute_chebyshev(self.inverse_link_function, self.approx_interval)
        b = (k - self.suff[0] * coefs[1])
        weights_cf = jnp.linalg.solve(2 * coefs[2] * self.suff[1], b)

        return (weights_cf[:-1], weights_cf[-1])

    def _negative_log_likelihood(
            self,
            X,
            y,
            params: Optional = None,
            random_key: Optional = None,
            suff: Optional = None,
    ):

        log_lam_y = self.compute_summed_ll(
            X,
            y,
            params,
            log=True
        )

        estimated_rate = self.integral_approximation(params)

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
            bin_size: Optional=0.001,
            time_sec: Optional=None,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        @jax.jit
        def intensity_function(X, t):
            w = jnp.hstack((weights, bias))
            spk_in_window = utils.slice_array(
                X, t[1].astype(int), self.max_window
            )
            dts = spk_in_window[0] - t[0]
            dts_eval = self.eval_function(dts)
            l = jnp.zeros((N,J)).at[spk_in_window[1].astype(int)].set(dts_eval)
            return utils.quadratic(
                jnp.dot(
                    jnp.append(l.ravel(), jnp.ones(1)),
                    w
                ),
                self.inverse_link_function,
                self.approx_interval
            )
        intensity_function_vmap = jax.vmap(lambda idx: intensity_function(X, idx), in_axes=(1,))

        if time_sec is None:
            time_sec = self.T

        weights, bias = params
        X = utils.adjust_indices_and_spike_times(X, self.history_window, self.max_window)
        N, J = weights.reshape(-1, self.n_basis_funcs).shape

        bin_times = jnp.linspace(bin_size, time_sec, int(time_sec / bin_size)) - bin_size/2
        bin_idx = jnp.searchsorted(X[0], bin_times, "right")
        bin_array =  jnp.vstack((bin_times, bin_idx))

        predicted_rate = intensity_function_vmap(bin_array).squeeze()

        return predicted_rate

