# required to get ArrayLike to render correctly
from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union
from numpy.typing import ArrayLike

import jax
import jax.numpy as jnp
import jaxopt
from functools import partial

import poisson_point_process.poisson_process_obs as obs
from .basis import raised_cosine_log_eval
from .utils import adjust_indices_and_spike_times, reshape_for_vmap, slice_array
from .base_regressor import BaseRegressor

from nemos import tree_utils, validation
from nemos.pytrees import FeaturePytree
from nemos.regularizer import GroupLasso, Lasso, Regularizer, Ridge
from nemos.typing import DESIGN_INPUT_TYPE

ModelParams = Tuple[jnp.ndarray, jnp.ndarray]

class ContinuousGLM(BaseRegressor):
    def __init__(
        self,
        obs_model_kwargs: dict,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[float] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        # initialize to None fit output
        self.intercept_ = None
        self.coef_ = None
        self.solver_state_ = None
        self.scale_ = None
        self.dof_resid_ = None

        # required to cmpute neg ll - no separate observation model class
        self.n_basis_funcs = obs_model_kwargs["n_basis_funcs"]
        self.n_batches_scan = obs_model_kwargs["n_batches_scan"]
        self.max_window = obs_model_kwargs["max_window"]
        self.history_window = obs_model_kwargs["history_window"]
        self.random_key = obs_model_kwargs["mc_random_key"]
        self.inverse_link_function = obs_model_kwargs["inverse_link_function"]

    @staticmethod
    def _check_params(
            params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
            data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        # check params has length two
        validation.check_length(params, 2, "Params must have length two.")
        # convert to jax array (specify type if needed)
        params = validation.convert_tree_leaves_to_jax_array(
            params,
            "Initial parameters must be array-like objects (or pytrees of array-like objects) "
            "with numeric data-type!",
            data_type,
        )

        validation.check_tree_leaves_dimensionality(
            params[0],
            expected_dim=1,
            err_message="params[0] must be an array or nemos.pytree.FeaturePytree "
                        "with array leafs of shape (n_features,).",
        )
        # check the dimensionality of intercept
        validation.check_tree_leaves_dimensionality(
            params[1],
            expected_dim=1,
            err_message="params[1] must be of shape (1,)"
        )
        if params[1].shape[0] != 1:
            raise ValueError(
                "Intercept term should be a single valued one-dimensional array."
            )
        return params

    @staticmethod
    def _check_input_n_timepoints(
            X: Union[DESIGN_INPUT_TYPE, jnp.ndarray], y: jnp.ndarray
    ):
        if y.shape[1] > X.shape[1]:
            raise ValueError(
                "The number of spikes in y cannot exceed the number of spikes in X."
                f"X has {X.shape[1]} spikes, "
                f"y has {y.shape[1]} spikes!"
            )

    @staticmethod
    def _check_input_dimensionality(
            X: Optional[DESIGN_INPUT_TYPE] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        if y is not None:
            validation.check_tree_leaves_dimensionality(
                y,
                expected_dim=2,
                err_message="y must be two-dimensional, with shape (2, n_target_spikes).",
            )
        if y.shape[0] != 2:
            raise ValueError(
                "y must have shape 2 at dimension 0 corresponding to postsynaptic spike times and indices"
            )
        if X is not None:
            validation.check_tree_leaves_dimensionality(
                X,
                expected_dim=2,
                err_message="X must be two-dimensional, with shape "
                "(2, n_spikes) or pytree of the same shape.",
            )
        if X.shape[0] != 2:
            raise ValueError(
                "X must have shape 2 at dimension 0 corresponding to spike times and neurons ids"
            )

    @staticmethod
    def _check_input_and_params_consistency(
            params: Tuple[Union[DESIGN_INPUT_TYPE, jnp.ndarray], jnp.ndarray],
            X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
            y: Optional[jnp.ndarray] = None,
    ):
        """Validate the number of features in model parameters and input arguments.

        Raises
        ------
        ValueError
            - if the number of features is inconsistent between params[1] and X
              (when provided).

        """
        if X is not None:
            # check that X and params[0] have the same structure
            if isinstance(X, FeaturePytree):
                data = X.data
            else:
                data = X

            validation.check_tree_structure(
                data,
                params[0],
                err_message=f"X and params[0] must be the same type, but X is "
                            f"{type(X)} and params[0] is {type(params[0])}",
            )

    def _get_coef_and_intercept(self) -> Tuple[Any, Any]:
        """Pack coef_ and intercept_  into a params pytree."""
        return self.coef_, self.intercept_

    def _set_coef_and_intercept(self, params: Any):
        """Unpack and store params pytree to coef_ and intercept_."""
        self.coef_: DESIGN_INPUT_TYPE = params[0]
        self.intercept_: jnp.ndarray = params[1]

    @partial(jax.jit, static_argnames=("self",))
    def sum_basis_and_dot(self, weights, dts):
        """compilable linear-non-linear transform"""
        fx = raised_cosine_log_eval(dts, self.history_window, self.n_basis_funcs)
        return jnp.sum(fx * weights)

    def linear_non_linear(self, dts, weights, bias):
        ll = self.inverse_link_function((self.sum_basis_and_dot(weights, dts) + bias))
        return ll

    def draw_mc_sample(self, X, M, random_key):
        """draw sample for a Monte-Carlo estimate of /int_0^T(lambda(t))dt"""
        keys = jax.random.split(random_key, 2)
        valid_spikes = X[0, :-self.max_window]
        s_m = jax.random.choice(keys[0], valid_spikes, shape=(M,), replace=False)
        epsilon_m = jax.random.uniform(keys[1], shape=(M,), minval=0.0, maxval=self.history_window)
        tau_m = s_m + epsilon_m
        tau_m_idx = jnp.searchsorted(X[0], tau_m, "right")
        mc_spikes = jnp.vstack((tau_m, tau_m_idx))

        return mc_spikes

    def compute_summed_ll(
            self,
            X,
            shifted_idx,
            params,
            log=True
    ):
        optional_log = jnp.log if log else lambda x: x

        weights, bias = params
        weights = weights.reshape(-1, self.n_basis_funcs)

        shifted_idx_array, padding = reshape_for_vmap(shifted_idx, self.n_batches_scan)

        # body of the scan function
        def scan_fn(lam_s, i):
            spk_in_window = slice_array(
                X, i + 1, self.max_window + 1
            )

            dts = spk_in_window[0] - jax.lax.dynamic_slice(X, (0, i), (1, 1))

            ll = optional_log(
                self.linear_non_linear(dts[0], weights[spk_in_window[1].astype(int)], bias)
            )
            lam_s += jnp.sum(ll)
            return jnp.sum(lam_s), None

        scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs))
        out, _ = scan_vmap(shifted_idx_array)
        sub, _ = scan_vmap(padding[:, None])
        return jnp.sum(out) - jnp.sum(sub)

    def compute_log_likelihood_term(self, X, y, params, optional_log=True):
        X_spikes_new, shifted_idx = \
            adjust_indices_and_spike_times(
                X,
                y,
                self.history_window,
                self.max_window
            )

        summed_ll = self.compute_summed_ll(
            X_spikes_new,
            shifted_idx,
            params,
            log=optional_log
        )

        return summed_ll

    def _negative_log_likelihood(
            self, y, X, params, random_key,
    ):
        log_lam_y = self.compute_log_likelihood_term(X, y, params, optional_log=True)

        M = y.shape[1] * 2
        mc_samples = self.draw_mc_sample(X, M, random_key)
        mc_estimate = self.compute_log_likelihood_term(X, mc_samples, params, optional_log=False)

        self.predicted_rate = mc_estimate / M

        return self.predicted_rate - log_lam_y

    def _predict_and_compute_loss(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        random_key: Optional=None,
    ) -> Tuple:
        """Loss function for a given model to be optimized over."""

        random_key, subkey = jax.random.split(random_key)

        neg_ll = self._negative_log_likelihood(y, X, params, subkey)

        return neg_ll, random_key

    def get_optimal_solver_params_config(self):
        """Return the functions for computing default step and batch size for the solver."""
        compute_optimal_params = None
        compute_smoothness = None
        strong_convexity = None
        return compute_optimal_params, compute_smoothness, strong_convexity

    @staticmethod
    def _initialize_intercept_matching_mean_rate(
            inverse_link_function: Callable,
            y: jnp.ndarray,
    ) -> jnp.ndarray:

        INVERSE_FUNCS = {
            jnp.exp: jnp.log,
            jax.nn.softplus: lambda x: jnp.log(jnp.exp(x) - 1.0),
        }

        analytical_inv = INVERSE_FUNCS.get(inverse_link_function, None)

        out = analytical_inv(y.shape[1] / int(jnp.ceil(y[0, -1])))
        print(out)

        return jnp.array([out,])

    def _initialize_parameters(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray
    )-> Tuple[Union[dict, jnp.ndarray], jnp.ndarray]:
        #set initial intercept as the inverse of firing rate
        initial_intercept = self._initialize_intercept_matching_mean_rate(
            self.inverse_link_function, y
        )
        # get coef dimensions
        n_neurons = len(jnp.unique(X[1]))
        n_basis_funcs = self.n_basis_funcs

        #initialize parameters
        init_params = (
            jnp.zeros(n_neurons * n_basis_funcs),
            initial_intercept,
        )
        return init_params

    def initialize_params(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            init_params: Optional[ModelParams] = None,
    ) -> Tuple[ModelParams, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        if init_params is None:
            init_params = self._initialize_parameters(X, y)  # initialize
        else:
            err_message = "Initial parameters must be array-like objects (or pytrees of array-like objects) "
            "with numeric data-type!"
            init_params = validation.convert_tree_leaves_to_jax_array(
                init_params, err_message=err_message, data_type=float
            )

        # validate input
        self._validate(X, y, init_params)

        return init_params

    def initialize_state(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            init_params,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        if isinstance(self.regularizer, GroupLasso):
            if self.regularizer.mask is None:
                warnings.warn(
                    UserWarning(
                        "Mask has not been set. Defaulting to a single group for all parameters. "
                        "Please see the documentation on GroupLasso regularization for defining a "
                        "mask."
                    )
                )
                self.regularizer.mask = jnp.ones_like(init_params[0])

        # this should do nothing
        opt_solver_kwargs = self.optimize_solver_params(data, y)

        self.instantiate_solver(solver_kwargs=opt_solver_kwargs)

        opt_state = self._solver_init_state(init_params, self.random_key, data, y,)

        return opt_state


    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: jnp.ndarray,
        init_params: Optional[Tuple[Union[dict, ArrayLike], ArrayLike]] = None,
    ):
        """Fit the model to neural activity."""
        init_params = self.initialize_params(X, y, init_params=init_params)

        self.initialize_state(X, y, init_params)

        params, state = self._solver_run(init_params, self.random_key, X, y)

        if tree_utils.pytree_map_and_reduce(
            lambda x: jnp.any(jnp.isnan(x)), any, params
        ):
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters, specifically try decreasing the `stepsize` "
                "and/or setting `acceleration=False`."
            )

        self._set_coef_and_intercept(params)

        self.solver_state_ = state
        return self

    def update(
            self,
            params: Tuple[jnp.ndarray, jnp.ndarray],
            opt_state: NamedTuple,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            *args,
            **kwargs,
    ) -> jaxopt.OptStep:
        """Run a single update step of the jaxopt solver."""
        # perform a one-step update
        opt_step = self._solver_update(params, opt_state, self.random_key, X, y, *args, **kwargs)

        # store params and state
        self._set_coef_and_intercept(opt_step[0])
        self.solver_state_ = opt_step[1]
        self.random_key = opt_step[1].aux

        return opt_step

    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters."""
        pass

    def score(
            self,
            X: DESIGN_INPUT_TYPE,
            y: Union[jnp.ndarray, jnp.ndarray],
            # may include score_type or other additional model dependent kwargs
            **kwargs,
    ) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target neural activity."""
        pass

    def simulate(
        self,
        random_key: jax.Array,
        feed_forward_input: DESIGN_INPUT_TYPE,
    ):
        """Simulate neural activity in response to a feed-forward input and recurrent activity."""
        pass