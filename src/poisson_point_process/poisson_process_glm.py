# required to get ArrayLike to render correctly
from __future__ import annotations

import warnings
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union
from numpy.typing import ArrayLike

import jax
import jax.numpy as jnp
import jaxopt

from . import utils
from . import poisson_process_obs_model as obs
from .base_regressor_MC import BaseRegressor

from nemos import tree_utils, validation
from nemos.pytrees import FeaturePytree
from nemos.regularizer import GroupLasso, Lasso, Regularizer, Ridge
from nemos.typing import DESIGN_INPUT_TYPE

ModelParams = Tuple[jnp.ndarray, jnp.ndarray]

from time import perf_counter

class ContinuousMC(BaseRegressor):
    def __init__(
        self,
        observation_model: obs.MonteCarloApproximation,
        random_key: ArrayLike=jax.random.PRNGKey(0),
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
        self.T = None
        self.max_window = None

        self.observation_model = observation_model
        self.random_key = random_key

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
        if y[1,-1] > X.shape[1]+1:
            raise ValueError(
                    "The position index in y cannot exceed the number of spikes in X."
                    f"X has {X.shape[1]} spikes, "
                    f"the largest index in y is {y.shape[1]}!"
                )
        pass

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

    def _predict_and_compute_loss(
            self,
            params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            random_key: Optional = None,
    ) -> Tuple:
        """Loss function for a given model to be optimized over."""

        random_key, subkey = jax.random.split(random_key)

        neg_ll = self.observation_model._negative_log_likelihood(X, y, params, subkey)

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

        return jnp.array([out,])

    def _initialize_parameters(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray
    )-> Tuple[Union[dict, jnp.ndarray], jnp.ndarray]:
        #set initial intercept as the inverse of firing rate
        initial_intercept = self._initialize_intercept_matching_mean_rate(
            self.observation_model.inverse_link_function, y
        )
        # get coef dimensions
        n_neurons = len(jnp.unique(X[1]))

        #initialize parameters
        init_params = (
            jnp.zeros(n_neurons * self.observation_model.n_basis_funcs),
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

    def _initialize_data_params(self,X,y):
        if self.max_window is None:
            self.max_window = int(
                jnp.maximum(
                    utils.compute_max_window_size(jnp.array([-self.observation_model.history_window, 0]), y[0], X[0]),
                    utils.compute_max_window_size(jnp.array([-self.observation_model.history_window, 0]), X[0], X[0])
                )
            )
        if self.T is None:
            self.T = jnp.ceil(X[0, -1])

    def initialize_state(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            init_params,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        # set data dependent parameters
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T,self.max_window)

        #add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

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
        # set data dependent parameters
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)

        #add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        init_params = self.initialize_params(data, y, init_params=init_params)

        self.initialize_state(X, y, init_params)

        params, state = self._solver_run(init_params, self.random_key, data, y)

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
        # set data dependent parameters
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)

        #add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        # perform a one-step update
        opt_step = self._solver_update(params, opt_state, self.random_key, data, y, *args, **kwargs)

        # store params and state
        self._set_coef_and_intercept(opt_step[0])
        self.solver_state_ = opt_step[1]
        self.random_key = opt_step[1].aux

        return opt_step

    def predict(
            self,
            X: Union[DESIGN_INPUT_TYPE, ArrayLike],
            bin_size: Optional=0.001,
            time_sec: Optional=None,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        return self.observation_model._predict(X, (self.coef_, self.intercept_), bin_size, time_sec)

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

class ContinuousPA(ContinuousMC):
    def __init__(
        self,
        observation_model: obs.PolynomialApproximation,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[float] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            observation_model=observation_model,
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
        self.T = None
        self.max_window = None

        self.observation_model = observation_model

    def _predict_and_compute_loss(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        random_key: Optional=None,
    ) -> jnp.ndarray:
        """Loss function for a given model to be optimized over."""

        neg_ll = self.observation_model._negative_log_likelihood(X, y, params)

        return neg_ll

    def initialize_state(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            init_params,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        # set data dependent parameters
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)

        # add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        # compute sufficient statistics
        self.observation_model._check_suff(X)

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

        opt_state = self._solver_init_state(init_params, None, data, y)

        return opt_state


    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: jnp.ndarray,
        init_params: Optional[Tuple[Union[dict, ArrayLike], ArrayLike]] = None,
    ):
        """Fit the model to neural activity."""

        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)
        self.observation_model._check_suff(X)

        #add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        init_params = self.initialize_params(data, y, init_params=init_params)

        self.initialize_state(data, y, init_params)

        params, state = self._solver_run(init_params, None, data, y)

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
        tt0 = perf_counter()
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)
        self.observation_model._check_suff(X)

        #add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        # perform a one-step update
        t0 = perf_counter()
        opt_step = self._solver_update(params, opt_state, None, data, y, *args, **kwargs)

        t0 = perf_counter()
        if tree_utils.pytree_map_and_reduce(
            lambda x: jnp.any(jnp.isnan(x)), any, opt_step[0]
        ):
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters, specifically try decreasing the `stepsize` "
                "and/or setting `acceleration=False`."
            )

        # store params and state
        self._set_coef_and_intercept(opt_step[0])
        self.solver_state_ = opt_step[1]
        # print(f"opt step, {perf_counter() - tt0}")

        return opt_step

    def fit_closed_form(self, X, y):
        if self.observation_model.inverse_link_function is not jnp.exp:
            raise ValueError(
                f"Closed form solution requires exponential inverse link function, "
                f"the inverse link provided is {self.observation_model.inverse_link_function}!"
            )
        # set data dependent parameters
        self._initialize_data_params(X, y)
        self.observation_model._initialize_data_params(self.T, self.max_window)

        # add max window padding to X and y
        X, y = utils.adjust_indices_and_spike_times(X, self.observation_model.history_window, self.max_window, y)

        # compute sufficient statistics
        self.observation_model._check_suff(X)

        params = self.observation_model._closed_form_solution(X, y)

        self._set_coef_and_intercept(params)

        return self

    def predict(
            self,
            X: Union[DESIGN_INPUT_TYPE, ArrayLike],
            bin_size: Optional=0.001,
            time_sec: Optional = None,
    ) -> jnp.ndarray:
        """Predict rates based on fit parameters."""

        return self.observation_model._predict(X, (self.coef_, self.intercept_), bin_size, time_sec)

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