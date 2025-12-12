"""Solvers wrapping Optax solvers with Optimistix for use with NeMoS."""

from typing import Any, Callable, ClassVar

import equinox as eqx
import optax
import optimistix as optx
from optax import contrib

from nemos.regularizer import Regularizer
from nemos.typing import Pytree

from nemos.tree_utils import tree_sub
from nemos.solvers._optax_optimistix_solvers import AbstractOptimistixOptaxSolver
from nemos.solvers._optimistix_solvers import (
    OptimistixStepResult,
    Params,
)

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100_000

DEFAULT_PATIENCE = 5
DEFAULT_COOLDOWN = 0
DEFAULT_FACTOR = 0.5
DEFAULT_ACCUMULATION_SIZE = 200

class OptimistixOptaxAdamReduceOnPlateau(AbstractOptimistixOptaxSolver):
    """
    Implementation of ADAM with reduce-on-plateau lr schedule using optax.chain wrapped by optimistix.OptaxMinimiser.

    Convergence criterion is implemented by Optimistix, so the Cauchy criterion is used.
    """

    fun: Callable
    fun_with_aux: Callable

    # stats: dict[str, PyTree[ArrayLike]]
    stats: dict[str, Pytree]

    _optax_solver = optax.chain

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        maxiter: int = DEFAULT_MAX_STEPS,
        stepsize: float | None = None,
        patience: int = DEFAULT_PATIENCE,
        cooldown: int = DEFAULT_COOLDOWN,
        factor: int = DEFAULT_FACTOR,
        accumulation_size: int = DEFAULT_ACCUMULATION_SIZE,
        **solver_init_kwargs,

    ):

        _adam_rop = optax.chain(
            optax.adam(stepsize),
            contrib.reduce_on_plateau(
                patience=patience,
                cooldown=cooldown,
                factor=factor,
                rtol=rtol,
                accumulation_size=accumulation_size,
            ),
        )

        solver_init_kwargs["optim"] = _adam_rop

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            tol=tol,
            rtol=rtol,
            maxiter=maxiter,
            **solver_init_kwargs,
        )

class OptimistixOptaxProximalAdamReduceOnPlateau(AbstractOptimistixOptaxSolver):
    """
    Implementation of ADAM with reduce-on-plateau lr schedule using optax.chain wrapped by optimistix.OptaxMinimiser.
    Applies proximal operator after the ADAM update.

    Convergence criterion is implemented by Optimistix, so the Cauchy criterion is used.
    """

    fun: Callable
    fun_with_aux: Callable
    prox: Callable

    # stats: dict[str, PyTree[ArrayLike]]
    stats: dict[str, Pytree]

    _optax_solver = optax.chain
    _proximal: ClassVar[bool] = True

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        maxiter: int = DEFAULT_MAX_STEPS,
        stepsize: float | None = None,
        patience: int = DEFAULT_PATIENCE,
        cooldown: int = DEFAULT_COOLDOWN,
        factor: int = DEFAULT_FACTOR,
        accumulation_size: int = DEFAULT_ACCUMULATION_SIZE,
        **solver_init_kwargs,

    ):
        _adam_rop = optax.chain(
            optax.adam(stepsize),
            contrib.reduce_on_plateau(
                patience=patience,
                cooldown=cooldown,
                factor=factor,
                rtol=rtol,
                accumulation_size=accumulation_size,
            ),
        )

        solver_init_kwargs["optim"] = _adam_rop

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            tol=tol,
            rtol=rtol,
            maxiter=maxiter,
            **solver_init_kwargs,
        )

    def get_learning_rate(self, state: optx._solver.optax._OptaxState) -> float:
        """
        Read out the learning rate for scaling within the proximal operator.

        This learning rate is either a static learning rate or was found by a linesearch.
        """
        return state.opt_state[-1].learning_rate

    def step(
        self,
        fn: Callable,
        y: Params,
        args: Pytree,
        options: dict[str, Any],
        state: optx._solver.optax._OptaxState,
        tags: frozenset[object],
    ):
        # take gradient step
        new_params, new_state, new_aux = self._solver.step(
            fn, y, args, options, state, tags
        )

        # apply the proximal operator
        new_params = self.prox(
            new_params,
            self.regularizer_strength,
            self.get_learning_rate(new_state),
        )

        # reevaluate function value at the new point
        new_state = eqx.tree_at(lambda s: s.f, new_state, fn(new_params, args)[0])

        # recheck convergence criteria with the projected point
        updates = tree_sub(new_params, y)

        # replicating the jaxopt stopping criterion
        terminate = (
            optx.two_norm(updates) / self.get_learning_rate(new_state)
            < self._solver.atol
        )

        new_state = eqx.tree_at(lambda s: s.terminate, new_state, terminate)

        return new_params, new_state, new_aux

    def run(
        self,
        init_params: Params,
        *args,
    ) -> OptimistixStepResult:
        solution = optx.minimise(
            fn=self.fun,
            solver=self,  # pyright: ignore
            y0=init_params,
            args=args,
            options=self.config.options,
            has_aux=self.config.has_aux,
            max_steps=self.config.maxiter,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
            tags=self.config.tags,
        )

        self.stats.update(solution.stats)

        return solution.value, solution.state
