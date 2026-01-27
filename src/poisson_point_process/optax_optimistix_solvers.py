"""Solvers wrapping Optax solvers with Optimistix for use with NeMoS."""

from typing import Any, ClassVar, Callable

import jax
import jax.numpy as jnp

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

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100_000

DEFAULT_PATIENCE = 15
DEFAULT_COOLDOWN = 0
DEFAULT_FACTOR = 0.5
DEFAULT_ACCUMULATION_SIZE = 100
DEFAULT_ATOL_ROP = 0.0
DEFAULT_RTOL_ROP = 1e-3


class OptimistixOptaxStochasticAdamRoP(AbstractOptimistixOptaxSolver):
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
        atol_rop: float = DEFAULT_ATOL_ROP,
        rtol_rop: float = DEFAULT_RTOL_ROP,
        **solver_init_kwargs,

    ):
        self.stepsize = stepsize

        base_optimizer = optax.adam(stepsize)
        masked_optimizer = self.masked_optimizer_adaptive(base_optimizer)

        _adam_rop = optax.chain(
            masked_optimizer,
            self.split_key_transform(),
            contrib.reduce_on_plateau(
                patience=patience,
                cooldown=cooldown,
                factor=factor,
                rtol=rtol_rop,
                atol=atol_rop,
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

    def masked_optimizer_adaptive(self, base_optimizer):
        """A masked optimizer that automatically adapts to params length.
        Assumes that random key is the last param."""

        def init_fn(params):
            mask = tuple([True] * (len(params) - 1) + [False])
            masked_opt = optax.masked(base_optimizer, mask)
            return masked_opt.init(params)

        def update_fn(updates, state, params):
            mask = tuple([True] * (len(params) - 1) + [False])
            masked_opt = optax.masked(base_optimizer, mask)
            return masked_opt.update(updates, state, params)

        return optax.GradientTransformation(init_fn, update_fn)

    def split_key_transform(self):
        def init_fn(params):
            return {}  # no state needed

        def update_fn(updates, state, params):
            # Split the key in params and put the new key in updates
            # this assumes that key exists
            key = params[-1].astype(jnp.uint32)
            new_key, _ = jax.random.split(key)
            key_delta = new_key - key
            updates = updates[:-1] + (key_delta,)

            return updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    def get_learning_rate(self, state: optx._solver.optax._OptaxState) -> float:
        """
        Read out the learning rate for scaling within the proximal operator.
        Returns static baseline adam LR multiplied by the current ROP scale
        """

        rop_scale = state.opt_state[-1].scale # assuming rop is the last in chain

        return float(self.stepsize) * rop_scale

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

        # convergence based on loss
        updates = tree_sub(new_state.f, state.f)

        # recheck convergence criteria with the projected point
        # updates = tree_sub(new_params, y)

        # replicating the jaxopt stopping criterion
        terminate = (
            optx.two_norm(updates) / self.get_learning_rate(new_state)
            < self._solver.atol
        )

        new_state = eqx.tree_at(lambda s: s.terminate, new_state, terminate)

        return new_params, new_state, new_aux

class OptimistixOptaxStochasticProximalAdamRoP(OptimistixOptaxStochasticAdamRoP):
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
        atol_rop: float = DEFAULT_ATOL_ROP,
        rtol_rop: float = DEFAULT_RTOL_ROP,
        **solver_init_kwargs,

    ):

        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            tol=tol,
            rtol=rtol,
            maxiter=maxiter,
            stepsize=stepsize,
            patience=patience,
            cooldown=cooldown,
            factor=factor,
            accumulation_size=accumulation_size,
            atol_rop=atol_rop,
            rtol_rop=rtol_rop,
            **solver_init_kwargs,
        )

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

        # convergence based on loss
        updates = tree_sub(new_state.f, state.f)

        # recheck convergence criteria with the projected point
        # updates = tree_sub(new_params, y)

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
