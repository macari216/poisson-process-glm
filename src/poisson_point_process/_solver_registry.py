"""Registry of optimization algorithms and their implementations."""

from dataclasses import dataclass
from typing import Type

from nemos.solvers._abstract_solver import SolverProtocol
from nemos.solvers._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from nemos.solvers._svrg import WrappedProxSVRG, WrappedSVRG


@dataclass
class SolverSpec:
    algo_name: str
    backend: str
    implementation: Type[SolverProtocol]

    @property
    def full_name(self) -> str:
        return f"{self.algo_name}[{self.backend}]"

    def __repr__(self) -> str:
        return (
            f"{self.full_name!r} - "
            f"{self.__class__.__name__}("
            f"algo_name={self.algo_name!r}, "
            f"backend={self.backend!r}, "
            f"implementation={self.implementation})"
        )


# mapping is {algo_name : {backend : implementation}}
_registry: dict[str, dict[str, SolverSpec]] = {}
# mapping is {algo_name : backend}
_defaults: dict[str, str] = {}


def _parse_name(name: str) -> tuple[str, str | None]:
    """Parse an algo_name[backend] string."""
    algo_name = name
    backend = None
    if "[" in name and name.endswith("]"):
        algo_name = name[: name.index("[")]
        backend = name[name.index("[") + 1 : -1]

    return algo_name, backend


def _raise_if_not_in_registry(algo_name: str):
    """Raise an error if an algorithm is not in the registry."""
    if algo_name not in _registry:
        raise ValueError(f"No solver registered for algorithm {algo_name}.")


def get_solver(name: str) -> Type[SolverProtocol]:
    """Fetch the solver implementation from the registry."""
    algo_name, backend = _parse_name(name)

    # make sure we have the algorithm
    _raise_if_not_in_registry(algo_name)
    algo_versions = _registry[algo_name]

    # if not specified, try getting the default backend for the algorithm
    if backend is None:
        backend = _defaults.get(algo_name, None)
    if backend is None:
        if len(algo_versions) == 1:
            backend = next(iter(algo_versions.keys()))
        else:
            raise ValueError(
                f"Multiple backends and no default found for {algo_name}. Please specify or set a default backend."
            )
    if backend not in algo_versions:
        raise ValueError(
            f"{backend} backend not available for {algo_name}. "
            f"Available backends: {list_algo_backends(algo_name)}"
        )

    return algo_versions[backend].implementation


def __getitem__(name: str) -> Type[SolverProtocol]:
    """Fetch the solver implementation with nicer syntax."""
    return get_solver(name)


def register(
    algo_name: str,
    implementation: Type[SolverProtocol],
    backend: str = "custom",
    replace: bool = False,
    default: bool = False,
):
    """
    Register a solver implementation in the registry.

    algo_name:
        Name of the optimization algorithm.
    implementation:
        Class implementing the solver.
        Has to adhere to the AbstractSolver interface.
    backend:
        Backend name. Defaults to "custom".
        When wrapping and registering an existing solver from an external
        package, this would be the package name.
    replace:
        If an implementation for the given algorithm and backend names
        is already present in the registry, overwrite it.
    default:
        Set this implementation as the default for the algorithm.
        Can also be done with `set_default`.
    """
    if not replace and backend in _registry.get(algo_name, {}):
        raise ValueError(
            f"{algo_name}[{backend}] already registered. Use replace=True to overwrite."
        )
    if algo_name not in _registry:
        _registry[algo_name] = {}

    _registry[algo_name][backend] = SolverSpec(algo_name, backend, implementation)

    if default:
        set_default(algo_name, backend)


def set_default(algo_name: str, backend: str):
    """Set the default backend for a given algorithm."""
    _raise_if_not_in_registry(algo_name)

    if backend not in _registry[algo_name]:
        raise ValueError(
            f"{backend} backend not available for {algo_name}."
            f"Available backends: {list_algo_backends(algo_name)}"
        )
    _defaults[algo_name] = backend


def list_algo_backends(algo_name: str) -> list[str]:
    """List the available backend for an algorithm."""
    return list(_registry[algo_name].keys())


# TODO: Add doctest
# TODO: Return full_name instead of the SolverSpec
def list_available_solvers() -> list[SolverSpec]:
    """List all available solvers."""
    return [
        spec for algo_versions in _registry.values() for spec in algo_versions.values()
    ]


# TODO: Add doctest
def list_available_algorithms() -> list[str]:
    """
    List the available algorithms that can be used for fitting models.

    To list the available backends for a given algorithm,
    see `list_algo_backends`.

    To access an extended documentation about a specific solver,
    see `nemos.solvers.get_solver_documentation`.
    """
    return list(_registry.keys())


register("GradientDescent", JaxoptGradientDescent, "jaxopt", default=True)
register("ProximalGradient", JaxoptProximalGradient, "jaxopt", default=True)
register("LBFGS", JaxoptLBFGS, "jaxopt", default=True)
register("BFGS", JaxoptBFGS, "jaxopt", default=True)
register("NonlinearCG", JaxoptNonlinearCG, "jaxopt", default=True)
register("SVRG", WrappedSVRG, "nemos", default=True)
register("ProxSVRG", WrappedProxSVRG, "nemos", default=True)