"""
Optimization drivers.

Re-exports the public API of the NSGA-III driver so importing is a
one-liner::

    from workflows.optimization import run_nsga3, Bounds, RunConfig, RunResult

The driver itself requires the rcarson3 DEAP fork (``pip install
workflow-common[nsga3]``), but this package deliberately imports it
lazily so that ``import workflows.optimization`` itself does not
blow up when DEAP is unavailable. Code that doesn't actually invoke
the driver — an IDE doing autocompletion, a script inspecting
``workflows.__file__``, a setup.py discovering subpackages — keeps
working regardless.

The first attribute access (``run_nsga3``, ``Bounds``, etc.) on this
module triggers the real import and will raise the familiar
``ModuleNotFoundError: No module named 'deap'`` if the extra wasn't
installed.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Bounds",
    "RunConfig",
    "RunResult",
    "build_reference_points",
    "derive_population_size",
    "run_nsga3",
]


def __getattr__(name: str) -> Any:
    """PEP 562 module-level attribute access hook.

    Defers the driver import until the user actually reaches for one
    of its public names. This keeps ``import workflows.optimization``
    cheap and DEAP-free.
    """
    if name in __all__:
        from . import nsga3_driver  # deferred; triggers DEAP import

        return getattr(nsga3_driver, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__():
    """Advertise the lazy attributes for tooling (tab completion, IDEs)."""
    return sorted(list(globals().keys()) + list(__all__))
