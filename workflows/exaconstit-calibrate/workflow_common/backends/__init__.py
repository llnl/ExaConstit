"""
Job-execution backends.

This subpackage contains the abstraction that lets the framework
launch simulations without knowing whether they are going to run as
local subprocesses, Flux jobs, SLURM job-steps, or anything else in
the future.

The contract
------------
Every backend implements :class:`JobBackend` (defined in
:mod:`workflow_common.backends.base`). A backend accepts a sequence
of :class:`SimJobSpec` values and produces one :class:`JobResult`
per spec. The driver does not need to know how that mapping happened
internally; it just consumes results.

Choosing a backend
------------------
* :class:`LocalBackend` - run simulations as subprocesses on the
  current machine. Useful for desktops, CI, unit tests, and small
  debugging runs. Supports serial (``max_workers=1``) or threadpool-
  parallel execution. Does NOT do MPI on its own, though it can
  invoke an MPI launcher if one is provided.
* :class:`FluxBackend` - run simulations as Flux jobs inside the
  current Flux instance. Intended for real HPC runs. Requires the
  ``flux`` python bindings to be importable.

Live status
-----------
Backends expose an iterator-style ``stream_batch`` method that yields
results as cases complete, in completion order. This makes it easy to
log "case X finished" events on the fly without polling the backend's
native API. If you just want "submit everything and wait", call
``submit_batch`` instead, which returns results in submission order
and blocks until all cases finish.

Flux availability
-----------------
``FluxBackend`` is imported lazily to avoid hard-failing on machines
without flux bindings. The module-level flag ``HAS_FLUX`` is True when
the import succeeded, False when it didn't, so callers can branch on
availability explicitly::

    from workflow_common.backends import HAS_FLUX, LocalBackend
    if HAS_FLUX:
        from workflow_common.backends import FluxBackend
        backend = FluxBackend(...)
    else:
        backend = LocalBackend(max_workers=4)
"""
from __future__ import annotations

from .base import (  # noqa: F401
    BackendStats,
    JobBackend,
    JobOutcome,
    JobResult,
    SimJobSpec,
)
from .local import LocalBackend  # noqa: F401

# Flux is optional. Calling code that explicitly requires it should
# import FluxBackend from .flux_backend itself so the ImportError
# propagates with its original message.
try:
    from .flux_backend import FluxBackend  # noqa: F401

    HAS_FLUX = True
except ImportError:  # pragma: no cover
    HAS_FLUX = False


__all__ = [
    "BackendStats",
    "JobBackend",
    "JobOutcome",
    "JobResult",
    "SimJobSpec",
    "LocalBackend",
    "HAS_FLUX",
    "FluxBackend",
]
