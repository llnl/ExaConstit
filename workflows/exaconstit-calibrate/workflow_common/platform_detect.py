"""
Small machine-specific detection helpers.

These exist because some HPC systems require slightly different
scheduler configuration (an extra shell flag, a different MPI launcher
name, etc.) from others. Rather than inlining ``if "lassen" in
hostname`` checks all over the code, we gather the detection logic
here and let backends call into it.

Adding support for a new machine should be a single-file edit in this
module.
"""
from __future__ import annotations

import os

# Substrings that appear in the hostname of IBM Spectrum-MPI systems.
# These machines need flux jobs launched with the ``mpi=spectrum``
# shell option so the Spectrum MPI PMIX shim is used. Matching is
# case-insensitive and looks for substrings because real hostnames are
# things like "lassen708.llnl.gov" or "summit1234.ccs.ornl.gov" with
# trailing numbers and domain suffixes.
_SPECTRUM_HOST_SUBSTRINGS = (
    "lassen",
    "sierra",
    "ansel",
    "summit",
    "andes",
)


def is_spectrum_machine(hostname: str = None) -> bool:
    """Return True if running on an IBM Spectrum-MPI host.

    Spectrum-MPI machines require the ``mpi=spectrum`` flux shell
    option. Backends that build jobspecs can call this to decide
    whether to set that option automatically.

    Args:
        hostname: Optional explicit hostname to check. If omitted,
            the current host's nodename (as returned by
            ``os.uname().nodename``) is used. Passing an explicit
            name is useful in tests.

    Returns:
        True if the hostname contains any of the known Spectrum-MPI
        substrings (case-insensitive), False otherwise.

    Example:
        ::

            from workflow_common.platform_detect import is_spectrum_machine
            if is_spectrum_machine():
                jobspec.setattr_shell_option("mpi", "spectrum")
    """
    if hostname is None:
        hostname = os.uname().nodename
    hostname = hostname.lower()
    return any(s in hostname for s in _SPECTRUM_HOST_SUBSTRINGS)


def detect_runtime_model(has_gpus: bool) -> str:
    """Return the default ExaConstit ``Solvers.rtmodel`` string.

    ExaConstit's options file has a top-level ``rtmodel`` choice
    (``"CPU"`` or ``"CUDA"`` or ``"OPENMP"``, etc.) that selects the
    runtime backend. This helper encodes the default policy: CUDA if
    the user asked for GPUs, CPU otherwise. Workflows that want a
    different policy (e.g. force OPENMP on some nodes) should
    override this with their own logic.

    Args:
        has_gpus: Whether the caller is configuring a GPU-enabled
            case (``gpus_per_task > 0`` in the job spec).

    Returns:
        Either ``"CUDA"`` or ``"CPU"``.
    """
    return "CUDA" if has_gpus else "CPU"
