"""
workflow_common
===============

Shared infrastructure for driving external simulation codes from Python
workflows such as parameter optimization, uncertainty quantification,
and parameter sweeps.

Why this package exists
-----------------------
Most computational-mechanics workflows end up writing the same four
pieces of code over and over:

1. A way to render per-case input files from a template.
2. A way to lay out working directories and know where output files
   will land afterwards.
3. A way to launch simulations (locally, via Flux, via SLURM, ...) and
   collect results back.
4. A way to survive an HPC allocation running out of time partway
   through a multi-day optimization and restart cleanly.

Each of those pieces is small in isolation but they are almost always
tangled together inside a single monolithic driver script, which makes
them hard to reuse and hard to test. ``workflow_common`` separates them
into independent, self-contained modules that cooperate through small
well-defined interfaces.

The package is intentionally code-agnostic. Nothing here knows the name
of a specific simulation binary, the format of a specific input file,
or the conventions of a specific output file. All of that is supplied
by the caller - either as a template, a path pattern, or a user-written
callback. That lets the same infrastructure drive ExaConstit today and
a completely different code tomorrow without modification.

Module map
----------
``_fs``
    Filesystem helpers: the ``cd`` context manager, atomic text writes.
``logging_utils``
    Stdlib-logging-based replacement for the old ``ExaConstit_Logger``
    plus a compatibility shim so existing call sites keep working.
``templates``
    ``%%key%%`` placeholder substitution for rendering per-case input
    files. Framework-agnostic: works on any text file.
``paths``
    ``PathResolver`` abstraction plus a default implementation driven by
    Python ``str.format`` patterns.
``manifest``
    JSONL manifest for persistent, crash-safe tracking of which cases
    have been submitted, completed, or failed. Used for restart.
``sentinel``
    Per-case ``.done`` marker files written atomically once outputs are
    validated. The authoritative signal for case completion.
``platform_detect``
    Small helpers for machine-specific quirks (Spectrum MPI hosts, etc.).
``backends``
    Job-execution abstraction. Contains the ``JobBackend`` Protocol,
    ``SimJobSpec`` / ``JobResult`` dataclasses, a local-subprocess
    backend, and a Flux backend.

Quick tour for new developers
-----------------------------
A minimal driver that runs a batch of simulations looks like::

    from workflow_common import (
        CaseContext, TemplatePathResolver, SimJobSpec, LocalBackend,
        Manifest, configure_logging, get_logger,
    )

    configure_logging(level="info", logfile="run.log")
    logger = get_logger(__name__)

    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}",
        output_file_patterns={"stress": "{working_dir}/avg_stress.txt"},
    )
    manifest = Manifest("wf/manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=4)

    specs = [
        SimJobSpec(
            working_dir=resolver.working_dir(CaseContext(0, i)),
            binary="/path/to/mechanics",
            args=("-opt", "options.toml"),
        )
        for i in range(10)
    ]
    for result in backend.stream_batch(specs):
        logger.info("case done: rc=%d", result.rc)

A more complete example that also wires the manifest and sentinel
machinery together for restartable workflows lives in
``tests/demo_workflow.py``.

For a decision table ("I need to X - which module do I reach
for?"), an overview of the layering of the package, and answers
to common design questions, see ``ARCHITECTURE.md``. For a
walk-through aimed at users of the pre-refactor code who want to
migrate existing drivers, see ``MIGRATION.md``.

Glossary for non-CS readers
---------------------------
A few terms come up repeatedly in the docs below; these are one-line
translations for readers more comfortable with mechanics than with
software engineering.

* **Protocol** - a class-like thing that only describes which methods
  an object must have (not its inheritance). Any object with matching
  methods can be used where the Protocol is expected. Similar to
  abstract base classes but without needing explicit inheritance.
* **Dataclass** - a class whose fields are declared as annotated
  attributes, with ``__init__`` and friends auto-generated. Think of
  it as a named, typed record.
* **Context manager** - an object used with a ``with`` statement that
  guarantees cleanup even if an exception is raised. ``open()`` is the
  canonical example.
* **Atomic write** - a file-writing pattern where, from the point of
  view of any other process, the file either has its new contents or
  its old contents but is never observed half-written.
* **JSONL** - "JSON Lines": a text file with one JSON object per line,
  appended to as events happen. Great for durable event logs on shared
  filesystems because each line is written as a single small append.
"""
from __future__ import annotations

from ._fs import cd, atomic_write_text, atomic_replace  # noqa: F401
from .logging_utils import get_logger, configure_logging  # noqa: F401
from .templates import render_template, render_template_file  # noqa: F401
from .paths import PathResolver, TemplatePathResolver, CaseContext  # noqa: F401
from .manifest import Manifest, ManifestEntry, CaseState  # noqa: F401
from .sentinel import write_sentinel, read_sentinel, Sentinel  # noqa: F401
from .results import (  # noqa: F401
    CaseLayout,
    CaseResultSet,
    ResultReader,
    TabularResult,
    TextTableReader,
    TextTableSpec,
    common_time_range,
    interpolate_to,
    load_experimental_csv,
)
from .smoothing import (  # noqa: F401
    ArcLengthSmoother,
    LegacyLinearSmoother,
    PchipSmoother,
    SmoothedCurve,
    Smoother,
    arc_length,
    auto_smoother,
    is_monotonic,
)
from .case_setup import (  # noqa: F401
    CallablePropertyWriter,
    CaseTemplater,
    DelimitedPropertyWriter,
    PropertyWriter,
    TemplatePropertyWriter,
    TemplateTarget,
)
from .objectives import (  # noqa: F401
    ConstantPenaltyFailureHandler,
    ERROR_METRICS,
    ErrorMetric,
    FailureHandler,
    InfinityFailureHandler,
    ObjectiveEvaluator,
    PartialProgressFailureHandler,
    StressStrainExtractor,
    StressStrainObjective,
    mae,
    max_abs_error,
    rmse,
)
from .problem import (  # noqa: F401
    ObjectiveSpec,
    Problem,
    ProblemConfig,
    SimCase,
)
from .postprocess import (  # noqa: F401
    BestSol,
    CheckpointData,
    GeneResult,
    best_solution_asf,
    best_solution_eudist,
    extract_gene_results,
    load_case_results,
    load_case_results_from_archive,
    load_checkpoint,
)
from .archive import (  # noqa: F401
    ArchiveDB,
    GeneRecord,
    GenerationSummary,
    RunSummary,
)
from .backends.base import (  # noqa: F401
    BackendStats,
    SimJobSpec,
    JobResult,
    JobBackend,
    JobOutcome,
)
from .backends.local import LocalBackend  # noqa: F401
from .progress import ProgressReporter  # noqa: F401

# Flux is optional. Users on a machine without Flux python bindings can
# still use everything else in the package; only the import of FluxBackend
# itself is guarded. HAS_FLUX is a module-level boolean so callers can
# branch on availability without catching ImportError themselves.
try:
    from .backends.flux_backend import FluxBackend  # noqa: F401

    HAS_FLUX = True
except ImportError:
    HAS_FLUX = False

__all__ = [
    # filesystem
    "cd",
    "atomic_write_text",
    "atomic_replace",
    # logging
    "get_logger",
    "configure_logging",
    # templating
    "render_template",
    "render_template_file",
    # paths
    "PathResolver",
    "TemplatePathResolver",
    "CaseContext",
    # manifest / sentinel
    "Manifest",
    "ManifestEntry",
    "CaseState",
    "write_sentinel",
    "read_sentinel",
    "Sentinel",
    # results
    "CaseLayout",
    "CaseResultSet",
    "ResultReader",
    "TabularResult",
    "TextTableReader",
    "TextTableSpec",
    "common_time_range",
    "interpolate_to",
    "load_experimental_csv",
    # smoothing
    "ArcLengthSmoother",
    "LegacyLinearSmoother",
    "PchipSmoother",
    "SmoothedCurve",
    "Smoother",
    "arc_length",
    "auto_smoother",
    "is_monotonic",
    # case setup
    "CallablePropertyWriter",
    "CaseTemplater",
    "DelimitedPropertyWriter",
    "PropertyWriter",
    "TemplatePropertyWriter",
    "TemplateTarget",
    # objectives
    "ConstantPenaltyFailureHandler",
    "ERROR_METRICS",
    "ErrorMetric",
    "FailureHandler",
    "InfinityFailureHandler",
    "ObjectiveEvaluator",
    "PartialProgressFailureHandler",
    "StressStrainExtractor",
    "StressStrainObjective",
    "mae",
    "max_abs_error",
    "rmse",
    # problem
    "ObjectiveSpec",
    "Problem",
    "ProblemConfig",
    "SimCase",
    # postprocess
    "BestSol",
    "CheckpointData",
    "GeneResult",
    "best_solution_asf",
    "best_solution_eudist",
    "extract_gene_results",
    "load_case_results",
    "load_case_results_from_archive",
    "load_checkpoint",
    # archive
    "ArchiveDB",
    "GeneRecord",
    "GenerationSummary",
    "RunSummary",
    # backends
    "BackendStats",
    "SimJobSpec",
    "JobResult",
    "JobBackend",
    "JobOutcome",
    "LocalBackend",
    "HAS_FLUX",
    "FluxBackend",
    # progress
    "ProgressReporter",
]
