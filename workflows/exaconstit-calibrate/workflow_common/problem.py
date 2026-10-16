"""
The :class:`Problem` orchestrator — a thin replacement for the legacy
:class:`ExaProb` class.

Why this module exists
----------------------
:class:`ExaProb` did seven things in one class: directory management,
input-file rendering, property-file writing, job submission, output
validation, result reading, error-metric computation, and failure
handling. That monolith was testable only by running full
simulations, impossible to reuse with a different simulation code,
and fragile to modify because every change risked subtle interactions
with the others.

This module wires the decomposed pieces together into a minimal
orchestrator. :class:`Problem` does no domain work of its own —
every step is delegated to a purpose-built component from the
surrounding modules:

* :mod:`case_setup` for rendering inputs and writing property files
* :mod:`paths` and :mod:`results` for resolving and cleaning up cases
* :mod:`backends` for running simulations
* :mod:`sentinel` and :mod:`manifest` for crash-safe state
* :mod:`results` for reading simulation output
* :mod:`objectives` for scoring results and handling failures

The payoff: every one of those components can be unit-tested in
isolation, swapped out for a different implementation without
touching the orchestrator, and documented independently.

What a :class:`Problem` represents
----------------------------------
One :class:`Problem` represents a whole multi-objective optimization
problem: for each gene the optimizer proposes, the problem runs
N simulations (one per objective), reads them back, and returns
N scalar error values. The N is set by the ``objective_specs`` list
handed to the constructor. Multi-objective is the general case;
single-objective is the N=1 special case handled uniformly.

Why each objective gets its own simulation
------------------------------------------
In realistic material-calibration workflows, different objectives
correspond to different loading conditions (e.g. tension at 298 K
vs. shear at 77 K). Each requires its own simulation run with its
own options file. Conflating them by running one sim and computing
multiple errors from it would only work for objectives that are
linear functions of the same output — a narrow special case.
The :class:`Problem` design makes N sims per gene the default and
trusts the backend to parallelize them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from .case_setup import CaseTemplater, PropertyWriter
from .backends.base import JobBackend, JobOutcome, JobResult, SimJobSpec
from .logging_utils import get_logger
from .manifest import CaseState, Manifest, ManifestEntry
from .objectives import FailureHandler, ObjectiveEvaluator
from .paths import CaseContext, PathResolver
from .results import CaseLayout, CaseResultSet, ResultReader
from .sentinel import (
    Sentinel,
    is_case_complete,
    read_sentinel,
    validate_outputs,
    write_sentinel,
)
from .templates import UnresolvedPlaceholderError

logger = get_logger(__name__)


# --- Configuration --------------------------------------------------------


@dataclass(frozen=True)
class ProblemConfig:
    """Static per-problem settings (ones that do not vary per gene or obj).

    Bundled into one record because :class:`Problem` has a lot of
    small knobs that rarely change and stuffing them into the
    constructor signature as individual kwargs would be noisy.

    Fields:
        binary: Path to the simulation executable. Passed through
            to each :class:`SimJobSpec`.
        binary_args: Command-line args appended after ``binary``.
            Default ``("-opt", "options.toml")`` matches ExaConstit.
        num_nodes: Nodes to request per simulation. Default 1.
        num_tasks: MPI ranks per simulation. Default 1.
        cores_per_task: Cores per rank. Default 1.
        gpus_per_task: GPUs per rank. Default 0.
        duration_s: Per-simulation wall-time budget in seconds.
            Default 3600 (one hour). Set higher for long sims; the
            backend enforces it as a kill-after deadline.
        stdout: Per-simulation stdout capture filename, relative to
            the case's working directory. Default ``"stdout.log"``.
            Set to ``None`` to discard.
        stderr: Per-simulation stderr capture filename. Default
            ``"stderr.log"``. Set to ``None`` to discard.
        required_outputs: Sequence of relative paths
            (e.g. ``("avg_stress.txt",)`` or
            ``("results/avg_stress.txt",)``) that must exist and be
            nonempty in each case's working directory for the case
            to count as successful. The :class:`Problem` validates
            these after the sim runs; if any are missing the case
            is marked FAILED regardless of the rc. Default empty
            (no post-validation).

            **Paths are treated as relative to each case's working
            directory.** The framework joins each entry with the
            resolved working-dir path at validation time. Absolute
            paths, ``..`` parent references, and ``{...}``
            placeholder syntax are rejected at construction time
            because they produce nonsense or path-doubling when
            joined. If your output file truly lives outside the
            case working directory, mark it ``required=True`` in
            the :class:`reader` spec — the reader knows the real
            paths via the resolver's ``output_file_patterns`` and
            validates them correctly without this footgun.
        skip_completed: If True (default), cases whose working
            directory already has a valid sentinel are not re-run.
            This is how restart works. Set False to force re-run
            every case (e.g. after a code change).
        clear_outputs_on_rerun: If True, delete any pre-existing
            output files before launching a rerun. Prevents
            confusion when a partial previous run left stale data.
            Default True. Set False if your simulation resumes
            from checkpoint files.
    """

    binary: Path
    binary_args: Tuple[str, ...] = ("-opt", "options.toml")
    num_nodes: int = 1
    num_tasks: int = 1
    cores_per_task: int = 1
    gpus_per_task: int = 0
    duration_s: int = 3600
    stdout: Optional[str] = "stdout.log"
    stderr: Optional[str] = "stderr.log"
    required_outputs: Tuple[str, ...] = ()
    skip_completed: bool = True
    clear_outputs_on_rerun: bool = True

    def __post_init__(self):
        """Validate ``required_outputs`` entries at construction.

        Historically the framework silently joined every entry with
        the case's working directory, producing paths like
        ``<workdir>/<user_supplied_path>``. If a user passed an
        absolute path, or a path that already included the working
        dir prefix, or a path using ``{working_dir}`` placeholder
        syntax (confusing this field with the resolver's richer
        patterns), the join produced the doubled
        ``<workdir>/<workdir>/...`` pattern — a silent correctness
        hazard that took a full run to surface. Reject those shapes
        at construction so the failure mode is loud and immediate.

        Multi-segment paths like ``"results/avg_stress.txt"`` are
        fine — binaries that write into subdirectories of their
        working dir are common and legitimate.
        """
        for i, entry in enumerate(self.required_outputs):
            if not isinstance(entry, str):
                raise TypeError(
                    f"required_outputs[{i}]: expected str, "
                    f"got {type(entry).__name__}"
                )
            if Path(entry).is_absolute():
                raise ValueError(
                    f"required_outputs[{i}]={entry!r}: must be "
                    f"relative to the case's working directory, "
                    f"not absolute. The framework joins each entry "
                    f"with the working dir; an absolute path here "
                    f"produces nonsense. If you need to validate a "
                    f"file at an absolute location, use the reader "
                    f"instead."
                )
            if ".." in Path(entry).parts:
                raise ValueError(
                    f"required_outputs[{i}]={entry!r}: parent-"
                    f"directory references ('..') are not "
                    f"supported. Files must live inside the case's "
                    f"working directory."
                )
            if "{" in entry or "}" in entry:
                raise ValueError(
                    f"required_outputs[{i}]={entry!r}: contains "
                    f"'{{' or '}}'. Placeholder syntax is supported "
                    f"by the resolver's output_file_patterns, not "
                    f"here. Use a plain relative path like "
                    f"'avg_stress.txt' or 'results/avg_stress.txt', "
                    f"or migrate the check to the reader where "
                    f"placeholders work."
                )


@dataclass
class SimCase:
    """One simulation unit. Zero or more objectives evaluate its output.

    A :class:`SimCase` is everything that distinguishes one
    simulation run from another: its per-case data overlay
    (strain rate, temperature, boundary conditions, RVE name —
    anything case-specific that's not part of the gene), its
    label for logging, and any per-case resource overrides.
    Multiple :class:`ObjectiveSpec` s may reference the same
    SimCase by index, in which case ONE simulation is executed
    and each objective evaluates against the shared
    :class:`CaseResultSet`.

    This is the right abstraction because different objectives
    often come from the same loading case:

    * Stress-strain RMSE AND slope-matching RMSE both come from one
      uniaxial tension sim.
    * Peak-stress error AND strain-to-failure error both come from
      one load-to-failure sim.
    * Voce-yield error, Voce-hardening error, AND Voce-saturation
      error all come from one fitting sim if you decompose the
      stress-strain residual into multiple objectives.

    Making sim count independent of objective count avoids spurious
    re-runs of identical simulations.

    Fields:
        case_data: One mapping for ALL per-case data. Visible to
            three places that need per-case context:

            * the **templater** for ``%%key%%`` substitution into
              rendered files (master_options.toml etc.)
            * the **path resolver** for ``{key}`` substitution in
              ``working_dir_pattern`` and ``output_file_patterns``
            * the **property writer** via ``sim_case.case_data``
              (the ``CallablePropertyWriter`` callable receives
              the whole SimCase)

            Typical contents: ``temperature_k``, ``strain_rate``,
            ``ori_file``, ``rve_name`` — anything specific to this
            experiment that the gene doesn't carry. The data is the
            same regardless of who reads it; one dict serves all
            three consumers.

            Values that aren't referenced by any template or path
            pattern are silently ignored by those consumers but
            remain available to the property writer. This means
            it's safe to put any helper data here that only the
            writer needs (e.g. lookup keys for an elastic-constants
            table).
        label: Optional short human-readable label for logging and
            for the backend's job tag. Default falls back to
            ``"sc{index}"``.
        num_nodes: Per-case override of
            :attr:`ProblemConfig.num_nodes`. ``None`` (default)
            inherits from the ProblemConfig. Useful when different
            experiments need different node allocations — e.g.
            a high-rate dynamic case needs 4 nodes but a
            quasi-static case only needs 1.
        num_tasks: Per-case override of
            :attr:`ProblemConfig.num_tasks` (MPI ranks).
        cores_per_task: Per-case override of
            :attr:`ProblemConfig.cores_per_task`.
        gpus_per_task: Per-case override of
            :attr:`ProblemConfig.gpus_per_task`. Set to 0 to run
            the case on CPU while other cases run on GPU; flux
            handles this automatically.
        duration_s: Per-case wall-time override. Expensive loading
            conditions can ask for more time without bloating the
            timeout for everything.
        binary: Per-case override of :attr:`ProblemConfig.binary`.
            Rarely needed, but useful if one experiment uses a
            different simulation code entirely.
        binary_args: Per-case override of
            :attr:`ProblemConfig.binary_args`. Useful for passing
            experiment-specific command-line flags.

    Example:
        Two loading conditions, three objectives total - the
        quasi-static sim is scored by BOTH stress-RMSE AND
        slope-RMSE, while the dynamic sim is scored only by
        stress-RMSE::

            sim_cases = [
                SimCase(
                    case_data={"strain_rate": 1e-3, "temperature_k": 298.0},
                    label="quasi_static",
                ),
                SimCase(
                    case_data={"strain_rate": 1e1, "temperature_k": 298.0},
                    label="dynamic",
                    num_nodes=4,        # override just for this case
                    duration_s=7200,
                ),
            ]
            objective_specs = [
                ObjectiveSpec(stress_evaluator_qs, sim_case=0, label="stress_qs"),
                ObjectiveSpec(slope_evaluator_qs,  sim_case=0, label="slope_qs"),
                ObjectiveSpec(stress_evaluator_dyn, sim_case=1, label="stress_dyn"),
            ]

        Per gene: 2 sims are run, 3 errors are returned.

        Per-case constants — including material values that vary
        across cases (e.g. elastic constants differing with
        temperature) — go directly in ``case_data``. There is no
        Python lookup table to maintain in the writer; the data
        IS the dict::

            sim_cases = [
                SimCase(case_data={
                    "temperature_k": 298.0,
                    "c11": 168.4, "c12": 121.4, "c44": 75.4,
                    # ... loading conditions etc ...
                }, label="cold"),
                SimCase(case_data={
                    "temperature_k": 600.0,
                    "c11": 156.0, "c12": 117.0, "c44": 72.0,
                    # ...
                }, label="hot"),
            ]

        Inside a property-writer callable, just read the keys::

            def write_properties(case_dir, gene, names, sim_case):
                d = sim_case.case_data
                # ... write properties.txt using d['c11'], etc ...

        Or if your sim reads its constants from a TOML/INI file,
        put the placeholders ``%%c11%%`` etc. directly in the
        master template — the templater fills them from
        ``case_data`` automatically with no writer code at all.
    """

    case_data: Mapping[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    # Optional per-case resource overrides. None -> inherit from ProblemConfig.
    num_nodes: Optional[int] = None
    num_tasks: Optional[int] = None
    cores_per_task: Optional[int] = None
    gpus_per_task: Optional[int] = None
    duration_s: Optional[int] = None
    binary: Optional[Path] = None
    binary_args: Optional[Sequence[str]] = None


@dataclass
class ObjectiveSpec:
    """One objective - an evaluator that scores a SimCase's output.

    The objective count determines the dimensionality of the
    optimizer's objective vector. The simulation count is
    determined separately by the number of :class:`SimCase` s.
    Multiple objectives may evaluate the same SimCase; a single
    objective always evaluates exactly one SimCase.

    Fields:
        evaluator: The :class:`ObjectiveEvaluator` used to score
            this objective. The experimental data and error-metric
            details live inside the evaluator.
        sim_case: Index into the :class:`Problem` 's ``sim_cases``
            list. The default (0) is correct for single-SimCase
            setups; multi-SimCase setups should set this explicitly.
        label: Optional short human-readable label for logging and
            result identification. Default falls back to
            ``"obj{index}"`` at use time.

    Example:
        Three objectives, two sim cases (see :class:`SimCase` for
        the full picture)::

            ObjectiveSpec(evaluator=e1, sim_case=0, label="stress_qs")
            ObjectiveSpec(evaluator=e2, sim_case=0, label="slope_qs")
            ObjectiveSpec(evaluator=e3, sim_case=1, label="stress_dyn")
    """

    evaluator: ObjectiveEvaluator
    sim_case: int = 0
    label: Optional[str] = None


# --- Problem -------------------------------------------------------------


class Problem:
    """Orchestrates per-gene evaluation across N objectives and M sim cases.

    A :class:`Problem` is the single object an optimizer interacts
    with. It exposes exactly two public methods:

    * :meth:`evaluate_gene` — run all M sim cases for one gene,
      evaluate all N objectives against the resulting outputs,
      return a list of N error values.
    * :meth:`evaluate_population` — run a batch of genes (each
      across M sim cases), evaluate all N objectives for each,
      return a list of per-gene error lists.

    The distinction between sim cases and objectives matters: one
    sim case can feed multiple objectives (e.g. stress RMSE AND
    slope RMSE from one uniaxial sim), so the sim count is NOT
    always equal to the objective count. See :class:`SimCase`.

    Everything else is handled internally by delegating to the
    framework components.

    Args:
        config: :class:`ProblemConfig` with static per-problem
            settings (binary path, resource request, timeouts).
        param_names: Names of the genes. Length must match each
            gene vector passed to :meth:`evaluate_gene`. Used for
            property file rendering.
        sim_cases: One :class:`SimCase` per distinct simulation
            that must run per gene. Length determines the M of
            "M sims per gene". Must be non-empty.
        objective_specs: One :class:`ObjectiveSpec` per objective.
            Length determines the N of the multi-objective run.
            Each spec's ``sim_case`` field must be a valid index
            into ``sim_cases``. Must be non-empty.
        templater: :class:`CaseTemplater` that renders input files
            per case.
        property_writer: :class:`PropertyWriter` that writes the
            gene vector to disk per case.
        resolver: :class:`PathResolver` mapping case contexts to
            directories and output file paths. Its path patterns
            should reference the ``obj`` coordinate to differentiate
            sim-case directories (despite the name, ``obj`` in
            CaseContext is now the sim-case index — kept for
            backward compatibility with existing path patterns).
        backend: :class:`JobBackend` to launch simulations.
        reader: :class:`ResultReader` for parsing output files.
        failure_handler: :class:`FailureHandler` policy for failed
            cases. Defaults to :class:`InfinityFailureHandler`.
        manifest: :class:`Manifest` for state tracking. Optional;
            if not supplied a new one is created at
            ``<resolver_root>/manifest.jsonl``.

    Example:
        Three-objective, two-sim-case problem (stress and slope
        from quasi-static, stress from dynamic)::

            problem = Problem(
                config=ProblemConfig(binary=Path("mechanics")),
                param_names=["yield_stress", "hardening"],
                sim_cases=[
                    SimCase(case_data={"strain_rate": 1e-3},
                            label="quasi"),
                    SimCase(case_data={"strain_rate": 1e1},
                            label="dynamic"),
                ],
                objective_specs=[
                    ObjectiveSpec(stress_eval_qs,  sim_case=0),
                    ObjectiveSpec(slope_eval_qs,   sim_case=0),
                    ObjectiveSpec(stress_eval_dyn, sim_case=1),
                ],
                templater=CaseTemplater([TemplateTarget(...)]),
                property_writer=TemplatePropertyWriter(...),
                resolver=TemplatePathResolver(...),
                backend=LocalBackend(max_workers=4),
                reader=TextTableReader({...}),
                manifest=Manifest("opt/manifest.jsonl"),
            )

            errs = problem.evaluate_gene(
                np.array([210.0, 1900.0]),
                generation=0,
                gene_idx=0,
            )
            assert len(errs) == 3  # three objectives, not three sims
    """

    def __init__(
        self,
        *,
        config: ProblemConfig,
        param_names: Sequence[str],
        sim_cases: Sequence[SimCase],
        objective_specs: Sequence[ObjectiveSpec],
        templater: CaseTemplater,
        property_writer: PropertyWriter,
        resolver: PathResolver,
        backend: JobBackend,
        reader: ResultReader,
        failure_handler: Optional[FailureHandler] = None,
        manifest: Optional[Manifest] = None,
        archive: Optional["ArchiveDB"] = None,
        archive_run_id: Optional[str] = None,
    ):
        if not sim_cases:
            raise ValueError("Problem requires at least one SimCase")
        if not objective_specs:
            raise ValueError("Problem requires at least one ObjectiveSpec")
        if not param_names:
            raise ValueError("Problem requires at least one param_name")

        # Validate the sim_case references in each objective spec.
        # An out-of-bounds index is a configuration error we would
        # rather catch at construction than on the first gene.
        n_sims = len(sim_cases)
        for i, spec in enumerate(objective_specs):
            if not (0 <= spec.sim_case < n_sims):
                raise ValueError(
                    f"ObjectiveSpec #{i} references sim_case={spec.sim_case} "
                    f"but only {n_sims} SimCase(s) were provided "
                    f"(valid indices: 0..{n_sims - 1})"
                )

        # An archive may come without a run_id at construction time
        # when the caller is about to enter the driver's resume
        # path: the pickle holds the run_id, and ``run_nsga3`` will
        # assign ``problem.archive_run_id`` from the pickle BEFORE
        # any archive writes happen. So we accept the orphan case
        # here and verify at write time instead
        # (see :meth:`_archive_case_outputs`).
        #
        # The inverse — a run_id without an archive — is still a
        # user error: there's nowhere to write to. Caught below.
        if archive is None and archive_run_id is not None:
            raise ValueError(
                "archive_run_id=... requires archive=... (you "
                "supplied a run_id but no archive to write to)"
            )

        self.config = config
        self.param_names = tuple(param_names)
        self.sim_cases = tuple(sim_cases)
        self.objective_specs = tuple(objective_specs)
        self.templater = templater
        self.property_writer = property_writer
        self.resolver = resolver
        self.backend = backend
        self.reader = reader
        self.archive = archive
        self.archive_run_id = archive_run_id

        # Per-SimCase cache for the extractor used to archive
        # full-range (independent, dependent) curves. Populated
        # lazily by ``_archive_case`` on first hit. We cache because
        # the same extractor is reused across every gene; a missing
        # entry (None) records "no extractor available for this
        # SimCase, skip curve archiving from now on."
        self._curve_extractor_cache: Dict[int, "Optional[Any]"] = {}
        # Parallel cache for the experimental DataFrame paired with
        # each SimCase. Used only by ``_archive_case_curve`` for
        # defensive sign-matching of the simulated curve before
        # storage. Pulled from the same evaluator that supplied the
        # extractor (first ObjectiveSpec on this SimCase exposing
        # both ``.extractor`` and ``.experimental``).
        self._curve_experimental_cache: Dict[
            int, "Optional[Any]"
        ] = {}

        # Lazy-default FailureHandler. Importing here rather than at
        # module top avoids circular imports - objectives.py may grow
        # additional dependencies on modules that import problem.py.
        if failure_handler is None:
            from .objectives import InfinityFailureHandler

            failure_handler = InfinityFailureHandler()
        self.failure_handler = failure_handler

        # Default manifest lives at a well-known path under the
        # resolver's root (if the resolver has one). If not, under cwd.
        if manifest is None:
            root = getattr(resolver, "_root", None) or Path.cwd()
            manifest = Manifest(Path(root) / "manifest.jsonl")
        self.manifest = manifest

    @property
    def n_objectives(self) -> int:
        """Number of objectives this problem evaluates per gene."""
        return len(self.objective_specs)

    @property
    def n_sim_cases(self) -> int:
        """Number of distinct simulations this problem runs per gene."""
        return len(self.sim_cases)

    def cleanup_generation_dirs(
        self,
        gen_idx: int,
        pop_size: int,
    ) -> int:
        """Delete every case directory belonging to one generation.

        Used by drivers implementing the rolling-cleanup pattern:
        at the end of generation ``N``, delete the dirs from
        generation ``N - 2``. Keeping the ``N - 1`` dirs around
        gives a one-generation safety margin — if generation ``N``
        crashes mid-evaluation, the previous generation's outputs
        are still on disk for debugging. This mitigates the
        "millions of small files" problem on networked filesystems
        without risking the most-recent output.

        The per-case sentinel and manifest entries are left on
        disk after deletion: sentinel files live inside the case
        directory and go with it, manifest entries are JSONL lines
        that never referenced the deleted data.

        Args:
            gen_idx: Generation whose case directories to delete.
            pop_size: Population size at that generation. Needed
                because the Problem does not persistently know how
                big each generation was; the driver (which does
                know) passes it in.

        Returns:
            Number of directories actually removed. Missing
            directories are silently skipped so this is safe to
            call twice.

        Example:
            Driver-side rolling-cleanup, invoked right before
            starting generation ``gen``::

                if gen >= 2:
                    problem.cleanup_generation_dirs(
                        gen_idx=gen - 2, pop_size=pop_size,
                    )
        """
        import shutil

        removed = 0
        for gene_idx in range(pop_size):
            for sc_idx in range(self.n_sim_cases):
                ctx = CaseContext(
                    generation=gen_idx, gene=gene_idx, obj=sc_idx,
                )
                layout = CaseLayout(ctx=ctx, resolver=self.resolver)
                wd = layout.working_dir
                if wd.is_dir():
                    # ignore_errors=True swallows permission / NFS
                    # flakes. The archive already has the data we
                    # care about; an unremovable dir is a nuisance
                    # but not a correctness issue.
                    shutil.rmtree(wd, ignore_errors=True)
                    if not wd.exists():
                        removed += 1
        logger.info(
            "cleanup: removed %d case directories from generation %d",
            removed, gen_idx,
        )
        return removed

    # --- Public API ------------------------------------------------------

    def evaluate_gene(
        self,
        gene: Sequence[float],
        *,
        generation: int,
        gene_idx: int,
    ) -> List[float]:
        """Run every sim case for one gene, score all N objectives.

        Pipeline, per gene:

        1. For each :class:`SimCase`, either run the sim or use an
           already-completed run (restart path).
        2. For each :class:`ObjectiveSpec`, look up its SimCase's
           result and call the evaluator. If the SimCase failed or
           the evaluator raises, route through the failure handler.
        3. Return the list of N error values in ``objective_specs``
           order.

        Failures at any step are routed through the configured
        :class:`FailureHandler`. :meth:`evaluate_gene` never raises
        on a per-case failure — the optimizer always gets N floats.
        Unrecoverable infrastructure failures (backend cannot submit
        at all) do propagate.

        Args:
            gene: Parameter vector. Length must match
                ``len(self.param_names)``.
            generation: GA generation index.
            gene_idx: Index of this gene within the generation.

        Returns:
            A list of ``len(self.objective_specs)`` float error
            values in the order the objectives were configured.
        """
        if len(gene) != len(self.param_names):
            raise ValueError(
                f"gene has {len(gene)} entries but param_names has "
                f"{len(self.param_names)}: {list(self.param_names)}"
            )

        # Run every sim case; collect (ctx, results, failure_reason)
        # per sim_case index. Results is None if the sim or read
        # failed entirely; failure_reason is None on success.
        sim_runs: Dict[int, Tuple[CaseContext, Optional[CaseResultSet], Optional[str]]] = {}
        for sc_idx, sim_case in enumerate(self.sim_cases):
            ctx = CaseContext(
                generation=generation,
                gene=gene_idx,
                obj=sc_idx,  # 'obj' in CaseContext == sim_case index
                extra=dict(sim_case.case_data),
                sim_case=sim_case,
            )
            rs, reason = self._run_sim_case(gene, ctx, sim_case)
            sim_runs[sc_idx] = (ctx, rs, reason)

        # Score every objective. Each looks up its SimCase's results
        # and runs its evaluator.
        return self._score_objectives(sim_runs)

    def evaluate_population(
        self,
        genes: Sequence[Sequence[float]],
        *,
        generation: int,
        progress: Optional[Any] = None,
    ) -> List[List[float]]:
        """Evaluate a whole population with one backend batch submission.

        Submits ``len(genes) * n_sim_cases`` jobs to the backend in
        one call so the backend can parallelize them. After results
        stream back, each objective is scored against its sim case's
        output.

        Args:
            genes: Sequence of gene vectors. Each must have length
                ``len(self.param_names)``.
            generation: GA generation index.
            progress: Optional progress reporter with a ``tick()``
                method. Called once per completed simulation (not
                per gene — a gene with three SimCases produces
                three ticks). Set to ``None`` (default) to disable
                progress reporting entirely.

        Returns:
            A list with one entry per gene, in submission order.
            Each entry is a list of N objective values in
            objective-spec order.
        """
        for i, g in enumerate(genes):
            if len(g) != len(self.param_names):
                raise ValueError(
                    f"gene #{i} has {len(g)} entries but param_names has "
                    f"{len(self.param_names)}"
                )

        # Stage 1: prepare every sim-case that needs to run.
        #
        # Bookkeeping: for each (gene_idx, sim_case_idx) pair we want
        # to end up with a (CaseContext, Optional[CaseResultSet],
        # Optional[failure_reason]) triple. We populate it here from
        # the skip path (sentinel present) and from backend results
        # in stage 2.

        # sim_runs[gene_idx][sim_case_idx] -> (ctx, results, reason)
        sim_runs: List[Dict[int, Tuple[CaseContext, Optional[CaseResultSet], Optional[str]]]] = [
            {} for _ in genes
        ]

        class _Pending:
            __slots__ = ("ctx", "sim_spec", "sim_case", "layout", "gene_idx", "sc_idx")

            def __init__(self, ctx, sim_spec, sim_case, layout, gene_idx, sc_idx):
                self.ctx = ctx
                self.sim_spec = sim_spec
                self.sim_case = sim_case
                self.layout = layout
                self.gene_idx = gene_idx
                self.sc_idx = sc_idx

        pending: List[_Pending] = []

        for gene_idx, gene in enumerate(genes):
            for sc_idx, sim_case in enumerate(self.sim_cases):
                ctx = CaseContext(
                    generation=generation,
                    gene=gene_idx,
                    obj=sc_idx,
                    extra=dict(sim_case.case_data),
                    sim_case=sim_case,
                )
                layout = CaseLayout(ctx=ctx, resolver=self.resolver)

                # Skip path: sentinel present + skip_completed True.
                if self.config.skip_completed and is_case_complete(
                    layout.working_dir
                ):
                    sen = read_sentinel(layout.working_dir)
                    rs, reason = self._load_existing_results(layout, sen, ctx)
                    sim_runs[gene_idx][sc_idx] = (ctx, rs, reason)
                    continue

                # Prepare-and-submit path.
                if self.config.clear_outputs_on_rerun:
                    layout.clear_outputs()
                sim_spec = self._prepare_case(gene, ctx, sim_case, layout)
                pending.append(_Pending(
                    ctx, sim_spec, sim_case, layout, gene_idx, sc_idx,
                ))
                self.manifest.record(ManifestEntry(
                    generation=ctx.generation,
                    gene=ctx.gene,
                    obj=ctx.obj,
                    state=CaseState.SUBMITTED,
                    case_dir=str(layout.working_dir),
                ))

        # Stage 2: submit the pending batch to the backend.
        if pending:
            pending_by_id = {id(p.sim_spec): p for p in pending}
            specs = [p.sim_spec for p in pending]
            for result in self.backend.stream_batch(specs):
                p = pending_by_id[id(result.spec)]
                rs, reason = self._handle_backend_result(
                    result, p.ctx, p.layout,
                )
                sim_runs[p.gene_idx][p.sc_idx] = (p.ctx, rs, reason)
                # One tick per completed simulation. A gene with
                # multiple SimCases produces multiple ticks; this is
                # the granularity the user actually cares about when
                # watching a long run.
                if progress is not None:
                    progress.tick()

        # Cases skipped via the `skip_completed` fast path in Stage 1
        # don't pass through stream_batch, but from the progress
        # reporter's point of view they are completed. Account for
        # them so the bar doesn't lie about the denominator.
        if progress is not None:
            total_cases = len(genes) * self.n_sim_cases
            skipped = total_cases - len(pending)
            for _ in range(skipped):
                progress.tick()

        # Stage 3: score every objective for every gene.
        errors: List[List[float]] = []
        for gene_idx in range(len(genes)):
            # Guard against any sim_case that somehow didn't get
            # populated; indicates a logic bug, not a user error.
            for sc_idx in range(self.n_sim_cases):
                if sc_idx not in sim_runs[gene_idx]:
                    raise RuntimeError(
                        f"internal: gene {gene_idx} sim_case {sc_idx} "
                        f"was not populated"
                    )
            errors.append(self._score_objectives(sim_runs[gene_idx]))
        return errors

    # --- Internals -------------------------------------------------------

    def _run_sim_case(
        self,
        gene: Sequence[float],
        ctx: CaseContext,
        sim_case: SimCase,
    ) -> Tuple[Optional[CaseResultSet], Optional[str]]:
        """Run one SimCase end-to-end.

        Returns:
            ``(CaseResultSet, None)`` on success,
            ``(None_or_partial, failure_reason_str)`` on failure.
            The partial result (if any) is supplied to the failure
            handler downstream so progress-aware handlers can use it.
        """
        layout = CaseLayout(ctx=ctx, resolver=self.resolver)

        # Restart path: sentinel already present and skip=True.
        if self.config.skip_completed and is_case_complete(layout.working_dir):
            sen = read_sentinel(layout.working_dir)
            return self._load_existing_results(layout, sen, ctx)

        if self.config.clear_outputs_on_rerun:
            layout.clear_outputs()

        sim_spec = self._prepare_case(gene, ctx, sim_case, layout)
        self.manifest.record(ManifestEntry(
            generation=ctx.generation,
            gene=ctx.gene,
            obj=ctx.obj,
            state=CaseState.SUBMITTED,
            case_dir=str(layout.working_dir),
        ))

        result = self.backend.submit_one(sim_spec)
        return self._handle_backend_result(result, ctx, layout)

    def _prepare_case(
        self,
        gene: Sequence[float],
        ctx: CaseContext,
        sim_case: SimCase,
        layout: CaseLayout,
    ) -> SimJobSpec:
        """Render inputs/properties for one SimCase and build a SimJobSpec."""
        layout.working_dir.mkdir(parents=True, exist_ok=True)

        # Build the templater values. SimCase supplies the per-case
        # data overlay; context identifiers are added as conveniences.
        # Gene-derived values go through the property writer, not
        # here. The path resolver also reads sim_case.case_data via
        # ctx.extra (populated upstream); we re-merge from ctx.extra
        # here so any overrides a caller may have layered onto the
        # context before reaching this method are honored.
        values: Dict[str, Any] = dict(sim_case.case_data)
        values.setdefault("generation", ctx.generation)
        values.setdefault("gen", ctx.generation)
        values.setdefault("gene", ctx.gene)
        values.setdefault("obj", ctx.obj)
        values.setdefault("sim_case", ctx.obj)  # semantic alias
        for k, v in ctx.extra.items():
            values.setdefault(k, v)

        # Render templates. If the master template references a
        # placeholder that isn't in case_data, raise a message that
        # tells the user exactly where to add it. The bare
        # UnresolvedPlaceholderError from render_template would
        # mention the key and the available-keys list, but won't
        # say "case_data" — and that's the field a user would edit
        # to fix it. Re-raise with the pointer.
        try:
            self.templater.render(layout, values)
        except UnresolvedPlaceholderError as e:
            # The original message format is:
            #   Template placeholder %%key%% has no value in the
            #   substitution mapping (available keys: [...])
            # We append a one-line hint without losing the original
            # detail. e.args[0] is the formatted message string.
            orig = e.args[0] if e.args else str(e)
            raise UnresolvedPlaceholderError(
                f"{orig}\n"
                f"Hint: the template placeholder is filled from "
                f"the SimCase's case_data dict. Add the missing "
                f"key to case_data on SimCase "
                f"label={sim_case.label!r} (sim_case index "
                f"{ctx.obj}) and re-run."
            ) from e
        self.property_writer.write(layout, gene, self.param_names)

        label = sim_case.label or f"sc{ctx.obj}"
        # Per-case overrides fall through to ProblemConfig when None.
        # Centralizing this "inherit-or-override" logic here keeps
        # SimCase's resource fields strictly optional at the call
        # site and gives users one documented place to look when a
        # setting doesn't land where they expected.
        def _or(per_case, default):
            return per_case if per_case is not None else default

        return SimJobSpec(
            working_dir=layout.working_dir,
            binary=_or(sim_case.binary, self.config.binary),
            args=tuple(
                _or(sim_case.binary_args, self.config.binary_args)
            ),
            num_nodes=_or(sim_case.num_nodes, self.config.num_nodes),
            num_tasks=_or(sim_case.num_tasks, self.config.num_tasks),
            cores_per_task=_or(sim_case.cores_per_task, self.config.cores_per_task),
            gpus_per_task=_or(sim_case.gpus_per_task, self.config.gpus_per_task),
            duration_s=_or(sim_case.duration_s, self.config.duration_s),
            stdout=self.config.stdout,
            stderr=self.config.stderr,
            tag=f"gen{ctx.generation}_g{ctx.gene}_{label}",
        )

    def _handle_backend_result(
        self,
        result: JobResult,
        ctx: CaseContext,
        layout: CaseLayout,
    ) -> Tuple[Optional[CaseResultSet], Optional[str]]:
        """Finalize a backend JobResult: validate, persist state, read output.

        Returns:
            ``(CaseResultSet, None)`` on success; ``(partial, reason)``
            on failure (partial may be None if nothing is readable).
        """
        # Validate required output files. validate_outputs resolves
        # each relative entry against the case's working directory
        # internally — do NOT pre-join here. The old code did the
        # pre-join AND validate_outputs did the join, producing
        # double-joined paths like `<workdir>/<workdir>/...` whenever
        # the working dir itself was relative (which it normally is
        # when the user's WORKSPACE is relative, e.g.
        # `Path("./calibration_run")`). Trust validate_outputs to
        # handle the join; its docstring is explicit that relative
        # entries are resolved against `case_dir`.
        ok_files, bad = validate_outputs(
            layout.working_dir, list(self.config.required_outputs),
        )
        ok = result.outcome == JobOutcome.OK and ok_files

        terminal = CaseState.COMPLETED if ok else CaseState.FAILED
        reason = (
            None if ok else (
                f"rc={result.rc}; outcome={result.outcome.value}; "
                f"missing_or_empty={bad or ()}"
            )
        )

        # Sentinel first, then manifest — crash-safety ordering.
        write_sentinel(
            layout.working_dir,
            Sentinel(
                rc=result.rc,
                wall_time_s=result.wall_time_s,
                jobid=result.jobid,
                output_files={
                    n: str(layout.output_file(n))
                    for n in layout.known_outputs()
                },
                status="ok" if ok else "bad",
                message=result.error_message if result.error_message else (reason or "ok"),
            ),
        )
        self.manifest.record(ManifestEntry(
            generation=ctx.generation,
            gene=ctx.gene,
            obj=ctx.obj,
            state=terminal,
            rc=result.rc,
            jobid=result.jobid,
            case_dir=str(layout.working_dir),
            message=reason,
        ))

        if ok:
            # Read the full result set. If reading itself fails, treat
            # the whole case as failed (truncated / corrupt output).
            try:
                rs = self.reader.read(layout)
            except Exception as e:
                logger.warning(
                    "reader failed on successful sim "
                    "(gen=%d gene=%d obj=%d): %s",
                    ctx.generation, ctx.gene, ctx.obj, e,
                )
                partial = self._try_partial_read(layout)
                return partial, f"reader error: {e}"
            # Archive the CaseResultSet - this is what went INTO the
            # evaluator, so post-processing has exactly the data the
            # optimizer saw. Archive failures should not break the
            # optimization; log and proceed.
            self._archive_case(ctx, rs)
            return rs, None

        # Failed sim: best-effort partial read for the handler.
        partial = self._try_partial_read(layout)
        return partial, reason

    def _archive_case(
        self,
        ctx: CaseContext,
        rs: CaseResultSet,
    ) -> None:
        """Write a successful CaseResultSet to the archive, if one is configured.

        Private helper so both the fresh-run path and the
        restart-skip path can share it. ``ctx.obj`` is the sim_case
        index because :class:`CaseContext` reuses the ``obj`` field
        for that purpose (documented in ARCHITECTURE.md).
        """
        if self.archive is None:
            return
        if self.archive_run_id is None:
            # Deferred construction-time check: at write time the
            # run_id MUST be populated. The driver's resume path
            # assigns this from the pickle before any writes
            # happen; a fresh run's caller assigned it at
            # construction. If neither happened, something is
            # wrong in the driver layer — complain loudly rather
            # than silently dropping archive writes.
            raise RuntimeError(
                "archive is set but archive_run_id is still None "
                "at archive-write time. Either the driver failed "
                "to resolve the run_id from a resume checkpoint, "
                "or the caller passed archive=... without either "
                "setting archive_run_id at construction or "
                "entering a resume path that would set it."
            )
        try:
            self.archive.record_case_outputs(
                self.archive_run_id,
                birth_gen=ctx.generation,
                birth_gene=ctx.gene,
                sim_case_idx=ctx.obj,
                results=rs,
            )
        except Exception as e:
            # Don't let a DB write failure abort the optimization.
            # The user would rather have a run that finished without
            # an archive than a run that died because the archive
            # was briefly unavailable.
            logger.warning(
                "archive write failed (gen=%d gene=%d sim_case=%d): %s",
                ctx.generation, ctx.gene, ctx.obj, e,
            )

        # Also archive the extracted (independent, dependent) curve
        # so post-run analyses can compare to experimental data, run
        # alternative metrics, or train surrogate models without
        # re-extracting from the raw simulator outputs every time.
        # The window is stripped before extraction: storage covers
        # the FULL extraction range so future analyses with
        # different windows can use the same source data.
        self._archive_case_curve(ctx, rs)

    def _archive_case_curve(
        self,
        ctx: CaseContext,
        rs: CaseResultSet,
    ) -> None:
        """Run the SimCase's extractor (window stripped) and archive the curve.

        Quietly does nothing when:
        * no extractor is available for this SimCase, or
        * extraction raises (the case probably failed in a way
          that the case_outputs alone capture). The optimizer
          itself will see the same failure via the evaluator, so
          this just declines to add a noisy duplicate to the log.

        Called once per (gene, sim_case). The chosen extractor is
        cached per ``sim_case_idx`` on the Problem; it doesn't
        change between genes.

        If experimental data is available (the same evaluator that
        supplied the extractor also has an ``.experimental``
        DataFrame), the simulated arrays are sign-matched against
        the experimental columns before archiving via
        :func:`workflow_common.objectives.match_sign_to_reference`.
        This catches extractors that accidentally absolute-value
        their output (e.g. a sign-stripped strain rate that
        produces positive strain on a compression run) — without
        the correction, downstream plots show the simulated curve
        flipped relative to the experimental reference. The fix is
        also applied at plot time as a backstop, but applying it
        here means archived data is also correct for any other
        consumer (Bayesian surrogate trainers, alternative
        post-processing scripts).
        """
        sim_case_idx = ctx.obj
        if sim_case_idx not in self._curve_extractor_cache:
            extractor, exp_df = self._resolve_case_curve_state(sim_case_idx)
            self._curve_extractor_cache[sim_case_idx] = extractor
            self._curve_experimental_cache[sim_case_idx] = exp_df
        extractor = self._curve_extractor_cache[sim_case_idx]
        if extractor is None:
            return
        try:
            independent, dependent = extractor.extract(rs)
        except Exception as e:  # noqa: BLE001
            # Logged at debug, not warning — curve archiving is a
            # bonus feature, not a primary path. The evaluator's
            # own failure handler is the place to surface
            # extraction failures to the user.
            logger.debug(
                "case_curve extraction failed gen=%d gene=%d "
                "sim_case=%d: %s",
                ctx.generation, ctx.gene, sim_case_idx, e,
            )
            return
        # Defensive sign-matching against experimental reference,
        # if available. Done BEFORE archiving so the stored curve
        # has the right sign for any downstream consumer (not just
        # the plotter). Uses the framework's PCHIP smoother to
        # align the experimental dependent values onto sim's grid,
        # then ``np.copysign`` element-wise — preserves any
        # cyclic / multi-step structure that elementwise comparison
        # would otherwise destroy.
        exp_df = self._curve_experimental_cache.get(sim_case_idx)
        if exp_df is not None:
            from .smoothing import PchipSmoother
            spec_evaluator = self._curve_evaluator_for(sim_case_idx)
            strain_col = getattr(
                spec_evaluator, "experimental_strain_col", "strain",
            )
            stress_col = getattr(
                spec_evaluator, "experimental_stress_col", "stress",
            )
            if (strain_col in exp_df.columns
                    and stress_col in exp_df.columns):
                exp_ind = exp_df[strain_col].to_numpy()
                exp_dep = exp_df[stress_col].to_numpy()
                independent = np.asarray(independent, dtype=float)
                dependent = np.asarray(dependent, dtype=float)
                try:
                    smoother = PchipSmoother(strict_monotonic=False)
                    exp_dep_at_sim = smoother.sample_at(
                        np.abs(exp_ind), exp_dep,
                        np.abs(independent),
                    ).y
                    dependent = np.copysign(dependent, exp_dep_at_sim)
                    if exp_ind.size > 0:
                        independent = np.copysign(
                            independent, exp_ind[-1],
                        )
                except (ValueError, RuntimeError) as e:
                    logger.debug(
                        "case_curve sign-match skipped on sim_case=%d "
                        "(gen=%d gene=%d): %s",
                        sim_case_idx, ctx.generation, ctx.gene, e,
                    )
        try:
            self.archive.record_case_curve(
                self.archive_run_id,
                birth_gen=ctx.generation,
                birth_gene=ctx.gene,
                sim_case_idx=sim_case_idx,
                independent=independent,
                dependent=dependent,
                independent_label="strain",
                dependent_label="stress",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "case_curve archive write failed "
                "(gen=%d gene=%d sim_case=%d): %s",
                ctx.generation, ctx.gene, sim_case_idx, e,
            )

    def _resolve_case_curve_state(
        self, sim_case_idx: int,
    ) -> "Tuple[Optional[Any], Optional[Any]]":
        """Find the (extractor, experimental_df) pair for one SimCase.

        Walks ``objective_specs`` in order and returns the FIRST
        evaluator on this SimCase that has an ``.extractor``
        attribute. The matching ``.experimental`` DataFrame from
        the same evaluator (if present) is returned alongside.
        Resolving both from the same evaluator keeps them
        consistent — a user who configured both fields with the
        same calibration intent gets matching behavior at archive
        time.

        The extractor is window-stripped via the to_dict/from_dict
        round-trip so the archived curve covers the full extraction
        range — windowing is purely an objective-scoring concern;
        downstream analyses may want to look at regions outside the
        optimization window.

        Returns ``(None, None)`` if no spec on this SimCase has an
        ``.extractor``.
        """
        for spec in self.objective_specs:
            if spec.sim_case != sim_case_idx:
                continue
            ext = getattr(spec.evaluator, "extractor", None)
            if ext is None:
                continue
            # Strip the window via to_dict/from_dict round-trip so
            # the archived curve isn't cropped. Falls back to
            # returning the original extractor if to_dict/from_dict
            # aren't available — better than nothing.
            to_dict = getattr(ext, "to_dict", None)
            from_dict = getattr(type(ext), "from_dict", None)
            if callable(to_dict) and callable(from_dict):
                try:
                    cfg = to_dict()
                    cfg = {**cfg, "window": None}
                    ext = from_dict(cfg)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "case_curve: extractor.to_dict round-trip "
                        "failed on sim_case=%d (%s); using original "
                        "extractor — archived curves may be windowed.",
                        sim_case_idx, e,
                    )
            exp_df = getattr(spec.evaluator, "experimental", None)
            return ext, exp_df
        return None, None

    def _curve_evaluator_for(self, sim_case_idx: int):
        """Return the evaluator chosen for curve archiving on this SimCase.

        Used by :meth:`_archive_case_curve` to look up
        ``experimental_strain_col`` / ``experimental_stress_col``
        attributes — column names default to "strain" / "stress"
        but a custom evaluator can override them. Returns ``None``
        if no spec on this SimCase has a usable extractor.
        """
        for spec in self.objective_specs:
            if spec.sim_case != sim_case_idx:
                continue
            if getattr(spec.evaluator, "extractor", None) is None:
                continue
            return spec.evaluator
        return None

    def _load_existing_results(
        self,
        layout: CaseLayout,
        sentinel: Optional[Sentinel],
        ctx: CaseContext,
    ) -> Tuple[Optional[CaseResultSet], Optional[str]]:
        """Load results from an already-completed case (restart path)."""
        if sentinel is not None and sentinel.status != "ok":
            partial = self._try_partial_read(layout)
            return partial, f"previous run failed (status={sentinel.status})"
        try:
            rs = self.reader.read(layout)
        except Exception as e:
            logger.warning(
                "re-read from existing sentinel failed "
                "(gen=%d gene=%d obj=%d): %s",
                ctx.generation, ctx.gene, ctx.obj, e,
            )
            partial = self._try_partial_read(layout)
            return partial, f"re-read error: {e}"
        # Archive on the skip path too: a restart that picks up cases
        # from a previous (non-archived) invocation should populate
        # the archive going forward. INSERT OR REPLACE on the archive
        # side makes this safe to do unconditionally.
        self._archive_case(ctx, rs)
        return rs, None

    def _score_objectives(
        self,
        sim_runs: Mapping[int, Tuple[CaseContext, Optional[CaseResultSet], Optional[str]]],
    ) -> List[float]:
        """Evaluate every ObjectiveSpec against its SimCase's results.

        Called once per gene after all SimCases for that gene have
        either succeeded or exhausted their recovery paths. Iterates
        through ``self.objective_specs`` in order and produces the
        final N-length error vector for the optimizer.
        """
        errors: List[float] = []
        for i, obj_spec in enumerate(self.objective_specs):
            ctx, rs, reason = sim_runs[obj_spec.sim_case]

            # SimCase failed entirely — route straight through the
            # handler. All objectives tied to this SimCase take the
            # same failure path (but each gets its own call to the
            # handler so the handler's logging shows one entry per
            # objective).
            if rs is None or reason is not None:
                err = float(self.failure_handler.on_failure(
                    ctx, reason or "sim case failed", rs,
                ))
                errors.append(err)
                continue

            # SimCase succeeded — run the evaluator. Evaluator
            # exceptions route through the failure handler with the
            # full result set as partial data, which is usually
            # plenty for progress-aware handlers.
            try:
                err = float(obj_spec.evaluator.evaluate(rs, ctx))
            except Exception as e:
                logger.warning(
                    "evaluator #%d (%s) failed on successful sim "
                    "(gen=%d gene=%d obj=%d): %s",
                    i, obj_spec.label or f"obj{i}",
                    ctx.generation, ctx.gene, ctx.obj, e,
                )
                err = float(self.failure_handler.on_failure(
                    ctx, f"evaluator error: {e}", rs,
                ))
            errors.append(err)
        return errors

    def _try_partial_read(self, layout: CaseLayout) -> Optional[CaseResultSet]:
        """Best-effort read of whatever files the case produced.

        Returns the successfully-parsed subset, or ``None`` if
        nothing could be read. Swallows all exceptions so this
        method never breaks the failure path.
        """
        try:
            return self.reader.read(layout)
        except Exception:
            return None
