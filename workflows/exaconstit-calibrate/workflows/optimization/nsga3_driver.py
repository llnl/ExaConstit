"""
NSGA-III / U-NSGA-III driver built on the DEAP fork at
https://github.com/rcarson3/deap and the new
:class:`workflow_common.Problem` orchestrator.

Purpose
-------
This module is the modernized counterpart to the pre-refactor
``ExaConstit_NSGA3.py``. It preserves DEAP as the genetic-algorithm
library (the fork includes U-NSGA-III's niching selection, which is
not in stock DEAP and which the pre-refactor code relies on) and
swaps out only the pieces that interacted with the old
``ExaProb`` + ``normal_map`` monolith, replacing them with the
decomposed framework.

What stayed the same
--------------------
Everything the old driver used to drive the GA itself is unchanged,
so results of a converged run should match the old driver bit-for-bit
when given the same seed and the same problem definition:

* ``tools.uniform_reference_points`` — Das-Dennis reference-point
  generation with optional two-hyperplane support.
* ``tools.selNSGA3(nd="standard")`` — non-dominated sorting +
  reference-point-directed niching for environmental selection.
* ``tools.niching_selection_UNSGA3`` — the fork-only niching step
  applied before variation when ``config.unsga3=True``. Required
  for single-objective runs, recommended for two-objective runs.
* ``tools.cxSimulatedBinaryBounded(eta=mate_eta)`` — SBX crossover
  with bound clipping.
* ``tools.mutPolynomialBounded(eta=mut_eta, indpb=1/ndim)`` —
  polynomial mutation with default per-gene probability 1/ndim.
* ``algorithms.varAnd(pop, toolbox, cxpb=1, mutpb=1)`` — variation
  operator, always applies crossover AND mutation per pair.
* ``deap.benchmarks.tools.hypervolume`` — hypervolume indicator
  with unit reference point.
* Pickle-based checkpoint format with the same keys as the old
  driver (so old checkpoints can be loaded into the new driver
  for a clean mid-run migration).
* Twin logbook layout (``logbook1`` stats, ``logbook2`` solutions).
* Stopping criteria: fail_limit on consecutive sim failures, and
  stop_limit on consecutive "ND == NPOP" generations after Imin.

What changed
------------
The two-line diff at the bottom of the previous driver's
evaluation block — the call to ``toolbox.map_custom(problem, gen,
invalid_ind)`` — is replaced by a call to
``wf_problem.evaluate_population(genes, generation=gen)``. That
bridge is the only seam between DEAP and the new framework, and it
is implemented by :func:`_deap_evaluate`. Everything upstream
(gene construction, DEAP bookkeeping) and everything downstream
(sim orchestration, restart, manifest) lives in its respective
layer.

What is gone
------------
* ``toolbox.map_custom_fail`` — the retry-with-random-replacement
  path is now implemented inside :func:`run_nsga3` using
  ``toolbox.individual()`` directly. One less object to register.
* ``ind.stress`` — the pre-refactor driver stashed the raw
  simulated stress history on each DEAP individual for post-
  processing. The new framework does not carry the CaseResultSet
  through the fitness path (it is consumed by the evaluator and
  discarded). If post-processing needs the sim output, read it
  back from the case directory using the framework's
  :class:`ResultReader`; the directory is discoverable from the
  problem's resolver. This is called out in MIGRATION.md.
* ``problem.is_simulation_done(igene)`` — failure signalling now
  flows through :class:`FailureHandler`, which converts a sim
  failure into a distinguished sentinel fitness value (typically
  inf or a large penalty). The driver detects failed genes by
  comparing returned values against
  :attr:`RunConfig.failure_threshold`.

For a step-by-step port of an existing ``ExaConstit_NSGA3.py`` to
this driver, see ``MIGRATION.md``.
"""
from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass, field
from math import factorial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap import algorithms, base, creator, tools
from deap.benchmarks.tools import hypervolume

from workflow_common import Problem as WorkflowProblem
from workflow_common.logging_utils import get_logger
from workflow_common.progress import ProgressReporter

logger = get_logger(__name__)


# --- Public dataclasses -------------------------------------------------


@dataclass
class Bounds:
    """Per-gene lower and upper bounds for the optimizer.

    DEAP's SBX crossover and polynomial mutation both accept bounds
    per-gene, so the bounds are always carried as arrays matching
    the gene length. Dependent-per-experiment parameters (the old
    ``DEP_LOW`` / ``DEP_UP`` pattern) should be constructed by the
    caller by extending the per-experiment repeats before handing
    them to this dataclass; see the MIGRATION.md section
    "Gene-structure handling" for the exact construction.

    Fields:
        lower: 1-D array of lower bounds, one per gene entry.
        upper: 1-D array of upper bounds, one per gene entry.

    Raises (at construction):
        ValueError: If shapes disagree, if the arrays are not 1-D,
            or if any upper is not strictly greater than the
            corresponding lower.

    Example:
        The old driver's BOUND_LOW/BOUND_UP with no dependent
        parameters::

            bounds = Bounds(
                lower=np.array([150., 100.,  50., 1500., 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]),
                upper=np.array([200., 150., 100., 2500., 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]),
            )

        With per-experiment dependent params (DEP_LOW duplicated
        NEXP times)::

            ind_low = [...]
            dep_low = [lo_a, lo_b]
            nexp = 2
            bounds = Bounds(
                lower=np.array(ind_low + dep_low * nexp),
                upper=np.array(ind_up  + dep_up  * nexp),
            )
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower.shape {self.lower.shape} != upper.shape {self.upper.shape}"
            )
        if self.lower.ndim != 1:
            raise ValueError(
                f"bounds must be 1-D, got {self.lower.ndim}-D"
            )
        if not np.all(self.upper > self.lower):
            raise ValueError(
                "upper bounds must be strictly greater than lower"
            )

    @property
    def n_params(self) -> int:
        return self.lower.size

    def as_deap_lists(self) -> Tuple[List[float], List[float]]:
        """DEAP's SBX/PM operators want plain Python lists of floats."""
        return self.lower.tolist(), self.upper.tolist()


@dataclass
class RunConfig:
    """GA-level knobs. All defaults match the pre-refactor driver.

    Anything that controls DEAP's behavior lives here; anything that
    controls the simulation-launching side lives on
    :class:`workflow_common.Problem`.

    Fields:
        n_generations: Total generation count. With DEAP, the
            initial sample is generation 0; this field counts the
            number of generation transitions after that, matching
            the old ``NGEN`` variable. Default 100.
        population_size: Optional explicit population size. If
            ``None`` (the default), the driver computes it from
            the reference-point count using the NSGA-III paper's
            recipe: ``NPOP = ceil(H / 4) * 4`` where ``H`` is the
            total number of reference points. Setting it explicitly
            is useful for reproducing runs that pinned NPOP.
        unsga3: If True, apply the fork's
            :func:`tools.niching_selection_UNSGA3` before each
            variation step. Required for single-objective runs
            (the default reference-point scheme degenerates
            without niching) and recommended for two-objective.
            Default True.
        ref_dirs_partitions: Two Das-Dennis partition counts. The
            first hyperplane's partitions go in ``p[0]``; the
            second (inner) hyperplane's in ``p[1]``. Set ``p[1]=0``
            to use a single hyperplane. Default (10, 0).
        ref_dirs_scaling: Matched scaling factors for the two
            hyperplanes. ``scaling[1]=0`` disables the second
            hyperplane. Default (1.0, 0.0).
        seed: Random seed passed to ``random.seed(...)``. Default 0.
        mate_eta: SBX distribution index. Larger = children more
            similar to parents. Default 30.
        mut_eta: Polynomial mutation distribution index. Larger =
            smaller perturbations. Default 20.
        mut_indpb: Per-gene mutation probability. If ``None`` (the
            default), uses ``1.0 / n_params`` which gives one
            expected mutation per child — the DEAP+NSGA-III
            convention.
        cx_prob: Crossover probability passed to varAnd. Default
            1.0 (match the old driver which calls
            ``varAnd(pop, toolbox, 1, 1)``).
        mut_prob: Mutation probability passed to varAnd. Default
            1.0 (same reason).
        fail_limit: After this many total sim failures (cumulative
            across all generations), run terminates via exception.
            Default 10.
        failure_threshold: Fitness value at or above which a gene
            is considered to have failed. The framework's default
            :class:`InfinityFailureHandler` emits inf; a
            :class:`ConstantPenaltyFailureHandler` emits a large
            number. Set this above normal objective values, below
            penalty values. Default ``np.inf`` — works out of the
            box with the default InfinityFailureHandler.
        imin_fraction: Fraction of n_generations after which the
            "ND == NPOP" stopping criterion activates. Default
            0.5, matching the old Imin = round(NGEN / 2).
        stop_limit: Consecutive "ND == NPOP" generations after
            Imin before declaring convergence. Default 5.
        checkpoint_dir: Directory for pickle checkpoints. None
            disables checkpointing. Default ``None``.
        checkpoint_freq: Save every N generations. Default 1 (every
            generation), matching the old driver.
        resume_from: Path to a specific checkpoint pickle to
            resume from. The driver reads it, restores random
            state, population, counters, and continues from
            ``gen + 1``. Default ``None`` (start fresh).
        track_hypervolume: Track per-generation hypervolume with
            reference point ``[1]*n_obj``. Default True; disable
            for very high n_obj where HV computation is expensive.
        cleanup_keep_generations: Number of trailing generations
            whose case directories stay on disk. ``None`` (default)
            disables filesystem cleanup - matches the old driver's
            "keep everything forever" behavior. Set to 2 for the
            recommended rolling-cleanup pattern: after generation
            N completes, delete the dirs from generation ``N - 2``,
            keeping the current and previous generations as a
            crash-recovery safety margin. Requires the
            :class:`~workflow_common.Problem` to be configured with
            an archive, otherwise the data would be permanently
            lost; the driver raises if cleanup is requested without
            an archive.
    """

    n_generations: int = 100
    population_size: Optional[int] = None
    unsga3: bool = True
    ref_dirs_partitions: Tuple[int, int] = (10, 0)
    ref_dirs_scaling: Tuple[float, float] = (1.0, 0.0)
    seed: int = 0
    mate_eta: float = 30.0
    mut_eta: float = 20.0
    mut_indpb: Optional[float] = None
    cx_prob: float = 1.0
    mut_prob: float = 1.0
    fail_limit: int = 10
    failure_threshold: float = float("inf")
    imin_fraction: float = 0.5
    stop_limit: int = 5
    checkpoint_dir: Optional[Path] = None
    checkpoint_freq: int = 1
    resume_from: Optional[Path] = None
    track_hypervolume: bool = True
    cleanup_keep_generations: Optional[int] = None
    # Progress reporting. Default True because the cost is near-zero
    # and the upside (catching a misconfigured launcher on generation
    # 0 instead of after a multi-day run) is enormous. Disable for
    # tests or for non-TTY log-only runs where the per-generation
    # log line is already sufficient.
    show_progress: bool = True
    # Directory for logbook text files. The driver writes two files
    # here: ``logbook1_stats.log`` (per-generation aggregate stats
    # — avg/std/min/max fitness, plus ND/GD/HV for multi-objective
    # runs) and ``logbook2_solutions.log`` (every evaluated
    # individual's gene vector + fitness). These are DEAP-style
    # tab-delimited text: human-readable, greppable, ``less``-able.
    # Matches the pre-refactor driver's output exactly so existing
    # downstream tooling keeps working.
    #
    # Default None means "next to the checkpoint if checkpointing is
    # on; else the current working directory." Set to ``False``
    # (via ``write_logbook_files=False``) if you want the in-memory
    # logbooks only.
    log_dir: Optional[Path] = None
    # Hard switch to disable the two .log files. The logbooks
    # themselves are still built and pickled into the checkpoint;
    # only the human-readable text files are skipped. Default True
    # because users seeing "no logs" on a multi-day run is worse
    # than a few hundred KB of disk.
    write_logbook_files: bool = True


@dataclass
class RunResult:
    """Output of a completed NSGA-III run.

    Fields:
        final_pop: The final DEAP population (list of Individuals).
            Each entry has ``.fitness.values``, ``.rank``, and
            ``.generation`` attributes set.
        pareto_front: Subset of ``final_pop`` with rank 0.
            Convenience accessor; identical to filtering final_pop
            by rank == 0.
        pop_library: Full generation history. ``pop_library[g]``
            is the NPOP-long list of Individuals at generation g.
            Pickled as part of the checkpoint so runs are fully
            reconstructible.
        logbook_stats: DEAP Logbook with per-generation summary
            stats (std, min, avg, max) plus ND/GD/HV for
            multi-objective runs. Populated by the driver.
        logbook_solutions: DEAP Logbook with per-individual records
            (fitness, gene, generation of origin, solution vector).
        stopped_early: True if the stop_limit criterion triggered
            convergence-based termination; False if the run
            completed all n_generations.
        generations_run: Actual number of generations executed.
            Equals n_generations when stopped_early is False.
        seed: The seed used (echoed back for reproducibility).
    """

    final_pop: List
    pareto_front: List
    pop_library: List[List]
    logbook_stats: Any
    logbook_solutions: Any
    stopped_early: bool
    generations_run: int
    seed: int


# --- DEAP creator setup (module-once) -----------------------------------

# DEAP's `creator` uses class-level side effects to build the Individual
# and FitnessMin classes. We lazily build them on first use and cache
# the result on the module so multiple run_nsga3() calls with the same
# NOBJ don't re-create the class (DEAP complains about that). Each
# distinct NOBJ value gets its own class.
_creator_cache: Dict[int, Tuple[type, type]] = {}


def _ensure_creator_classes(n_obj: int) -> Tuple[type, type]:
    """Return ``(FitnessMin, Individual)`` types for ``n_obj`` objectives.

    Building these is DEAP-idiomatic. FitnessMin inherits from
    base.Fitness with negative weights so that DEAP's default
    MAXIMIZE semantics behave as MINIMIZE on our error values.
    Individual is a list subclass carrying the fitness plus all the
    extra attributes the pre-refactor code hung on it (rank, nich,
    nich_dist, generation, gene). ``stress`` is dropped; see module
    docstring.

    Registering on ``creator`` adds these as attributes of the
    creator module; the pickle path and DEAP's internals rely on
    that. Calling this function twice with the same n_obj is a
    no-op after the first call.
    """
    cached = _creator_cache.get(n_obj)
    if cached is not None:
        return cached

    fitness_name = f"FitnessMin_{n_obj}"
    individual_name = f"Individual_{n_obj}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,) * n_obj)
    if not hasattr(creator, individual_name):
        creator.create(
            individual_name,
            list,
            fitness=getattr(creator, fitness_name),
            rank=None,
            nich=None,
            nich_dist=None,
            generation=None,
            gene=None,
        )
    fitness_cls = getattr(creator, fitness_name)
    individual_cls = getattr(creator, individual_name)
    _creator_cache[n_obj] = (fitness_cls, individual_cls)
    return fitness_cls, individual_cls


# --- Reference points ---------------------------------------------------


def build_reference_points(
    n_obj: int, partitions: Tuple[int, int], scaling: Tuple[float, float],
) -> Tuple[np.ndarray, int]:
    """Build Das-Dennis reference points, optionally with a second hyperplane.

    Faithful to the old driver's ``p = [10, 0]; scaling = [1, 0]``
    pattern. If ``partitions[1] == 0`` or ``scaling[1] == 0`` the
    second hyperplane is skipped.

    Exposed publicly so example scripts can print the expected
    population size up front. The return value matches what the
    driver itself uses internally.

    Args:
        n_obj: Number of objectives.
        partitions: Two Das-Dennis partition counts ``(p_outer, p_inner)``.
            Set ``p_inner=0`` to use a single hyperplane.
        scaling: Matched scaling factors ``(s_outer, s_inner)``.
            ``s_inner=0`` disables the second hyperplane.

    Returns:
        ``(ref_points, H)`` where H is the total number of reference
        points. NPOP is typically chosen so H divides it per the
        NSGA-III paper's recipe.

    Raises:
        ValueError: If n_obj == 1 and partitions[0] < 1.
    """
    p = partitions
    s = scaling
    if n_obj == 1:
        # Single-objective runs: one reference "direction" — the
        # first partition count doubles as P for population sizing.
        if p[0] < 1:
            raise ValueError(
                "single-objective runs need partitions[0] >= 1"
            )
        ref = tools.uniform_reference_points(n_obj, p[0])
        return ref, p[0]

    ref1 = tools.uniform_reference_points(n_obj, p[0], s[0])
    if p[1] != 0 and s[1] != 0:
        ref2 = tools.uniform_reference_points(n_obj, p[1], s[1])
        ref = np.concatenate((ref1, ref2), axis=0)
    else:
        ref = ref1

    # H count per NSGA-III paper recipe. For the dual-hyperplane
    # case H is the total count of rows in ``ref``; that already
    # equals the combinatorial sum so we read it off the array
    # directly rather than recomputing.
    h = int(ref.shape[0])
    return ref, h


def derive_population_size(n_obj: int, h: int) -> int:
    """NSGA-III paper recipe: ``NPOP = round_up_to_multiple_of_4(H)``.

    Exposed publicly so example scripts can print NPOP at config
    time. The framework uses the same formula internally.
    """
    # Round H up to the next multiple of 4. The ``(+ (4 - H % 4))``
    # form from the old code doesn't short-circuit when H is already
    # a multiple of 4; ``(-(-H // 4)) * 4`` is the clean way.
    return -(-h // 4) * 4


# Internal aliases kept for backward compatibility with the driver's
# existing test suite and any user code that imported them while
# they were underscore-prefixed. Direct uses should migrate to the
# public names above.
_build_reference_points = build_reference_points
_derive_population_size = derive_population_size


# --- DEAP toolbox wiring ------------------------------------------------


def _build_toolbox(
    bounds: Bounds, config: RunConfig, ref_points: np.ndarray,
    individual_cls: type, n_params: int,
):
    """Wire up a DEAP Toolbox matching the pre-refactor driver.

    Uses the exact operators and the exact parameters from the old
    driver; the only non-obvious bit is that `attr_float` returns
    a fresh random individual-length list per call (via the
    ``uniform`` closure), which is what DEAP's ``initIterate``
    expects.
    """
    low, up = bounds.as_deap_lists()
    indpb = (
        config.mut_indpb
        if config.mut_indpb is not None
        else 1.0 / n_params
    )

    def uniform(lo: List[float], hi: List[float], size: int) -> List[float]:
        # Faithful to the old driver's `uniform` helper. random.uniform
        # is seeded by random.seed(...) once at run_nsga3 entry.
        try:
            return [random.uniform(a, b) for a, b in zip(lo, hi)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([lo] * size, [hi] * size)]

    tb = base.Toolbox()
    tb.register("attr_float", uniform, low, up, n_params)
    tb.register("individual", tools.initIterate, individual_cls, tb.attr_float)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register(
        "mate", tools.cxSimulatedBinaryBounded,
        low=low, up=up, eta=config.mate_eta,
    )
    tb.register(
        "mutate", tools.mutPolynomialBounded,
        low=low, up=up, eta=config.mut_eta, indpb=indpb,
    )
    tb.register(
        "select", tools.selNSGA3,
        ref_points=ref_points, nd="standard",
    )
    return tb


# --- Bridge: DEAP population -> framework evaluate_population ---------


def _deap_evaluate(
    wf_problem: WorkflowProblem,
    generation: int,
    invalid_ind: Sequence,
    progress: Optional[Any] = None,
) -> List[Tuple[float, ...]]:
    """Adapter between DEAP's invalid_ind list and the framework.

    DEAP passes a list of Individuals (list-subclasses holding
    gene floats). The framework expects a plain list of lists of
    floats and a generation index. After evaluation, DEAP wants
    fitness values as tuples so they can be assigned to
    ``ind.fitness.values`` (a DEAP Fitness expects a tuple).

    Args:
        wf_problem: The workflow_common.Problem doing the real work.
        generation: GA generation index; forwarded so the framework
            groups case directories by generation for clean on-disk
            layout and restart.
        invalid_ind: DEAP list of Individuals lacking valid fitness.
        progress: Optional progress reporter; forwarded to
            ``Problem.evaluate_population`` so each completed sim
            ticks the bar. Constructed and managed by the caller.

    Returns:
        List of tuples aligned with ``invalid_ind``. One tuple per
        Individual, one float per objective.
    """
    genes: List[List[float]] = [list(ind) for ind in invalid_ind]
    results: List[List[float]] = wf_problem.evaluate_population(
        genes, generation=generation, progress=progress,
    )
    # Validate shape early - a subtle off-by-one from the framework
    # surfaces much more clearly here than three layers down inside
    # DEAP's fitness bookkeeping.
    if len(results) != len(genes):
        raise RuntimeError(
            f"evaluate_population returned {len(results)} entries "
            f"for {len(genes)} genes"
        )
    for i, r in enumerate(results):
        if len(r) != wf_problem.n_objectives:
            raise RuntimeError(
                f"gene {i}: evaluate_population returned {len(r)} values, "
                f"expected {wf_problem.n_objectives}"
            )
    return [tuple(r) for r in results]


def _is_failed(fit: Sequence[float], threshold: float) -> bool:
    """True if any objective in this fitness exceeds the failure threshold.

    Matches the old driver's ``problem.is_simulation_done(igene) != 0``
    signal. The framework's FailureHandler writes one fixed value
    on failure (inf by default, or a configurable penalty). We
    test "any value >= threshold" rather than equality because
    some handlers emit slightly different penalties per objective
    (e.g. PartialProgressFailureHandler).
    """
    return any((v is None) or (not np.isfinite(v)) or (v >= threshold) for v in fit)


# --- Logbook file writing ----------------------------------------------


class _LogbookWriter:
    """Writes DEAP logbook streams to two human-readable text files.

    The pre-refactor driver wrote ``logbook1_stats.log`` (aggregate
    per-generation stats) and ``logbook2_solutions.log`` (every
    individual's gene/fitness/solution vector) as DEAP's default
    tab-delimited stream format. Downstream team tooling parses
    these as tab-separated text, so the refactor preserves the
    exact byte shape.

    The driver calls ``flush_stats(lb1)`` and ``flush_solutions(lb2)``
    after every ``lb.record(...)`` site. DEAP's ``logbook.stream``
    emits only records added since the previous ``stream`` access,
    so repeated calls produce an append-friendly delta.

    On resume-from-checkpoint, call ``rewrite_from(lb1, lb2)`` once
    to truncate both files and replay the entire logbook history.
    This avoids the "some records written twice, some never
    written" failure mode that would happen if we tried to
    synchronize incremental writes across a crash + resume.
    """

    def __init__(self, log_dir: Path):
        """Open the two log files in append mode, creating them if absent.

        Args:
            log_dir: Directory to write into. Created if missing.
                The two filenames are hard-coded to
                ``logbook1_stats.log`` and ``logbook2_solutions.log``
                to match the pre-refactor driver's output exactly.
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.stats_path = self._log_dir / "logbook1_stats.log"
        self.solutions_path = self._log_dir / "logbook2_solutions.log"
        # Touch the files so they exist even if no records are
        # written (unlikely but possible if a run aborts before
        # the first generation).
        self.stats_path.touch(exist_ok=True)
        self.solutions_path.touch(exist_ok=True)

    def flush_stats(self, logbook: "tools.Logbook") -> None:
        """Append the stats-logbook's unprinted records to the stats file.

        Relies on DEAP's ``logbook.stream`` returning only records
        added since the previous access. Callers must invoke this
        immediately after ``logbook.record(...)`` so the delta is
        exactly one new generation.
        """
        stream = logbook.stream
        if not stream:
            return
        with self.stats_path.open("a") as f:
            f.write(stream + "\n")

    def flush_solutions(self, logbook: "tools.Logbook") -> None:
        """Append the solutions-logbook's unprinted records.

        Same ``.stream`` delta trick as ``flush_stats``. The
        solutions logbook receives one record per individual, so the
        delta written here is ``npop`` records — one whole
        generation's population.
        """
        stream = logbook.stream
        if not stream:
            return
        with self.solutions_path.open("a") as f:
            f.write(stream + "\n")

    def rewrite_from(
        self,
        stats_logbook: "tools.Logbook",
        solutions_logbook: "tools.Logbook",
    ) -> None:
        """Truncate both files and replay every record from scratch.

        Called exactly once at resume-from-checkpoint time. The
        logbooks loaded from the pickle hold the full history; this
        method rewrites the ``.log`` files so they match the
        in-memory state. After this call, ``.stream`` still works
        (DEAP tracks what's been streamed internally, and on a
        freshly-loaded logbook nothing has been streamed yet, so
        one ``.stream`` call will emit the entire history — which
        is exactly what we want here).
        """
        # Truncate by reopening in write mode.
        self.stats_path.write_text("")
        self.solutions_path.write_text("")
        # A freshly-loaded logbook's .stream cursor (``buffindex``)
        # points wherever the last process left it. On resume we
        # want to re-emit EVERY record — including ones already
        # streamed pre-crash — so reset the cursor to zero.
        # ``buffindex`` is the attribute name used by DEAP's
        # ``Logbook.stream`` property (verified by reading
        # ``deap.tools.Logbook.stream``'s source). It's a public
        # field in practice; resetting it is stable across DEAP
        # releases that keep the stream mechanism.
        stats_logbook.buffindex = 0
        solutions_logbook.buffindex = 0
        stats_stream = stats_logbook.stream
        solutions_stream = solutions_logbook.stream
        if stats_stream:
            with self.stats_path.open("w") as f:
                f.write(stats_stream + "\n")
        if solutions_stream:
            with self.solutions_path.open("w") as f:
                f.write(solutions_stream + "\n")





def _save_checkpoint(
    path: Path, *,
    generation: int,
    pop_library: List,
    iter_tot: int,
    fail_count: int,
    stop_count: int,
    logbook1: Any,
    logbook2: Any,
    archive_run_id: Optional[str] = None,
) -> None:
    """Pickle the GA state. Keys match the pre-refactor driver exactly.

    Backward-compatible on purpose: a mid-run team can keep using
    old checkpoint files with this driver, or switch drivers
    between runs. The ``archive_run_id`` entry is the one addition
    — older checkpoints simply don't have it, and the load path
    tolerates its absence.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ckp = dict(
        pop_library=pop_library,
        iter_tot=iter_tot,
        generation=generation,
        fail_count=fail_count,
        stop_count=stop_count,
        logbook1=logbook1,
        logbook2=logbook2,
        rndstate=random.getstate(),
        archive_run_id=archive_run_id,
    )
    # Tempfile-then-rename so a crash mid-pickle cannot corrupt the
    # last-good checkpoint. Same pattern as atomic_write_text but we
    # need binary mode for pickle.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(ckp, f)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint pickle written by :func:`_save_checkpoint`.

    **Security note.** This uses :func:`pickle.load`, which executes
    arbitrary Python during unpickling. Only ever call this on
    checkpoint files you wrote yourself in your own workspace. A
    malicious pickle planted in a shared HPC scratch directory,
    served from an S3 bucket, or otherwise sourced externally can
    run arbitrary code as your user. If you need to exchange
    checkpoints across users or hosts, convert to a safe
    serialization format (JSON for the metadata plus a numpy
    ``.npz`` for the gene arrays) at the boundary.
    """
    with path.open("rb") as f:
        return pickle.load(f)


def _build_gene_records(
    run_id: str,
    gen_idx: int,
    pop: Sequence[Any],
) -> "List[Any]":
    """Translate a DEAP population into :class:`GeneRecord` rows for the archive.

    One record per individual in ``pop``. Reads
    ``ind.fitness.values``, ``ind.generation`` (birth gen),
    ``ind.gene`` (birth offspring idx), and ``ind.rank`` — the same
    attributes the driver assigns inside the evaluation loop.

    The ``GeneRecord`` import is deferred so callers who never use
    archiving don't pay the cost (and don't need archive.py to
    import cleanly — if numpy/pandas/sqlite3 were missing for any
    reason the archive-free path would still work).
    """
    from workflow_common.archive import GeneRecord

    records = []
    for pop_idx, ind in enumerate(pop):
        records.append(GeneRecord(
            run_id=run_id,
            gen_idx=gen_idx,
            pop_idx=pop_idx,
            birth_gen=getattr(ind, "generation", gen_idx),
            birth_gene=getattr(ind, "gene", pop_idx),
            gene_vector=np.asarray(list(ind), dtype=float),
            fitness=tuple(ind.fitness.values),
            rank=getattr(ind, "rank", None),
        ))
    return records


# --- Main entry point ---------------------------------------------------


def _record_experiments_from_problem(
    archive,
    run_id: str,
    wf_problem,
) -> None:
    """Walk objective_specs and store any experimental DataFrames.

    The driver runs this once at run-start (both fresh and resume
    paths). For each unique ``sim_case_idx``, it picks the FIRST
    evaluator that exposes an ``.experimental`` attribute holding
    a DataFrame, and writes that DataFrame to the archive's
    ``experiments`` table.

    Why first-only: if two evaluators on the same SimCase both
    have an ``.experimental`` field, they are scoring the same
    simulation against the same reference (e.g. stress and slope
    objectives both come from the same uniaxial sim), so the
    experimental DataFrame they hold MUST be the same data.
    Recording it once is enough. We also emit a warning if a
    later evaluator on the same SimCase exposes a *different*
    DataFrame — that's a misconfiguration the user should know
    about, even if the framework can't auto-fix it.

    Why this lives here, not on Problem: the archive isn't always
    present (``archive=None`` is a valid mode), and Problem has no
    write capability. The driver is the only place that owns the
    archive lifecycle; it's the right layer to do the recording.

    Silently skips SimCases whose evaluators don't carry
    experimental data (custom evaluators, evaluators that pull from
    a different source). The plotting tools handle missing data
    gracefully by falling back to user-supplied CSV paths.
    """
    if archive is None:
        return
    seen_for_case: Dict[int, Any] = {}
    for spec in wf_problem.objective_specs:
        sc_idx = spec.sim_case
        ev = spec.evaluator
        df = getattr(ev, "experimental", None)
        if df is None:
            continue
        if sc_idx in seen_for_case:
            # Same SimCase already has experimental data recorded.
            # Verify it's the same (or at least reasonably compatible)
            # — divergent data here is a configuration smell.
            prev = seen_for_case[sc_idx]
            if prev is df:
                continue  # exact same object, fine
            # DataFrames don't compare with == cleanly when shapes
            # differ; use shape + column-name check as a cheap
            # divergence signal.
            try:
                same_shape = prev.shape == df.shape
                same_cols = list(prev.columns) == list(df.columns)
            except Exception:  # noqa: BLE001
                same_shape = same_cols = False
            if not (same_shape and same_cols):
                logger.warning(
                    "experimental data divergence on sim_case_idx=%d: "
                    "two evaluators carry different DataFrames for "
                    "the same SimCase; archiving the first.",
                    sc_idx,
                )
            continue
        # First evaluator for this SimCase that has experimental data.
        sc_label: Optional[str] = None
        sc_window: Optional[Tuple[Optional[float], Optional[float]]] = None
        if 0 <= sc_idx < len(wf_problem.sim_cases):
            sc = wf_problem.sim_cases[sc_idx]
            sc_label = sc.label
            # Optimization window from case_data, if any. The user
            # supplies (lo, hi) (either side may be None for
            # "unbounded"). The framework stores it verbatim; the
            # plotter shades the corresponding region against the
            # full experimental curve so users can sanity-check
            # what the optimizer was actually fitting against.
            #
            # Defensive parse: case_data is user-provided, so accept
            # tuples, lists, or anything that unpacks to two items.
            # Anything else logs a warning and is dropped (better
            # to record the experiment with no window than fail
            # to record at all).
            mm_raw = sc.case_data.get("minmax_strain")
            if mm_raw is not None:
                try:
                    lo, hi = mm_raw
                    sc_window = (
                        None if lo is None else float(lo),
                        None if hi is None else float(hi),
                    )
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "archive: case_data['minmax_strain']=%r on "
                        "sim_case=%d is not a 2-tuple of floats/Nones; "
                        "dropping window from archive (%s)",
                        mm_raw, sc_idx, e,
                    )
                    sc_window = None
        # Pull extractor config from the evaluator if present. Most
        # framework-supplied evaluators (StressStrainObjective and
        # the example's _StdNormalized*Evaluator) expose ``.extractor``
        # holding a StressStrainExtractor; user-defined evaluators
        # may or may not. Missing or non-serializable → store None
        # and the plotter falls back to its default extractor (with
        # a warning, not a crash). The serialize itself is wrapped
        # in try/except because user-defined extractors may not have
        # a ``to_dict`` method.
        sc_extractor_config: Optional[Dict[str, object]] = None
        ev_extractor = getattr(ev, "extractor", None)
        if ev_extractor is not None:
            to_dict = getattr(ev_extractor, "to_dict", None)
            if callable(to_dict):
                try:
                    sc_extractor_config = to_dict()
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "archive: evaluator.extractor.to_dict() "
                        "failed on sim_case=%d: %s — extractor "
                        "config will not be archived; plot tools "
                        "will fall back to defaults.",
                        sc_idx, e,
                    )
        try:
            archive.record_experiment(
                run_id, sim_case_idx=sc_idx, df=df,
                label=sc_label, minmax_strain=sc_window,
                extractor_config=sc_extractor_config,
            )
            seen_for_case[sc_idx] = df
            logger.info(
                "archive: stored experimental data for sim_case=%d "
                "(label=%s, shape=%s, window=%s)",
                sc_idx, sc_label, getattr(df, "shape", None), sc_window,
            )
        except Exception as e:  # noqa: BLE001
            # Don't let an experiment-write failure abort the run.
            # Plotting tools can fall back to user-supplied CSVs.
            logger.warning(
                "archive: failed to store experimental data for "
                "sim_case=%d: %s", sc_idx, e,
            )


def run_nsga3(
    wf_problem: WorkflowProblem,
    bounds: Bounds,
    config: RunConfig,
) -> RunResult:
    """Run (U-)NSGA-III against a ``workflow_common.Problem``.

    Faithful to the pre-refactor ``ExaConstit_NSGA3.py::main`` in
    algorithm, operator choices, population sizing, stopping
    criteria, and checkpoint format. The one intentional difference
    is the failure-detection mechanism (see module docstring).

    Args:
        wf_problem: Ready-to-use :class:`workflow_common.Problem`
            with sim_cases and objective_specs already configured.
        bounds: Per-gene lower/upper bounds matching
            ``wf_problem.param_names`` length.
        config: GA-level knobs. See :class:`RunConfig`.

    Returns:
        A :class:`RunResult` with the final population, Pareto
        front, full history, and DEAP logbooks.

    Raises:
        ValueError: If bounds.n_params disagrees with problem
            param_names length.
        RuntimeError: If fail_count exceeds config.fail_limit.

    Example:
        Two-experiment, four-objective run (stress + slope per
        experiment, two experiments - the old driver's default
        topology)::

            sim_cases = [
                SimCase(case_data={"strain_rate": 1e-3}, label="exp1"),
                SimCase(case_data={"strain_rate": 1e-2}, label="exp2"),
            ]
            objective_specs = [
                ObjectiveSpec(stress_eval_1, sim_case=0, label="stress_1"),
                ObjectiveSpec(slope_eval_1,  sim_case=0, label="slope_1"),
                ObjectiveSpec(stress_eval_2, sim_case=1, label="stress_2"),
                ObjectiveSpec(slope_eval_2,  sim_case=1, label="slope_2"),
            ]
            problem = Problem(
                ..., sim_cases=sim_cases, objective_specs=objective_specs,
            )

            result = run_nsga3(
                problem,
                bounds=Bounds(lower=np.array(IND_LOW), upper=np.array(IND_UP)),
                config=RunConfig(
                    n_generations=100, unsga3=True,
                    ref_dirs_partitions=(10, 0), seed=42,
                    checkpoint_dir=Path("checkpoint_files"),
                ),
            )
            print("Pareto size:", len(result.pareto_front))
    """
    # --- Validation ---
    if bounds.n_params != len(wf_problem.param_names):
        raise ValueError(
            f"bounds has {bounds.n_params} entries but problem.param_names "
            f"has {len(wf_problem.param_names)}"
        )

    # Rolling cleanup can only run safely if the archive is capturing
    # the data we are about to delete. Fail early rather than silently
    # destroy the user's outputs.
    if config.cleanup_keep_generations is not None:
        if wf_problem.archive is None:
            raise ValueError(
                "cleanup_keep_generations requires the Problem to have "
                "an archive configured (archive=... on Problem); "
                "otherwise cleaned-up case directories would be "
                "permanently lost. Either set up an archive or disable "
                "cleanup_keep_generations."
            )
        if config.cleanup_keep_generations < 1:
            raise ValueError(
                "cleanup_keep_generations must be >= 1 "
                "(0 would delete the current generation's dirs mid-run)"
            )

    n_obj = wf_problem.n_objectives

    # --- Reference points + population size ---
    ref_points, h = _build_reference_points(
        n_obj, config.ref_dirs_partitions, config.ref_dirs_scaling,
    )
    npop = (
        config.population_size
        if config.population_size is not None
        else _derive_population_size(n_obj, h)
    )

    # --- Build DEAP classes and toolbox ---
    _, individual_cls = _ensure_creator_classes(n_obj)
    toolbox = _build_toolbox(
        bounds, config, ref_points, individual_cls, bounds.n_params,
    )

    # --- Stats + logbooks ---
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook1 = tools.Logbook()
    logbook2 = tools.Logbook()
    if n_obj == 1:
        logbook1.header = "gen", "iter", "simRuns", "std", "min", "avg", "max"
    else:
        logbook1.header = (
            "gen", "iter", "simRuns", "ND", "GD", "HV",
            "std", "min", "avg", "max",
        )
    logbook2.header = "gen", "fitness", "sol_generation", "sol_gene", "solutions"

    # Logbook text-file writer. Writes DEAP-style tab-delimited
    # per-generation stats and per-individual solutions to two
    # .log files that match the pre-refactor driver's output
    # exactly. Downstream tooling parses these as tab-separated.
    # Can be disabled via ``RunConfig(write_logbook_files=False)``.
    if config.write_logbook_files:
        # Pick a log dir: explicit config > next-to-checkpoint > cwd.
        # The "next to checkpoint" default keeps the logs and the
        # pickle co-located, which is what users tend to expect and
        # what the pre-refactor driver effectively did (it wrote
        # to cwd alongside checkpoint.pkl).
        if config.log_dir is not None:
            log_dir = Path(config.log_dir)
        elif config.checkpoint_dir is not None:
            log_dir = Path(config.checkpoint_dir)
        else:
            log_dir = Path.cwd()
        logbook_writer: Optional[_LogbookWriter] = _LogbookWriter(log_dir)
        logger.info(
            "logbook files: stats=%s solutions=%s",
            logbook_writer.stats_path, logbook_writer.solutions_path,
        )
    else:
        logbook_writer = None

    # --- Start-state: fresh, or resumed from checkpoint ---
    iter_tot = 0
    fail_count = 0
    stop_count = 0
    stop_optimization = False
    pop_library: List[List] = []
    archive = wf_problem.archive  # convenience alias

    if config.resume_from is not None:
        logger.info(
            "resume path: loading checkpoint from %s",
            config.resume_from,
        )
        ckp = _load_checkpoint(Path(config.resume_from))
        random.setstate(ckp["rndstate"])
        pop_library = ckp["pop_library"]
        last_gen = ckp["generation"]
        pop = pop_library[last_gen]
        if len(pop) != npop:
            raise ValueError(
                f"checkpoint's NPOP ({len(pop)}) differs from computed "
                f"NPOP ({npop}); did ref_dirs_partitions change?"
            )
        iter_tot = ckp["iter_tot"]
        fail_count = ckp["fail_count"]
        stop_count = ckp["stop_count"]
        logbook1 = ckp["logbook1"]
        logbook2 = ckp["logbook2"]
        # Diagnostic summary of what was recovered. This runs on
        # EVERY resume and costs nothing; when resume misbehaves
        # in the wild, having these numbers in the log turns a
        # "it started from gen 0 somehow" report into a tractable
        # debugging session.
        logger.info(
            "resume: last completed gen=%d, pop_library has %d entries, "
            "logbook1 has %d records, logbook2 has %d records, "
            "pickled archive_run_id=%s",
            last_gen, len(pop_library), len(logbook1), len(logbook2),
            ckp.get("archive_run_id"),
        )
        # Rewrite the two .log files from the just-loaded logbooks
        # so incremental writes going forward pick up seamlessly.
        # Without this, a mid-run crash followed by resume would
        # leave the .log files truncated to the pre-crash state and
        # subsequent .stream calls would append only NEW gens —
        # leaving a gap where the last pre-crash generation's
        # partial write used to be. Rewriting from scratch is
        # cheap (the logbooks are in memory) and eliminates that
        # failure mode.
        if logbook_writer is not None:
            logbook_writer.rewrite_from(logbook1, logbook2)
        start_gen = last_gen + 1
        logger.info(
            "resume: next generation to run is %d", start_gen,
        )
        if stop_count >= config.stop_limit:
            stop_optimization = True

        # Archive-side resume:
        # - If the checkpoint carries the archive_run_id (new format),
        #   continue writing to that run. The Problem's archive_run_id
        #   must match, otherwise the caller assembled Problem + pickle
        #   inconsistently.
        # - Drop any archive rows for generations > last_gen — those
        #   are half-written from the crash that preceded resume.
        pickled_run_id = ckp.get("archive_run_id")
        if archive is not None:
            if pickled_run_id is None:
                logger.warning(
                    "resume: checkpoint has no archive_run_id (older "
                    "format). Starting a fresh archive run; the "
                    "pre-crash generations won't be in the new run."
                )
                wf_problem.archive_run_id = archive.start_run(
                    seed=config.seed,
                    param_names=wf_problem.param_names,
                )
            else:
                if (
                    wf_problem.archive_run_id is not None
                    and wf_problem.archive_run_id != pickled_run_id
                ):
                    raise ValueError(
                        f"archive_run_id mismatch on resume:\n"
                        f"  checkpoint's run_id:  {pickled_run_id!r}\n"
                        f"  Problem's run_id:     {wf_problem.archive_run_id!r}\n"
                        f"\n"
                        f"This usually means the caller invoked "
                        f"ArchiveDB.start_run() unconditionally before "
                        f"handing the Problem to run_nsga3(). On a "
                        f"fresh run that's correct; on a resume, the "
                        f"new UUID from start_run() collides with the "
                        f"one already recorded in the checkpoint.\n"
                        f"\n"
                        f"Fix: in your driver, only call start_run() "
                        f"when NOT resuming:\n"
                        f"    if args.resume_from is None:\n"
                        f"        problem.archive_run_id = "
                        f"archive.start_run(...)\n"
                        f"    # else: leave archive_run_id = None, "
                        f"and the framework will adopt the pickled "
                        f"run's UUID."
                    )
                wf_problem.archive_run_id = pickled_run_id
                # Count what's in the archive BEFORE the discard so
                # the operator can see what was preserved (gens
                # 0..last_gen of this run are kept) vs what was
                # pruned (gens > last_gen, stale from the pre-crash
                # state). If a user ever reports "my archive got
                # wiped" again, these three log lines make it
                # trivial to confirm whether the framework is
                # touching anything besides the current run.
                _pre_gens = [
                    g.gen_idx for g in archive.list_generations(pickled_run_id)
                ]
                logger.info(
                    "archive resume: run=%s has gens %s before discard; "
                    "keeping gens <= %d, dropping >= %d",
                    pickled_run_id, _pre_gens, last_gen, last_gen + 1,
                )
                archive.discard_from_generation(
                    pickled_run_id, last_gen + 1,
                )
                _post_gens = [
                    g.gen_idx for g in archive.list_generations(pickled_run_id)
                ]
                logger.info(
                    "archive resume: after discard, run=%s has gens %s",
                    pickled_run_id, _post_gens,
                )
    else:
        random.seed(config.seed)

        # Archive-side fresh start: the driver is the authority on
        # run identity, so it creates the run row here. The
        # Problem's archive_run_id gets set so subsequent
        # record_case_outputs calls land in the right run.
        if archive is not None and wf_problem.archive_run_id is None:
            wf_problem.archive_run_id = archive.start_run(
                seed=config.seed,
                param_names=wf_problem.param_names,
            )

        # Generation 0: produce and evaluate the initial population.
        logger.info(
            "NSGA-III starting: n_gen=%d pop=%d n_obj=%d unsga3=%s seed=%d",
            config.n_generations, npop, n_obj, config.unsga3, config.seed,
        )
        pop = toolbox.population(n=npop)
        pop_library.append(pop)

        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # Per-generation reporter. Total = genes * sim_cases because
        # the Problem produces one sim per (gene, SimCase) pair.
        reporter0 = (
            ProgressReporter(
                total=len(invalid_ind) * wf_problem.n_sim_cases,
                label=f"gen 0/{config.n_generations}",
                backend=wf_problem.backend,
            )
            if config.show_progress else None
        )
        fits = _deap_evaluate(wf_problem, 0, invalid_ind, progress=reporter0)
        if reporter0 is not None:
            reporter0.finish()

        iter_pgen = 0
        for ind_idx, (ind, fit) in enumerate(zip(invalid_ind, fits)):
            iter_pgen += 1
            iter_tot += 1
            # Fail-retry loop: if this gene failed, draw a new
            # random individual in-place and re-evaluate JUST that
            # one. Cumulative fail_count; hit fail_limit → abort.
            while _is_failed(fit, config.failure_threshold):
                if fail_count >= config.fail_limit:
                    raise RuntimeError(
                        f"reached fail_limit={config.fail_limit} on gen 0 "
                        f"gene {ind_idx}; framework terminating"
                    )
                fail_count += 1
                logger.warning(
                    "gen 0 gene %d failed (fit=%s); retry #%d with fresh random gene",
                    ind_idx, fit, fail_count,
                )
                # Replace in-place so the surrounding population
                # list still points at this object.
                ind[:] = toolbox.individual()
                fit = _deap_evaluate(wf_problem, 0, [ind])[0]
            ind.fitness.values = fit
            ind.gene = ind_idx
            ind.generation = 0

        # Initial logbook records.
        record = stats.compile(pop)
        if n_obj == 1:
            logbook1.record(
                gen=0, iter=iter_pgen, simRuns=iter_pgen * n_obj, **record,
            )
        else:
            logbook1.record(
                gen=0, iter=iter_pgen, simRuns=iter_pgen * n_obj,
                ND="None", GD="None", HV="None", **record,
            )
        if logbook_writer is not None:
            logbook_writer.flush_stats(logbook1)
        # Initial "selection" — the old driver calls select on pop
        # alone here, which just sorts by non-domination rank without
        # changing size (selNSGA3 on NPOP candidates returns NPOP).
        pop = toolbox.select(pop, npop)
        for ind in pop:
            logbook2.record(
                gen=0,
                sol_generation=ind.generation,
                sol_gene=ind.gene,
                fitness=list(ind.fitness.values),
                solutions=list(ind),
            )
        if logbook_writer is not None:
            logbook_writer.flush_solutions(logbook2)

        # Archive gen 0 if archiving is enabled. Ranks are assigned
        # by selNSGA3 above, so the records capture them.
        if archive is not None:
            archive.record_generation(
                wf_problem.archive_run_id,
                gen_idx=0,
                genes=_build_gene_records(
                    wf_problem.archive_run_id, 0, pop,
                ),
                stats=record,
            )

        # Initial checkpoint, just in case the first real generation
        # blows up before saving. Matches old driver.
        if config.checkpoint_dir is not None:
            _save_checkpoint(
                Path(config.checkpoint_dir) / "checkpoint_gen_0.pkl",
                generation=0, pop_library=pop_library, iter_tot=iter_tot,
                fail_count=fail_count, stop_count=stop_count,
                logbook1=logbook1, logbook2=logbook2,
                archive_run_id=wf_problem.archive_run_id,
            )
        start_gen = 1

    # Capture the experimental reference DataFrames once
    # archive_run_id is locked in (either freshly created above or
    # adopted from the resume pickle). Re-recording on resume is
    # safe — record_experiment is INSERT OR REPLACE, so the call
    # is idempotent and writes the same data on every invocation.
    # Doing it before the main loop guarantees plotting tools that
    # open the archive mid-run see the experimental data; doing it
    # AFTER gen 0 in the fresh path is fine because experimental
    # data is read-only reference, not consumed by evaluation.
    if archive is not None and wf_problem.archive_run_id is not None:
        _record_experiments_from_problem(
            archive, wf_problem.archive_run_id, wf_problem,
        )

    # --- Main generational loop ---
    imin = int(round(config.n_generations * config.imin_fraction))
    gen = start_gen
    while gen <= config.n_generations and not stop_optimization:
        logger.info("NSGA-III generation %d", gen)

        # U-NSGA-III pre-variation niching step from the fork.
        if config.unsga3:
            src = tools.niching_selection_UNSGA3(pop)
        else:
            src = pop

        # varAnd applies crossover + mutation; offspring lose their
        # parents' fitness.valid. The cx_prob=1, mut_prob=1 defaults
        # match the old driver.
        offspring = algorithms.varAnd(
            src, toolbox, config.cx_prob, config.mut_prob,
        )

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        reporter_g = (
            ProgressReporter(
                total=len(invalid_ind) * wf_problem.n_sim_cases,
                label=f"gen {gen}/{config.n_generations}",
                backend=wf_problem.backend,
            )
            if config.show_progress else None
        )
        fits = _deap_evaluate(wf_problem, gen, invalid_ind, progress=reporter_g)
        if reporter_g is not None:
            reporter_g.finish()

        iter_pgen = 0
        for ind_idx, (ind, fit) in enumerate(zip(invalid_ind, fits)):
            iter_pgen += 1
            iter_tot += 1
            while _is_failed(fit, config.failure_threshold):
                if fail_count >= config.fail_limit:
                    raise RuntimeError(
                        f"reached fail_limit={config.fail_limit} at gen {gen} "
                        f"gene {ind_idx}; framework terminating"
                    )
                fail_count += 1
                logger.warning(
                    "gen %d gene %d failed (fit=%s); retry #%d",
                    gen, ind_idx, fit, fail_count,
                )
                ind[:] = toolbox.individual()
                fit = _deap_evaluate(wf_problem, gen, [ind])[0]
            ind.fitness.values = fit
            ind.gene = ind_idx
            ind.generation = gen

        # Environmental selection: combine parent pop with offspring
        # and select down to NPOP using NSGA-III's rank+niche sort.
        pop = toolbox.select(pop + offspring, npop)
        pop_library.append(pop)

        # Stats + stopping criteria.
        record = stats.compile(pop)
        if n_obj == 1:
            logbook1.record(
                gen=gen, iter=iter_pgen, simRuns=iter_pgen * n_obj, **record,
            )
        else:
            best_front = [ind for ind in pop if ind.rank == 0]
            nd = len(best_front)
            best_front_fit = [ind.fitness.values for ind in best_front]
            # GD: average Euclidean distance from the non-dominated
            # front to the origin. The old driver uses this as a
            # progress indicator (smaller is better for minimization).
            if best_front_fit:
                arr = np.asarray(best_front_fit, dtype=float)
                gd = float(np.sqrt((1 / nd) * np.sum(arr ** 2)))
            else:
                gd = float("nan")
            hv = (
                float(hypervolume(pop, [1.0] * n_obj))
                if config.track_hypervolume
                else float("nan")
            )
            logbook1.record(
                gen=gen, iter=iter_pgen, simRuns=iter_pgen * n_obj,
                ND=nd, GD=gd, HV=hv, **record,
            )

            # Consecutive "ND == NPOP" counter, only active after Imin.
            if gen > imin:
                if nd == npop:
                    stop_count += 1
                    logger.info(
                        "stopping criterion: consecutive stop_count = %d",
                        stop_count,
                    )
                else:
                    stop_count = 0
                if stop_count >= config.stop_limit:
                    stop_optimization = True
                    logger.info(
                        "stop_limit=%d reached at gen %d; ending early",
                        config.stop_limit, gen,
                    )

        for ind in pop:
            logbook2.record(
                gen=gen,
                sol_generation=ind.generation,
                sol_gene=ind.gene,
                fitness=list(ind.fitness.values),
                solutions=list(ind),
            )

        # Flush the two logbooks to disk. DEAP's .stream property
        # returns only records added since the previous .stream
        # access, so each call here writes exactly one generation's
        # worth of new entries.
        if logbook_writer is not None:
            logbook_writer.flush_stats(logbook1)
            logbook_writer.flush_solutions(logbook2)

        # Archive this generation's data. Ranks and GD/HV/ND are
        # already set; capture them in the stats dict so the archive
        # is self-describing.
        if archive is not None:
            archive.record_generation(
                wf_problem.archive_run_id,
                gen_idx=gen,
                genes=_build_gene_records(
                    wf_problem.archive_run_id, gen, pop,
                ),
                stats=record,
            )

        # Rolling filesystem cleanup. The archive has the data we
        # need; delete the case directories from (gen - keep) on
        # disk to keep the inode count bounded. The "keep" window
        # retains the most recent generations in case of crash
        # mid-run — we don't want to delete dirs we might need for
        # debugging if the next generation blows up.
        keep = config.cleanup_keep_generations
        if keep is not None and gen >= keep:
            # Example: keep=2 means when gen=2 we clean gen=0, when
            # gen=3 we clean gen=1, etc. The current gen and the
            # keep-1 previous ones stay on disk.
            wf_problem.cleanup_generation_dirs(
                gen_idx=gen - keep, pop_size=npop,
            )

        # Checkpoint.
        if (
            config.checkpoint_dir is not None
            and gen % config.checkpoint_freq == 0
        ):
            _save_checkpoint(
                Path(config.checkpoint_dir) / f"checkpoint_gen_{gen}.pkl",
                generation=gen, pop_library=pop_library, iter_tot=iter_tot,
                fail_count=fail_count, stop_count=stop_count,
                logbook1=logbook1, logbook2=logbook2,
                archive_run_id=wf_problem.archive_run_id,
            )

        gen += 1

    # --- Package results ---
    final_pop = pop
    pareto = [ind for ind in final_pop if getattr(ind, "rank", None) == 0]
    # Fallback when rank is not set (single-obj runs; DEAP only sets
    # rank on individuals that went through selNSGA3 in a multi-obj
    # context). Fall back to the minimum-fitness entry.
    if not pareto:
        pareto = [min(final_pop, key=lambda i: i.fitness.values)]

    # Mark run complete in the archive. Exceptions raised before
    # this point leave completed_at NULL, which lets tooling
    # distinguish crashed runs from clean ones.
    if archive is not None and wf_problem.archive_run_id is not None:
        archive.end_run(wf_problem.archive_run_id)

    return RunResult(
        final_pop=final_pop,
        pareto_front=pareto,
        pop_library=pop_library,
        logbook_stats=logbook1,
        logbook_solutions=logbook2,
        stopped_early=stop_optimization,
        generations_run=gen - 1,
        seed=config.seed,
    )
