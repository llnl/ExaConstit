"""
Post-processing utilities for NSGA-III runs.

Replaces the pre-refactor ``ExaConstit_PostProcess.py``,
``ExaConstit_SolPicker.py``, and the stress-strain portion of
``ExaPlots.py``. Three responsibilities, one module because they
are almost always used together.

What is here
------------
1. **Checkpoint loading** — :func:`load_checkpoint` reads a pickle
   written by :mod:`workflows.optimization.nsga3_driver` (same
   format the pre-refactor driver produced) and returns a typed
   :class:`CheckpointData` record. The driver's raw dict is kept
   as-is on purpose so old pickles continue to load.

2. **Best-solution picking** — :func:`best_solution_eudist` and
   :func:`best_solution_asf` reproduce the two strategies in the
   old ``ExaConstit_SolPicker.BestSol`` class. Both accept
   weights, optional normalization, and an ``nsmallest`` k-best
   cutoff. The legacy :class:`BestSol` class wrapper is kept for
   drop-in compatibility with existing post-processing scripts.

3. **Re-reading sim output on demand** — :func:`extract_gene_results`
   walks a ``pop_library`` from a checkpoint and builds a list of
   :class:`GeneResult` records that locate each gene's case
   directory on disk. :func:`load_case_results` then reads the
   simulation output back through the framework's
   :class:`~workflow_common.results.ResultReader`. This is the
   replacement for the pre-refactor ``ind.stress`` attribute.

4. **Plotting** — :func:`plot_stress_strain_overlay` and
   :func:`plot_pareto_front` replace ``ExaPlots.StressStrain`` and
   the 2-D portion of ``ExaPlots.ObjFun2D``. Matplotlib is a soft
   dependency and imported on first use so the rest of the module
   works in headless environments.

What is NOT here
----------------
* The old ``ExaPlots.ObjFun3D`` (3-D Pareto scatter). For >2
  objectives the 2-D scatter doesn't generalize usefully;
  parallel-coordinate or petal plots are better. Users who need
  them can call matplotlib / pymoo's visualization tools directly
  on the output of :func:`extract_gene_results`.
* Pickle-state patching to restore ``ind.stress``. The pre-refactor
  approach of stashing stress histories on DEAP Individuals meant
  every checkpoint grew by ``pop * n_sim_cases * n_timesteps *
  n_stress_components * 8`` bytes per generation — megabytes
  quickly. The new approach keeps sim output on disk where it
  already lives and pulls it back on demand.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from .logging_utils import get_logger
from .paths import CaseContext, PathResolver
from .results import CaseLayout, CaseResultSet, ResultReader

logger = get_logger(__name__)


# --- Checkpoint loading --------------------------------------------------


@dataclass
class CheckpointData:
    """Parsed contents of one driver checkpoint pickle.

    Field names follow the driver's internal naming
    (``logbook_stats`` / ``logbook_solutions`` rather than the
    on-disk ``logbook1`` / ``logbook2``) so post-processing code
    reads more clearly. The raw on-disk keys are preserved by
    :func:`load_checkpoint` below; the renaming happens at the
    record-construction boundary.

    Fields:
        pop_library: ``pop_library[gen]`` is the
            ``population_size``-long list of DEAP Individuals at
            generation ``gen``. Each Individual is itself a list
            of gene floats, plus ``fitness.values``,
            ``generation`` (birth gen), ``gene`` (within-gen
            offspring index at birth), and ``rank``.
        iter_tot: Cumulative count of evaluations across the run.
            Mostly useful for logging and sanity checks.
        generation: The generation at which this checkpoint was
            written (equals ``len(pop_library) - 1`` for
            end-of-generation checkpoints).
        fail_count: Cumulative simulation failures at checkpoint
            time. Used to drive the ``fail_limit`` stopping
            criterion on resume.
        stop_count: Consecutive-ND-equals-NPOP counter. Used by
            the ``stop_limit`` stopping criterion on resume.
        logbook_stats: DEAP ``Logbook`` with per-gen stats.
            ``gen`` / ``iter`` / ``simRuns`` / ``ND`` / ``GD`` /
            ``HV`` / ``std`` / ``min`` / ``avg`` / ``max``.
        logbook_solutions: DEAP ``Logbook`` with per-individual
            solution records (one entry per (gen, ind)).
        rndstate: ``random.getstate()`` tuple. Restoring this on
            resume gives bit-for-bit trajectory continuation.

    Properties:
        n_generations: Number of generations captured (includes
            gen 0; a 5-generation run has ``n_generations == 6``
            at its final checkpoint).
        n_pop: Population size (length of the last generation's
            list).
        n_dim: Gene dimensionality (length of one Individual).
    """

    pop_library: List[List[Any]]
    iter_tot: int
    generation: int
    fail_count: int
    stop_count: int
    logbook_stats: Any
    logbook_solutions: Any
    rndstate: tuple

    @property
    def n_generations(self) -> int:
        return len(self.pop_library)

    @property
    def n_pop(self) -> int:
        return len(self.pop_library[-1]) if self.pop_library else 0

    @property
    def n_dim(self) -> int:
        if self.n_pop == 0:
            return 0
        return len(self.pop_library[-1][0])


def load_checkpoint(path: Union[str, Path]) -> CheckpointData:
    """Load a checkpoint pickle into a typed record.

    Accepts any pickle written by the driver or by the pre-refactor
    ``ExaConstit_NSGA3.py`` (they use the same on-disk keys).

    Args:
        path: Filesystem path to the ``.pkl`` file.

    Returns:
        :class:`CheckpointData` with fields renamed from the
        on-disk keys.

    Raises:
        FileNotFoundError: If the path does not exist.
        KeyError: If any expected key is missing from the pickle
            (indicates a corrupt file or a non-checkpoint pickle).
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = pickle.load(f)
    # Map on-disk keys to record fields. The explicit indexing
    # rather than ``**raw`` protects against an old checkpoint
    # having extra keys we don't know about - we just ignore them.
    return CheckpointData(
        pop_library=raw["pop_library"],
        iter_tot=raw["iter_tot"],
        generation=raw["generation"],
        fail_count=raw["fail_count"],
        stop_count=raw["stop_count"],
        logbook_stats=raw["logbook1"],
        logbook_solutions=raw["logbook2"],
        rndstate=raw["rndstate"],
    )


# --- Best-solution picking ----------------------------------------------


def _maybe_normalize(pop_fit: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1]. Degenerate columns (all equal) stay at 0.

    Private helper shared by the two best-solution functions below.
    Handling the degenerate case matters because a multi-objective
    run's early generations often have objectives that all have the
    same value (e.g. every gene failed and the penalty is
    constant). Division by zero would produce NaN, which breaks
    argpartition.
    """
    approx_ideal = pop_fit.min(axis=0)
    approx_nadir = pop_fit.max(axis=0)
    spread = approx_nadir - approx_ideal
    # A zero spread means the objective is constant across the
    # population; in that case all rows get value 0 on that axis
    # (no information). Using np.where avoids the division entirely.
    denom = np.where(spread > 0, spread, 1.0)
    return (pop_fit - approx_ideal) / denom


def best_solution_eudist(
    pop_fit: Union[np.ndarray, Sequence[Sequence[float]]],
    weights: Optional[Sequence[float]] = None,
    p: int = 2,
    nsmallest: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    """Pick the ``nsmallest`` solutions closest to the utopian origin.

    Faithful to the pre-refactor ``BestSol.EUDIST``: computes
    ``(sum(w * fit**p))**(1/p)`` per row and returns the indices of
    the k smallest distances.

    For the canonical p=2 case this is weighted Euclidean distance
    from ``[0, ..., 0]`` — which is the utopian point of a
    minimization problem where every objective is a non-negative
    error metric. Smaller = better.

    Args:
        pop_fit: Shape ``(n, m)`` — n individuals, m objectives.
            Accepts any array-like.
        weights: Per-objective weights. ``None`` (default) uses
            uniform weights ``[1] * m``. Use this to emphasize
            particular objectives.
        p: Norm exponent. ``2`` (default) is Euclidean. Use
            ``p=np.inf`` for Chebyshev distance.
        nsmallest: Return the indices of this many best solutions.
            Default 1.
        normalize: If True, scale each objective to [0, 1] across
            the population before distance computation. Useful
            when objectives have very different scales (e.g. a
            stress-RMSE of order 100 and a slope-RMSE of order 0.1).

    Returns:
        A 1-D int numpy array of indices, length ``nsmallest``.
        Order among returned indices is unspecified (matches
        :func:`numpy.argpartition` behavior) — sort the result
        yourself if you need a deterministic order.

    Raises:
        ValueError: If ``pop_fit`` is not 2-D, or if ``weights``
            length doesn't match the objective count.

    Example:
        Pick the 3 best solutions of a 4-objective population,
        normalizing because stress and slope have different scales::

            idx = best_solution_eudist(
                pop_fit, normalize=True, nsmallest=3,
            )
            best_genes = [pop_library[-1][i] for i in idx]
    """
    fit_arr = np.asarray(pop_fit, dtype=float)
    if fit_arr.ndim != 2:
        raise ValueError(
            f"pop_fit must be 2-D (n_ind, n_obj); got {fit_arr.ndim}-D"
        )
    n, m = fit_arr.shape
    if n == 0:
        raise ValueError("pop_fit is empty")

    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,):
            raise ValueError(
                f"weights shape {w.shape} != ({m},) for n_obj={m}"
            )

    if normalize:
        fit_arr = _maybe_normalize(fit_arr)

    # Weighted p-norm. For p=2 this is Euclidean. Using np.abs
    # before the power makes odd p values sensible if someone asks
    # for p=1 — otherwise a negative residual raised to an odd
    # power would be negative and corrupt the sum.
    dist = np.sum(w * np.abs(fit_arr) ** p, axis=1) ** (1.0 / p)

    # argpartition with kth=nsmallest puts the k smallest at the
    # front (in unspecified order among themselves). Clamp to
    # len-1 for correctness when nsmallest >= n.
    k = min(nsmallest, n - 1)
    return np.argpartition(dist, k)[:nsmallest]


def best_solution_asf(
    pop_fit: Union[np.ndarray, Sequence[Sequence[float]]],
    weights: Optional[Sequence[float]] = None,
    nsmallest: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    """Pick solutions minimizing the worst weighted objective.

    Faithful to the pre-refactor ``BestSol.ASF`` (achievement
    scalarizing function): picks the gene whose ``max`` over
    objectives, after multiplying by weights, is smallest. A good
    choice when you care about worst-case performance and don't
    want one great objective to compensate for a terrible one.

    Args:
        pop_fit: Shape ``(n, m)`` — n individuals, m objectives.
        weights: Per-objective weights. ``None`` = uniform.
        nsmallest: Return the top-k best.
        normalize: Scale to [0, 1] per objective before taking max.

    Returns:
        Array of indices of the k best solutions.
    """
    fit_arr = np.asarray(pop_fit, dtype=float)
    if fit_arr.ndim != 2:
        raise ValueError(
            f"pop_fit must be 2-D; got {fit_arr.ndim}-D"
        )
    n, m = fit_arr.shape
    if n == 0:
        raise ValueError("pop_fit is empty")

    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (m,):
            raise ValueError(
                f"weights shape {w.shape} != ({m},) for n_obj={m}"
            )
    if normalize:
        fit_arr = _maybe_normalize(fit_arr)

    asf = (fit_arr * w).max(axis=1)
    k = min(nsmallest, n - 1)
    return np.argpartition(asf, k)[:nsmallest]


class BestSol:
    """Legacy class wrapper preserving the pre-refactor API.

    Same constructor signature and method names as
    ``ExaConstit_SolPicker.BestSol``, implemented on top of
    :func:`best_solution_eudist` and :func:`best_solution_asf`.
    Existing post-processing scripts that import ``BestSol`` can
    swap the import path and keep working.

    Args:
        pop_fit: 2-D array of objective values.
        weights: Per-objective weights or ``None`` for uniform.
        normalize: Scale to [0, 1] per objective before ranking.
        nsmallest: Return this many indices from each method call.

    Example:
        Drop-in migration of a pre-refactor script::

            # old
            # from ExaConstit_SolPicker import BestSol
            # new
            from workflow_common.postprocess import BestSol
            # everything below is unchanged
            best = BestSol(pop_fit, weights=[1]*nobj, nsmallest=3)
            best_idx = best.EUDIST()
    """

    def __init__(
        self,
        pop_fit: Union[np.ndarray, Sequence[Sequence[float]]],
        weights: Optional[Sequence[float]] = None,
        normalize: bool = False,
        nsmallest: int = 1,
    ):
        # Coerce to numpy once, at construction, so repeated method
        # calls don't re-do the work.
        self._pop_fit = np.asarray(pop_fit, dtype=float)
        self._weights = weights
        self._normalize = normalize
        self._nsmallest = nsmallest

    def EUDIST(self, p: int = 2) -> np.ndarray:
        """Weighted p-norm distance from origin. See :func:`best_solution_eudist`."""
        return best_solution_eudist(
            self._pop_fit, self._weights, p=p,
            nsmallest=self._nsmallest, normalize=self._normalize,
        )

    def ASF(self) -> np.ndarray:
        """Max weighted objective. See :func:`best_solution_asf`."""
        return best_solution_asf(
            self._pop_fit, self._weights,
            nsmallest=self._nsmallest, normalize=self._normalize,
        )


# --- Re-reading sim output on demand ------------------------------------


@dataclass
class GeneResult:
    """Lightweight per-gene record for post-processing.

    A small bridge between DEAP Individuals (which carry DEAP
    machinery the post-processor doesn't need) and the framework's
    path-and-result layer. Building a ``List[List[GeneResult]]``
    from a pop_library frees downstream code from needing DEAP
    imports just to iterate results.

    Fields:
        generation: The generation in which this gene was BORN.
            Critical: an individual from gen 3 may still be in the
            population at gen 10 (via selection), but its case
            directory was written at gen 3. This field gets the
            lookup right.
        gene: Within-generation offspring index at birth. Combined
            with ``generation`` it uniquely identifies the case
            directory via the PathResolver.
        gene_vector: Parameter values (what the optimizer was
            evolving).
        fitness: Tuple of objective values. Length equals the
            problem's ``n_objectives``.
    """

    generation: int
    gene: int
    gene_vector: np.ndarray
    fitness: Tuple[float, ...]


def extract_gene_results(
    pop_library: Sequence[Sequence[Any]],
) -> List[List[GeneResult]]:
    """Convert a DEAP pop_library to a list-of-lists of :class:`GeneResult`.

    Input is the ``pop_library`` field of a :class:`CheckpointData`
    (or any equivalent: ``[[Individual, ...], ...]``). Output
    mirrors its 2-D shape so ``results[gen][i]`` corresponds to
    ``pop_library[gen][i]``.

    Each GeneResult captures the BIRTH generation (``ind.generation``)
    and BIRTH offspring index (``ind.gene``), which are what the
    case directory was indexed by, NOT the current-generation position
    in the pop_library. This matters for any individual that survived
    selection across multiple generations.

    Args:
        pop_library: Per-generation populations from a checkpoint.

    Returns:
        A list-of-lists mirroring the input shape. Each inner entry
        is a :class:`GeneResult`.
    """
    results: List[List[GeneResult]] = []
    for gen_idx, pop in enumerate(pop_library):
        gen_results: List[GeneResult] = []
        for ind in pop:
            gene_vec = np.asarray(list(ind), dtype=float)
            # ``ind.generation`` is the birth gen. getattr fallback
            # lets this function work on plain list-of-list input
            # (e.g. hand-constructed test data) where no Individual
            # wrapper is used.
            birth_gen = getattr(ind, "generation", gen_idx)
            birth_gene = getattr(ind, "gene", 0)
            fitness_vals = tuple(ind.fitness.values)
            gen_results.append(GeneResult(
                generation=birth_gen,
                gene=birth_gene,
                gene_vector=gene_vec,
                fitness=fitness_vals,
            ))
        results.append(gen_results)
    return results


def load_case_results(
    result: GeneResult,
    sim_case_idx: int,
    resolver: PathResolver,
    reader: ResultReader,
) -> Optional[CaseResultSet]:
    """Re-read a gene's sim output from disk on demand.

    The pre-refactor code cached the stress history on
    ``ind.stress`` inside DEAP Individuals. That had two
    disadvantages: it grew every checkpoint by megabytes per
    generation, and it tied post-processing to having DEAP
    installed. The new approach keeps sim output on disk (where
    the framework already writes it) and pulls it back through
    the standard ResultReader on demand.

    See :func:`load_case_results_from_archive` for the archive-
    based alternative (when case directories have been cleaned up
    from disk but the data was archived).

    Args:
        result: A :class:`GeneResult` from
            :func:`extract_gene_results`. Carries the ``generation``
            and ``gene`` indices needed for the PathResolver lookup.
        sim_case_idx: Which SimCase's output to load. For the
            common ``NOBJ = NEXP * 2`` pattern (stress + slope per
            experiment sharing a sim), this would be the experiment
            index, not the objective index.
        resolver: The same :class:`PathResolver` the original run
            used. If the patterns changed between the run and the
            post-processor, directories won't be found.
        reader: A :class:`ResultReader` configured for the sim
            code's output format. For ExaConstit, typically a
            :class:`TextTableReader`.

    Returns:
        The parsed :class:`CaseResultSet`, or ``None`` if the
        case directory is missing or the reader fails. Returning
        None rather than raising lets post-processors iterate
        over many genes without a single missing directory
        aborting the whole analysis.

    Example:
        Plot the stress-strain curve of the best gene in the
        last generation::

            ckp = load_checkpoint("checkpoint_gen_50.pkl")
            results = extract_gene_results(ckp.pop_library)
            final_gen = results[-1]

            fits = np.array([r.fitness for r in final_gen])
            best_idx = best_solution_eudist(fits, nsmallest=1)[0]
            best = final_gen[best_idx]

            case = load_case_results(
                best, sim_case_idx=0, resolver=resolver, reader=reader,
            )
            if case is not None:
                df = case.df("avg_stress")
                plot_stress_strain_overlay(
                    df["Time"].to_numpy() * strain_rate,
                    df["Szz"].to_numpy(),
                    exp_strain, exp_stress,
                )
    """
    ctx = CaseContext(
        generation=result.generation,
        gene=result.gene,
        obj=sim_case_idx,
    )
    layout = CaseLayout(ctx=ctx, resolver=resolver)
    if not layout.working_dir.is_dir():
        # Case directory wiped between run and analysis, or the
        # resolver patterns differ from what the run used. Warn
        # but don't raise so batch post-processing continues.
        logger.warning(
            "case directory missing: %s (gen=%d gene=%d sim_case=%d)",
            layout.working_dir, result.generation, result.gene, sim_case_idx,
        )
        return None
    try:
        return reader.read(layout)
    except Exception as e:
        # Same reasoning as above - reader errors on one case
        # should not halt processing of others.
        logger.warning(
            "failed to read case %s: %s", layout.working_dir, e,
        )
        return None


def load_case_results_from_archive(
    archive,  # ArchiveDB, forward-typed to avoid circular import
    result: "GeneResult",
    sim_case_idx: int,
    run_id: Optional[str] = None,
) -> Optional[CaseResultSet]:
    """Archive-based equivalent of :func:`load_case_results`.

    Use this when the filesystem case directories have been cleaned
    up (the rolling-cleanup pattern with
    ``cleanup_keep_generations`` enabled during the run) but the
    data was archived to a SQLite database. The returned
    :class:`CaseResultSet` is reconstructed from the stored
    DataFrames; ``source_path`` on each table is synthesized as
    ``archive://<output_name>`` since there's no on-disk file.

    Args:
        archive: An open :class:`~workflow_common.archive.ArchiveDB`.
            Can be read-only or writable.
        result: A :class:`GeneResult` from
            :func:`extract_gene_results`. The
            ``result.generation`` (birth gen) and ``result.gene``
            (birth offspring idx) fields select the case.
        sim_case_idx: Which SimCase's output to pull. Same
            semantics as the disk variant.
        run_id: The archive's run ID. If ``None``, uses the single
            run in the archive — convenient for the common case of
            "archive has one run in it". Raises if the archive
            holds multiple runs and ``run_id`` is not specified.

    Returns:
        The reconstructed :class:`CaseResultSet`, or ``None`` if
        no outputs were archived for those coordinates.

    Raises:
        ValueError: If ``run_id`` is None and the archive contains
            zero or multiple runs.

    Example:
        End-to-end flow for an archive-only post-process::

            from workflow_common import ArchiveDB
            from workflow_common.postprocess import (
                extract_gene_results, load_checkpoint,
                load_case_results_from_archive,
                best_solution_eudist,
            )
            import numpy as np

            ckp = load_checkpoint("ck/checkpoint_gen_50.pkl")
            results = extract_gene_results(ckp.pop_library)
            final_gen = results[-1]

            fits = np.array([r.fitness for r in final_gen])
            best = final_gen[best_solution_eudist(fits)[0]]

            with ArchiveDB("opt.db", readonly=True) as archive:
                case = load_case_results_from_archive(
                    archive, best, sim_case_idx=0,
                )
                df = case.df("avg_stress")
    """
    if run_id is None:
        runs = archive.list_runs()
        if len(runs) == 0:
            raise ValueError("archive is empty (no runs)")
        if len(runs) > 1:
            raise ValueError(
                f"archive has {len(runs)} runs; pass run_id explicitly. "
                f"Available: {[r.run_id for r in runs]}"
            )
        run_id = runs[0].run_id
    return archive.load_case_outputs(
        run_id,
        birth_gen=result.generation,
        birth_gene=result.gene,
        sim_case_idx=sim_case_idx,
    )


# --- Plotting (matplotlib soft dep) -------------------------------------


def _lazy_import_matplotlib():
    """Import matplotlib.pyplot on demand.

    Matplotlib is a soft dependency: the data-analysis functions
    above work fine without it. Only the plot functions need it,
    and those are called explicitly by the user. Deferring the
    import here lets the module load in headless CI environments
    where matplotlib might not be installed at all.
    """
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for postprocess plotting; "
            "install with `pip install matplotlib`"
        ) from e


def plot_stress_strain_overlay(
    sim_strain: np.ndarray,
    sim_stress: np.ndarray,
    exp_strain: np.ndarray,
    exp_stress: np.ndarray,
    *,
    title: Optional[str] = None,
    ax=None,
    label_prefix: str = "",
    show: bool = False,
):
    """Overlay sim and exp stress-strain curves on one axis.

    Minimal replacement for the pre-refactor ``ExaPlots.StressStrain``.
    Plots experimental as solid black and simulated as dashed red,
    with grid, legend, and optional title. Returns the figure and
    axis so callers can stack more plots or customize.

    Args:
        sim_strain, sim_stress: Simulated curve arrays (same length).
        exp_strain, exp_stress: Experimental reference arrays.
        title: Optional plot title. Default None.
        ax: Existing matplotlib Axes to draw on. ``None`` (default)
            creates a new figure.
        label_prefix: String prepended to the legend labels. Useful
            when plotting multiple cases on one figure — pass e.g.
            ``"exp1_"`` and ``"exp2_"`` on successive calls sharing
            an axis.
        show: If True, call ``plt.show()`` at the end. Default
            False so scripts can batch multiple plots before
            displaying (or save to file without displaying).

    Returns:
        ``(fig, ax)`` tuple. Even if ``ax`` was supplied, the
        parent figure is returned so the caller can save it.
    """
    plt = _lazy_import_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    ax.plot(exp_strain, exp_stress, "k-", label=f"{label_prefix}exp")
    ax.plot(sim_strain, sim_stress, "r--", label=f"{label_prefix}sim")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return fig, ax


def plot_pareto_front(
    pop_fit: np.ndarray,
    *,
    best_idx: Optional[Sequence[int]] = None,
    axis_labels: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax=None,
    show: bool = False,
):
    """2-D scatter of a Pareto front with optional "best" highlight.

    Minimal replacement for the 2-D portion of the pre-refactor
    ``ExaPlots.ObjFun2D``. Plots all population points as blue
    crosses, optionally highlights a subset as red circles, and
    marks the utopian origin ``[0, 0]`` as a black plus.

    The 3-D variant (``ExaPlots.ObjFun3D``) is intentionally NOT
    ported: for more than 2 objectives scatter plots do not scale
    usefully. Use parallel-coordinates or radar plots from
    matplotlib / pymoo visualization instead. This function
    raises on 3+ dim inputs to fail loudly rather than truncate.

    Args:
        pop_fit: Shape ``(n, 2)`` objective values.
        best_idx: Optional indices to highlight. Typically the
            output of :func:`best_solution_eudist`.
        axis_labels: ``(x_label, y_label)``. Default
            ``("f_1", "f_2")``.
        title: Optional plot title.
        ax: Existing Axes to draw on. ``None`` creates a new figure.
        show: Call ``plt.show()`` at the end. Default False.

    Returns:
        ``(fig, ax)`` tuple.

    Raises:
        ValueError: If ``pop_fit`` is not shape ``(n, 2)``.
    """
    plt = _lazy_import_matplotlib()
    fit_arr = np.asarray(pop_fit, dtype=float)
    if fit_arr.ndim != 2 or fit_arr.shape[1] != 2:
        raise ValueError(
            f"plot_pareto_front only supports 2-D objective space; "
            f"got shape {fit_arr.shape}. For >2 objectives use "
            f"parallel-coordinates or radar plots."
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.scatter(
        fit_arr[:, 0], fit_arr[:, 1],
        c="tab:blue", marker="x", label="population",
    )
    if best_idx is not None and len(best_idx) > 0:
        best = fit_arr[list(best_idx)]
        ax.scatter(
            best[:, 0], best[:, 1],
            s=80, facecolors="none", edgecolors="red",
            linewidths=1.5, label="best",
        )
    # Utopian origin — faithful to the pre-refactor plot.
    ax.scatter([0], [0], c="black", marker="+", s=100, label="utopia")
    labels = axis_labels if axis_labels is not None else ("f_1", "f_2")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return fig, ax
