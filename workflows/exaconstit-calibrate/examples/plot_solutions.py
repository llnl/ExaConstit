"""
Plot the top-N optimized solutions vs the experimental reference.

Companion to ``nsga3_calibration.py``. Once a calibration finishes,
point this script at the workspace and it will:

1. Find the SQLite archive automatically (resolves
   ``calibration.db`` in the workspace, or any unique ``*.db`` file).
2. Pull the top-N gene records from the archive — ranked either by
   L2 norm of the fitness tuple (the "balanced winner" — closest to
   the utopian origin) or by a single objective. Reuses
   :func:`workflows.optimization.inspect_archive.select_top_genes`,
   which is the same selection logic that powers the inspect-archive
   CLI's ``--pareto-only`` view.
3. For each selected gene and each SimCase, reconstruct the
   stress-strain curve from the archived ``avg_stress`` /
   ``avg_def_grad`` tables. The case directories on disk DO NOT need
   to still exist; the archive carries everything required.
4. Draw an interactive matplotlib figure overlaying:
   * the experimental reference curve (thick black);
   * each selected gene's simulated curve (semi-transparent line,
     red-to-blue gradient by rank);
   * a slider that fades curves below a chosen rank, so users can
     interactively narrow focus from N to K without re-running;
   * a click panel that prints the gene's parameter values when
     the user clicks a curve.
5. Optionally produce a 2-D Pareto-front scatter (``--pareto``) for
   any pair of objectives, with the L2-closest gene highlighted.

Usage examples
--------------

    # Top 10 by L2 norm (the headline plot):
    python examples/plot_solutions.py calibration_run \\
        --top 10 --experimental experiments/exp1.csv experiments/exp2.csv

    # Top 5 ranked by objective 0 (e.g. stress-RMSE for exp1):
    python examples/plot_solutions.py calibration_run \\
        --top 5 --objective 0 \\
        --experimental experiments/exp1.csv experiments/exp2.csv

    # 2-D Pareto-front scatter for objectives 0 vs 2:
    python examples/plot_solutions.py calibration_run \\
        --pareto 0,2 --experimental experiments/exp1.csv experiments/exp2.csv

    # Save without showing (headless / batch / CI):
    python examples/plot_solutions.py calibration_run \\
        --top 10 --save out.png --no-show \\
        --experimental experiments/exp1.csv experiments/exp2.csv

Programmatic use (from a notebook):

    from examples.plot_solutions import plot_top_solutions_overlay
    plot_top_solutions_overlay(
        "calibration_run", top_n=10, mode="l2",
        experimental_paths=["experiments/exp1.csv",
                            "experiments/exp2.csv"],
    )

Architectural note
------------------
This module is deliberately a thin orchestrator over functionality
that already lives in the framework:

* archive resolution + run selection + top-N ranking →
  :mod:`workflows.optimization.inspect_archive` (its public selection
  API: ``select_top_genes``, ``pick_run``, ``resolve_archive_path``,
  ``collect_all_genes``).
* simulated stress-strain reconstruction →
  :meth:`ArchiveDB.load_case_outputs` (reconstitutes the on-disk
  output tables from the archive's blob storage).
* strain/stress extraction → :class:`StressStrainExtractor`.
* curve overlay + Pareto scatter →
  :func:`workflow_common.postprocess.plot_stress_strain_overlay`
  and :func:`plot_pareto_front`.

The only NEW logic in this file is the matplotlib interactivity
glue (Slider for opacity, pick events for click-to-show parameters)
and the per-SimCase subplot layout. Everything else is composed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import (
    Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np

from workflow_common import (
    ArchiveDB,
    GeneRecord,
    PchipSmoother,
    StressStrainExtractor,
    load_experimental_csv,
)
from workflow_common.postprocess import plot_pareto_front
from workflow_common.results import CaseResultSet
from workflows.optimization.inspect_archive import (
    RankedGene,
    collect_all_genes,
    pick_run,
    resolve_archive_path,
    select_top_genes,
)


# --- Helpers -------------------------------------------------------------


def _resolve_objective_index(
    objective: Union[int, str], objective_labels: Sequence[str],
) -> int:
    """Map an int-or-name objective specifier to an integer index.

    Names take precedence (so ``--objective stress_1`` works); a
    bare numeric string falls through to int parse. Raises
    :class:`ValueError` with the available labels listed if neither
    matches.
    """
    if isinstance(objective, str):
        if objective in objective_labels:
            return list(objective_labels).index(objective)
        try:
            idx = int(objective)
        except ValueError:
            raise ValueError(
                f"objective {objective!r} not found. Available: "
                f"{list(objective_labels)}"
            )
        if not 0 <= idx < len(objective_labels):
            raise ValueError(
                f"objective index {idx} out of range "
                f"[0, {len(objective_labels)})"
            )
        return idx
    if not 0 <= objective < len(objective_labels):
        raise ValueError(
            f"objective index {objective} out of range "
            f"[0, {len(objective_labels)})"
        )
    return objective


def _ranked_genes_for_mode(
    archive: ArchiveDB,
    run_id: str,
    *,
    objective_labels: Sequence[str],
    top_n: int,
    mode: str,
    objective: Optional[Union[int, str]],
) -> List[RankedGene]:
    """Pick the genes to plot using inspect_archive's selection logic.

    The three modes map to ``select_top_genes`` outputs:

    * ``"l2"`` — return the ``"l2"`` category list directly.
    * ``"objective"`` — pick a single per-objective category by
      label/index.
    * ``"last-gen"`` — bypass ``select_top_genes`` and return every
      gene from the highest gen_idx (no ranking; sorted by pop_idx
      so the order matches what a user would see in
      ``inspect_archive --gen <last>``).
    """
    if mode == "last-gen":
        all_genes = collect_all_genes(archive, run_id)
        if not all_genes:
            return []
        last_gen = max(g.gen_idx for g in all_genes)
        last = sorted(
            (g for g in all_genes if g.gen_idx == last_gen),
            key=lambda g: g.pop_idx,
        )
        l2_full = np.array([
            float(np.linalg.norm(g.fitness)) for g in last
        ])
        if top_n <= 0 or top_n > len(last):
            take = len(last)
        else:
            take = top_n
        return [
            RankedGene(
                gene=g, rank=i, category=f"gen={last_gen}",
                score=float(g.pop_idx), l2_norm=float(l2_full[i]),
            )
            for i, g in enumerate(last[:take])
        ]

    all_genes = collect_all_genes(archive, run_id)
    if not all_genes:
        return []

    if mode == "l2":
        selection = select_top_genes(
            all_genes,
            objective_labels=objective_labels,
            top_n=top_n,
            categories=("l2",),
        )
        return selection.get("l2", [])

    if mode == "objective":
        if objective is None:
            raise ValueError("mode='objective' requires --objective")
        idx = _resolve_objective_index(objective, objective_labels)
        label = objective_labels[idx]
        selection = select_top_genes(
            all_genes,
            objective_labels=objective_labels,
            top_n=top_n,
            categories=("per_objective",),
        )
        return selection.get(f"obj:{label}", [])

    raise ValueError(
        f"mode={mode!r} not recognized. Use 'l2', 'objective', or 'last-gen'."
    )


def _discover_sim_case_count(
    archive: ArchiveDB, run_id: str, sample_gene: GeneRecord,
    *, max_check: int = 16,
) -> int:
    """Probe how many SimCases were archived for a sample gene.

    The archive doesn't store an explicit "n_sim_cases" field — it's
    a derived property of which sim_case_idx values have rows under
    a given ``(run_id, birth_gen, birth_gene)`` triple. We probe
    incrementally until ``load_case_outputs`` returns ``None``.

    ``max_check`` caps the probe to avoid pathological loops if some
    archive becomes sparse; 16 covers any reasonable calibration
    setup (most have 1-4 SimCases).
    """
    count = 0
    for idx in range(max_check):
        outputs = archive.load_case_outputs(
            run_id,
            birth_gen=sample_gene.birth_gen,
            birth_gene=sample_gene.birth_gene,
            sim_case_idx=idx,
        )
        if outputs is None:
            break
        count += 1
    return count


def _extract_curve(
    case_result: CaseResultSet, extractor: StressStrainExtractor,
    *, rank: int, sim_case_idx: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run an extractor against one CaseResultSet, swallow + warn on failure.

    Bad data shouldn't crash the whole plot; print a one-line warning
    and let the caller skip the curve.
    """
    try:
        strain, stress = extractor.extract(case_result)
        return np.asarray(strain), np.asarray(stress)
    except Exception as e:  # noqa: BLE001
        print(
            f"warning: skipping rank={rank} sim_case={sim_case_idx} "
            f"(extractor failed: {e})",
            file=sys.stderr,
        )
        return None


def _load_or_extract_curve(
    archive: ArchiveDB,
    run_id: str,
    *,
    birth_gen: int,
    birth_gene: int,
    sim_case_idx: int,
    extractor: StressStrainExtractor,
    rank_label: int,
    experimental_reference: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get an extracted curve for one (gene, sim_case).

    Preference order:

    1. Archived curve from the ``case_curves`` table. This is the
       FULL-range curve the framework stored at run time (window
       stripped before extraction). Faster than re-extracting and
       guaranteed to match what the optimizer scored against,
       modulo the optimizer's own windowing on top.
    2. Re-extract from raw ``case_outputs`` using the supplied
       extractor. Used for archives written before the
       ``case_curves`` table existed, or for cases where the
       run-time extraction failed but the raw outputs are
       still useful.

    If ``experimental_reference`` is supplied as ``(exp_ind,
    exp_dep)``, both the loaded ``ind`` and ``dep`` arrays are
    sign-matched against their respective exp axes via
    :func:`workflow_common.objectives.match_sign_to_reference`.
    This catches the case where an extractor accidentally
    absolute-values its output (a sign-stripped strain rate, an
    evaluator that calls ``np.abs`` on its arrays before scoring
    RMSE) — without correction, the plot shows the simulated curve
    flipped relative to the experimental reference, which is
    confusing. The correction is purely cosmetic; it doesn't
    re-write the archive.

    Returns ``(independent, dependent)`` arrays or ``None`` if
    neither path produces data.
    """
    archived = archive.load_case_curve(
        run_id,
        birth_gen=birth_gen, birth_gene=birth_gene,
        sim_case_idx=sim_case_idx,
    )
    if archived is not None:
        ind, dep, _ind_label, _dep_label = archived
    else:
        outputs = archive.load_case_outputs(
            run_id,
            birth_gen=birth_gen, birth_gene=birth_gene,
            sim_case_idx=sim_case_idx,
        )
        if outputs is None:
            return None
        pair = _extract_curve(
            outputs, extractor,
            rank=rank_label, sim_case_idx=sim_case_idx,
        )
        if pair is None:
            return None
        ind, dep = pair
    if experimental_reference is not None:
        exp_ind = np.asarray(experimental_reference[0], dtype=float)
        exp_dep = np.asarray(experimental_reference[1], dtype=float)
        ind = np.asarray(ind, dtype=float)
        dep = np.asarray(dep, dtype=float)
        try:
            # Dependent axis: bring exp_dep onto sim's grid via
            # PCHIP, then element-wise np.copysign so each sim
            # point inherits the sign of the experimental
            # response at the same strain magnitude. ``np.abs``
            # on both x-arrays handles the cross-sign case
            # (sim positive from absolute-valued extractor, exp
            # negative from compression test) — the smoother
            # works on monotonic |strain|, sign comes back via
            # copysign.
            smoother = PchipSmoother(strict_monotonic=False)
            exp_dep_at_sim = smoother.sample_at(
                np.abs(exp_ind), exp_dep, np.abs(ind),
            ).y
            dep = np.copysign(dep, exp_dep_at_sim)
            # Independent axis: a scalar sign drawn from the
            # experimental's last strain value gives the test
            # direction unambiguously. (For cyclic data ending
            # back at zero this fails; the framework targets
            # monotonic loading.)
            if exp_ind.size > 0:
                ind = np.copysign(ind, exp_ind[-1])
        except (ValueError, RuntimeError) as e:
            # Degenerate exp data (e.g. single point, all
            # duplicates) → leave the curve as-is rather than
            # crashing the plot.
            print(
                f"warning: sim/exp sign-match for sim_case={sim_case_idx} "
                f"skipped — {e}", file=sys.stderr,
            )
    return ind, dep


def _format_param_panel(
    rg: RankedGene, param_names: Sequence[str],
    objective_labels: Sequence[str],
) -> str:
    """Pretty-print a gene's parameters for the click panel.

    Format is dense (monospace, line per param) since the panel sits
    in a tight bottom strip of the figure. Includes fitness values
    too because users almost always want to see "what trade-off did
    this winner make?" alongside the parameters.
    """
    lines = [
        f"rank {rg.rank}  category={rg.category}  "
        f"score={rg.score:.4g}  L2={rg.l2_norm:.4g}",
    ]
    for name, val in zip(param_names, rg.gene.gene_vector):
        lines.append(f"  {name} = {val:.6g}")
    lines.append(
        "  fitness: "
        + ", ".join(
            f"{lbl}={v:.4g}"
            for lbl, v in zip(objective_labels, rg.gene.fitness)
        )
    )
    lines.append(
        f"  birth: gen={rg.gene.birth_gen}, "
        f"gene={rg.gene.birth_gene}"
    )
    return "\n".join(lines)


def _format_gene_panel(
    g: GeneRecord, param_names: Sequence[str],
    objective_labels: Sequence[str],
    *, header: str = "",
) -> str:
    """Pretty-print a plain GeneRecord for the Pareto-click panel.

    Same shape as :func:`_format_param_panel` but works directly on
    a ``GeneRecord`` (no rank/category metadata, since Pareto-front
    points aren't ranked relative to each other in the same way the
    overlay's top-N list is). The optional ``header`` line lets the
    caller annotate which point was clicked (e.g. its scatter
    index or "L2 winner").
    """
    lines: List[str] = []
    if header:
        lines.append(header)
    lines.append(
        f"L2={float(np.linalg.norm(g.fitness)):.4g}  "
        f"birth: gen={g.birth_gen}, gene={g.birth_gene}"
    )
    for name, val in zip(param_names, g.gene_vector):
        lines.append(f"  {name} = {val:.6g}")
    lines.append(
        "  fitness: "
        + ", ".join(
            f"{lbl}={v:.4g}"
            for lbl, v in zip(objective_labels, g.fitness)
        )
    )
    return "\n".join(lines)


def _resolve_experimental_for_case(
    archive: ArchiveDB,
    run_id: str,
    sim_case_idx: int,
    csv_fallback: Optional[Sequence[Optional[Union[str, Path]]]],
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[str]]]:
    """Get experimental ``(strain, stress, label)`` for one SimCase.

    Resolution order:

    1. Archive's ``experiments`` table — written by the driver at
       run start. This is the "no manual ordering" path: the
       SimCase index is the key, so users don't have to remember
       which CSV maps to which experiment.
    2. CSV path at ``csv_fallback[sim_case_idx]`` — for archives
       written before this feature existed, or for users who want
       to override with different reference data post-hoc.
    3. ``None`` — no experimental overlay for this SimCase.

    Returns ``(strain, stress, label)`` on success, ``None`` if no
    source resolved.
    """
    # Archive first.
    result = archive.load_experiment(run_id, sim_case_idx)
    if result is not None:
        label, df = result
        if {"strain", "stress"}.issubset(df.columns):
            return (
                df["strain"].to_numpy(), df["stress"].to_numpy(), label,
            )
        cols = list(df.columns)
        return (
            df[cols[0]].to_numpy(), df[cols[1]].to_numpy(), label,
        )
    # CSV fallback.
    if (csv_fallback is not None
            and sim_case_idx < len(csv_fallback)
            and csv_fallback[sim_case_idx] is not None):
        path = Path(csv_fallback[sim_case_idx])
        try:
            df = load_experimental_csv(path)
            if {"strain", "stress"}.issubset(df.columns):
                return (
                    df["strain"].to_numpy(), df["stress"].to_numpy(),
                    str(path.name),
                )
            cols = list(df.columns)
            return (
                df[cols[0]].to_numpy(), df[cols[1]].to_numpy(),
                str(path.name),
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: could not load CSV {path} for SimCase "
                f"{sim_case_idx}: {e}",
                file=sys.stderr,
            )
    return None


def _resolve_extractor_for_case(
    archive: ArchiveDB,
    run_id: str,
    sim_case_idx: int,
    *,
    factory: Optional[Callable[[int], StressStrainExtractor]] = None,
) -> StressStrainExtractor:
    """Pick a StressStrainExtractor for the given SimCase.

    Resolution order:

    1. Archive's stored extractor config (written by the driver
       at run start). This is the source of truth: it captures
       the EXACT settings the optimizer used to score curves,
       including ``strain_source``, ``strain_rate``, column-name
       overrides, etc. Without this, a default-built extractor
       can silently produce different curves from what the
       optimizer saw.
    2. ``factory(sim_case_idx)`` if the caller supplied one.
       Lets a Python caller of plot_top_solutions_overlay /
       plot_pareto_front_with_l2_winner pass per-SimCase
       extractor configs explicitly.
    3. ``StressStrainExtractor()`` default (ExaConstit z-axis
       Biot-strain conventions). The "no idea what they used,
       try defaults" fallback for archives written before
       extractor configs were stored.

    The window field is STRIPPED from any archived extractor
    before return: the plotter wants full-curve visibility (so
    the user sees what's going on outside the optimization
    region) and shades the window separately via
    ``case_data["minmax_strain"]`` from the archive's
    experiments.minmax_strain column.
    """
    cfg = archive.load_extractor_config(run_id, sim_case_idx)
    if cfg is not None:
        try:
            ext = StressStrainExtractor.from_dict(cfg)
            # Plotter wants full curves with the window shaded as
            # an overlay rather than data cropped at extraction.
            if ext.window is not None:
                ext = StressStrainExtractor.from_dict({
                    **cfg, "window": None,
                })
            return ext
        except Exception as e:  # noqa: BLE001
            print(
                f"warning: archived extractor config for SimCase "
                f"{sim_case_idx} failed to load ({e}); falling back",
                file=sys.stderr,
            )
    if factory is not None:
        return factory(sim_case_idx)
    return StressStrainExtractor()


def _slope_of(strain: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Numerical slope dStress/dStrain via finite difference.

    Uses ``np.gradient`` so endpoints are handled with one-sided
    differences and the slope array has the same length as the
    inputs. This matches what ``StressStrainObjective``'s
    slope-extraction does internally — deliberately so, because
    if the optimizer scored a slope objective, the slope values
    we plot here must be derived the same way to be comparable.
    """
    return np.gradient(np.asarray(stress, dtype=float),
                       np.asarray(strain, dtype=float))


def _lazy_matplotlib():
    """Soft dep on matplotlib with a useful error if missing."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        return plt, Slider
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plot_solutions; install via "
            "`pip install -e \".[plot]\"` or `pip install matplotlib`."
        ) from e


# --- Public plotting functions -------------------------------------------


def plot_top_solutions_overlay(
    workspace: Union[str, Path],
    *,
    top_n: int = 10,
    mode: str = "l2",
    objective: Optional[Union[int, str]] = None,
    run_id: Optional[str] = None,
    experimental_paths: Optional[Sequence[Optional[Union[str, Path]]]] = None,
    extractor_factory: Optional[
        Callable[[int], StressStrainExtractor]
    ] = None,
    show_slopes: bool = True,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """Build the headline interactive figure.

    Layout: a 2-row × N-SimCase-column grid of subplots.

    * Top row — stress-strain. Top-N simulated curves overlaid on
      the experimental reference (thick black). Curves colored by
      rank with a coolwarm-reversed colormap so rank 0 is red.
    * Bottom row — slope-strain (dStress/dStrain). Same overlay,
      same colors. Computed via ``np.gradient`` for both sim and
      exp so the curves correspond to what a slope objective
      would have scored. Hidden if ``show_slopes=False`` or if
      no experimental data is available.

    Below the subplots:

    * a slider that fades curves below a chosen rank, so users can
      narrow visual focus from N to K interactively;
    * a click panel that prints the gene's parameter values when
      the user clicks a curve.

    Experimental data resolution (per SimCase):

    1. The archive's ``experiments`` table (written by the driver
       at run start). This is the canonical source — keyed by
       ``sim_case_idx``, so users don't have to remember CSV
       ordering.
    2. ``experimental_paths[sim_case_idx]`` if archive doesn't
       have it. Lets archives written before the experiments
       table existed still produce useful plots.
    3. None — the experimental overlay is omitted for that case.

    Args:
        workspace: Workspace dir, or direct ``.db`` archive path.
        top_n: Number of solutions to plot. ``0`` or negative shows
            all available.
        mode: ``"l2"`` (default), ``"objective"``, or ``"last-gen"``.
        objective: Required for ``mode='objective'``; integer index
            or label string.
        run_id: Optional explicit run UUID (default: latest).
        experimental_paths: Per-SimCase CSV fallbacks; only used
            when the archive doesn't carry experimental data for
            a given SimCase.
        extractor_factory: ``f(sim_case_idx) -> StressStrainExtractor``;
            lets per-SimCase extractor configs match what was used
            at run time. Default: vanilla
            ``StressStrainExtractor()`` for every case.
        show_slopes: Include the slope-strain row. Default True.
        save: Path to write the figure to (PNG / PDF / SVG).
        show: Whether to call ``plt.show()`` blocking. Set False
            for headless / batch.

    Returns:
        The matplotlib ``Figure``.
    """
    plt, Slider = _lazy_matplotlib()

    db_path = resolve_archive_path(Path(workspace), "calibration.db")
    with ArchiveDB(db_path, readonly=True) as archive:
        meta = pick_run(archive, run_id, latest_if_none=True)
        run_id = meta.run_id

        ranked = _ranked_genes_for_mode(
            archive, run_id,
            objective_labels=meta.objective_labels,
            top_n=top_n, mode=mode, objective=objective,
        )
        if not ranked:
            raise RuntimeError(
                f"no plottable solutions in run {run_id} "
                f"(mode={mode!r}). Either no genes have finite "
                f"fitness, or the run hasn't completed gen 0."
            )

        n_sim_cases = _discover_sim_case_count(
            archive, run_id, sample_gene=ranked[0].gene,
        )
        if n_sim_cases == 0:
            raise RuntimeError(
                "best gene has no archived case outputs. "
                "Was archiving disabled at run time?"
            )

        # Per-(sim_case, rank) (strain, stress) for simulated curves.
        per_case_curves: List[
            List[Tuple[RankedGene, np.ndarray, np.ndarray]]
        ] = []
        # Per-sim_case experimental (strain, stress, label) or None.
        per_case_exp: List[
            Optional[Tuple[np.ndarray, np.ndarray, Optional[str]]]
        ] = []
        # Per-sim_case (lo, hi) optimization window or None. Either
        # side of the tuple may itself be None for "unbounded on
        # that side." Plotter shades the corresponding strain region
        # so users can see what range the optimizer scored against.
        per_case_window: List[
            Optional[Tuple[Optional[float], Optional[float]]]
        ] = []

        for sc_idx in range(n_sim_cases):
            extractor = _resolve_extractor_for_case(
                archive, run_id, sc_idx, factory=extractor_factory,
            )
            # Resolve experimental data FIRST so we can sign-match
            # the simulated curves against it as we load them. This
            # corrects archived data where an extractor accidentally
            # absolute-values its output (e.g. a sign-stripped
            # strain rate producing positive strain on a compression
            # run): without correction the simulated curve plots
            # mirrored relative to the experimental reference.
            # The correction is applied in :func:`_load_or_extract_curve`.
            exp_resolved = _resolve_experimental_for_case(
                archive, run_id, sc_idx, experimental_paths,
            )
            exp_reference: Optional[Tuple[np.ndarray, np.ndarray]] = None
            if exp_resolved is not None:
                exp_strain, exp_stress, _exp_label = exp_resolved
                exp_reference = (exp_strain, exp_stress)
            sc_curves: List[
                Tuple[RankedGene, np.ndarray, np.ndarray]
            ] = []
            for rg in ranked:
                pair = _load_or_extract_curve(
                    archive, run_id,
                    birth_gen=rg.gene.birth_gen,
                    birth_gene=rg.gene.birth_gene,
                    sim_case_idx=sc_idx,
                    extractor=extractor,
                    rank_label=rg.rank,
                    experimental_reference=exp_reference,
                )
                if pair is None:
                    continue
                sc_curves.append((rg, pair[0], pair[1]))
            per_case_curves.append(sc_curves)
            per_case_exp.append(exp_resolved)
            # Pull the optimization window the optimizer was
            # constrained to. Returns None on archives that pre-date
            # the minmax_strain column (the read path tolerates that
            # gracefully) and on SimCases that didn't supply one.
            per_case_window.append(
                archive.load_experiment_window(run_id, sc_idx)
            )

    # --- Lay out the figure ----------------------------------------
    n_cols = max(n_sim_cases, 1)
    # Row layout: stress (top), slope (middle, optional), spacer,
    # slider, panel.
    has_slopes = show_slopes
    if has_slopes:
        # 5 stress + spacer + 5 slope + spacer + slider + panel.
        height_ratios = [1] * 5 + [0.15] + [1] * 5 + [0.2, 0.5, 0.7]
        fig = plt.figure(figsize=(6 * n_cols, 10))
    else:
        height_ratios = [1] * 7 + [0.2, 0.5, 0.7]
        fig = plt.figure(figsize=(6 * n_cols, 7.5))

    gs = fig.add_gridspec(
        nrows=len(height_ratios), ncols=n_cols,
        height_ratios=height_ratios,
        hspace=0.35, wspace=0.25,
    )
    if has_slopes:
        stress_axes = [fig.add_subplot(gs[0:5, c]) for c in range(n_cols)]
        slope_axes = [fig.add_subplot(gs[6:11, c]) for c in range(n_cols)]
        slider_ax = fig.add_subplot(gs[12, :])
        panel_ax = fig.add_subplot(gs[13, :])
    else:
        stress_axes = [fig.add_subplot(gs[0:7, c]) for c in range(n_cols)]
        slope_axes = []
        slider_ax = fig.add_subplot(gs[8, :])
        panel_ax = fig.add_subplot(gs[9, :])

    panel_ax.axis("off")
    panel_text = panel_ax.text(
        0.01, 0.95,
        "click any curve to see its parameter values",
        transform=panel_ax.transAxes,
        va="top", ha="left", family="monospace", fontsize=9,
    )

    n_total = len(ranked)
    cmap = plt.get_cmap("coolwarm_r")
    line_to_rg: Dict[object, RankedGene] = {}
    # line_groups[rank] = list of every Line2D for that rank across
    # both stress and slope rows. Slider toggles them all together.
    line_groups: List[List[object]] = [[] for _ in range(n_total)]

    for sc_idx in range(n_cols):
        stress_ax = stress_axes[sc_idx]
        slope_ax = slope_axes[sc_idx] if has_slopes else None
        curves = per_case_curves[sc_idx]
        exp_data = per_case_exp[sc_idx]

        for rg, strain, stress in curves:
            color = cmap(rg.rank / max(n_total - 1, 1))
            line, = stress_ax.plot(
                strain, stress,
                color=color, alpha=0.7, linewidth=1.4,
                picker=5,
                label=f"rank {rg.rank}",
            )
            line_to_rg[line] = rg
            line_groups[rg.rank].append(line)

            if slope_ax is not None:
                slope = _slope_of(strain, stress)
                slope_line, = slope_ax.plot(
                    strain, slope,
                    color=color, alpha=0.7, linewidth=1.4,
                    picker=5,
                    label=f"rank {rg.rank}",
                )
                line_to_rg[slope_line] = rg
                line_groups[rg.rank].append(slope_line)

        # Experimental overlay — same source for both rows.
        if exp_data is not None:
            ex, ey, exp_label = exp_data
            stress_ax.plot(
                ex, ey, "k-", linewidth=2.5, label="experimental",
                zorder=10,
            )
            if slope_ax is not None:
                exp_slope = _slope_of(ex, ey)
                slope_ax.plot(
                    ex, exp_slope, "k-", linewidth=2.5,
                    label="experimental", zorder=10,
                )

        # Per-SimCase title using the experimental label if available.
        title = f"SimCase {sc_idx}"
        if exp_data is not None and exp_data[2]:
            title += f" — {exp_data[2]}"
        stress_ax.set_xlabel("Strain")
        stress_ax.set_ylabel("Stress")
        stress_ax.set_title(title)
        stress_ax.grid(True, alpha=0.3)

        if slope_ax is not None:
            slope_ax.set_xlabel("Strain")
            slope_ax.set_ylabel("dStress/dStrain")
            slope_ax.set_title(f"SimCase {sc_idx} — slope")
            slope_ax.grid(True, alpha=0.3)
            # Stress-strain slopes mix orders of magnitude AND
            # signs: the elastic regime hits 10^5 MPa, plastic is
            # 10^0 - 10^2, and the very first time steps often
            # produce brief negative numerical artifacts when dt
            # is tiny. Auto-scaling against ``max(|slope|)``
            # symmetrically lets those artifacts dictate the
            # y-axis on BOTH sides — the majority-positive bulk
            # gets squashed because matplotlib reserves equal
            # vertical real estate for an all-but-empty negative
            # half.
            #
            # The right rule: per-side decision. The side where
            # the data actually lives gets its full range so the
            # elastic spike stays visible. The other side, which
            # is mostly outliers/numerical noise, gets clipped to
            # the bulk's 90th-percentile envelope so it doesn't
            # eat half the chart for a handful of points.
            #
            # When the data is genuinely two-sided (cyclic
            # loading, repeated load reversals), neither side is
            # a "minority" and both get the bulk clip — preserves
            # symmetric behavior for the case where it's right.
            all_slopes = []
            for _rg, s_strain, s_stress in curves:
                all_slopes.append(_slope_of(s_strain, s_stress))
            if exp_data is not None:
                all_slopes.append(_slope_of(exp_data[0], exp_data[1]))
            if all_slopes:
                concat = np.concatenate(all_slopes)
                finite = concat[np.isfinite(concat)]
                if finite.size:
                    abs_finite = np.abs(finite)
                    nonzero = abs_finite[abs_finite > 0]
                    # Bulk envelope — used to clip the minority
                    # side. 90th percentile catches the majority
                    # of the smooth response while excluding the
                    # top ~5% (typically elastic spikes or
                    # start-of-test artifacts).
                    bulk_clip = min(
                        0,
                        1.0,
                    )
                    positives = finite[finite > 0]
                    negatives = finite[finite < 0]
                    n_pos, n_neg = positives.size, negatives.size
                    n_total = n_pos + n_neg
                    # 10% threshold: if one side has fewer than
                    # 10% of all signed points, treat it as the
                    # minority. Empirically this catches start-
                    # of-test numerical noise without misfiring
                    # on real two-sided responses.
                    MINORITY = 0.10
                    if n_total == 0:
                        upper, lower = 1.0, -1.0
                    elif n_neg == 0:
                        # All non-negative: full positive range,
                        # zero floor (nothing to show below).
                        upper = float(positives.max()) * 1.05
                        lower = 0.0
                    elif n_pos == 0:
                        upper = 0.0
                        lower = float(negatives.min()) * 1.05
                    elif n_neg / n_total < MINORITY:
                        # Mostly positive: full positive range,
                        # clip the negative tail to bulk envelope
                        # so the plastic bulk doesn't get crushed
                        # by a couple of dt-too-small noise dips.
                        upper = max(
                            float(positives.max()) * 1.05,
                            bulk_clip,
                        )
                        lower = -bulk_clip
                    elif n_pos / n_total < MINORITY:
                        upper = bulk_clip
                        lower = min(
                            float(negatives.min()) * 1.05,
                            -bulk_clip,
                        )
                    else:
                        # Genuinely two-sided: clip both ends to
                        # the bulk envelope; outliers extend off
                        # the chart visibly on whichever side
                        # they appeared.
                        upper = bulk_clip
                        lower = -bulk_clip
                    slope_ax.set_ylim(lower, upper)
                    # linthresh from the small end of the
                    # distribution — keeps the bulk in the LOG
                    # region rather than the LINEAR band, where
                    # log spacing actually helps with the
                    # multi-decade variation that the bulk
                    # exhibits.
                    if nonzero.size:
                        linthresh = max(
                            float(np.percentile(nonzero, 5)) * 0.5,
                            1e-9,
                        )
                    else:
                        linthresh = 1e-9
                    slope_ax.set_yscale("symlog", linthresh=linthresh)
                else:
                    # All non-finite slope values; keep a sensible
                    # default so the axis renders rather than
                    # crashing on the symlog setup.
                    slope_ax.set_yscale("symlog", linthresh=1.0)

        # Optimization-window shading. Shows the strain region the
        # optimizer was constrained to via case_data["minmax_strain"].
        # Drawn under everything else (low zorder) so curves stay
        # legible. None on either side means "unbounded on that
        # side" — extend the shading to the axis edge there.
        win = per_case_window[sc_idx] if sc_idx < len(per_case_window) else None
        if win is not None and (win[0] is not None or win[1] is not None):
            for ax_target in [stress_ax] + ([slope_ax] if slope_ax is not None else []):
                xlo, xhi = ax_target.get_xlim()
                lo = abs(win[0]) if win[0] is not None else xlo
                hi = abs(win[1]) if win[1] is not None else xhi
                # If the curves are negative-strain (compression),
                # mirror the window onto the negative side so the
                # shading lines up. We pick the sign of the
                # experimental data if available, else of the first
                # simulated curve.
                ref = exp_data[0] if exp_data is not None else (
                    curves[0][1] if curves else None
                )
                if ref is not None and len(ref) > 0 and float(np.median(ref)) < 0:
                    lo, hi = -hi, -lo
                ax_target.axvspan(
                    lo, hi, color="tab:green", alpha=0.08, zorder=0,
                    label="optimization window" if ax_target is stress_ax else None,
                )

    # --- Slider --------------------------------------------------------
    # The slider only makes sense when there are at least 2 ranks to
    # toggle between. With n_total=1 matplotlib would raise an
    # "identical low/high xlim" UserWarning trying to build a slider
    # with valmin=valmax=1; the slider would also be functionless
    # for the user. Hide the slider axes and skip widget creation.
    slider = None
    if n_total >= 2:
        slider = Slider(
            ax=slider_ax,
            label="show top K",
            valmin=1, valmax=n_total, valinit=n_total, valstep=1,
        )

        def _apply_visibility(k: int) -> None:
            for r in range(n_total):
                alpha = 0.7 if r < k else 0.07
                for line in line_groups[r]:
                    line.set_alpha(alpha)
            fig.canvas.draw_idle()

        slider.on_changed(lambda val: _apply_visibility(int(val)))
    else:
        slider_ax.axis("off")

    # --- Click handler -------------------------------------------------
    def _on_pick(event):
        rg = line_to_rg.get(event.artist)
        if rg is None:
            return
        panel_text.set_text(_format_param_panel(
            rg, meta.param_names, meta.objective_labels,
        ))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)

    label_for_title = ranked[0].category if ranked else "unranked"
    fig.suptitle(
        f"Top {n_total} solutions  |  ranked by {label_for_title}  "
        f"|  run {run_id[:8]}",
        fontsize=11,
    )

    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
        print(f"saved figure to {save}", file=sys.stderr)
    if show:
        plt.show()
    fig._plot_solutions_slider = slider  # type: ignore[attr-defined]
    return fig


def plot_pareto_front_with_l2_winner(
    workspace: Union[str, Path],
    *,
    objective_pair: Tuple[int, int] = (0, 1),
    top_n: int = 0,
    color_by: str = "subset_l2",
    run_id: Optional[str] = None,
    experimental_paths: Optional[Sequence[Optional[Union[str, Path]]]] = None,
    extractor_factory: Optional[
        Callable[[int], StressStrainExtractor]
    ] = None,
    save: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """2-D Pareto-front scatter, clickable, with response-curve inset.

    Layout: a two-column figure.

    * Left column — the 2-D scatter of fitness pairs. Points are
      colored by L2 norm (smaller = darker), the L2-closest gene
      is marked in red, every point has ``picker=5`` so a click
      fires a pick event.
    * Right column — an inset axes that initially shows the
      L2-closest gene's stress-strain response. When a user clicks
      any scatter point, the inset redraws to show that gene's
      simulated curves (one line per SimCase) overlaid against
      the experimental references. A monospace text panel below
      the inset shows the clicked gene's parameters and fitness.

    Selection: by default plots EVERY rank-0 (Pareto-front) gene
    so the picture is the actual front. Pass ``top_n > 0`` to
    restrict the scatter to the ``top_n`` lowest-L2 genes — useful
    when the front has hundreds of points and the user only wants
    to see the best balanced trade-offs.

    Experimental data is pulled archive-first via
    :func:`_resolve_experimental_for_case` so users never need to
    keep CSVs around once a run has completed.

    Args:
        workspace: Workspace directory or direct ``.db`` path.
        objective_pair: ``(x_idx, y_idx)`` objective indices to scatter.
        top_n: If positive, restrict to the ``top_n`` lowest-L2 genes.
            ``0`` (default) shows every rank-0 gene.
        run_id: Optional explicit run UUID.
        experimental_paths: CSV fallbacks per SimCase, used only when
            the archive doesn't have experimental data.
        extractor_factory: Per-SimCase extractor configs.
        save: Path to write the figure.
        show: Whether to ``plt.show()`` blocking.

    Returns:
        The matplotlib ``Figure``.
    """
    plt, _ = _lazy_matplotlib()

    db_path = resolve_archive_path(Path(workspace), "calibration.db")

    # Pre-load EVERYTHING the click handler will need: gene records,
    # per-gene case_outputs for all SimCases, experimental refs. The
    # alternative — keeping the DB connection open across user
    # interaction — risks half-open connections and SQLite locking
    # surprises. A few MB of in-memory DataFrames is the right
    # tradeoff for a post-run analysis tool.
    with ArchiveDB(db_path, readonly=True) as archive:
        meta = pick_run(archive, run_id, latest_if_none=True)
        run_id = meta.run_id

        all_genes = collect_all_genes(archive, run_id)
        finite = [
            g for g in all_genes
            if all(np.isfinite(v) for v in g.fitness)
        ]
        if not finite:
            raise RuntimeError("no finite-fitness genes to plot.")

        a_idx, b_idx = objective_pair
        n_obj = len(finite[0].fitness)
        if not (0 <= a_idx < n_obj and 0 <= b_idx < n_obj):
            raise ValueError(
                f"objective_pair {objective_pair} out of range for "
                f"{n_obj} objectives."
            )

        # Default selection: every rank-0 gene (the actual Pareto
        # front). The archive stores rank in GeneRecord.rank, which
        # is what selNSGA3 assigns — rank 0 is non-dominated.
        if top_n is not None and top_n > 0:
            # Top-N by L2 norm, dedup'd via shared logic with
            # inspect_archive so the same genes appear in the
            # `--pareto-only` table and on this plot.
            selection = select_top_genes(
                finite,
                objective_labels=meta.objective_labels,
                top_n=top_n,
                categories=("l2",),
            )
            chosen = [rg.gene for rg in selection.get("l2", [])]
            mode_note = f"top-{top_n} by L2 norm"
        else:
            chosen = [
                g for g in finite
                if g.rank is not None and g.rank == 0
            ]
            # Dedup on gene-vector — elitism keeps the same
            # solution across generations, and we don't want N
            # copies of the same point sitting on top of each
            # other in the scatter.
            from workflows.optimization.inspect_archive import (
                dedup_on_gene_vector,
            )
            chosen = dedup_on_gene_vector(chosen)
            mode_note = "rank-0 Pareto front"

        if not chosen:
            raise RuntimeError(
                "no genes selected for Pareto plot. With top_n=0 "
                "the run had no rank-0 genes (rank may not have "
                "been recorded — check archive's gene table)."
            )

        # Discover SimCases via the first chosen gene.
        n_sim_cases = _discover_sim_case_count(
            archive, run_id, sample_gene=chosen[0],
        )

        # Resolve per-SimCase extractors ONCE — they don't vary
        # across genes, so building them inside the gene loop is
        # wasted work. Archive-stored configs are preferred over
        # the user-supplied factory and the default fallback.
        per_case_extractor: Dict[int, StressStrainExtractor] = {
            sc_idx: _resolve_extractor_for_case(
                archive, run_id, sc_idx, factory=extractor_factory,
            )
            for sc_idx in range(n_sim_cases)
        }

        # Per-SimCase experimental references — loaded BEFORE the
        # gene loop so each loaded simulated curve can be
        # sign-matched against its experimental reference at load
        # time. Without sign-matching, archived curves where the
        # extractor accidentally absolute-valued its output appear
        # mirrored in the inset relative to the experimental
        # reference, which is confusing.
        per_case_exp: List[
            Optional[Tuple[np.ndarray, np.ndarray, Optional[str]]]
        ] = [
            _resolve_experimental_for_case(
                archive, run_id, sc_idx, experimental_paths,
            )
            for sc_idx in range(n_sim_cases)
        ]
        per_case_exp_reference: Dict[
            int, Optional[Tuple[np.ndarray, np.ndarray]]
        ] = {}
        for sc_idx, exp_resolved in enumerate(per_case_exp):
            if exp_resolved is None:
                per_case_exp_reference[sc_idx] = None
            else:
                exp_strain, exp_stress, _exp_label = exp_resolved
                per_case_exp_reference[sc_idx] = (exp_strain, exp_stress)

        # Pre-load per-gene case curves for the inset. Same shape
        # as the overlay's per_case_curves, but indexed by clicked
        # gene index rather than rank.
        gene_to_curves: Dict[
            int, List[Tuple[int, np.ndarray, np.ndarray]]
        ] = {}
        for g_idx, g in enumerate(chosen):
            sc_curves: List[Tuple[int, np.ndarray, np.ndarray]] = []
            for sc_idx in range(n_sim_cases):
                pair = _load_or_extract_curve(
                    archive, run_id,
                    birth_gen=g.birth_gen, birth_gene=g.birth_gene,
                    sim_case_idx=sc_idx,
                    extractor=per_case_extractor[sc_idx],
                    rank_label=g_idx,
                    experimental_reference=per_case_exp_reference[sc_idx],
                )
                if pair is None:
                    continue
                sc_curves.append((sc_idx, pair[0], pair[1]))
            gene_to_curves[g_idx] = sc_curves

    # --- Build the figure ----------------------------------------------
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(
        nrows=10, ncols=2,
        height_ratios=[1] * 7 + [0.2, 0.5, 0.7],
        width_ratios=[1.0, 1.1],
        hspace=0.35, wspace=0.25,
    )
    scatter_ax = fig.add_subplot(gs[0:7, 0])
    inset_ax = fig.add_subplot(gs[0:7, 1])
    panel_ax = fig.add_subplot(gs[9, :])
    panel_ax.axis("off")
    panel_text = panel_ax.text(
        0.01, 0.95,
        "click any scatter point to see its parameters and response",
        transform=panel_ax.transAxes,
        va="top", ha="left", family="monospace", fontsize=9,
    )

    # --- Scatter -------------------------------------------------------
    fits = np.array([g.fitness for g in chosen])
    pop_fit_2d = fits[:, [a_idx, b_idx]]

    # Two L2 norms are useful, depending on the question being asked:
    #   subset_l2: distance from utopian origin in the projected pair
    #     (a,b). This is what users typically want for "the best
    #     balanced point ON THIS PROJECTION" — the Pareto winner of
    #     what they're actually looking at.
    #   full_l2: distance across all objectives. Useful as a
    #     secondary signal showing how a point ranks globally vs
    #     just on the displayed pair.
    # The L2 winner ring uses subset_l2 by default — Robert's ask:
    # "the L2 circle should really be based on the objectives
    # chosen and that should be what's most useful."
    x_vals = pop_fit_2d[:, 0]
    y_vals = pop_fit_2d[:, 1]
    subset_l2 = np.sqrt(x_vals ** 2 + y_vals ** 2)
    full_l2 = np.linalg.norm(fits, axis=1)
    best_local_idx = int(np.argmin(subset_l2))

    # Pick the color metric. Each mode highlights a different aspect
    # of where points sit relative to the X=0 / Y=0 planes:
    #   - subset_l2: iso-curves are circles centered at origin in
    #     the projected pair. Points along a true Pareto front sit
    #     along an iso-curve at constant distance, so a uniform
    #     color along the front is the visual signal that the
    #     front is genuinely tradeoff-shaped here.
    #   - full_l2: same idea but with all objectives folded in.
    #     A point that looks balanced on (a,b) but is bad on the
    #     hidden objectives shows up as light here.
    #   - x: just the projected X value. Iso-curves are vertical
    #     lines. Reveals proximity to the Y=0 plane (how
    #     specialized-on-Y a point is — small X means good on
    #     X regardless of Y).
    #   - y: symmetric to x. Iso-curves are horizontal lines.
    #     Shows proximity to the X=0 plane.
    #   - asymmetry: |x - y| / (x + y). Zero means perfectly
    #     balanced on the pair, larger means more lopsided. Lets
    #     users see at a glance which points are specialists vs
    #     compromises.
    color_metric_options = ("subset_l2", "full_l2", "x", "y", "asymmetry")
    if color_by not in color_metric_options:
        raise ValueError(
            f"unknown color_by={color_by!r}; valid options: "
            f"{color_metric_options}"
        )
    if color_by == "subset_l2":
        color_values = subset_l2
        cbar_label = (
            f"L2 of ({meta.objective_labels[a_idx]}, "
            f"{meta.objective_labels[b_idx]})"
        )
    elif color_by == "full_l2":
        color_values = full_l2
        cbar_label = "L2 (all objectives)"
    elif color_by == "x":
        color_values = x_vals
        cbar_label = meta.objective_labels[a_idx]
    elif color_by == "y":
        color_values = y_vals
        cbar_label = meta.objective_labels[b_idx]
    else:  # asymmetry
        # Guard against division by zero — x+y=0 only if both are
        # zero (utopia point), which is a degenerate edge case.
        denom = x_vals + y_vals
        with np.errstate(divide="ignore", invalid="ignore"):
            asym = np.where(denom > 0, np.abs(x_vals - y_vals) / denom, 0.0)
        color_values = asym
        cbar_label = (
            f"|{meta.objective_labels[a_idx]} - "
            f"{meta.objective_labels[b_idx]}| / sum"
        )

    sc = scatter_ax.scatter(
        pop_fit_2d[:, 0], pop_fit_2d[:, 1],
        c=color_values, cmap="viridis",
        s=40, picker=5, edgecolors="black", linewidths=0.4,
    )
    # Mark the subset-L2 winner with a red ring on top. This is the
    # gene most balanced on the displayed projection — answering
    # "which point is the best compromise on what I'm looking at?"
    scatter_ax.scatter(
        pop_fit_2d[best_local_idx, 0], pop_fit_2d[best_local_idx, 1],
        s=160, facecolors="none", edgecolors="red", linewidths=2.0,
        label="L2 winner (subset)", zorder=5,
    )
    # Utopian origin marker — same convention as plot_pareto_front
    # in workflow_common.postprocess.
    scatter_ax.scatter(
        [0], [0], c="black", marker="+", s=80, label="utopia",
        zorder=4,
    )
    cbar = fig.colorbar(sc, ax=scatter_ax, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label)
    scatter_ax.set_xlabel(meta.objective_labels[a_idx])
    scatter_ax.set_ylabel(meta.objective_labels[b_idx])
    scatter_ax.set_title(
        f"Pareto: {meta.objective_labels[a_idx]} vs "
        f"{meta.objective_labels[b_idx]}\n"
        f"({mode_note}, color={color_by})",
        fontsize=10,
    )
    scatter_ax.legend(loc="upper right", fontsize=8)
    scatter_ax.grid(True, alpha=0.3)

    # --- Inset: response curve for the clicked gene --------------------
    inset_ax.set_xlabel("Strain")
    inset_ax.set_ylabel("Stress")
    inset_ax.set_title("Response of clicked gene", fontsize=10)
    inset_ax.grid(True, alpha=0.3)

    def _draw_response(g_idx: int) -> None:
        """Redraw the inset showing gene g_idx's sim curves vs experimental."""
        inset_ax.clear()
        inset_ax.set_xlabel("Strain")
        inset_ax.set_ylabel("Stress")
        inset_ax.grid(True, alpha=0.3)
        # Deterministic per-SimCase coloring.
        sc_cmap = plt.get_cmap("tab10")
        sim_curves = gene_to_curves.get(g_idx, [])
        for sc_idx, strain, stress in sim_curves:
            color = sc_cmap(sc_idx % 10)
            inset_ax.plot(
                strain, stress,
                color=color, linewidth=1.6,
                label=f"sim sc={sc_idx}",
            )
            # Experimental reference (if any) for this SimCase.
            if (sc_idx < len(per_case_exp)
                    and per_case_exp[sc_idx] is not None):
                ex, ey, _exp_label = per_case_exp[sc_idx]
                inset_ax.plot(
                    ex, ey,
                    color=color, linestyle="--", linewidth=2.0,
                    alpha=0.8,
                    label=f"exp sc={sc_idx}",
                )
        gene = chosen[g_idx]
        is_winner = g_idx == best_local_idx
        title_extra = " (L2 winner)" if is_winner else ""
        inset_ax.set_title(
            f"Response — gene g{gene.birth_gen}.{gene.birth_gene}"
            f"{title_extra}",
            fontsize=10,
        )
        if sim_curves:
            inset_ax.legend(loc="best", fontsize=8)

    # Initialize the inset with the L2 winner's response.
    _draw_response(best_local_idx)
    panel_text.set_text(_format_gene_panel(
        chosen[best_local_idx], meta.param_names, meta.objective_labels,
        header=f"L2 winner — {len(chosen)} points on plot",
    ))

    # --- Click handler -------------------------------------------------
    def _on_pick(event):
        # PathCollection (scatter) pick events arrive with `ind` —
        # an array of indices into the collection's data. A single
        # click usually returns one index, but a dense cluster can
        # return several; pick the first.
        if event.artist is not sc:
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return
        g_idx = int(event.ind[0])
        _draw_response(g_idx)
        panel_text.set_text(_format_gene_panel(
            chosen[g_idx], meta.param_names, meta.objective_labels,
            header=f"clicked point #{g_idx} of {len(chosen)}",
        ))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)

    fig.suptitle(
        f"Pareto front (run {run_id[:8]}) — click points for details",
        fontsize=11,
    )

    if save is not None:
        fig.savefig(str(save), dpi=120, bbox_inches="tight")
        print(f"saved figure to {save}", file=sys.stderr)
    if show:
        plt.show()
    return fig


# --- CLI -----------------------------------------------------------------


def _parse_pareto_pair(s: str) -> Tuple[int, int]:
    """Parse ``--pareto 0,1`` into a tuple of ints with friendly errors."""
    try:
        a, b = (int(x.strip()) for x in s.split(","))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"--pareto must be 'i,j' (e.g. '0,1'); got {s!r}"
        )
    return (a, b)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot top-N optimized solutions vs experimental "
            "data from a calibration archive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Top 10 by L2 norm:\n"
            "    python examples/plot_solutions.py calibration_run \\\n"
            "        --top 10 --experimental experiments/exp1.csv \\\n"
            "                                 experiments/exp2.csv\n\n"
            "  Top 5 by objective 0:\n"
            "    python examples/plot_solutions.py calibration_run \\\n"
            "        --top 5 --objective 0 \\\n"
            "        --experimental experiments/exp1.csv\n\n"
            "  Pareto front for objectives 0 vs 2:\n"
            "    python examples/plot_solutions.py calibration_run \\\n"
            "        --pareto 0,2\n"
        ),
    )
    p.add_argument(
        "workspace", type=Path,
        help="Calibration workspace dir, or direct .db file path.",
    )
    p.add_argument(
        "--top", dest="top_n", type=int, default=10,
        help="Number of top solutions to plot (default: 10). "
             "Pass 0 or negative for all available.",
    )
    p.add_argument(
        "--mode", dest="mode",
        choices=("l2", "objective", "last-gen"), default="l2",
        help="Ranking mode (default: l2).",
    )
    p.add_argument(
        "--objective", dest="objective", default=None,
        help="For --mode=objective: integer index or label name.",
    )
    p.add_argument(
        "--run-id", dest="run_id", default=None,
        help="Archive run UUID (default: latest run).",
    )
    p.add_argument(
        "--experimental", dest="experimental",
        nargs="*", type=Path, default=None,
        help="CSV paths, one per SimCase (in SimCase order). "
             "If fewer paths than SimCases are given, later cases "
             "show no experimental overlay.",
    )
    p.add_argument(
        "--pareto", dest="pareto",
        type=_parse_pareto_pair, default=None,
        help="Produce a Pareto-front scatter for objectives i,j "
             "(e.g. '0,1'). The subset-L2 winner is highlighted, all "
             "points are clickable to reveal parameters and the "
             "stress-strain response of every SimCase. Restricted to "
             "--top N points if --top is set; otherwise plots every "
             "rank-0 gene. NOTE: passing --pareto suppresses the "
             "headline overlay by default — pass --overlay to also "
             "plot the overlay.",
    )
    p.add_argument(
        "--overlay", dest="overlay", action="store_true", default=None,
        help="Force the headline stress-strain overlay figure on "
             "even when --pareto is given. Without this flag, "
             "passing --pareto suppresses the overlay (the bare "
             "command without --pareto still shows the overlay by "
             "default).",
    )
    p.add_argument(
        "--color-by", dest="color_by",
        choices=("subset_l2", "full_l2", "x", "y", "asymmetry"),
        default="subset_l2",
        help="Pareto-scatter color metric. 'subset_l2' (default) "
             "colors by L2 distance in the projected (X,Y) plane — "
             "iso-curves are circles, uniform color along the "
             "front signals a true tradeoff curve. 'full_l2' uses "
             "all objectives. 'x'/'y' color by a single axis "
             "(showing proximity to the Y=0/X=0 plane). 'asymmetry' "
             "highlights points that are specialists on one axis "
             "vs balanced compromises.",
    )
    p.add_argument(
        "--save", dest="save", type=Path, default=None,
        help="Save the headline figure to this path. Use with "
             "--no-show for headless/CI usage.",
    )
    p.add_argument(
        "--save-pareto", dest="save_pareto",
        type=Path, default=None,
        help="Save the Pareto-front figure to this path.",
    )
    p.add_argument(
        "--no-show", dest="show", action="store_false", default=True,
        help="Don't open a window (useful with --save in batch).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Auto-promote mode: if the user passed --objective without
    # --mode, they almost certainly meant `--mode objective`. The
    # alternative — silently using L2 ranking with the objective
    # flag ignored — was the source of a real user-reported bug
    # ("changing --objective shows the same plot"). So when
    # --objective is supplied AND --mode is still the default
    # "l2", flip mode to "objective". Users who really want L2
    # ranking with an objective set (a strange combination) can
    # pass --mode l2 explicitly... and they'd see the same L2
    # plot they get without --objective, which is consistent.
    #
    # last-gen mode is incompatible with --objective (the former
    # ignores ranking entirely). Error rather than silently
    # picking one over the other.
    if args.objective is not None:
        if args.mode == "last-gen":
            print(
                "error: --objective is not compatible with "
                "--mode last-gen (last-gen has no ranking step).",
                file=sys.stderr,
            )
            return 1
        if args.mode == "l2":
            args.mode = "objective"

    # Resolve overlay default. Robert's report: passing --pareto
    # alone should suppress the overlay, since users asking for a
    # Pareto plot usually want only that. The --overlay flag is the
    # explicit opt-in to also draw the overlay alongside Pareto.
    # When --pareto is NOT given, the overlay is the headline
    # figure and shows by default (the "user just runs the script
    # to look at their results" path).
    #
    # args.overlay is None when neither flag was set; True when
    # --overlay was passed. Resolve to a concrete bool here.
    if args.overlay is True:
        show_overlay = True
    elif args.pareto is not None:
        show_overlay = False
    else:
        show_overlay = True

    try:
        if show_overlay:
            plot_top_solutions_overlay(
                args.workspace,
                top_n=args.top_n,
                mode=args.mode,
                objective=args.objective,
                run_id=args.run_id,
                experimental_paths=args.experimental,
                save=args.save,
                show=args.show,
            )
        if args.pareto is not None:
            plot_pareto_front_with_l2_winner(
                args.workspace,
                objective_pair=args.pareto,
                top_n=args.top_n,
                color_by=args.color_by,
                run_id=args.run_id,
                experimental_paths=args.experimental,
                save=args.save_pareto,
                show=args.show,
            )
        # No-op runs (neither overlay nor pareto) shouldn't happen
        # given the resolution above, but guard for paranoia: the
        # only way to reach this is --pareto unset AND --overlay
        # explicitly false (which the new CLI doesn't expose), or
        # the resolution logic above getting confused.
        if not show_overlay and args.pareto is None:
            print(
                "error: nothing to plot. Pass --pareto i,j or "
                "remove conflicting flags.",
                file=sys.stderr,
            )
            return 1
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except SystemExit as e:
        return int(e.code) if e.code is not None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
