"""Command-line tool for inspecting an exaconstit-calibrate SQLite archive.

Writes archive contents to stdout in whatever format the user asks for
(text table, CSV, or JSON). No SQL knowledge required on the reader's
side. Complements the binary archive.db file so teammates can audit a
run's results from a terminal, spreadsheet, or analysis notebook
without connecting a SQLite client.

Invocation:

    python -m workflows.optimization.inspect_archive RUN_DIR
    python -m workflows.optimization.inspect_archive RUN_DIR --format csv
    python -m workflows.optimization.inspect_archive RUN_DIR --run RUN_ID
    python -m workflows.optimization.inspect_archive RUN_DIR --gen 0 --pareto-only

Subcommands / flags:

    RUN_DIR              Workspace dir (the one containing archive.db)
    --runs               List all runs in the archive and exit
    --gens               List generations for a run (default: latest run)
    --genes              Dump the gene records for one generation
    --run RUN_ID         Restrict to a specific run_id (default: latest)
    --gen GEN_IDX        Restrict to a specific generation (default: latest)
    --pareto-only        Only dump rank-0 individuals (the Pareto front)
    --format FORMAT      'table' (default), 'csv', or 'json'
    --no-header          Suppress the header row (useful for piping)

Exit codes:
    0  normal
    1  user error (missing file, bad run_id, etc.)
    2  no data found matching the query
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, IO, List, Optional, Sequence

import numpy as np

from workflow_common.archive import (
    ArchiveDB,
    GeneRecord,
    GenerationSummary,
    RunSummary,
)


# --- Public selection / loading API --------------------------------------
#
# These helpers exist so other tools (the plot_solutions example, user
# notebooks, third-party scripts) can reuse the archive-loading and
# top-N-ranking logic that this CLI was built around. The CLI internally
# calls these too — there's no second copy of the ranking math.
#
# Library-style behavior:
#   * Errors raise plain exceptions (ValueError, FileNotFoundError),
#     not SystemExit. The CLI wraps these and converts to exit codes.
#   * No printing to stderr by default. Diagnostic output is on the
#     CLI's side; library callers handle their own UX.
#   * Functions accept already-open ArchiveDB instances rather than
#     paths, so callers can batch multiple operations on one connection
#     without paying the open-connect-close cost per call.


from dataclasses import dataclass


@dataclass(frozen=True)
class RankedGene:
    """One gene paired with the score that placed it in a top-N selection.

    Returned by :func:`select_top_genes`. Carries everything a downstream
    consumer needs to identify the gene (``gene.birth_gen`` /
    ``gene.birth_gene`` / ``sim_case_idx`` are the archive-lookup
    coordinates), explain why it was selected (``category``, ``score``),
    and display it (``rank``, ``l2_norm``).

    Fields:
        gene: The underlying :class:`GeneRecord` from the archive.
        rank: 0-based position within ``category``. ``rank=0`` means
            "best in this category".
        category: Either ``"l2"`` (closest to utopian origin in
            objective space) or ``"obj:<label>"`` (best on a single
            objective). Useful for grouping in displays.
        score: The numeric value the rank was based on — the L2 norm
            for L2 rows, the objective value for per-objective rows.
            Same units as the underlying objective.
        l2_norm: The full Euclidean norm of the fitness tuple,
            always populated. Lets per-objective rows show "winner
            but also balanced?" at a glance.
    """
    gene: GeneRecord
    rank: int
    category: str
    score: float
    l2_norm: float


def dedup_on_gene_vector(
    genes: Sequence[GeneRecord],
) -> List[GeneRecord]:
    """Deduplicate gene records on their parameter-vector values.

    NSGA-III carries good individuals across generations via elitism,
    so the same gene vector can appear at gens 4, 5, 6, ... — which
    would fill a top-N list with copies. This helper keeps the FIRST
    sighting (i.e. the earliest birth gen) and drops the duplicates.

    Float equality is the right comparison: DEAP stores genes as lists
    of floats that compare bit-exactly, and SQLite round-trips the
    same repr on archive read.

    The returned list preserves the input order (with duplicates
    removed). Callers wanting "most recent first" should reverse
    after deduping.
    """
    seen: set = set()
    out: List[GeneRecord] = []
    for g in genes:
        key = tuple(float(x) for x in g.gene_vector)
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def select_top_genes(
    all_genes: Sequence[GeneRecord],
    *,
    objective_labels: Sequence[str],
    top_n: int,
    categories: Sequence[str] = ("l2", "per_objective"),
    dedup: bool = True,
) -> Dict[str, List[RankedGene]]:
    """Pick the top ``top_n`` genes per requested category.

    The headline ranking strategies for an NSGA-III archive:

    * ``"l2"`` — sort ascending by L2 norm of the fitness tuple. This
      is the "balanced winner" notion: small on every axis, no narrow
      specialists.
    * ``"per_objective"`` — for each objective, the top ``top_n`` genes
      ascending on that objective alone. Reveals specialists.

    Genes with non-finite fitness (``inf`` from failed sims) are filtered
    out before ranking. Ties are broken by ``(birth_gen, birth_gene)``
    via stable sort, so the earliest-born of a tied cluster ranks first.

    Args:
        all_genes: Every :class:`GeneRecord` from the run, typically
            from :meth:`ArchiveDB.load_all_genes`.
        objective_labels: Names for the M fitness components, in the
            same order as ``GeneRecord.fitness``. Used to build the
            ``"obj:<label>"`` category keys.
        top_n: How many genes to return per category. If larger than
            the available count, all available genes are returned.
        categories: Which categories to populate. Default is both;
            pass ``("l2",)`` for just the L2 ranking, or
            ``("per_objective",)`` for the per-axis specialists only.
        dedup: Whether to deduplicate on gene-vector before ranking.
            Default True. Set False if your archive somehow contains
            non-elite duplicates that you want preserved.

    Returns:
        Dict mapping category-name → ranked ``RankedGene`` list. Keys
        are ``"l2"`` (if requested) and one ``"obj:<label>"`` key per
        objective (if ``"per_objective"`` was requested). Empty list
        values for objectives with no finite-fitness genes.
    """
    pool = dedup_on_gene_vector(all_genes) if dedup else list(all_genes)
    if not pool:
        return {}

    n_obj = len(objective_labels)
    fit_matrix = np.asarray(
        [list(g.fitness) for g in pool], dtype=float,
    )
    finite_all = np.all(np.isfinite(fit_matrix), axis=1)
    finite_per_obj = np.isfinite(fit_matrix)
    l2_norms = np.linalg.norm(fit_matrix, axis=1)

    out: Dict[str, List[RankedGene]] = {}

    if "l2" in categories:
        l2_list: List[RankedGene] = []
        if np.any(finite_all):
            l2_masked = np.where(finite_all, l2_norms, np.inf)
            n_valid = int(np.sum(finite_all))
            take = min(top_n, n_valid)
            order = np.argsort(l2_masked, kind="stable")[:take]
            for rank, idx in enumerate(order):
                i = int(idx)
                l2_list.append(RankedGene(
                    gene=pool[i],
                    rank=rank,
                    category="l2",
                    score=float(l2_norms[i]),
                    l2_norm=float(l2_norms[i]),
                ))
        out["l2"] = l2_list

    if "per_objective" in categories:
        for obj_idx, label in enumerate(objective_labels):
            col = fit_matrix[:, obj_idx]
            valid = finite_per_obj[:, obj_idx]
            if not np.any(valid):
                out[f"obj:{label}"] = []
                continue
            col_masked = np.where(valid, col, np.inf)
            n_valid = int(np.sum(valid))
            take = min(top_n, n_valid)
            order = np.argsort(col_masked, kind="stable")[:take]
            sub: List[RankedGene] = []
            for rank, idx in enumerate(order):
                i = int(idx)
                sub.append(RankedGene(
                    gene=pool[i],
                    rank=rank,
                    category=f"obj:{label}",
                    score=float(col[i]),
                    l2_norm=float(l2_norms[i]),
                ))
            out[f"obj:{label}"] = sub

    return out


def pick_run(
    archive: ArchiveDB,
    run_id: Optional[str] = None,
    *,
    latest_if_none: bool = True,
) -> RunSummary:
    """Select a run from the archive.

    Library counterpart of the CLI's interactive run selection. Behavior:

    * ``run_id`` given → return that run. Raise :class:`ValueError` if
      no such run exists, listing the available IDs.
    * ``run_id`` None and ``latest_if_none=True`` (default) → the
      newest run by ``started_at``. Tied timestamps fall through to
      list-tail order (matches the CLI behavior).
    * ``run_id`` None and ``latest_if_none=False`` → raise
      :class:`ValueError` if there's more than one run.
    * Empty archive → raise :class:`ValueError`.

    Args:
        archive: An open :class:`ArchiveDB` (read-only is fine).
        run_id: Optional explicit run UUID.
        latest_if_none: How to behave when no ``run_id`` is supplied
            and the archive holds multiple runs. Default ``True``
            picks the latest, matching the CLI; pass ``False`` for
            strict "must be a single run" semantics.

    Raises:
        ValueError: As described above.
    """
    runs = archive.list_runs()
    if not runs:
        raise ValueError("archive contains no runs")
    if run_id is not None:
        for r in runs:
            if r.run_id == run_id:
                return r
        raise ValueError(
            f"run_id {run_id!r} not in archive; available: "
            f"{[r.run_id for r in runs]}"
        )
    if len(runs) == 1:
        return runs[0]
    if latest_if_none:
        return runs[-1]
    raise ValueError(
        f"archive has {len(runs)} runs; pass run_id explicitly. "
        f"Available: {[r.run_id for r in runs]}"
    )


def collect_all_genes(
    archive: ArchiveDB, run_id: str,
) -> List[GeneRecord]:
    """Load every gene from a run in a single SQLite query.

    Thin wrapper over :meth:`ArchiveDB.load_all_genes` with a more
    discoverable name from the public-API surface.
    """
    return archive.load_all_genes(run_id)


# --- Formatters -----------------------------------------------------------


def _format_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    *,
    with_header: bool = True,
) -> str:
    """Render rows as a fixed-width text table.

    Computes column widths from the widest rendered value per column.
    Floats and numpy arrays get compact default repr. Nothing
    terminal-aware; the output is plain ASCII so it survives tee /
    less / pipe to file.
    """
    if not rows:
        return ""
    # Pre-render everything to strings so we can measure widths.
    cell_strs: List[List[str]] = []
    for row in rows:
        cells = [_cell_str(row.get(col, "")) for col in columns]
        cell_strs.append(cells)
    widths = [len(col) for col in columns]
    for cells in cell_strs:
        for i, c in enumerate(cells):
            if len(c) > widths[i]:
                widths[i] = len(c)

    lines = []
    if with_header:
        header = "  ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
        sep = "  ".join("-" * widths[i] for i in range(len(columns)))
        lines.append(header)
        lines.append(sep)
    for cells in cell_strs:
        lines.append("  ".join(cells[i].ljust(widths[i]) for i in range(len(columns))))
    return "\n".join(lines)


def _format_csv(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    *,
    with_header: bool = True,
) -> str:
    """Standard CSV output. Gene vectors and fitness tuples get joined
    with semicolons inside a single cell so the result is still a
    proper CSV (one row = one record).
    """
    from io import StringIO

    buf = StringIO()
    w = csv.writer(buf)
    if with_header:
        w.writerow(columns)
    for row in rows:
        w.writerow([_cell_str(row.get(col, "")) for col in columns])
    return buf.getvalue().rstrip("\n")


def _format_json(rows: Sequence[Dict[str, Any]]) -> str:
    """JSON array of objects. ndarrays flatten to lists."""
    return json.dumps(
        [_jsonify(row) for row in rows],
        indent=2,
        default=_json_default,
    )


def _cell_str(v: Any) -> str:
    """Stringify one cell for table/CSV output.

    Gene vectors and fitness tuples get joined with ``;`` so the row
    stays one line. Floats use a short but precise repr (no trailing
    e+00 noise for whole numbers).
    """
    if isinstance(v, np.ndarray):
        return ";".join(f"{x:g}" for x in v)
    if isinstance(v, (tuple, list)) and v and isinstance(v[0], (int, float)):
        return ";".join(f"{x:g}" for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    if v is None:
        return ""
    return str(v)


def _jsonify(row: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare a row dict for ``json.dumps``.

    The ``default`` handler in json.dumps handles numpy arrays, but
    nested dicts / tuples inside the row are easier to convert
    eagerly so the output structure is cleaner.
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _json_default(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    raise TypeError(f"not JSON-serializable: {type(v).__name__}")


# --- Query helpers --------------------------------------------------------


def _pick_run(archive: ArchiveDB, run_id: Optional[str]) -> RunSummary:
    """CLI wrapper around :func:`pick_run` — converts ValueError to SystemExit.

    Library callers should use :func:`pick_run` directly; this exists
    purely so the CLI's exit-code convention (1 = user error) keeps
    working without sprinkling try/except around every call site.
    """
    try:
        return pick_run(archive, run_id, latest_if_none=True)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)


def _run_rows(runs: Sequence[RunSummary]) -> List[Dict[str, Any]]:
    return [
        dict(
            run_id=r.run_id,
            started_at=r.started_at,
            completed_at=r.completed_at,
            seed=r.seed,
            n_generations=r.n_generations,
            param_names=r.param_names,
            objective_labels=r.objective_labels,
        )
        for r in runs
    ]


def _gen_rows(gens: Sequence[GenerationSummary]) -> List[Dict[str, Any]]:
    rows = []
    for g in gens:
        row: Dict[str, Any] = dict(
            gen_idx=g.gen_idx,
            recorded_at=g.recorded_at,
            n_pop=g.n_pop,
        )
        # Flatten the stats dict into top-level fields. Common keys:
        # avg, std, min, max (each an array), plus ND, GD, HV in
        # multi-objective runs. Array-valued entries stringify as
        # ``;``-joined for table/CSV or stay list-valued for JSON.
        for k, v in g.stats.items():
            row[k] = v
        rows.append(row)
    return rows


def _gene_rows(
    genes: Sequence[GeneRecord],
    *,
    param_names: Sequence[str],
    objective_labels: Sequence[str],
    pareto_only: bool,
) -> List[Dict[str, Any]]:
    rows = []
    for g in genes:
        if pareto_only and (g.rank is None or g.rank != 0):
            continue
        row: Dict[str, Any] = dict(
            gen_idx=g.gen_idx,
            pop_idx=g.pop_idx,
            birth_gen=g.birth_gen,
            birth_gene=g.birth_gene,
            rank=g.rank,
        )
        # Expand gene vector and fitness into one column each, using
        # the provided names. A user with param_names=["yield", "H"]
        # gets columns "yield", "H" rather than a cryptic gene[0]/[1].
        for i, name in enumerate(param_names):
            row[name] = float(g.gene_vector[i])
        for i, label in enumerate(objective_labels):
            row[label] = float(g.fitness[i])
        rows.append(row)
    return rows


def _collect_all_genes(
    archive: ArchiveDB, run_id: str,
) -> List[GeneRecord]:
    """CLI-internal alias for :func:`collect_all_genes`."""
    return collect_all_genes(archive, run_id)


def _dedup_on_gene_vector(
    genes: Sequence[GeneRecord],
) -> List[GeneRecord]:
    """CLI-internal alias for :func:`dedup_on_gene_vector`."""
    return dedup_on_gene_vector(genes)


def _pareto_history_rows(
    all_genes: Sequence[GeneRecord],
    *,
    param_names: Sequence[str],
    objective_labels: Sequence[str],
    top_n: int,
) -> List[Dict[str, Any]]:
    """Build the "best across all generations" table.

    Layout of the returned rows:

    * ``top_n`` rows tagged ``category="l2"`` — smallest L2 norm
      across the whole fitness tuple. These are the
      "most-balanced" genes; for single-objective problems this
      reduces to the same set as ranking by that sole objective.
    * ``top_n`` rows tagged ``category="obj:<label>"`` per
      objective — the genes that are individually best on each
      axis. For a run with 4 objectives this is ``4 * top_n``
      extra rows.

    Genes are deduplicated on their vector values before ranking
    so survivors carried across generations don't fill the table
    with copies. Non-finite fitness values are filtered out so they
    can't game the rankings. The ranking itself is delegated to
    :func:`select_top_genes`; this function is the
    GeneRecord-to-display-row formatter.

    Args:
        all_genes: Every :class:`GeneRecord` in the run (see
            :func:`collect_all_genes`).
        param_names: Column labels for the gene-vector components.
        objective_labels: Column labels for the fitness-tuple
            components; drives both the per-objective rankings
            and the column headers.
        top_n: How many rows to show per category.

    Returns:
        A list of row dicts ready for :func:`_emit`. Each row
        carries every parameter and every objective as named
        columns, plus a ``category`` string.
    """
    selection = select_top_genes(
        all_genes,
        objective_labels=objective_labels,
        top_n=top_n,
        categories=("l2", "per_objective"),
    )
    if not selection:
        return []

    rows: List[Dict[str, Any]] = []
    # L2 first — matches the historical row ordering that callers
    # have come to expect from --pareto-only output.
    for rg in selection.get("l2", []):
        rows.append(_pareto_row(
            rg.gene, rg.category, param_names, objective_labels,
            l2_norm=rg.l2_norm,
        ))
    # Per-objective categories follow, in objective order.
    for label in objective_labels:
        for rg in selection.get(f"obj:{label}", []):
            rows.append(_pareto_row(
                rg.gene, rg.category, param_names, objective_labels,
                l2_norm=rg.l2_norm,
            ))
    return rows


def _pareto_row(
    g: GeneRecord,
    category: str,
    param_names: Sequence[str],
    objective_labels: Sequence[str],
    l2_norm: float,
) -> Dict[str, Any]:
    """Format one GeneRecord into a row dict for the cross-gen view.

    Kept separate from :func:`_gene_rows` because the columns
    differ: this view leads with ``category`` and ``birth_gen``
    (the gene's actual provenance, which is more useful in a
    historical context than the ``gen_idx`` of where we happened
    to find it).

    The ``l2_norm`` field is the Euclidean norm of the gene's
    fitness tuple. It's the same number the L2 category sorts by,
    and it's useful on per-objective rows too: a gene that wins on
    obj:stress but has a huge L2 is a narrow specialist, while one
    with a small L2 is balanced-AND-a-specialist. Showing it makes
    the ranking self-verifying — users can scan down the L2 block
    and see the norms increasing monotonically.
    """
    row: Dict[str, Any] = dict(
        category=category,
        birth_gen=g.birth_gen,
        birth_gene=g.birth_gene,
        rank=g.rank,
    )
    for i, name in enumerate(param_names):
        row[name] = float(g.gene_vector[i])
    for i, label in enumerate(objective_labels):
        row[label] = float(g.fitness[i])
    row["l2_norm"] = float(l2_norm)
    return row


def _gens_best_rows(
    all_genes: Sequence[GeneRecord],
    *,
    objective_labels: Sequence[str],
) -> List[Dict[str, Any]]:
    """Build the per-generation convergence table.

    For each generation G in the run, emit one row summarizing the
    best-ever fitness values seen across all genes in generations
    0..G. Columns are:

    * ``gen_idx`` — the generation this row summarizes
    * ``n_seen`` — cumulative unique gene-record count through
      this generation (helpful for sanity-checking archive gaps)
    * ``<label>_best`` per objective — the running minimum across
      generations 0..G on that single objective
    * ``champion_l2`` — the L2 norm of the fitness tuple of the
      gene currently holding the "most-balanced" crown
    * ``champion_birth_gen`` — the generation that gene was born
      in. A long plateau (same value for many gens) means nobody
      has dethroned the current L2 champion

    Non-finite fitness values are filtered out before the running
    minimums so failed/penalty genes don't permanently fix a
    column to ``inf``.

    Note on semantics: the per-objective "best" columns track
    each objective independently — they may belong to different
    genes from one another AND from the L2 champion. That's
    intentional; it lets the user distinguish "one gene is
    dominating on everything" (all three best columns match the
    champion's fitness) from "we're winning different components
    with different genes" (they don't).

    Args:
        all_genes: Every :class:`GeneRecord` in the run, ordered
            ``(gen_idx ASC, pop_idx ASC)`` (which
            :func:`_collect_all_genes` guarantees).
        objective_labels: Column labels for the fitness tuple;
            drives the ``<label>_best`` column names.

    Returns:
        One dict per generation with the columns described above.
        Empty if ``all_genes`` is empty.
    """
    if not all_genes:
        return []

    n_obj = len(objective_labels)
    # Group by generation. Lists of fitness tuples per gen.
    by_gen: Dict[int, List[GeneRecord]] = {}
    for g in all_genes:
        by_gen.setdefault(g.gen_idx, []).append(g)

    # Running best-per-objective and best-L2, updated incrementally
    # so the loop stays O(total genes) rather than O(gens * cumulative).
    running_best_obj = np.full(n_obj, np.inf, dtype=float)
    running_best_l2 = float("inf")
    running_champion_birth_gen: Optional[int] = None
    cumulative_seen = 0

    rows: List[Dict[str, Any]] = []
    for gen_idx in sorted(by_gen.keys()):
        gen_genes = by_gen[gen_idx]
        cumulative_seen += len(gen_genes)

        # Stack this generation's fitnesses and mask non-finite.
        fit = np.asarray(
            [list(g.fitness) for g in gen_genes], dtype=float,
        )
        finite_all_mask = np.all(np.isfinite(fit), axis=1)

        # Update per-objective running minimum. For each column,
        # pull only finite values from this generation.
        for j in range(n_obj):
            col_mask = np.isfinite(fit[:, j])
            if np.any(col_mask):
                col_min = float(np.min(fit[col_mask, j]))
                if col_min < running_best_obj[j]:
                    running_best_obj[j] = col_min

        # Update the L2 champion (requires all fitness components
        # finite — partially-failed genes don't qualify).
        if np.any(finite_all_mask):
            norms = np.linalg.norm(fit[finite_all_mask], axis=1)
            best_local_idx = int(np.argmin(norms))
            best_local_l2 = float(norms[best_local_idx])
            if best_local_l2 < running_best_l2:
                running_best_l2 = best_local_l2
                # Map from the filtered array back to the original
                # gen_genes list to read the champion's birth_gen.
                # ``np.where`` on a 1-D bool mask gives positions
                # within gen_genes.
                original_positions = np.where(finite_all_mask)[0]
                original_idx = int(original_positions[best_local_idx])
                running_champion_birth_gen = (
                    gen_genes[original_idx].birth_gen
                )

        row: Dict[str, Any] = dict(
            gen_idx=gen_idx,
            n_seen=cumulative_seen,
        )
        for j, label in enumerate(objective_labels):
            row[f"{label}_best"] = (
                float(running_best_obj[j])
                if np.isfinite(running_best_obj[j])
                else float("inf")
            )
        row["champion_l2"] = (
            running_best_l2 if np.isfinite(running_best_l2) else float("inf")
        )
        row["champion_birth_gen"] = running_champion_birth_gen
        rows.append(row)
    return rows


# --- main -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Inspect an exaconstit-calibrate SQLite archive from the "
            "command line. Useful when you don't want to install a "
            "SQLite client just to check how a run went."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Show every run in the archive:\n"
            "    python -m workflows.optimization.inspect_archive ./wf --runs\n"
            "\n"
            "  # Every generation's summary stats for the latest run:\n"
            "    python -m workflows.optimization.inspect_archive ./wf --gens\n"
            "\n"
            "  # Convergence view: per-gen best-ever fitness on each\n"
            "  # objective and the L2 champion's birth generation.\n"
            "  # A long run of the same champion_birth_gen means the\n"
            "  # GA has plateaued on the most-balanced gene:\n"
            "    python -m workflows.optimization.inspect_archive ./wf --gens-best\n"
            "\n"
            "  # Best-across-all-generations view: top 3 by balanced L2\n"
            "  # norm, plus top 3 per objective. Great for catching strong\n"
            "  # genes that NSGA-III dropped while exploring the space:\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --genes --pareto-only\n"
            "\n"
            "  # Same as above but top 5 per category, piped to CSV:\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --genes --pareto-only --top 5 --format csv > best.csv\n"
            "\n"
            "  # Single-generation rank-0 set at gen 10 (old behavior):\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --genes --gen 10 --pareto-only\n"
            "\n"
            "  # Raw dump of one generation's population (default: first\n"
            "  # 200 rows; pass --limit 0 to disable the cap):\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --genes --gen 10 --limit 50\n"
            "\n"
            "  # Preview what would be pruned, then actually prune.\n"
            "  # Useful after a debugging session leaves several\n"
            "  # aborted runs behind:\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --clean-empty-runs --dry-run\n"
            "    python -m workflows.optimization.inspect_archive ./wf \\\n"
            "        --clean-empty-runs\n"
        ),
    )
    p.add_argument(
        "path",
        type=Path,
        help=(
            "Path to the archive. Accepts three shapes: (a) a .db "
            "file directly — used as-is; (b) a directory containing "
            "'archive.db' — opens that; (c) a directory with no "
            "'archive.db' but exactly one other *.db — opens that; "
            "(d) a directory with multiple *.db files — prompts "
            "interactively for which one to use."
        ),
    )
    # Mutually-exclusive view selectors. --runs / --gens / --genes all
    # pick WHAT to show; default is --gens (the most commonly useful).
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--runs", action="store_true",
        help="List every run in the archive and exit",
    )
    group.add_argument(
        "--gens", action="store_true",
        help="Per-generation summary stats (default if nothing else given)",
    )
    group.add_argument(
        "--genes", action="store_true",
        help="Individual-level gene records (the Pareto front lives here)",
    )
    group.add_argument(
        "--gens-best", dest="gens_best", action="store_true",
        help=(
            "Convergence view: per-generation best-ever fitness "
            "on each objective, plus the generation where the "
            "current L2 champion was born. Great for spotting "
            "plateaus — if the champion_birth_gen column holds "
            "the same value for many generations, the GA is stuck."
        ),
    )
    group.add_argument(
        "--clean-empty-runs", dest="clean_empty_runs", action="store_true",
        help=(
            "Delete runs that have no generations and no case "
            "outputs. Useful after a debugging session that left "
            "several aborted starts in the archive. By default "
            "only touches runs that are either completed or older "
            "than 60 minutes (protects in-progress runs); override "
            "via --age-minutes. Pair with --dry-run to preview."
        ),
    )
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help=(
            "For --clean-empty-runs: list what would be deleted "
            "without actually deleting. No-op for other views."
        ),
    )
    p.add_argument(
        "--age-minutes", dest="age_minutes", type=float, default=60.0,
        help=(
            "For --clean-empty-runs: minimum age of a started-but-"
            "never-ended run before it becomes eligible for pruning. "
            "Default 60.0 protects in-progress runs. Pass 0 to "
            "consider all empty runs regardless of age. Ignored "
            "for runs that have explicitly called end_run."
        ),
    )
    p.add_argument(
        "--run", dest="run_id", default=None,
        help="Restrict to a specific run_id (default: the most recent)",
    )
    p.add_argument(
        "--gen", dest="gen_idx", type=int, default=None,
        help="Restrict --genes output to one generation (default: the last)",
    )
    p.add_argument(
        "--pareto-only", action="store_true",
        help=(
            "For --genes: show the best individuals across the WHOLE "
            "run history, not just the current generation. Produces "
            "one group per category — top N by L2 norm of the full "
            "fitness tuple (most balanced), plus top N per "
            "objective (best on each single axis). Useful because "
            "NSGA-III sometimes evicts strong individuals while "
            "exploring; this view resurrects them. If combined with "
            "--gen N, reverts to the old behavior of just that "
            "generation's rank-0 set."
        ),
    )
    p.add_argument(
        "--top", dest="top_n", type=int, default=3,
        help=(
            "How many genes to list per category in the "
            "cross-generation --pareto-only view (default: 3). "
            "Ignored when --pareto-only is off or when --gen is set."
        ),
    )
    p.add_argument(
        "--limit", dest="limit", type=int, default=200,
        help=(
            "Cap on the number of rows shown by the raw --genes "
            "view (default: 200). Prevents accidentally flooding "
            "the terminal when pointing at a run with a huge "
            "population. A notice is printed to stderr when rows "
            "are truncated. Pass 0 to disable the cap entirely. "
            "Ignored for --pareto-only (already bounded by --top) "
            "and for the other views."
        ),
    )
    p.add_argument(
        "--format", dest="fmt",
        choices=["table", "csv", "json"], default="table",
        help="Output format (default: table)",
    )
    p.add_argument(
        "--no-header", dest="with_header", action="store_false",
        help="Suppress the header row (useful when piping into another tool)",
    )
    p.add_argument(
        "--archive-name", default="archive.db",
        help=(
            "Filename to look for inside a directory path (default: "
            "archive.db). Ignored if ``path`` is a file. Retained "
            "for back-compat; prefer passing the file path directly."
        ),
    )
    return p


def resolve_archive_path(
    path: Path,
    default_name: str,
    *,
    input_fn: Callable[[str], str] = input,
    isatty_fn: Callable[[], bool] = lambda: sys.stdin.isatty(),
    stderr: Optional[IO[str]] = None,
) -> Path:
    """Resolve a path argument to an actual archive file.

    Public counterpart of :func:`_resolve_archive_path`. Same logic;
    exposed for use by other tools (e.g. the plot_solutions example)
    that want to accept the same flexible "workspace dir or .db file"
    user input.

    Note: a SystemExit-on-failure helper is sometimes inconvenient for
    library callers, but resolving an archive path is a fundamentally
    interactive operation in the multiple-db case (the prompt). Library
    callers who don't want the prompt should pass an isatty_fn that
    returns False; they'll then get a SystemExit they can catch with
    ``try/except SystemExit``. For pure non-interactive paths, point
    directly at a known db file and skip this resolver entirely.
    """
    return _resolve_archive_path(
        path, default_name,
        input_fn=input_fn, isatty_fn=isatty_fn, stderr=stderr,
    )


def _resolve_archive_path(
    path: Path,
    default_name: str,
    *,
    input_fn: Callable[[str], str] = input,
    isatty_fn: Callable[[], bool] = lambda: sys.stdin.isatty(),
    stderr: Optional[IO[str]] = None,
) -> Path:
    """Resolve the positional ``path`` argument to an actual archive file.

    Three shapes are accepted — in order of user effort:

    1. **File** — used as-is. Only validation: it has to exist.
    2. **Directory containing the default name** (``archive.db``) —
       silently uses that file. This is the common case.
    3. **Directory with no default but a different ``*.db``** —
       if there's exactly one, auto-pick it and mention the choice
       on stderr so the user sees what was inferred. If there are
       multiple, list them and prompt interactively for a 1-based
       index. Non-interactive contexts (stdin closed, piped input,
       CI) raise instead of hanging at the prompt.

    Args:
        path: Whatever the user typed as the positional arg.
        default_name: Filename to look for inside a directory
            (``"archive.db"`` is the driver's default).
        input_fn / isatty_fn / stderr: Indirection points for
            testing. Default behavior is the obvious stdlib bindings;
            tests override to exercise prompt flows deterministically.
            ``stderr`` defaults to the LIVE ``sys.stderr`` at call
            time, not import time — important because pytest's
            ``capsys`` rebinds ``sys.stderr`` mid-session and a
            static default would bypass the capture.

    Returns:
        An absolute Path to a real ``.db`` file on disk.

    Raises:
        SystemExit(1): if the path doesn't exist, is a directory
            with no ``.db`` files, or the user bails out of the
            prompt. Exit code 1 matches the pre-existing
            "user error" convention in :func:`main`.
    """
    if stderr is None:
        stderr = sys.stderr

    if not path.exists():
        print(f"error: path does not exist: {path}", file=stderr)
        raise SystemExit(1)

    # Case 1: explicit file. Users pasting the db path verbatim hit
    # this branch; it's also how the tests exercise the simple path.
    if path.is_file():
        return path

    # Case 2: the default ``archive.db`` is right there. No notice —
    # this is the expected shape for anyone using the framework's
    # defaults.
    default_path = path / default_name
    if default_path.is_file():
        return default_path

    # Cases 3a/3b/3c: directory, no default name. Enumerate *.db.
    candidates = sorted(path.glob("*.db"))
    if not candidates:
        print(
            f"error: no {default_name!r} and no other *.db files "
            f"under {path}",
            file=stderr,
        )
        raise SystemExit(1)

    # Case 3a: exactly one *.db — obvious intent, use it. Mention
    # the choice so the user isn't surprised when the output says
    # "run-xyz" but they typed no filename.
    if len(candidates) == 1:
        chosen = candidates[0]
        print(
            f"note: no {default_name!r} in {path}; using "
            f"{chosen.name}",
            file=stderr,
        )
        return chosen

    # Case 3b/3c: multiple *.db files. Prompt interactively if we
    # have a tty; otherwise refuse rather than hang.
    if not isatty_fn():
        listing = "\n  ".join(c.name for c in candidates)
        print(
            f"error: multiple *.db files under {path} and no "
            f"{default_name!r} to disambiguate:\n  {listing}\n"
            f"pass the desired file path directly, or run "
            f"interactively to be prompted.",
            file=stderr,
        )
        raise SystemExit(1)

    print(
        f"Multiple *.db files under {path}. Pick one:",
        file=stderr,
    )
    for i, c in enumerate(candidates, start=1):
        print(f"  [{i}] {c.name}", file=stderr)
    while True:
        try:
            raw = input_fn(f"Enter 1-{len(candidates)} (or q to quit): ")
        except (EOFError, KeyboardInterrupt):
            print("\naborted", file=stderr)
            raise SystemExit(1)
        raw = raw.strip().lower()
        if raw in ("q", "quit", "exit"):
            raise SystemExit(1)
        try:
            idx = int(raw)
        except ValueError:
            print(f"  not a number: {raw!r}", file=stderr)
            continue
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print(
            f"  out of range; pick 1-{len(candidates)}",
            file=stderr,
        )


def _handle_clean_empty_runs(
    db_path: Path,
    *,
    dry_run: bool,
    age_minutes: float,
) -> int:
    """Dispatch handler for ``--clean-empty-runs``.

    Opens the archive WRITABLE (the one CLI action that mutates),
    calls :meth:`ArchiveDB.prune_empty_runs`, and reports what
    happened on stderr. The body is kept in a separate helper so
    ``main()``'s main branch can stay read-only — a write-mode
    archive open elsewhere would be a subtle invariant to track.

    Output shape:

    * Prints the list of candidate run_ids to stderr (dry-run) or
      the list of run_ids actually deleted (normal mode).
    * Empty list on either branch → helper line pointing at
      ``--age-minutes 0`` in case the user expected something to
      happen and nothing did.
    * Exit code 0 in all normal paths.
    """
    from workflow_common.archive import ArchiveDB

    with ArchiveDB(db_path) as a:
        affected = a.prune_empty_runs(
            min_age_minutes=age_minutes, dry_run=dry_run,
        )

    verb = "would delete" if dry_run else "deleted"
    if not affected:
        print(
            f"no empty runs eligible for deletion "
            f"(min_age_minutes={age_minutes}). "
            f"Pass --age-minutes 0 to consider young empty runs too.",
            file=sys.stderr,
        )
        return 0

    print(f"{verb} {len(affected)} empty run(s):", file=sys.stderr)
    for rid in affected:
        print(f"  {rid}", file=sys.stderr)
    if dry_run:
        print(
            "dry run: no changes were made. "
            "Rerun without --dry-run to commit.",
            file=sys.stderr,
        )
    return 0


def _emit(
    rows: List[Dict[str, Any]],
    columns: List[str],
    fmt: str,
    with_header: bool,
) -> None:
    if not rows:
        print("(no rows matched the query)", file=sys.stderr)
        raise SystemExit(2)
    if fmt == "table":
        out = _format_table(rows, columns, with_header=with_header)
    elif fmt == "csv":
        out = _format_csv(rows, columns, with_header=with_header)
    else:  # json
        out = _format_json(rows)
    print(out)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve the positional path through the three-behavior rule:
    # explicit file > default-name-in-directory > auto-pick or
    # prompt among *.db. Any failure mode inside the resolver
    # raises SystemExit(1); we don't catch it because argparse is
    # already letting SystemExit propagate normally.
    db_path = _resolve_archive_path(args.path, args.archive_name)

    # --pareto-only only makes sense with --genes (that's where the
    # cross-gen-or-single-gen Pareto logic actually lives). Three
    # cases to handle cleanly:
    #
    # 1. ``--pareto-only`` on its own: interpret as "the user wants
    #    the cross-gen top-N view". Imply --genes so we hit the
    #    right dispatch branch. Prior behavior was to silently fall
    #    through to --gens, which ignored --pareto-only entirely —
    #    a classic silent-no-op trap.
    #
    # 2. ``--pareto-only --genes``: explicit and intended; continue.
    #
    # 3. ``--pareto-only`` combined with any non-gene view
    #    (--runs / --gens / --gens-best): logical contradiction.
    #    Fail loudly with a clear message so the user rewords;
    #    silently dropping the flag is how this bug class hides.
    any_nongene_view = (
        args.runs or args.gens or args.gens_best or args.clean_empty_runs
    )
    if args.pareto_only and any_nongene_view:
        print(
            "error: --pareto-only only applies to --genes; it cannot "
            "be combined with --runs, --gens, --gens-best, or "
            "--clean-empty-runs.",
            file=sys.stderr,
        )
        return 1
    if args.pareto_only and not args.genes:
        # Case 1: imply --genes so the dispatch hits the Pareto branch.
        args.genes = True

    # --gens is the sensible default; preserve user intent if they
    # gave anything explicit (including --gens itself).
    any_view = (
        args.runs or args.gens or args.genes
        or args.gens_best or args.clean_empty_runs
    )
    if not any_view:
        args.gens = True

    # --clean-empty-runs is the only view that mutates the archive,
    # so it gets its own write-mode opener. Handled here before the
    # read-only block below. Dry-run still goes through this branch
    # because prune_empty_runs() does the "what would be deleted"
    # query against the live DB — it just doesn't commit the
    # DELETEs.
    if args.clean_empty_runs:
        return _handle_clean_empty_runs(
            db_path,
            dry_run=args.dry_run,
            age_minutes=args.age_minutes,
        )

    with ArchiveDB(db_path, readonly=True) as a:
        if args.runs:
            rows = _run_rows(a.list_runs())
            columns = [
                "run_id", "started_at", "completed_at", "seed",
                "n_generations", "param_names", "objective_labels",
            ]
            _emit(rows, columns, args.fmt, args.with_header)
            return 0

        run = _pick_run(a, args.run_id)

        if args.gens:
            gens = a.list_generations(run.run_id)
            rows = _gen_rows(gens)
            # Column order: fixed fields first, then whatever stats
            # keys appear. Different runs may carry different stats
            # depending on single- vs multi-objective.
            fixed = ["gen_idx", "recorded_at", "n_pop"]
            stat_keys: List[str] = []
            seen = set(fixed)
            for r in rows:
                for k in r:
                    if k not in seen:
                        stat_keys.append(k)
                        seen.add(k)
            columns = fixed + stat_keys
            _emit(rows, columns, args.fmt, args.with_header)
            return 0

        if args.gens_best:
            # Convergence view: running best-per-objective + L2
            # champion's birth generation per gen. Needs every
            # gene, so we take the batch hit; budget ~1s at 50k
            # genes (dominated by the same JSON decode we hit in
            # --pareto-only).
            all_genes = _collect_all_genes(a, run.run_id)
            rows = _gens_best_rows(
                all_genes,
                objective_labels=run.objective_labels,
            )
            columns = (
                ["gen_idx", "n_seen"]
                + [f"{lab}_best" for lab in run.objective_labels]
                + ["champion_l2", "champion_birth_gen"]
            )
            _emit(rows, columns, args.fmt, args.with_header)
            return 0

        if args.genes:
            gens = a.list_generations(run.run_id)
            if not gens:
                print(
                    f"error: run {run.run_id!r} has no generations",
                    file=sys.stderr,
                )
                return 1

            # Dispatch branches:
            #   --pareto-only (no --gen)  -> cross-generation top-N
            #   --pareto-only --gen N     -> just gen N's rank-0 set
            #   (no pareto flag)          -> full population of one gen
            if args.pareto_only and args.gen_idx is None:
                all_genes = _collect_all_genes(a, run.run_id)
                rows = _pareto_history_rows(
                    all_genes,
                    param_names=run.param_names,
                    objective_labels=run.objective_labels,
                    top_n=args.top_n,
                )
                columns = (
                    ["category", "birth_gen", "birth_gene", "rank"]
                    + list(run.param_names)
                    + list(run.objective_labels)
                    + ["l2_norm"]
                )
                _emit(rows, columns, args.fmt, args.with_header)
                return 0

            # Single-generation path: either an explicit --gen N, or
            # default-to-latest. --pareto-only here (paired with a
            # specific gen) keeps the old "rank-0 in this gen"
            # filter so anyone who was scripting that behavior
            # before keeps working.
            if args.gen_idx is not None:
                target = args.gen_idx
            else:
                target = max(g.gen_idx for g in gens)
            genes = a.load_genes(run.run_id, target)
            rows = _gene_rows(
                genes,
                param_names=run.param_names,
                objective_labels=run.objective_labels,
                pareto_only=args.pareto_only,
            )
            # --limit caps the raw --genes dump. Skip the cap for
            # --pareto-only (already bounded by top_n * categories)
            # so the user's explicit Pareto-filtered request isn't
            # trimmed further. 0 means "no cap"; anything else
            # trims and notes the truncation on stderr so the user
            # sees why their row count is suspiciously round.
            if (
                not args.pareto_only
                and args.limit > 0
                and len(rows) > args.limit
            ):
                print(
                    f"note: truncated to {args.limit} of {len(rows)} "
                    f"rows; pass --limit 0 to disable or --limit N "
                    f"for a different cap.",
                    file=sys.stderr,
                )
                rows = rows[:args.limit]
            columns = (
                ["gen_idx", "pop_idx", "birth_gen", "birth_gene", "rank"]
                + list(run.param_names)
                + list(run.objective_labels)
            )
            _emit(rows, columns, args.fmt, args.with_header)
            return 0

    # Shouldn't be reachable because of the default-view fallback above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
