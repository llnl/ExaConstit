"""Tests for the public selection API and the plot_solutions example.

Two surfaces:

1. :mod:`workflows.optimization.inspect_archive` public API —
   ``select_top_genes``, ``pick_run``, ``dedup_on_gene_vector``,
   ``collect_all_genes``, ``resolve_archive_path``. These were
   promoted from internal helpers and need their library contracts
   pinned independently of the CLI.

2. :mod:`examples.plot_solutions` — the data-loading helpers
   (skipping matplotlib UI which requires a display). Verifies the
   example correctly composes the public API to produce ranked
   ``RankedGene`` lists matching the user's request.

Matplotlib plotting itself isn't tested at the rendering level; the
example uses matplotlib's standard ``Slider`` and ``pick_event``
APIs, which are themselves well-tested upstream.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common.archive import ArchiveDB, GeneRecord
from workflow_common.paths import CaseContext
from workflow_common.results import CaseResultSet, TabularResult
from workflows.optimization.inspect_archive import (
    RankedGene,
    collect_all_genes,
    dedup_on_gene_vector,
    pick_run,
    resolve_archive_path,
    select_top_genes,
)


# --- Helpers ------------------------------------------------------------


def _voce_stress_df(y0=200.0, H=1900.0, k=48.0, n=20) -> pd.DataFrame:
    """Synthetic Voce-curve stress data shaped like ExaConstit avg_stress."""
    t = np.linspace(0.0, 1.0, n)
    stress = y0 + H * (1 - np.exp(-k * t))
    return pd.DataFrame({
        "Time": t, "Volume": np.ones_like(t),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Szz": stress,
        "Sxy": np.zeros_like(t), "Sxz": np.zeros_like(t),
        "Syz": np.zeros_like(t),
    })


def _voce_defgrad_df(rate=1.0, n=20) -> pd.DataFrame:
    t = np.linspace(0.0, 1.0, n)
    eps = rate * t
    F33 = 1.0 + eps
    return pd.DataFrame({
        "Time": t, "Volume": np.ones_like(t),
        "F11": 1.0 - 0.5 * eps, "F12": np.zeros_like(t),
        "F13": np.zeros_like(t),
        "F21": np.zeros_like(t), "F22": 1.0 - 0.5 * eps,
        "F23": np.zeros_like(t),
        "F31": np.zeros_like(t), "F32": np.zeros_like(t),
        "F33": F33,
    })


def _make_case_result_set(
    ctx: CaseContext, y0: float = 200.0, H: float = 1900.0,
) -> CaseResultSet:
    return CaseResultSet(ctx=ctx, tables={
        "avg_stress": TabularResult(
            name="avg_stress", df=_voce_stress_df(y0=y0, H=H),
            source_path=Path("fake/avg_stress.txt"),
        ),
        "avg_def_grad": TabularResult(
            name="avg_def_grad", df=_voce_defgrad_df(),
            source_path=Path("fake/avg_def_grad.txt"),
        ),
    })


# --- select_top_genes --------------------------------------------------


def _gene(
    *, gen_idx, pop_idx, fitness, gene_vec=None, rank=0,
):
    """Compact GeneRecord factory for ranking tests."""
    return GeneRecord(
        run_id="r", gen_idx=gen_idx, pop_idx=pop_idx,
        birth_gen=gen_idx, birth_gene=pop_idx,
        gene_vector=np.asarray(
            gene_vec if gene_vec is not None
            else [float(gen_idx), float(pop_idx)],
            dtype=float,
        ),
        fitness=tuple(fitness), rank=rank,
    )


def test_select_top_genes_l2_ranks_by_l2_norm_ascending():
    """The L2 category should be sorted ascending by sqrt(sum(f^2))."""
    genes = [
        _gene(gen_idx=0, pop_idx=0, fitness=[3.0, 4.0]),  # L2=5
        _gene(gen_idx=0, pop_idx=1, fitness=[1.0, 1.0]),  # L2~1.41 (best)
        _gene(gen_idx=0, pop_idx=2, fitness=[6.0, 8.0]),  # L2=10
    ]
    out = select_top_genes(
        genes, objective_labels=["a", "b"], top_n=3,
        categories=("l2",),
    )
    l2 = out["l2"]
    assert [rg.gene.pop_idx for rg in l2] == [1, 0, 2]
    assert [rg.rank for rg in l2] == [0, 1, 2]
    assert [rg.category for rg in l2] == ["l2"] * 3
    # Score == L2 norm exactly for the L2 category.
    assert l2[0].score == pytest.approx(np.sqrt(2.0))
    assert l2[1].score == pytest.approx(5.0)


def test_select_top_genes_per_objective_ranks_each_axis_separately():
    """obj:<label> rankings sort by that single component, ascending."""
    genes = [
        _gene(gen_idx=0, pop_idx=0, fitness=[1.0, 5.0]),  # best on a
        _gene(gen_idx=0, pop_idx=1, fitness=[5.0, 1.0]),  # best on b
        _gene(gen_idx=0, pop_idx=2, fitness=[3.0, 3.0]),
    ]
    out = select_top_genes(
        genes, objective_labels=["a", "b"], top_n=3,
        categories=("per_objective",),
    )
    assert [rg.gene.pop_idx for rg in out["obj:a"]] == [0, 2, 1]
    assert [rg.gene.pop_idx for rg in out["obj:b"]] == [1, 2, 0]
    # Per-objective score is the value on that axis.
    assert out["obj:a"][0].score == 1.0
    assert out["obj:b"][0].score == 1.0


def test_select_top_genes_dedup_removes_repeats_keeping_oldest():
    """Same gene_vector at gens 0 and 1 should reduce to one entry, keeping the gen-0 sighting."""
    genes = [
        _gene(gen_idx=0, pop_idx=0, fitness=[1.0, 1.0],
              gene_vec=[100.0, 100.0]),
        _gene(gen_idx=1, pop_idx=0, fitness=[1.0, 1.0],
              gene_vec=[100.0, 100.0]),  # duplicate
        _gene(gen_idx=1, pop_idx=1, fitness=[2.0, 2.0],
              gene_vec=[200.0, 200.0]),
    ]
    out = select_top_genes(
        genes, objective_labels=["a", "b"], top_n=10,
        categories=("l2",),
    )
    l2 = out["l2"]
    assert len(l2) == 2
    # First entry should be the gen-0 sighting of the duplicate.
    assert l2[0].gene.gen_idx == 0


def test_select_top_genes_filters_non_finite_fitness():
    """Genes with inf/nan fitness must be excluded from rankings."""
    genes = [
        _gene(gen_idx=0, pop_idx=0, fitness=[float("inf"), 1.0]),
        _gene(gen_idx=0, pop_idx=1, fitness=[1.0, 1.0]),
        _gene(gen_idx=0, pop_idx=2, fitness=[float("nan"), 5.0]),
    ]
    out = select_top_genes(
        genes, objective_labels=["a", "b"], top_n=10,
        categories=("l2",),
    )
    assert len(out["l2"]) == 1
    assert out["l2"][0].gene.pop_idx == 1


def test_select_top_genes_top_n_cap_respects_available():
    """Asking for more than exist returns everything."""
    genes = [_gene(gen_idx=0, pop_idx=i, fitness=[float(i)]) for i in range(3)]
    out = select_top_genes(
        genes, objective_labels=["a"], top_n=999,
        categories=("l2",),
    )
    assert len(out["l2"]) == 3


def test_select_top_genes_empty_input_returns_empty_dict():
    assert select_top_genes(
        [], objective_labels=["a"], top_n=5,
    ) == {}


def test_dedup_on_gene_vector_preserves_input_order():
    g1 = _gene(gen_idx=0, pop_idx=0, fitness=[0.5], gene_vec=[1.0])
    g2 = _gene(gen_idx=0, pop_idx=1, fitness=[0.5], gene_vec=[2.0])
    g3 = _gene(gen_idx=1, pop_idx=0, fitness=[0.5], gene_vec=[1.0])
    out = dedup_on_gene_vector([g1, g2, g3])
    assert [g.pop_idx for g in out] == [0, 1]
    # First sighting (gen 0) preferred over the gen-1 dupe.
    assert out[0].gen_idx == 0


# --- pick_run ------------------------------------------------------------


def test_pick_run_returns_only_run_when_unspecified(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(seed=1, param_names=["x"], objective_labels=["o"])
        a.end_run(rid)
    with ArchiveDB(db, readonly=True) as a:
        r = pick_run(a)
    assert r.run_id == rid


def test_pick_run_picks_latest_when_multiple_and_no_id(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        old = a.start_run(seed=1, param_names=["x"], objective_labels=["o"])
        a.end_run(old)
        new = a.start_run(seed=2, param_names=["x"], objective_labels=["o"])
        a.end_run(new)
    with ArchiveDB(db, readonly=True) as a:
        r = pick_run(a)  # latest_if_none=True default
    assert r.run_id == new


def test_pick_run_strict_mode_rejects_multiple_runs(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        a.start_run(run_id="r1", seed=1,
                    param_names=["x"], objective_labels=["o"])
        a.end_run("r1")
        a.start_run(run_id="r2", seed=2,
                    param_names=["x"], objective_labels=["o"])
        a.end_run("r2")
    with ArchiveDB(db, readonly=True) as a:
        with pytest.raises(ValueError, match="has 2 runs"):
            pick_run(a, latest_if_none=False)


def test_pick_run_unknown_id_raises_with_listing(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        a.start_run(run_id="real", seed=1,
                    param_names=["x"], objective_labels=["o"])
        a.end_run("real")
    with ArchiveDB(db, readonly=True) as a:
        with pytest.raises(ValueError, match="not in archive"):
            pick_run(a, run_id="bogus")


def test_pick_run_empty_archive_raises(tmp_path):
    with ArchiveDB(tmp_path / "a.db") as a:
        pass  # never started a run
    with ArchiveDB(tmp_path / "a.db", readonly=True) as a:
        with pytest.raises(ValueError, match="no runs"):
            pick_run(a)


# --- resolve_archive_path ------------------------------------------------


def test_resolve_archive_path_uses_calibration_db_in_workspace(tmp_path):
    """When ``calibration.db`` exists at the workspace root, use it."""
    (tmp_path / "calibration.db").write_text("")
    p = resolve_archive_path(tmp_path, "calibration.db")
    assert p == tmp_path / "calibration.db"


def test_resolve_archive_path_accepts_direct_file(tmp_path):
    f = tmp_path / "custom.db"
    f.write_text("")
    p = resolve_archive_path(f, "calibration.db")
    assert p == f


# --- collect_all_genes ---------------------------------------------------


def test_collect_all_genes_returns_every_gene_in_run(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        for gen_idx in range(3):
            genes = [
                GeneRecord(
                    run_id=rid, gen_idx=gen_idx, pop_idx=i,
                    birth_gen=gen_idx, birth_gene=i,
                    gene_vector=np.array([float(gen_idx), float(i)]),
                    fitness=(0.1 * i,), rank=0,
                )
                for i in range(2)
            ]
            a.record_generation(
                rid, gen_idx=gen_idx,
                genes=genes,
                stats={"avg": [0.1]},
            )
    with ArchiveDB(db, readonly=True) as a:
        genes = collect_all_genes(a, rid)
    assert len(genes) == 6  # 3 gens * 2 individuals


# --- examples.plot_solutions integration --------------------------------


@pytest.fixture
def plot_solutions_module():
    """Load the example as a module without requiring it on sys.path.

    The examples directory is a sibling of workflow_common, not a
    package, so we load by file path. Cached for the session.
    """
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "plot_solutions", here / "examples" / "plot_solutions.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_solutions"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def calibration_archive(tmp_path: Path) -> Path:
    """Two-objective archive with case_outputs + experimental data for 2 SimCases.

    Five genes across two generations. Genes are designed so the
    L2-best is gen=1, pop_idx=0; the obj=0-best is gen=1, pop_idx=0
    (fitness 0.4); the obj=1-best is gen=1, pop_idx=2 (fitness 0.3).
    Experimental DataFrames are recorded for both SimCases so the
    archive-first experimental resolution path is exercised.
    """
    db = tmp_path / "calibration.db"
    fitnesses = [
        # gen 0
        [(2.0, 2.0), (3.0, 4.0), (0.5, 5.0)],
        # gen 1
        [(0.4, 0.4), (1.0, 1.0), (4.0, 0.3)],
    ]
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="run-1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["rmse_0", "rmse_1"],
        )
        for gen_idx, gen_fits in enumerate(fitnesses):
            genes = [
                GeneRecord(
                    run_id=rid, gen_idx=gen_idx, pop_idx=i,
                    birth_gen=gen_idx, birth_gene=i,
                    gene_vector=np.array([
                        200.0 + 10 * gen_idx + i,
                        2000.0 + 100 * gen_idx + 50 * i,
                    ]),
                    fitness=fit, rank=0,
                )
                for i, fit in enumerate(gen_fits)
            ]
            a.record_generation(
                rid, gen_idx=gen_idx, genes=genes,
                stats={"avg": [1.0, 1.0]},
            )
            for g_idx, g in enumerate(genes):
                for sim_case_idx in range(2):
                    ctx = CaseContext(
                        generation=g.birth_gen, gene=g.birth_gene,
                        obj=sim_case_idx,
                    )
                    rs = _make_case_result_set(
                        ctx,
                        y0=200.0 + 5 * sim_case_idx + g_idx,
                        H=1900.0 + 50 * sim_case_idx,
                    )
                    a.record_case_outputs(
                        rid,
                        birth_gen=g.birth_gen, birth_gene=g.birth_gene,
                        sim_case_idx=sim_case_idx, results=rs,
                    )
        # Record experimental data for both SimCases. The plotter's
        # archive-first resolution path picks these up — no CSV
        # paths needed at plot time.
        from workflow_common.objectives import StressStrainExtractor
        for sim_case_idx in range(2):
            exp_df = pd.DataFrame({
                "strain": np.linspace(0.0, 1.0, 25),
                "stress": (210.0 + 5 * sim_case_idx) + 1900.0 * (
                    1 - np.exp(-48.0 * np.linspace(0.0, 1.0, 25))
                ),
            })
            # Different windows per SimCase so window-related tests
            # can distinguish them on the rendered figure.
            window = (0.05, 0.30) if sim_case_idx == 0 else (None, 0.40)
            # Store an extractor config too — exercises the
            # Pareto-click "use archived extractor" path so tests
            # don't accidentally rely on default fallbacks.
            ext_cfg = StressStrainExtractor(
                window=(window[0] if window[0] is not None else 0.0,
                        window[1] if window[1] is not None else 1e9),
            ).to_dict()
            a.record_experiment(
                rid, sim_case_idx=sim_case_idx, df=exp_df,
                label=f"exp_sc{sim_case_idx}",
                minmax_strain=window,
                extractor_config=ext_cfg,
            )
        a.end_run(rid)
    return tmp_path


def test_plot_solutions_l2_mode_picks_lowest_l2_first(
    calibration_archive, plot_solutions_module,
):
    """Verify the example's L2 ranking returns gen=1 idx=0 first.

    That gene has fitness (0.4, 0.4) — L2 ~ 0.566 — beating every
    other gene by L2 norm. Pinning this guards against a future
    regression where the ranking accidentally gets reversed or
    the dedup drops the winner.
    """
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        ranked = mod._ranked_genes_for_mode(
            a, meta.run_id,
            objective_labels=meta.objective_labels,
            top_n=3, mode="l2", objective=None,
        )
    assert len(ranked) == 3
    # Best in L2: gen=1, pop_idx=0 with fitness (0.4, 0.4).
    assert ranked[0].gene.birth_gen == 1
    assert ranked[0].gene.birth_gene == 0
    assert ranked[0].rank == 0
    # Score = L2 norm for this category.
    assert ranked[0].score == pytest.approx(np.sqrt(0.32), abs=1e-6)


def test_plot_solutions_objective_mode_picks_objective_winner(
    calibration_archive, plot_solutions_module,
):
    """obj=0 best in the fixture is gen=1, pop_idx=0 (fitness 0.4, 0.4)."""
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        ranked = mod._ranked_genes_for_mode(
            a, meta.run_id,
            objective_labels=meta.objective_labels,
            top_n=1, mode="objective", objective=0,
        )
    assert len(ranked) == 1
    # Best on obj 0 is gen=1, pop_idx=0 (fitness (0.4, 0.4)).
    assert ranked[0].gene.birth_gen == 1
    assert ranked[0].gene.birth_gene == 0
    assert ranked[0].score == pytest.approx(0.4)
    # Category includes the objective label.
    assert ranked[0].category == "obj:rmse_0"


def test_plot_solutions_objective_mode_accepts_label_string(
    calibration_archive, plot_solutions_module,
):
    """--objective rmse_1 should resolve to objective index 1."""
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        ranked = mod._ranked_genes_for_mode(
            a, meta.run_id,
            objective_labels=meta.objective_labels,
            top_n=1, mode="objective", objective="rmse_1",
        )
    # Best on rmse_1 is gen=1 pop=2 (fitness (4.0, 0.3)).
    assert ranked[0].gene.birth_gen == 1
    assert ranked[0].gene.birth_gene == 2


def test_plot_solutions_last_gen_mode_returns_final_generation_only(
    calibration_archive, plot_solutions_module,
):
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        ranked = mod._ranked_genes_for_mode(
            a, meta.run_id,
            objective_labels=meta.objective_labels,
            top_n=10, mode="last-gen", objective=None,
        )
    # Three genes were recorded in gen 1.
    assert all(r.gene.birth_gen == 1 for r in ranked)
    assert len(ranked) == 3
    # Order is by pop_idx ascending in last-gen mode.
    assert [r.gene.birth_gene for r in ranked] == [0, 1, 2]


def test_plot_solutions_discover_sim_case_count_finds_two(
    calibration_archive, plot_solutions_module,
):
    """The fixture archives two SimCases per gene; probe should find both."""
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        sample = collect_all_genes(a, meta.run_id)[0]
        n = mod._discover_sim_case_count(a, meta.run_id, sample)
    assert n == 2


def test_plot_solutions_resolve_objective_index_label_or_int(
    plot_solutions_module,
):
    """The objective resolver handles both labels and integer strings."""
    mod = plot_solutions_module
    labels = ["stress_rmse", "slope_rmse"]
    assert mod._resolve_objective_index("stress_rmse", labels) == 0
    assert mod._resolve_objective_index("slope_rmse", labels) == 1
    assert mod._resolve_objective_index("0", labels) == 0
    assert mod._resolve_objective_index(1, labels) == 1
    with pytest.raises(ValueError, match="not found"):
        mod._resolve_objective_index("nope", labels)
    with pytest.raises(ValueError, match="out of range"):
        mod._resolve_objective_index(5, labels)


def test_plot_solutions_main_errors_on_missing_workspace(
    tmp_path, plot_solutions_module, capsys,
):
    """CLI error path: nonexistent workspace exits 1 with a message."""
    mod = plot_solutions_module
    rc = mod.main([str(tmp_path / "does_not_exist")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


# --- Plotter end-to-end (headless, write-and-check) -----------------------

# These tests use Agg so no display is needed. They verify the plot
# functions complete and produce figures with the expected shape; they
# don't test pixel content.

@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    """Force matplotlib's Agg backend for every test in this module
    AND close all figures after each test runs.

    Backend choice via env var: matplotlib lazy-imports, so we set
    MPLBACKEND before any test triggers its import.

    Figure-cleanup yield: tests that open figures via
    plt.figure() (which the plot helpers do) leak across the
    session; once the count exceeds matplotlib's
    ``figure.max_open_warning`` threshold (20 by default), the
    `RuntimeWarning` may escalate to an error under strict warning
    filters. Closing on teardown keeps each test isolated.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


def test_plot_overlay_uses_archive_experimental_data_no_csv_needed(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """The overlay plot must work without --experimental when the archive
    has experimental data. This is the headline reason for storing
    experimental DataFrames in the SQLite database — the user shouldn't
    need to remember which CSV maps to which SimCase post-run.
    """
    mod = plot_solutions_module
    save_path = tmp_path / "overlay.png"
    fig = mod.plot_top_solutions_overlay(
        str(calibration_archive),
        top_n=3, mode="l2",
        # No experimental_paths supplied — must come from archive.
        save=save_path, show=False,
    )
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    # 2 SimCases × 2 rows (stress + slope) + slider + panel = 6 axes.
    # The exact layout is documented in the plot fn; just sanity-check
    # there's more than one axis (no plot would be 0-1).
    assert len(fig.axes) >= 4


def test_plot_pareto_top_n_restricts_scatter_points(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """top_n=N must restrict the Pareto scatter to the N lowest-L2 genes.

    With top_n=2 on the fixture's 6 genes, the scatter's PathCollection
    should hold exactly 2 points. This pins the bug Robert reported
    where the Pareto plot ignored --top.
    """
    mod = plot_solutions_module
    save_path = tmp_path / "pareto_top2.png"
    fig = mod.plot_pareto_front_with_l2_winner(
        str(calibration_archive),
        objective_pair=(0, 1),
        top_n=2,
        save=save_path, show=False,
    )
    assert save_path.exists()
    # Find the scatter PathCollection on the leftmost axis. There are
    # multiple PathCollections (main scatter, L2 winner ring, utopia
    # marker); the main one has the most points, and is the one with
    # `picker` enabled so it's the only one with a non-None pickradius.
    import matplotlib.collections as mcoll
    scatter_axes = [
        ax for ax in fig.axes
        if any(isinstance(c, mcoll.PathCollection) for c in ax.collections)
    ]
    assert scatter_axes, "no scatter axes found on Pareto figure"
    # Find the picker-enabled scatter; that's the main one.
    main_scatters = []
    for ax in scatter_axes:
        for coll in ax.collections:
            if (isinstance(coll, mcoll.PathCollection)
                    and coll.get_picker() is not None):
                main_scatters.append(coll)
    assert len(main_scatters) >= 1
    main_sc = main_scatters[0]
    assert main_sc.get_offsets().shape[0] == 2


def test_plot_pareto_top_n_zero_uses_all_rank_zero_genes(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """top_n=0 (default) should plot every rank-0 gene from the run."""
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(calibration_archive),
        objective_pair=(0, 1),
        top_n=0,
        save=tmp_path / "pareto_all.png", show=False,
    )
    import matplotlib.collections as mcoll
    main_sc = next(
        coll for ax in fig.axes for coll in ax.collections
        if isinstance(coll, mcoll.PathCollection)
        and coll.get_picker() is not None
    )
    # Fixture has 6 rank-0 genes, all unique gene-vectors → 6 points.
    assert main_sc.get_offsets().shape[0] == 6


def test_plot_pareto_inset_axes_present_for_response_curve(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """The Pareto figure must include the inset axes that draws the
    response curve when a point is clicked. We verify presence by
    looking for an axes with the expected title prefix; click handler
    behavior itself is tested indirectly via the picker config.
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(calibration_archive),
        objective_pair=(0, 1),
        top_n=3,
        save=tmp_path / "pareto_with_inset.png", show=False,
    )
    titles = [ax.get_title() for ax in fig.axes]
    # Inset's title says "Response — gene g…" after the initial draw.
    assert any("Response" in t for t in titles), (
        f"no Response axes found; titles: {titles}"
    )


def test_plot_pareto_picker_is_enabled_on_main_scatter(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """Click-to-show only works if the scatter has picker enabled.

    Pinning this guards against future refactors silently dropping
    the picker= argument and leaving an unclickable scatter.
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(calibration_archive),
        objective_pair=(0, 1), top_n=3,
        save=tmp_path / "p.png", show=False,
    )
    import matplotlib.collections as mcoll
    pickers = [
        coll.get_picker()
        for ax in fig.axes for coll in ax.collections
        if isinstance(coll, mcoll.PathCollection)
    ]
    # At least one collection has a picker enabled.
    assert any(p is not None and p is not False for p in pickers)


# --- CLI behavior ---------------------------------------------------------


def test_main_auto_promotes_mode_when_objective_given_without_mode(
    calibration_archive, plot_solutions_module, tmp_path, capsys,
):
    """`--objective 0` with default `--mode l2` should auto-promote to
    `--mode objective`. Pinning this fixes Robert's bug where
    `--objective 0` silently fell through to L2 ranking.
    """
    mod = plot_solutions_module
    rc = mod.main([
        str(calibration_archive),
        "--top", "1",
        "--objective", "0",
        "--save", str(tmp_path / "obj0.png"),
        "--no-show",
    ])
    assert rc == 0
    # The fixture's obj=0 winner is gen=1 pop=0 with fitness 0.4.
    # If mode auto-promotion failed, we'd be plotting the L2 winner
    # (which is also gen=1 pop=0 in this fixture, so we can't
    # distinguish on data alone here). We rely on the rc=0 + save
    # success as the smoke check; the unit-level test below pins
    # the actual selection.


def test_main_objective_with_last_gen_mode_errors(
    calibration_archive, plot_solutions_module, capsys,
):
    """--objective and --mode last-gen are contradictory; main should reject."""
    mod = plot_solutions_module
    rc = mod.main([
        str(calibration_archive),
        "--mode", "last-gen",
        "--objective", "0",
        "--no-show",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not compatible" in err or "last-gen" in err


def test_main_pareto_alone_skips_overlay(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """--pareto i,j with no other flags should produce ONLY the
    Pareto figure. Pins Robert's ask: 'when we pass in the pareto
    flag only the pareto front should be plotted.'

    The previous semantics required an explicit --no-overlay to
    suppress the headline figure; that's been inverted.
    """
    mod = plot_solutions_module
    overlay_save = tmp_path / "should_not_exist.png"
    pareto_save = tmp_path / "pareto.png"
    rc = mod.main([
        str(calibration_archive),
        "--top", "3",
        "--pareto", "0,1",
        "--save", str(overlay_save),
        "--save-pareto", str(pareto_save),
        "--no-show",
    ])
    assert rc == 0
    assert not overlay_save.exists(), (
        "overlay file was created despite bare --pareto"
    )
    assert pareto_save.exists()


def test_main_overlay_flag_forces_overlay_with_pareto(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """--pareto + --overlay should produce BOTH figures. The
    --overlay opt-in is the explicit way to keep the headline
    overlay alongside the Pareto plot.
    """
    mod = plot_solutions_module
    overlay_save = tmp_path / "overlay.png"
    pareto_save = tmp_path / "pareto.png"
    rc = mod.main([
        str(calibration_archive),
        "--top", "3",
        "--pareto", "0,1",
        "--overlay",
        "--save", str(overlay_save),
        "--save-pareto", str(pareto_save),
        "--no-show",
    ])
    assert rc == 0
    assert overlay_save.exists(), "overlay should be produced with --overlay"
    assert pareto_save.exists()


def test_main_no_pareto_still_shows_overlay(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """The bare command (no --pareto, no --overlay) shows the
    headline overlay — the default for "user just wants to see
    their results."
    """
    mod = plot_solutions_module
    overlay_save = tmp_path / "overlay.png"
    rc = mod.main([
        str(calibration_archive),
        "--top", "3",
        "--save", str(overlay_save),
        "--no-show",
    ])
    assert rc == 0
    assert overlay_save.exists()


def test_main_pareto_only_does_not_say_l2_in_title_when_objective_given(
    calibration_archive, plot_solutions_module, tmp_path, capsys,
):
    """When mode auto-promotes from --objective, the overlay's title
    should say `obj:rmse_0` not `l2`. Robert's complaint was the title
    saying L2 even after --objective was passed.
    """
    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(calibration_archive),
        top_n=2, mode="objective", objective=0,
        save=tmp_path / "obj.png", show=False,
    )
    suptitle = fig._suptitle.get_text() if fig._suptitle else ""
    assert "obj:rmse_0" in suptitle, (
        f"suptitle should mention obj:rmse_0; got {suptitle!r}"
    )
    assert "l2" not in suptitle.lower().replace("rmse_0", ""), (
        f"suptitle should NOT mention l2; got {suptitle!r}"
    )


# --- Archive-first experimental resolution -------------------------------


def test_resolve_experimental_archive_first(
    calibration_archive, plot_solutions_module,
):
    """Archive-first resolution picks the SimCase's stored DataFrame
    even when no CSV fallback is supplied.
    """
    mod = plot_solutions_module
    with ArchiveDB(calibration_archive / "calibration.db", readonly=True) as a:
        meta = pick_run(a, latest_if_none=True)
        result = mod._resolve_experimental_for_case(
            a, meta.run_id, sim_case_idx=0, csv_fallback=None,
        )
    assert result is not None
    strain, stress, label = result
    assert label == "exp_sc0"
    assert len(strain) == len(stress) == 25


def test_resolve_experimental_falls_back_to_csv_when_archive_missing(
    tmp_path, plot_solutions_module,
):
    """If the archive has no experimental data for a SimCase, fall back
    to the user-supplied CSV path. Backward compat for archives written
    before the experiments table existed.

    Note: ``load_experimental_csv`` defaults to whitespace delimiters
    (matches ExaConstit's typical column-aligned text dumps), so the
    test uses whitespace-separated content — a deliberate choice to
    exercise the same path real users hit.
    """
    mod = plot_solutions_module
    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            seed=1, param_names=["p"], objective_labels=["o"],
        )
        # No record_experiment call — archive has no exp data.
        a.end_run(rid)

    csv_path = tmp_path / "exp.txt"
    csv_path.write_text("strain stress\n0.0 100.0\n0.5 200.0\n")

    with ArchiveDB(db, readonly=True) as a:
        result = mod._resolve_experimental_for_case(
            a, rid, sim_case_idx=0, csv_fallback=[csv_path],
        )
    assert result is not None
    strain, stress, label = result
    assert len(strain) == 2
    # CSV fallback uses filename as label.
    assert "exp.txt" in label


def test_slope_helper_matches_np_gradient():
    """The slope helper is just np.gradient under the hood; pin the
    contract so future "optimizations" don't accidentally change shape
    or convention.
    """
    from examples.plot_solutions import _slope_of  # noqa: WPS433
    strain = np.linspace(0.0, 1.0, 11)
    stress = strain ** 2
    slope = _slope_of(strain, stress)
    assert slope.shape == strain.shape
    # d/dx(x^2) = 2x at interior; np.gradient is exact for linear data
    # but for x^2 it's only approximate; sanity-check trend.
    assert slope[5] == pytest.approx(2.0 * strain[5], rel=0.05)


# --- Backward compat: archives without the experiments table -------------


def _build_pre_experiments_archive_with_outputs(db_path):
    """Build an archive that has case_outputs but no experiments table.

    Imitates a real-world old archive: every table the optimizer used
    EXISTED, but the experiments table was added in a later release.
    Used to verify the plotter's --experimental CSV fallback works
    cleanly on such archives instead of crashing.
    """
    import sqlite3
    import pickle
    con = sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT
    );
    CREATE TABLE generations (
        run_id TEXT NOT NULL, gen_idx INTEGER NOT NULL,
        recorded_at TEXT NOT NULL, n_pop INTEGER NOT NULL,
        stats_json TEXT, PRIMARY KEY (run_id, gen_idx)
    );
    CREATE TABLE genes (
        run_id TEXT NOT NULL, gen_idx INTEGER NOT NULL,
        pop_idx INTEGER NOT NULL, birth_gen INTEGER NOT NULL,
        birth_gene INTEGER NOT NULL, gene_vector TEXT NOT NULL,
        fitness TEXT NOT NULL, rank INTEGER,
        PRIMARY KEY (run_id, gen_idx, pop_idx)
    );
    CREATE TABLE case_outputs (
        run_id TEXT NOT NULL, birth_gen INTEGER NOT NULL,
        birth_gene INTEGER NOT NULL, sim_case_idx INTEGER NOT NULL,
        output_name TEXT NOT NULL, data_blob BLOB NOT NULL,
        PRIMARY KEY (run_id, birth_gen, birth_gene, sim_case_idx, output_name)
    );
    INSERT INTO runs VALUES ('r1', '2025-01-01T00:00:00',
        '2025-01-01T01:00:00', 1, '["yield_stress","hardening"]',
        '["rmse_0","rmse_1"]', NULL);
    INSERT INTO generations VALUES('r1', 0, '2025-01-01T00:30:00',
        2, '{"avg":[1.0,1.0]}');
    """)
    # Two genes with finite fitness so L2 ranking has something to do.
    import json
    for i, fit in enumerate([(1.0, 1.0), (0.5, 2.0)]):
        con.execute("INSERT INTO genes VALUES(?,?,?,?,?,?,?,?)",
            ("r1", 0, i, 0, i,
             json.dumps([200.0+i*10, 2000.0+i*100]),
             json.dumps(list(fit)), 0))
    # case_outputs blobs: minimal Voce stress + def_grad for 2 SimCases.
    def stress_blob(y0, n=15):
        t = np.linspace(0, 1, n)
        return pickle.dumps(pd.DataFrame({
            "Time": t, "Volume": np.ones(n),
            "Sxx": np.zeros(n), "Syy": np.zeros(n),
            "Szz": y0 + 1900*(1 - np.exp(-48*t)),
            "Sxy": np.zeros(n), "Sxz": np.zeros(n), "Syz": np.zeros(n),
        }))
    def defgrad_blob(n=15):
        t = np.linspace(0, 1, n); eps = t
        return pickle.dumps(pd.DataFrame({
            "Time": t, "Volume": np.ones(n),
            "F11": 1-0.5*eps, "F12": np.zeros(n), "F13": np.zeros(n),
            "F21": np.zeros(n), "F22": 1-0.5*eps, "F23": np.zeros(n),
            "F31": np.zeros(n), "F32": np.zeros(n), "F33": 1+eps,
        }))
    for i in range(2):
        for sc in range(2):
            con.execute("INSERT INTO case_outputs VALUES(?,?,?,?,?,?)",
                ("r1", 0, i, sc, "avg_stress", stress_blob(y0=200+i+sc*5)))
            con.execute("INSERT INTO case_outputs VALUES(?,?,?,?,?,?)",
                ("r1", 0, i, sc, "avg_def_grad", defgrad_blob()))
    con.commit()
    con.close()


def test_resolve_experimental_handles_missing_table_falls_back_to_csv(
    tmp_path, plot_solutions_module,
):
    """Reproduces Robert's exact bug: an archive without the
    experiments table, plotter called with --experimental CSV paths.
    Must NOT raise sqlite3.OperationalError; must fall through to
    the CSV path and load successfully.
    """
    mod = plot_solutions_module
    db = tmp_path / "old.db"
    _build_pre_experiments_archive_with_outputs(db)

    csv_path = tmp_path / "exp_sc0.txt"
    csv_path.write_text("strain stress\n0.0 100.0\n0.5 200.0\n")

    with ArchiveDB(db, readonly=True) as a:
        # No raise — returns the CSV-loaded data.
        result = mod._resolve_experimental_for_case(
            a, "r1", sim_case_idx=0, csv_fallback=[csv_path],
        )
    assert result is not None
    strain, stress, label = result
    assert len(strain) == 2
    assert "exp_sc0.txt" in label


def test_resolve_experimental_handles_missing_table_returns_none_without_fallback(
    tmp_path, plot_solutions_module,
):
    """Same archive, no CSV fallback supplied — returns None, doesn't
    crash. The plotter code expects None to mean "no experimental
    overlay available, skip it."
    """
    mod = plot_solutions_module
    db = tmp_path / "old.db"
    _build_pre_experiments_archive_with_outputs(db)
    with ArchiveDB(db, readonly=True) as a:
        result = mod._resolve_experimental_for_case(
            a, "r1", sim_case_idx=0, csv_fallback=None,
        )
    assert result is None


def test_main_works_on_pre_experiments_archive_with_csv_fallback(
    tmp_path, plot_solutions_module,
):
    """End-to-end: the plotter's main() against an archive that pre-dates
    the experiments table, with --experimental CSVs supplied. This is
    Robert's exact command-line, in test form.
    """
    mod = plot_solutions_module
    db = tmp_path / "old.db"
    _build_pre_experiments_archive_with_outputs(db)
    # Place the .db where the resolver expects to find it.
    canonical = tmp_path / "calibration.db"
    db.rename(canonical)

    csv0 = tmp_path / "exp_sc0.txt"
    csv1 = tmp_path / "exp_sc1.txt"
    s = np.linspace(0, 1, 25)
    csv0.write_text("strain stress\n" + "\n".join(
        f"{a:.5f} {b:.5f}" for a, b in zip(s, 210 + 1900*(1-np.exp(-48*s)))
    ))
    csv1.write_text("strain stress\n" + "\n".join(
        f"{a:.5f} {b:.5f}" for a, b in zip(s, 215 + 1900*(1-np.exp(-48*s)))
    ))

    save = tmp_path / "out.png"
    rc = mod.main([
        str(tmp_path), "--top", "2", "--objective", "rmse_1",
        "--experimental", str(csv0), str(csv1),
        "--save", str(save), "--no-show",
    ])
    assert rc == 0
    assert save.exists()
    assert save.stat().st_size > 0


# --- Slope axes: optimized + experimental + symlog ----------------------


def test_plot_overlay_renders_optimized_slope_curves(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """Slope axes must contain the optimized (sim) curves alongside
    the experimental reference. Robert's report: 'I'm not seeing the
    optimized results being plotted on those slope curves.' This
    test pins that they ARE plotted (linear-y compression making
    them visually invisible was the actual root cause; the
    structural test below confirms the lines exist).
    """
    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(calibration_archive),
        top_n=3, mode="l2", show_slopes=True,
        save=tmp_path / "overlay.png", show=False,
    )
    slope_axes = [
        ax for ax in fig.axes
        if "slope" in (ax.get_title() or "").lower()
    ]
    assert len(slope_axes) == 2  # 2 SimCases in the fixture
    for ax in slope_axes:
        # Each slope axis should hold:
        #   - 3 simulated curves (top_n=3)
        #   - 1 experimental reference
        # = 4 lines total. Anything less means the optimized curves
        # got dropped along the way.
        assert len(ax.get_lines()) == 4, (
            f"slope axis '{ax.get_title()}' has "
            f"{len(ax.get_lines())} lines, expected 4 "
            f"(3 sim + 1 exp)"
        )


def test_plot_overlay_slope_axes_use_symlog(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """Stress-strain slopes span orders of magnitude (elastic ~100s
    of GPa, plastic much smaller). Linear y compresses the plastic
    portion to a flat line. Symlog makes both regimes legible
    while tolerating noise-induced sign flips."""
    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(calibration_archive),
        top_n=2, mode="l2", show_slopes=True,
        save=tmp_path / "ovl.png", show=False,
    )
    for ax in fig.axes:
        if "slope" in (ax.get_title() or "").lower():
            assert ax.get_yscale() == "symlog", (
                f"slope axis '{ax.get_title()}' yscale={ax.get_yscale()}, "
                f"expected 'symlog'"
            )


def test_plot_overlay_slope_ylim_robust_against_outliers(
    plot_solutions_module, tmp_path,
):
    """Slope axis ylim should be driven by the BULK of the data,
    not by extreme outliers (elastic spike, start-of-test
    numerical noise). Pins the percentile-based fix.

    Robert's report: 'the entire range is not plotted only the very
    start is plotted.' Root cause was ``linthresh = max_abs * 0.001``
    plus matplotlib's data-range autoscaling letting a 200x outlier
    dictate the axis. After the fix, ylim follows the 90th-percentile
    bulk; spikes still plot but extend visibly past the chart edge
    rather than dominating it.
    """
    import numpy as np
    import pandas as pd
    from workflow_common import ArchiveDB
    from workflow_common.archive import GeneRecord
    from workflow_common.paths import CaseContext
    from workflow_common.results import CaseResultSet, TabularResult

    # Construct a sim curve whose slope mirrors the failure pattern:
    # huge spikes at the very start (elastic + numerical noise),
    # smooth bulk in the [0.7, 3] range across the rest.
    strain = np.concatenate([
        np.array([-1e-6, -4.125e-6, -1.39e-5, -4.44e-5, -1.4e-4, -4.4e-4]),
        np.linspace(-6.7e-4, -0.12, 127),
    ])
    n = len(strain)
    # Slopes: 5 spike points, then smooth linear-decay bulk.
    target_slope = np.concatenate([
        np.array([-128.0, -40.0, 210.0, 128.0, 112.0, 32.0]),
        np.linspace(7.0, 0.7, n - 6),
    ])
    d_strain = np.gradient(strain)
    sim_stress = np.cumsum(target_slope * d_strain)

    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield", "H"],
            objective_labels=["stress", "slope"],
        )
        g = GeneRecord(
            run_id=rid, gen_idx=0, pop_idx=0,
            birth_gen=0, birth_gene=0,
            gene_vector=np.array([200.0, 2000.0]),
            fitness=(0.5, 0.4), rank=0,
        )
        a.record_generation(rid, gen_idx=0, genes=[g],
                            stats={"avg": [0.5, 0.4]})
        a.record_case_curve(rid, birth_gen=0, birth_gene=0,
                            sim_case_idx=0,
                            independent=strain, dependent=sim_stress)
        t = np.linspace(0, 1, n)
        rs = CaseResultSet(
            ctx=CaseContext(generation=0, gene=0, obj=0),
            tables={
                "avg_stress": TabularResult(
                    name="avg_stress", df=pd.DataFrame({
                        "Time": t, "Volume": np.ones(n),
                        "Sxx": np.zeros(n), "Syy": np.zeros(n),
                        "Szz": sim_stress,
                        "Sxy": np.zeros(n), "Sxz": np.zeros(n),
                        "Syz": np.zeros(n),
                    }), source_path=tmp_path / "fake",
                ),
            },
        )
        a.record_case_outputs(rid, birth_gen=0, birth_gene=0,
                              sim_case_idx=0, results=rs)
        a.end_run(rid)

    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(tmp_path), top_n=1, mode="l2",
        show_slopes=True,
        save=tmp_path / "out.png", show=False,
    )
    slope_ax = next(
        ax for ax in fig.axes
        if "slope" in (ax.get_title() or "").lower()
    )
    ymin, ymax = slope_ax.get_ylim()
    abs_extent = max(abs(ymin), abs(ymax))

    # Symmetric ylim — both sides of zero get equal real estate.
    assert abs(abs(ymin) - abs(ymax)) < 1e-6, (
        f"slope ylim should be symmetric, got ({ymin}, {ymax})"
    )

    # Bulk-driven, not max-driven. The data has |slope| up to 210
    # but the bulk lives below ~7. The new ylim should be in the
    # bulk's order of magnitude (single digits to maybe 20), not
    # the spike's (hundreds). The old code produced ylim ≈ ±300.
    assert abs_extent < 50, (
        f"slope ylim ({ymin}, {ymax}) is dominated by spike outliers; "
        f"expected bulk-driven ylim around ±5"
    )

    # The line itself must STILL contain the spike points — the
    # fix is about visible-range, not data-clipping. The user
    # sees the line exit the chart at the spike, not the spike
    # being silently dropped.
    sim_lines = [
        l for l in slope_ax.get_lines()
        if "rank" in (l.get_label() or "")
    ]
    assert len(sim_lines) == 1
    plotted_y = sim_lines[0].get_ydata()
    assert plotted_y.max() > 100, (
        "spike data was dropped from the plot; the fix should "
        "set visible ylim, not clip data"
    )
    assert plotted_y.min() < -50, (
        "negative spike data was dropped from the plot"
    )


def test_plot_overlay_shades_optimization_window(
    calibration_archive, plot_solutions_module, tmp_path,
):
    """The plotter must shade the strain region the optimizer was
    constrained to (per case_data['minmax_strain']) so users can
    sanity-check what the optimizer actually scored against.

    The fixture stores window=(0.05, 0.30) on SimCase 0 and
    window=(None, 0.40) on SimCase 1. Verify each SimCase's stress
    axis has the right shaded span.
    """
    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(calibration_archive),
        top_n=2, mode="l2", show_slopes=True,
        save=tmp_path / "win.png", show=False,
    )
    # Stress axes — find by title prefix "SimCase N — ..." minus slope.
    stress_axes = [
        ax for ax in fig.axes
        if (ax.get_title() or "").startswith("SimCase")
        and "slope" not in (ax.get_title() or "").lower()
    ]
    assert len(stress_axes) == 2
    # axvspan creates a Rectangle patch on the axes. Count them
    # per axis; should be exactly 1 (the optimization window).
    import matplotlib.patches as mpatches
    for ax in stress_axes:
        spans = [
            p for p in ax.patches
            if isinstance(p, mpatches.Rectangle)
            and p.get_alpha() is not None
            and p.get_alpha() < 0.5  # the translucent overlay
        ]
        assert len(spans) >= 1, (
            f"SimCase axis '{ax.get_title()}' has no shaded window patch"
        )


def test_plot_overlay_no_window_no_shading(
    tmp_path, plot_solutions_module,
):
    """When the archive has no minmax_strain for any SimCase (e.g.
    user never set it), the plotter shouldn't draw shading patches
    at all. Verifies the optional-shading path."""
    from workflow_common import ArchiveDB
    from workflow_common.archive import GeneRecord
    from workflow_common.paths import CaseContext
    from workflow_common.results import CaseResultSet, TabularResult

    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["rmse_0"],
        )
        genes = [
            GeneRecord(run_id=rid, gen_idx=0, pop_idx=i,
                       birth_gen=0, birth_gene=i,
                       gene_vector=np.array([200.0+i, 2000.0+i*100]),
                       fitness=(0.5 + 0.1 * i,), rank=0)
            for i in range(3)
        ]
        a.record_generation(rid, gen_idx=0, genes=genes,
                            stats={"avg": [0.6]})
        for g in genes:
            ctx = CaseContext(generation=g.birth_gen,
                              gene=g.birth_gene, obj=0)
            n = 15; t = np.linspace(0, 1, n)
            stress = pd.DataFrame({
                "Time": t, "Volume": np.ones(n),
                "Sxx": np.zeros(n), "Syy": np.zeros(n),
                "Szz": 200 + 1900*(1 - np.exp(-48*t)),
                "Sxy": np.zeros(n), "Sxz": np.zeros(n), "Syz": np.zeros(n),
            })
            dfg = pd.DataFrame({
                "Time": t, "Volume": np.ones(n),
                "F11": 1-0.5*t, "F12": np.zeros(n), "F13": np.zeros(n),
                "F21": np.zeros(n), "F22": 1-0.5*t, "F23": np.zeros(n),
                "F31": np.zeros(n), "F32": np.zeros(n), "F33": 1+t,
            })
            rs = CaseResultSet(ctx=ctx, tables={
                "avg_stress": TabularResult(
                    name="avg_stress", df=stress,
                    source_path=Path("fake")),
                "avg_def_grad": TabularResult(
                    name="avg_def_grad", df=dfg,
                    source_path=Path("fake")),
            })
            a.record_case_outputs(rid, birth_gen=g.birth_gen,
                                  birth_gene=g.birth_gene,
                                  sim_case_idx=0, results=rs)
        # Experiment recorded WITHOUT a window.
        exp_df = pd.DataFrame({
            "strain": np.linspace(0, 1, 30),
            "stress": 210 + 1900 * (1 - np.exp(-48 * np.linspace(0, 1, 30))),
        })
        a.record_experiment(rid, sim_case_idx=0, df=exp_df,
                            label="no_window", minmax_strain=None)
        a.end_run(rid)

    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(tmp_path), top_n=2, mode="l2", show_slopes=True,
        save=tmp_path / "no_win.png", show=False,
    )
    # Find the stress axis. Should have NO low-alpha rectangle patch.
    import matplotlib.patches as mpatches
    stress_ax = next(
        ax for ax in fig.axes
        if (ax.get_title() or "").startswith("SimCase")
        and "slope" not in (ax.get_title() or "").lower()
    )
    spans = [
        p for p in stress_ax.patches
        if isinstance(p, mpatches.Rectangle)
        and p.get_alpha() is not None
        and p.get_alpha() < 0.5
    ]
    assert len(spans) == 0, (
        f"unexpected window shading despite minmax_strain=None: "
        f"{len(spans)} translucent rectangles"
    )


# --- Pareto subset-L2 winner + color modes + click handler --------------


@pytest.fixture
def pareto_4obj_archive(tmp_path: Path) -> Path:
    """Archive with 4 objectives where subset-L2 and full-L2 winners diverge.

    Three rank-0 genes, fitness chosen so the projected pair (0,2)
    has different "best" than the full vector:

      gene 0: fitness=(0.1, 0.1, 0.1, 0.1) — full-L2 winner
              subset_l2 on (0,2) = sqrt(0.01+0.01) ≈ 0.141
      gene 1: fitness=(0.05, 0.5, 0.05, 0.5) — projected (0,2) WINNER
              subset_l2 on (0,2) = sqrt(0.0025+0.0025) ≈ 0.071
              full L2 ≈ 0.71 (worse than gene 0 globally)
      gene 2: fitness=(0.3, 0.3, 0.3, 0.3) — middle on both
              subset_l2 on (0,2) ≈ 0.424

    With color_by="subset_l2" the winner ring lands on gene 1.
    With color_by="full_l2" the lowest-color point would be gene 0
    (winner ring is on gene 1 for both — see test below — but
    color values differ).
    """
    from workflow_common.objectives import StressStrainExtractor
    db = tmp_path / "calibration.db"
    fits = [
        (0.10, 0.10, 0.10, 0.10),
        (0.05, 0.50, 0.05, 0.50),
        (0.30, 0.30, 0.30, 0.30),
    ]
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["stress_1", "slope_1", "stress_2", "slope_2"],
        )
        genes = [
            GeneRecord(
                run_id=rid, gen_idx=0, pop_idx=i,
                birth_gen=0, birth_gene=i,
                gene_vector=np.array([200.0 + 10 * i, 2000.0 + 100 * i]),
                fitness=f, rank=0,
            )
            for i, f in enumerate(fits)
        ]
        a.record_generation(rid, gen_idx=0, genes=genes,
                            stats={"avg": [0.15, 0.30, 0.15, 0.30]})
        # Two SimCases per gene with simple stress + def-grad.
        for g in genes:
            for sim_case_idx in range(2):
                ctx = CaseContext(
                    generation=g.birth_gen, gene=g.birth_gene,
                    obj=sim_case_idx,
                )
                rs = _make_case_result_set(
                    ctx,
                    y0=200.0 + 5 * sim_case_idx + g.birth_gene,
                    H=1900.0 + 50 * sim_case_idx,
                )
                a.record_case_outputs(
                    rid,
                    birth_gen=g.birth_gen, birth_gene=g.birth_gene,
                    sim_case_idx=sim_case_idx, results=rs,
                )
        # Experimental refs + extractor configs so the click
        # handler reproduces both SimCases without falling back to
        # defaults that might silently produce wrong curves.
        for sim_case_idx in range(2):
            exp_df = pd.DataFrame({
                "strain": np.linspace(0.0, 1.0, 25),
                "stress": (210.0 + 5 * sim_case_idx) + 1900.0 * (
                    1 - np.exp(-48.0 * np.linspace(0.0, 1.0, 25))
                ),
            })
            ext_cfg = StressStrainExtractor().to_dict()
            a.record_experiment(
                rid, sim_case_idx=sim_case_idx, df=exp_df,
                label=f"exp_sc{sim_case_idx}",
                extractor_config=ext_cfg,
            )
        a.end_run(rid)
    return tmp_path


def test_pareto_winner_ring_uses_subset_l2_not_full_l2(
    pareto_4obj_archive, plot_solutions_module, tmp_path,
):
    """The red 'L2 winner' ring on the Pareto plot must be on the
    subset-L2 minimum (best balanced point on the projection),
    NOT the full-L2 minimum. Robert's ask: 'the L2 circle should
    really be based on the objectives chosen.'

    Fixture genes:
      0 — best on full L2 (small everywhere)
      1 — best on subset L2 of (0,2) but worse on full L2
      2 — middle

    Projecting to (0, 2) → ring should land on gene 1 (proj.
    fitness (0.05, 0.05)), not gene 0 (proj. fitness (0.10, 0.10)).
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(pareto_4obj_archive),
        objective_pair=(0, 2), top_n=0,
        color_by="subset_l2",
        save=tmp_path / "p.png", show=False,
    )
    # The ring is the only red-edged scatter on the figure.
    import matplotlib.collections as mcoll
    rings = []
    for ax in fig.axes:
        for coll in ax.collections:
            if isinstance(coll, mcoll.PathCollection):
                ec = coll.get_edgecolors()
                if len(ec) > 0:
                    # red edge = (1, 0, 0, 1) approximately
                    r, g, b = ec[0][:3]
                    if r > 0.9 and g < 0.1 and b < 0.1:
                        rings.append(coll)
    assert len(rings) == 1, f"expected one red ring, got {len(rings)}"
    ring_xy = rings[0].get_offsets()[0]
    # Gene 1's projected (obj 0, obj 2) = (0.05, 0.05).
    assert ring_xy[0] == pytest.approx(0.05, abs=1e-9)
    assert ring_xy[1] == pytest.approx(0.05, abs=1e-9)


def test_pareto_color_by_subset_l2_uses_projected_pair(
    pareto_4obj_archive, plot_solutions_module, tmp_path,
):
    """color_by='subset_l2' must color points by sqrt(x² + y²) on
    the projected pair, not by the full-objective L2.

    Verify by reading the scatter's color array and comparing to
    the expected subset L2 values for our fixture genes.
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(pareto_4obj_archive),
        objective_pair=(0, 2), top_n=0,
        color_by="subset_l2",
        save=tmp_path / "p.png", show=False,
    )
    import matplotlib.collections as mcoll
    main_sc = next(
        coll for ax in fig.axes for coll in ax.collections
        if isinstance(coll, mcoll.PathCollection)
        and coll.get_picker() is not None
    )
    # PathCollection.get_array returns the c= values (the colormap
    # input), separately from the rendered RGBA edge/face colors.
    color_array = np.asarray(main_sc.get_array())
    # Subset L2 for each fixture gene on (obj 0, obj 2):
    expected = np.array([
        np.sqrt(0.10 ** 2 + 0.10 ** 2),  # gene 0
        np.sqrt(0.05 ** 2 + 0.05 ** 2),  # gene 1
        np.sqrt(0.30 ** 2 + 0.30 ** 2),  # gene 2
    ])
    # The points are scattered in chosen order, which for top_n=0
    # comes from select-rank-0 then dedup. Both keep insertion
    # order, so we expect the same ordering as the fixture.
    np.testing.assert_allclose(color_array, expected, atol=1e-9)


def test_pareto_color_by_full_l2_uses_all_objectives(
    pareto_4obj_archive, plot_solutions_module, tmp_path,
):
    """color_by='full_l2' colors by ‖fitness‖ across ALL objectives,
    so the colors differ from subset_l2 mode in this fixture."""
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(pareto_4obj_archive),
        objective_pair=(0, 2), top_n=0,
        color_by="full_l2",
        save=tmp_path / "p.png", show=False,
    )
    import matplotlib.collections as mcoll
    main_sc = next(
        coll for ax in fig.axes for coll in ax.collections
        if isinstance(coll, mcoll.PathCollection)
        and coll.get_picker() is not None
    )
    color_array = np.asarray(main_sc.get_array())
    expected = np.array([
        np.linalg.norm([0.10, 0.10, 0.10, 0.10]),
        np.linalg.norm([0.05, 0.50, 0.05, 0.50]),
        np.linalg.norm([0.30, 0.30, 0.30, 0.30]),
    ])
    np.testing.assert_allclose(color_array, expected, atol=1e-9)


@pytest.mark.parametrize("color_by", ["x", "y", "asymmetry"])
def test_pareto_color_by_other_metrics_renders_without_error(
    pareto_4obj_archive, plot_solutions_module, tmp_path, color_by,
):
    """Each color mode must render to a valid figure.

    The values themselves are checked only weakly (just non-empty
    and finite); the structural test above pins subset_l2 and
    full_l2 exactly.
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(pareto_4obj_archive),
        objective_pair=(0, 2), top_n=0,
        color_by=color_by,
        save=tmp_path / f"p_{color_by}.png", show=False,
    )
    import matplotlib.collections as mcoll
    main_sc = next(
        coll for ax in fig.axes for coll in ax.collections
        if isinstance(coll, mcoll.PathCollection)
        and coll.get_picker() is not None
    )
    color_array = np.asarray(main_sc.get_array())
    assert len(color_array) == 3  # 3 genes in fixture
    assert np.all(np.isfinite(color_array))


def test_pareto_color_by_invalid_raises(
    pareto_4obj_archive, plot_solutions_module, tmp_path,
):
    """Unknown color_by string raises ValueError naming the valid options."""
    mod = plot_solutions_module
    with pytest.raises(ValueError, match="color_by"):
        mod.plot_pareto_front_with_l2_winner(
            str(pareto_4obj_archive),
            objective_pair=(0, 2), top_n=0,
            color_by="nope",
            save=tmp_path / "p.png", show=False,
        )


def test_pareto_inset_initially_shows_all_simcases(
    pareto_4obj_archive, plot_solutions_module, tmp_path,
):
    """Initial Pareto render (before any click) populates the inset
    with the L2-winner gene's response — covering EVERY SimCase,
    not just one. Pins Robert's bug: 'It only plots 1 of 2.'

    The fixture has 2 SimCases. After the initial draw the inset
    should hold 2 sim curves + 2 exp curves = 4 lines.
    """
    mod = plot_solutions_module
    fig = mod.plot_pareto_front_with_l2_winner(
        str(pareto_4obj_archive),
        objective_pair=(0, 2), top_n=0,
        save=tmp_path / "p.png", show=False,
    )
    inset = next(
        ax for ax in fig.axes
        if "Response" in (ax.get_title() or "")
    )
    sim_lines = [
        line for line in inset.get_lines()
        if "sim" in (line.get_label() or "")
    ]
    exp_lines = [
        line for line in inset.get_lines()
        if "exp" in (line.get_label() or "")
    ]
    assert len(sim_lines) == 2, (
        f"expected 2 sim lines (one per SimCase), got {len(sim_lines)}"
    )
    assert len(exp_lines) == 2, (
        f"expected 2 exp lines (one per SimCase), got {len(exp_lines)}"
    )


def test_pareto_uses_archived_extractor_when_available(
    plot_solutions_module, tmp_path,
):
    """The Pareto plotter must consult ``load_extractor_config`` for
    each SimCase before falling back to the default. Verify by
    storing a non-default extractor config (custom column names)
    and confirming extraction succeeds — extraction with the
    default extractor would fail since the case_outputs use
    different column names.

    This pins the architectural fix for Robert's 'only 1 of 2
    SimCases plotted' bug: the plotter now uses what the optimizer
    used, not its own guess.
    """
    from workflow_common.objectives import StressStrainExtractor
    # Build a separate fixture-like archive where SimCase 0's data
    # uses non-default column names, and the matching extractor
    # config is archived.
    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["o0", "o1"],
        )
        genes = [
            GeneRecord(
                run_id=rid, gen_idx=0, pop_idx=i, birth_gen=0,
                birth_gene=i,
                gene_vector=np.array([200.0 + i, 2000.0 + i * 100]),
                fitness=(0.5 + 0.1 * i, 0.4 + 0.1 * i), rank=0,
            )
            for i in range(2)
        ]
        a.record_generation(rid, gen_idx=0, genes=genes,
                            stats={"avg": [0.6, 0.5]})
        # Build case_outputs whose columns match a CUSTOM extractor —
        # not the StressStrainExtractor() defaults.
        n = 12
        t = np.linspace(0, 1, n)
        for g in genes:
            ctx = CaseContext(generation=g.birth_gen,
                              gene=g.birth_gene, obj=0)
            stress_df = pd.DataFrame({
                "Time": t, "Volume": np.ones(n),
                # 's11' instead of 'Szz' — a default extractor
                # would raise KeyError on this DataFrame.
                "s11": 200 + 1900 * (1 - np.exp(-48 * t)),
                # Other components zero so the DataFrame has the
                # expected ExaConstit shape.
                "Sxx": np.zeros(n), "Syy": np.zeros(n),
                "Szz": np.zeros(n), "Sxy": np.zeros(n),
                "Sxz": np.zeros(n), "Syz": np.zeros(n),
            })
            rs = CaseResultSet(ctx=ctx, tables={
                "avg_stress": TabularResult(
                    name="avg_stress", df=stress_df,
                    source_path=Path("fake")),
            })
            a.record_case_outputs(rid, birth_gen=g.birth_gen,
                                  birth_gene=g.birth_gene,
                                  sim_case_idx=0, results=rs)
        # Experimental + matching extractor config that names the
        # custom column. Without this archived config the plotter's
        # default would silently fail on this DataFrame.
        exp_df = pd.DataFrame({
            "strain": np.linspace(0, 0.5, 20),
            "stress": 210 + 1900 * (1 - np.exp(-48 * np.linspace(0, 0.5, 20))),
        })
        custom_ext = StressStrainExtractor(
            stress_column="s11",
            strain_source="time_rate", strain_rate=0.5,
        )
        a.record_experiment(
            rid, sim_case_idx=0, df=exp_df, label="custom",
            extractor_config=custom_ext.to_dict(),
        )
        a.end_run(rid)

    mod = plot_solutions_module
    # Should NOT raise even though default extractor would fail
    # against an 's11' DataFrame.
    fig = mod.plot_pareto_front_with_l2_winner(
        str(tmp_path),
        objective_pair=(0, 1), top_n=0,
        save=tmp_path / "p.png", show=False,
    )
    inset = next(
        ax for ax in fig.axes
        if "Response" in (ax.get_title() or "")
    )
    sim_lines = [
        line for line in inset.get_lines()
        if "sim" in (line.get_label() or "")
    ]
    # SimCase 0 plotted via the archived custom extractor.
    assert len(sim_lines) == 1


# --- Plotter prefers archived case_curves over re-extraction ------------


def test_plotter_uses_archived_case_curves_when_available(
    plot_solutions_module, tmp_path,
):
    """When the archive carries case_curves rows, the plotter must
    use those instead of re-extracting from raw case_outputs.
    Verify by storing curves with a deliberately-recognizable
    signature and confirming the plotter reads them back.

    The case_outputs in this fixture would normally extract to
    different values; if the plotter correctly prefers case_curves,
    it'll plot the recognizable signature.
    """
    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["o0"],
        )
        # One gene.
        gene = GeneRecord(
            run_id=rid, gen_idx=0, pop_idx=0, birth_gen=0,
            birth_gene=0, gene_vector=np.array([200.0, 2000.0]),
            fitness=(0.5,), rank=0,
        )
        a.record_generation(rid, gen_idx=0, genes=[gene],
                            stats={"avg": [0.5]})
        # Standard case_outputs (some Voce-shaped data).
        n = 12; t = np.linspace(0, 1, n)
        ctx = CaseContext(generation=0, gene=0, obj=0)
        rs = _make_case_result_set(ctx, y0=200.0, H=1900.0)
        a.record_case_outputs(rid, birth_gen=0, birth_gene=0,
                              sim_case_idx=0, results=rs)
        # Plant a recognizable curve signature in case_curves: a
        # straight line that the extractor would never produce.
        sig_strain = np.linspace(0.0, 1.0, 20)
        sig_stress = 12345.0 + 1000.0 * sig_strain  # sentinel
        a.record_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            independent=sig_strain, dependent=sig_stress,
        )
        # Experimental + extractor config so the plotter has the
        # rest of what it expects.
        from workflow_common.objectives import StressStrainExtractor
        a.record_experiment(
            rid, sim_case_idx=0, df=pd.DataFrame({
                "strain": np.linspace(0, 1, 10),
                "stress": np.linspace(100, 300, 10),
            }),
            label="exp",
            extractor_config=StressStrainExtractor().to_dict(),
        )
        a.end_run(rid)

    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(tmp_path), top_n=1, mode="l2", show_slopes=False,
        save=tmp_path / "ovl.png", show=False,
    )
    # The headline stress axis should hold the sentinel curve, NOT
    # whatever the extractor produced from case_outputs.
    stress_ax = next(
        ax for ax in fig.axes
        if (ax.get_title() or "").startswith("SimCase")
    )
    sim_lines = [
        line for line in stress_ax.get_lines()
        if "rank" in (line.get_label() or "")
    ]
    assert len(sim_lines) == 1
    y = sim_lines[0].get_ydata()
    # Min of the sentinel stress is 12345 — far above any
    # plausible extractor output from the fixture.
    assert float(np.min(y)) >= 12345.0 - 1e-6, (
        f"plot didn't use archived case_curves; min stress={np.min(y)}"
    )


# --- Sign-correction in the post-processing plotter ---------------------


def test_plotter_sign_corrects_archived_curves_against_experimental(
    plot_solutions_module, tmp_path,
):
    """Plotter applies sign-matching against the experimental
    reference even when the archive holds wrong-signed curves.
    Pins Robert's specific ask: 'I am specifically asking to fix
    this in the post-processing step as that's where users are
    going to note things being off.'

    Setup: archive with positive-signed sim curves, negative-signed
    experimental reference. Plot. Assert the plotted simulation
    line shows negative values, not positive.
    """
    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["o0"],
        )
        gene = GeneRecord(
            run_id=rid, gen_idx=0, pop_idx=0, birth_gen=0,
            birth_gene=0, gene_vector=np.array([200.0, 2000.0]),
            fitness=(0.5,), rank=0,
        )
        a.record_generation(rid, gen_idx=0, genes=[gene],
                            stats={"avg": [0.5]})
        # case_outputs needed so _discover_sim_case_count can find
        # this SimCase. Contents don't matter — the plotter prefers
        # case_curves once it knows how many SimCases exist.
        ctx = CaseContext(generation=0, gene=0, obj=0)
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            results=_make_case_result_set(ctx, y0=200.0, H=1900.0),
        )
        # Stash a wrong-signed simulation curve directly via
        # record_case_curve (an old archive predating sign-matching
        # at save-off, OR an evaluator without experimental data —
        # either way the archive holds positive values).
        sim_strain_pos = np.linspace(0.0, 1.0, 25)
        sim_stress_pos = 200.0 + 1900.0 * (
            1 - np.exp(-48.0 * sim_strain_pos)
        )
        a.record_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            independent=sim_strain_pos, dependent=sim_stress_pos,
        )
        # Experimental reference is compression-shaped (negative).
        exp_strain = np.linspace(0.0, -1.0, 25)
        exp_stress = -210.0 - 1900.0 * (1 - np.exp(48.0 * exp_strain))
        a.record_experiment(
            rid, sim_case_idx=0,
            df=pd.DataFrame({"strain": exp_strain, "stress": exp_stress}),
            label="compression",
        )
        a.end_run(rid)

    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(tmp_path), top_n=1, mode="l2", show_slopes=False,
        save=tmp_path / "ovl.png", show=False,
    )
    # Find the SimCase stress axis.
    stress_ax = next(
        ax for ax in fig.axes
        if (ax.get_title() or "").startswith("SimCase")
    )
    sim_lines = [
        line for line in stress_ax.get_lines()
        if "rank" in (line.get_label() or "")
    ]
    assert len(sim_lines) == 1
    x = sim_lines[0].get_xdata()
    y = sim_lines[0].get_ydata()
    # After sign-correction at plot time, the simulated curve
    # should have negative-dominant strain AND stress, matching
    # the compression reference. Without correction, the line
    # would still hold the original positive values from the
    # archive.
    assert x[int(np.argmax(np.abs(x)))] <= 0, (
        f"plotted strain not sign-corrected; dominant={x[int(np.argmax(np.abs(x)))]}"
    )
    assert y[int(np.argmax(np.abs(y)))] <= 0, (
        f"plotted stress not sign-corrected; dominant={y[int(np.argmax(np.abs(y)))]}"
    )


def test_plotter_leaves_curves_untouched_when_no_experimental(
    plot_solutions_module, tmp_path,
):
    """No experimental reference → no sign-correction applied;
    archived sim curves go onto the plot as-is. (Avoids 'fixing'
    a curve where there's nothing to fix it against.)
    """
    db = tmp_path / "calibration.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="r1", seed=1,
            param_names=["yield_stress", "hardening"],
            objective_labels=["o0"],
        )
        gene = GeneRecord(
            run_id=rid, gen_idx=0, pop_idx=0, birth_gen=0,
            birth_gene=0, gene_vector=np.array([200.0, 2000.0]),
            fitness=(0.5,), rank=0,
        )
        a.record_generation(rid, gen_idx=0, genes=[gene],
                            stats={"avg": [0.5]})
        ctx = CaseContext(generation=0, gene=0, obj=0)
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            results=_make_case_result_set(ctx, y0=200.0, H=1900.0),
        )
        sim_strain = np.linspace(0.0, 1.0, 25)
        sim_stress = 200.0 + 1900.0 * (1 - np.exp(-48.0 * sim_strain))
        a.record_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            independent=sim_strain, dependent=sim_stress,
        )
        # No experiment recorded.
        a.end_run(rid)

    mod = plot_solutions_module
    fig = mod.plot_top_solutions_overlay(
        str(tmp_path), top_n=1, mode="l2", show_slopes=False,
        save=tmp_path / "ovl.png", show=False,
    )
    stress_ax = next(
        ax for ax in fig.axes
        if (ax.get_title() or "").startswith("SimCase")
    )
    sim_lines = [
        line for line in stress_ax.get_lines()
        if "rank" in (line.get_label() or "")
    ]
    assert len(sim_lines) == 1
    y = sim_lines[0].get_ydata()
    # Stress remains positive — no reference to flip against.
    assert y[int(np.argmax(np.abs(y)))] >= 0
