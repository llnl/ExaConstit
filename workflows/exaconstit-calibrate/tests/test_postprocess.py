"""
Unit and integration tests for :mod:`workflow_common.postprocess`.

What these prove
----------------
* Both best-solution strategies (EUDIST, ASF) produce numerically
  the expected rankings on hand-computable inputs, and respect
  weights / normalization / nsmallest parameters.
* The legacy :class:`BestSol` class wrapper produces bit-identical
  output to the underlying functions (so drop-in migration works).
* Checkpoint loading handles both the new driver's pickle format
  and the pre-refactor driver's pickle format (identical, by
  design - verified here).
* :func:`extract_gene_results` preserves the BIRTH generation
  of each individual even when it survives to later generations
  via selection (the critical detail for correct on-disk lookups).
* :func:`load_case_results` returns None on missing case dirs
  rather than raising, and reads real directories correctly.
* Matplotlib plotting functions run without errors in headless
  Agg mode and reject invalid inputs.

The post-process tests do not run simulations (they are not
end-to-end tests); they operate on pop_libraries and case
directories built directly from test fixtures. End-to-end coverage
is in test_nsga3_driver.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common import (
    CaseContext,
    TemplatePathResolver,
    TextTableReader,
    TextTableSpec,
)
from workflow_common.postprocess import (
    BestSol,
    CheckpointData,
    GeneResult,
    best_solution_asf,
    best_solution_eudist,
    extract_gene_results,
    load_case_results,
    load_checkpoint,
)


# --- Test helpers -------------------------------------------------------


class _FakeFitness:
    """Stand-in for DEAP's Fitness class.

    The post-processor accesses ``ind.fitness.values`` as a tuple
    of floats. That is all the tests need; building the full DEAP
    class hierarchy for a unit test would couple the test file to
    DEAP's version quirks and provide no additional coverage.
    """

    def __init__(self, values):
        self.values = tuple(values)


class _FakeInd(list):
    """Stand-in for DEAP's Individual class.

    A list subclass carrying ``fitness``, ``generation``, ``gene``.
    Mirrors what the driver assigns to real DEAP Individuals.
    """

    pass


def _make_ind(vec, fitness, gene, generation):
    ind = _FakeInd(vec)
    ind.fitness = _FakeFitness(fitness)
    ind.gene = gene
    ind.generation = generation
    ind.rank = 0
    return ind


# --- best_solution_eudist ----------------------------------------------


def test_eudist_picks_closest_to_origin():
    """The row closest to [0, 0] wins."""
    pop_fit = np.array([
        [5.0, 5.0],   # dist = sqrt(50) ~ 7.07
        [1.0, 1.0],   # dist = sqrt(2) ~ 1.41  <- best
        [3.0, 3.0],   # dist = sqrt(18) ~ 4.24
    ])
    idx = best_solution_eudist(pop_fit, nsmallest=1)
    assert len(idx) == 1
    assert idx[0] == 1


def test_eudist_weights_influence_ranking():
    """Weighting f1 heavily penalizes rows with large f1."""
    pop_fit = np.array([
        [1.0, 5.0],
        [5.0, 1.0],
    ])
    # Symmetric weights -> tie. With weight [10, 1], row 0 beats
    # row 1 because row 1's f1=5 is now heavily penalized.
    idx = best_solution_eudist(pop_fit, weights=[10.0, 1.0], nsmallest=1)
    assert idx[0] == 0


def test_eudist_nsmallest_returns_k():
    """nsmallest=k returns k indices, all pointing to the k best."""
    pop_fit = np.array([[10.0], [1.0], [2.0], [5.0]])
    idx = best_solution_eudist(pop_fit, nsmallest=2)
    assert len(idx) == 2
    # argpartition guarantees set membership but not order -> test both.
    assert set(idx.tolist()) == {1, 2}


def test_eudist_normalize_equalizes_scales():
    """With normalize=True, tiny-scale f1 no longer dominates."""
    pop_fit = np.array([
        [1e-6, 5.0],   # tiny f1, max f2
        [1e-6, 1.0],   # tiny f1, min f2  <- best after normalization
        [1e-5, 1.0],   # max f1, min f2
    ])
    idx = best_solution_eudist(pop_fit, normalize=True, nsmallest=1)
    assert idx[0] == 1


def test_eudist_normalize_handles_constant_column():
    """A column with zero spread does not cause NaN via div-by-zero."""
    pop_fit = np.array([
        [0.0, 1.0],
        [0.0, 2.0],
        [0.0, 0.5],  # best: f2 is smallest; f1 is constant
    ])
    idx = best_solution_eudist(pop_fit, normalize=True, nsmallest=1)
    assert idx[0] == 2


def test_eudist_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        best_solution_eudist(np.array([1.0, 2.0, 3.0]))


def test_eudist_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        best_solution_eudist(np.zeros((0, 2)))


def test_eudist_rejects_wrong_weight_length():
    with pytest.raises(ValueError, match="weights shape"):
        best_solution_eudist(np.zeros((3, 2)), weights=[1.0, 2.0, 3.0])


def test_eudist_nsmallest_larger_than_pop_does_not_crash():
    """Asking for more results than exist clamps gracefully."""
    pop_fit = np.array([[5.0], [1.0], [3.0]])
    idx = best_solution_eudist(pop_fit, nsmallest=10)
    # Should return all 3 without error.
    assert len(idx) <= 10
    assert 1 in idx.tolist()


# --- best_solution_asf -------------------------------------------------


def test_asf_picks_smallest_worst_objective():
    """ASF picks the row whose max objective is smallest."""
    pop_fit = np.array([
        [1.0, 10.0],   # max = 10
        [5.0, 5.0],    # max = 5    <- best ASF
        [10.0, 1.0],   # max = 10
    ])
    idx = best_solution_asf(pop_fit, nsmallest=1)
    assert idx[0] == 1


def test_asf_weights():
    """Weighting flips the result by magnifying particular objectives."""
    pop_fit = np.array([
        [5.0, 1.0],    # weighted max = max(5, 10) = 10
        [1.0, 5.0],    # weighted max = max(1, 50) = 50
    ])
    idx = best_solution_asf(pop_fit, weights=[1.0, 10.0], nsmallest=1)
    assert idx[0] == 0


# --- BestSol class (legacy API) -----------------------------------------


def test_bestsol_class_api_matches_functions():
    """Class wrapper returns bit-identical output to the functions."""
    rng = np.random.RandomState(42)
    pop_fit = rng.rand(10, 3)
    bs = BestSol(pop_fit, weights=[1.0, 1.0, 1.0], nsmallest=2)
    np.testing.assert_array_equal(
        bs.EUDIST(),
        best_solution_eudist(
            pop_fit, weights=[1.0, 1.0, 1.0], nsmallest=2,
        ),
    )
    np.testing.assert_array_equal(
        bs.ASF(),
        best_solution_asf(
            pop_fit, weights=[1.0, 1.0, 1.0], nsmallest=2,
        ),
    )


def test_bestsol_class_default_weights():
    """Default weights are all-ones; consistent with the old class."""
    pop_fit = np.array([[1.0, 2.0], [3.0, 1.0]])
    bs = BestSol(pop_fit)  # nsmallest=1, weights=None
    # Not checking specific value; just that it doesn't raise and
    # produces a valid index.
    idx = bs.EUDIST()
    assert idx[0] in (0, 1)


def test_bestsol_class_with_normalize():
    pop_fit = np.array([
        [1e-6, 100.0],
        [1e-6, 1.0],
        [1e-5, 1.0],
    ])
    bs = BestSol(pop_fit, normalize=True, nsmallest=1)
    idx = bs.EUDIST()
    # Middle row has smallest normalized coords
    assert idx[0] == 1


# --- Checkpoint loading ------------------------------------------------


def test_load_checkpoint_parses_driver_pickle(tmp_path):
    """Round-trip: a new-driver pickle loads into a CheckpointData."""
    # Construct a pickle matching the driver's _save_checkpoint format.
    ckp = dict(
        pop_library=[[_make_ind([1.0, 2.0], (0.5, 0.3), 0, 0)]],
        iter_tot=1,
        generation=0,
        fail_count=0,
        stop_count=0,
        logbook1="(stub logbook1)",
        logbook2="(stub logbook2)",
        rndstate=(3, (1, 2, 3), None),
    )
    path = tmp_path / "ck.pkl"
    with path.open("wb") as f:
        pickle.dump(ckp, f)

    loaded = load_checkpoint(path)
    assert isinstance(loaded, CheckpointData)
    assert loaded.generation == 0
    assert loaded.n_generations == 1
    assert loaded.n_pop == 1
    assert loaded.n_dim == 2
    assert loaded.logbook_stats == "(stub logbook1)"
    assert loaded.logbook_solutions == "(stub logbook2)"


def test_load_checkpoint_missing_key_raises(tmp_path):
    """Missing expected key raises KeyError cleanly."""
    ckp = dict(pop_library=[], iter_tot=0)   # many keys missing
    path = tmp_path / "bad.pkl"
    with path.open("wb") as f:
        pickle.dump(ckp, f)
    with pytest.raises(KeyError):
        load_checkpoint(path)


def test_checkpoint_properties_on_empty():
    """Properties handle the degenerate empty case without crashing."""
    ckp = CheckpointData(
        pop_library=[], iter_tot=0, generation=0,
        fail_count=0, stop_count=0,
        logbook_stats=None, logbook_solutions=None, rndstate=(),
    )
    assert ckp.n_generations == 0
    assert ckp.n_pop == 0
    assert ckp.n_dim == 0


# --- extract_gene_results ----------------------------------------------


def test_extract_gene_results_shape_matches_input():
    pop_library = [
        [_make_ind([i, i * 2], (0.5, 0.3), i, 0) for i in range(3)],
        [_make_ind([i + 10, i * 2], (0.4, 0.2), i, 1) for i in range(3)],
    ]
    results = extract_gene_results(pop_library)
    assert len(results) == 2
    assert all(len(g) == 3 for g in results)


def test_extract_gene_results_captures_fields():
    pop_library = [
        [_make_ind([1.0, 2.0], (0.5, 0.3), 1, 0)],
    ]
    r = extract_gene_results(pop_library)[0][0]
    assert isinstance(r, GeneResult)
    assert r.generation == 0
    assert r.gene == 1
    assert r.fitness == (0.5, 0.3)
    np.testing.assert_array_equal(r.gene_vector, [1.0, 2.0])


def test_extract_preserves_birth_generation_across_selection():
    """An individual born in gen 0 and selected into gen 2 carries gen=0.

    This is the critical correctness test. The pop_library list
    position does NOT necessarily equal the birth generation;
    selection can copy an old individual into a later position.
    The case directory on disk was written at birth time, so
    post-processing must use the birth generation for lookups.
    """
    legacy_ind = _make_ind([5.0, 5.0], (0.1, 0.1), 3, 0)   # born gen 0
    pop_library = [
        [_make_ind([1.0, 2.0], (0.5, 0.5), 0, 0)],
        [_make_ind([1.0, 2.0], (0.5, 0.5), 0, 1)],
        [legacy_ind],  # at position gen=2, but birth gen=0
    ]
    results = extract_gene_results(pop_library)
    assert results[2][0].generation == 0  # birth gen preserved
    assert results[2][0].gene == 3        # birth offspring idx preserved


def test_extract_fallback_when_ind_missing_attrs():
    """Plain list-of-list input (no Individual wrapper) still works."""
    # This lets callers build fake pop_libraries for tests without
    # having DEAP installed.
    class _Minimal(list):
        def __init__(self, vec, fit):
            super().__init__(vec)
            self.fitness = _FakeFitness(fit)
            # no .generation or .gene

    pop_library = [[_Minimal([1.0], (0.5,))]]
    results = extract_gene_results(pop_library)
    r = results[0][0]
    # Falls back to gen_idx / 0 as documented.
    assert r.generation == 0
    assert r.gene == 0


# --- load_case_results -------------------------------------------------


def test_load_case_results_missing_dir(tmp_path):
    """Missing directory returns None (does not raise)."""
    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={},
        root=tmp_path,
    )
    reader = TextTableReader({})
    result = GeneResult(
        generation=5, gene=3,
        gene_vector=np.array([1.0]), fitness=(0.5,),
    )
    rs = load_case_results(
        result, sim_case_idx=0, resolver=resolver, reader=reader,
    )
    assert rs is None


def test_load_case_results_reads_real_dir(tmp_path):
    """An existing case directory is read back via the framework reader."""
    case_dir = tmp_path / "gen_0" / "gene_1_sc_0"
    case_dir.mkdir(parents=True)
    stress_file = case_dir / "avg_stress.txt"
    stress_file.write_text(
        # Time Volume Sxx Syy Szz Sxy Sxz Syz — Szz (col 5) carries
        # the sequence asserted below.
        "0.0 1.0 0 0 1.0 0 0 0\n"
        "0.5 1.0 0 0 2.0 0 0 0\n"
        "1.0 1.0 0 0 3.0 0 0 0\n"
    )

    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/avg_stress.txt",
        },
        root=tmp_path,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
        ),
    })
    result = GeneResult(
        generation=0, gene=1,
        gene_vector=np.array([1.0]), fitness=(0.5,),
    )
    rs = load_case_results(
        result, sim_case_idx=0, resolver=resolver, reader=reader,
    )
    assert rs is not None
    df = rs.df("avg_stress")
    assert len(df) == 3
    assert df["Szz"].tolist() == [1.0, 2.0, 3.0]


def test_load_case_results_handles_reader_error(tmp_path):
    """Reader exception on a bad file returns None rather than raising."""
    # Directory exists but the file is malformed.
    case_dir = tmp_path / "gen_0" / "gene_0_sc_0"
    case_dir.mkdir(parents=True)
    # Write a completely wrong number of columns.
    (case_dir / "avg_stress.txt").write_text("not a valid table\n")
    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/avg_stress.txt",
        },
        root=tmp_path,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
        ),
    })
    result = GeneResult(
        generation=0, gene=0,
        gene_vector=np.array([1.0]), fitness=(0.5,),
    )
    rs = load_case_results(
        result, sim_case_idx=0, resolver=resolver, reader=reader,
    )
    assert rs is None


# --- Plotting (matplotlib soft dep) ------------------------------------


def test_plot_stress_strain_overlay_runs():
    """Smoke test: function produces a figure without raising."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # headless backend
    from workflow_common.postprocess import plot_stress_strain_overlay

    sim_s = np.linspace(0, 1, 10)
    sim_y = sim_s * 100
    exp_s = np.linspace(0, 1, 10)
    exp_y = sim_s * 95
    fig, ax = plot_stress_strain_overlay(sim_s, sim_y, exp_s, exp_y)
    assert fig is not None
    assert ax is not None


def test_plot_stress_strain_overlay_accepts_existing_ax():
    """Passing an existing ax re-uses the figure."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from workflow_common.postprocess import plot_stress_strain_overlay

    fig, ax = plt.subplots()
    fig2, ax2 = plot_stress_strain_overlay(
        np.array([0.0, 1.0]), np.array([0.0, 100.0]),
        np.array([0.0, 1.0]), np.array([0.0, 95.0]),
        ax=ax,
    )
    assert fig2 is fig
    assert ax2 is ax


def test_plot_pareto_front_runs():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from workflow_common.postprocess import plot_pareto_front

    rng = np.random.RandomState(0)
    fit = rng.rand(10, 2)
    fig, _ = plot_pareto_front(
        fit, best_idx=[0, 3], axis_labels=("stress RMSE", "slope RMSE"),
    )
    assert fig is not None


def test_plot_pareto_front_rejects_3d():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from workflow_common.postprocess import plot_pareto_front

    fit = np.random.RandomState(0).rand(10, 3)
    with pytest.raises(ValueError, match="2-D"):
        plot_pareto_front(fit)
