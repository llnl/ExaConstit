"""
Unit tests for :mod:`workflow_common.objectives`.

Covers the extractor (with its three strain-derivation modes), the
shipped :class:`StressStrainObjective` evaluator, each error metric,
and the three :class:`FailureHandler` policies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common import (
    CaseContext,
    CaseResultSet,
    ConstantPenaltyFailureHandler,
    ERROR_METRICS,
    FailureHandler,
    InfinityFailureHandler,
    LegacyLinearSmoother,
    ObjectiveEvaluator,
    PartialProgressFailureHandler,
    PchipSmoother,
    StressStrainExtractor,
    StressStrainObjective,
    TabularResult,
    mae,
    max_abs_error,
    rmse,
)


# --- helpers -------------------------------------------------------------


def _make_result_set(
    stress_df: pd.DataFrame,
    def_grad_df: pd.DataFrame = None,
    *,
    strain_output_name: str = "avg_def_grad",
) -> CaseResultSet:
    """Build a CaseResultSet in memory for a test.

    The ``strain_output_name`` knob lets tests build a result set
    where the strain-source table is registered under a non-default
    name (e.g. ``"avg_lagrangian_strain"`` for a direct-strain test).
    """
    ctx = CaseContext(generation=0, gene=0, obj=0)
    tables = {
        "avg_stress": TabularResult(
            name="avg_stress", df=stress_df, source_path=Path("fake/avg_stress.txt"),
        ),
    }
    if def_grad_df is not None:
        tables[strain_output_name] = TabularResult(
            name=strain_output_name, df=def_grad_df,
            source_path=Path(f"fake/{strain_output_name}.txt"),
        )
    return CaseResultSet(ctx=ctx, tables=tables)


def _voce_stress(strain: np.ndarray, y0: float, H: float, k: float) -> np.ndarray:
    """Voce-law saturating stress response."""
    return y0 + H * (1 - np.exp(-k * strain))


# --- Error metrics -------------------------------------------------------


def test_rmse_zero_for_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert rmse(a, a) == 0.0


def test_rmse_known_value():
    """Sanity on a hand-computable example."""
    sim = np.array([1.0, 2.0, 3.0])
    exp = np.array([2.0, 2.0, 2.0])
    # residual: [-1, 0, 1], sqrt(mean(1, 0, 1)) = sqrt(2/3)
    assert rmse(sim, exp) == pytest.approx(np.sqrt(2 / 3))


def test_mae_and_max_abs():
    sim = np.array([1.0, 2.0, 3.0])
    exp = np.array([2.0, 2.0, 0.0])
    # residual: [-1, 0, 3], abs: [1, 0, 3]
    assert mae(sim, exp) == pytest.approx(4 / 3)
    assert max_abs_error(sim, exp) == pytest.approx(3.0)


def test_error_metrics_table_keys():
    """The ERROR_METRICS table should expose at least the three defaults."""
    assert {"rmse", "mae", "max_abs"}.issubset(ERROR_METRICS.keys())


# --- StressStrainExtractor -----------------------------------------------


def test_extractor_biot_strain():
    """Biot strain: strain = X - 1, where X is the axial stretch."""
    t = np.linspace(0, 1, 10)
    stress_df = pd.DataFrame({
        "Time": t,
        "Szz": _voce_stress(t * 0.01, 200, 2000, 50),
        "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    F33 = 1.0 + 0.01 * t
    dg_df = pd.DataFrame({"Time": t, "F33": F33})

    extractor = StressStrainExtractor()  # default: biot
    strain, stress = extractor.extract(_make_result_set(stress_df, dg_df))
    assert np.allclose(strain, 0.01 * t)
    assert len(stress) == len(t)


def test_extractor_log_strain():
    """Hencky / log strain: strain = log(F33)."""
    t = np.linspace(0, 1, 10)
    stress_df = pd.DataFrame({
        "Time": t,
        "Szz": np.ones_like(t),
        "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    F33 = 1.0 + 0.5 * t
    dg_df = pd.DataFrame({"Time": t, "F33": F33})

    extractor = StressStrainExtractor(strain_source="log")
    strain, _ = extractor.extract(_make_result_set(stress_df, dg_df))
    assert np.allclose(strain, np.log(F33))


def test_extractor_time_rate():
    """Constant-rate: strain = rate * time. No def-grad needed."""
    t = np.linspace(0, 1, 10)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.ones_like(t), "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t), "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t), "Sxz": np.zeros_like(t),
    })
    extractor = StressStrainExtractor(
        strain_source="time_rate", strain_rate=2e-3,
    )
    strain, _ = extractor.extract(_make_result_set(stress_df))
    assert np.allclose(strain, 2e-3 * t)


def test_extractor_time_rate_requires_strain_rate():
    t = np.linspace(0, 1, 5)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.zeros_like(t), "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t), "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t), "Sxz": np.zeros_like(t),
    })
    extractor = StressStrainExtractor(strain_source="time_rate")  # no rate
    with pytest.raises(ValueError, match="strain_rate"):
        extractor.extract(_make_result_set(stress_df))


def test_extractor_log_strain_rejects_nonpositive():
    """``log`` strain requires the axial stretch > 0 everywhere; zero or negative raises."""
    t = np.linspace(0, 1, 5)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.zeros_like(t), "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t), "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t), "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": np.array([1, 0.5, 0, 1, 1])})
    extractor = StressStrainExtractor(strain_source="log")
    with pytest.raises(ValueError, match="F33 > 0"):
        extractor.extract(_make_result_set(stress_df, dg_df))


def test_extractor_missing_def_grad_raises():
    """Biot/log strain need the strain-source file; without it we get a clear KeyError."""
    t = np.linspace(0, 1, 5)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.zeros_like(t), "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t), "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t), "Sxz": np.zeros_like(t),
    })
    # no dg passed in
    extractor = StressStrainExtractor()
    with pytest.raises(KeyError, match="avg_def_grad"):
        extractor.extract(_make_result_set(stress_df))


def test_extractor_detects_nan_in_output():
    """Non-finite values in sim output are a red flag; raise."""
    t = np.linspace(0, 1, 5)
    stress = np.array([100.0, 200.0, np.nan, 400.0, 500.0])
    stress_df = pd.DataFrame({
        "Time": t, "Szz": stress, "Sxx": np.zeros_like(t),
        "Syy": np.zeros_like(t), "Sxy": np.zeros_like(t),
        "Syz": np.zeros_like(t), "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + 0.01 * t})
    extractor = StressStrainExtractor()
    with pytest.raises(ValueError, match="NaN"):
        extractor.extract(_make_result_set(stress_df, dg_df))


# --- StressStrainObjective -----------------------------------------------


def _exp_df() -> pd.DataFrame:
    """Simple exp reference with matching Voce parameters."""
    strain = np.linspace(0, 0.01, 30)
    stress = _voce_stress(strain, 200, 2000, 50)
    return pd.DataFrame({"strain": strain, "stress": stress})


def _sim_result(y0=200, H=2000, k=50) -> CaseResultSet:
    """Sim result for Voce params y0, H, k over strain 0..0.01."""
    t = np.linspace(0, 1, 50)
    strain = 0.01 * t
    stress = _voce_stress(strain, y0, H, k)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": stress,
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + strain})
    return _make_result_set(stress_df, dg_df)


def test_objective_zero_when_sim_matches_exp():
    """Identical Voce params => near-zero RMSE (up to smoothing error)."""
    exp_df = _exp_df()
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    err = evaluator.evaluate(_sim_result(), CaseContext(0, 0, 0))
    # Won't be exactly 0 because PCHIP resamples on a different grid.
    assert err < 1.0, f"expected near-zero, got {err}"


def test_objective_positive_when_sim_differs():
    """Different Voce params produce a measurable error."""
    exp_df = _exp_df()
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    # Shift yield stress by 50 MPa.
    err = evaluator.evaluate(_sim_result(y0=250), CaseContext(0, 0, 0))
    assert err > 10.0


def test_objective_respects_custom_metric():
    """A callable metric is honored."""
    exp_df = _exp_df()

    def constant(sim, exp):
        return 42.0

    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
        metric=constant,
    )
    assert evaluator.evaluate(_sim_result(), CaseContext(0, 0, 0)) == 42.0


def test_objective_respects_named_metric():
    """String names 'rmse', 'mae', 'max_abs' resolve."""
    exp_df = _exp_df()
    for name in ("rmse", "mae", "max_abs"):
        evaluator = StressStrainObjective(
            experimental=exp_df,
            extractor=StressStrainExtractor(),
            metric=name,
        )
        err = evaluator.evaluate(_sim_result(y0=250), CaseContext(0, 0, 0))
        assert err > 0


def test_objective_unknown_metric_raises():
    exp_df = _exp_df()
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
        metric="nonexistent",
    )
    with pytest.raises(ValueError, match="unknown metric"):
        evaluator.evaluate(_sim_result(), CaseContext(0, 0, 0))


def test_objective_no_strain_overlap_raises():
    """Sim and exp with disjoint strain ranges should raise, not silently zero."""
    exp_df = pd.DataFrame({
        "strain": np.linspace(5.0, 10.0, 20),
        "stress": np.linspace(100.0, 200.0, 20),
    })
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluator.evaluate(_sim_result(), CaseContext(0, 0, 0))


def test_objective_protocol_conformance():
    exp_df = _exp_df()
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    assert isinstance(evaluator, ObjectiveEvaluator)


def test_objective_alternate_smoother():
    """LegacyLinearSmoother is an acceptable drop-in."""
    exp_df = _exp_df()
    evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
        smoother=LegacyLinearSmoother(n_samples=200),
    )
    err = evaluator.evaluate(_sim_result(y0=250), CaseContext(0, 0, 0))
    assert err > 0


# --- FailureHandlers -----------------------------------------------------


def test_infinity_handler():
    h = InfinityFailureHandler()
    v = h.on_failure(CaseContext(0, 0, 0), "timeout", None)
    assert v == float("inf")


def test_infinity_handler_custom_value():
    h = InfinityFailureHandler(value=1e18)
    v = h.on_failure(CaseContext(0, 0, 0), "timeout", None)
    assert v == 1e18


def test_constant_penalty_handler():
    h = ConstantPenaltyFailureHandler(penalty=999.0)
    assert h.on_failure(CaseContext(0, 0, 0), "rc=7", None) == 999.0


def test_partial_progress_no_results_returns_base_penalty():
    exp_df = _exp_df()
    inner = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    h = PartialProgressFailureHandler(
        inner_evaluator=inner, base_penalty=1e6,
    )
    v = h.on_failure(CaseContext(0, 0, 0), "no output", None)
    assert v == 1e6


def test_partial_progress_half_way():
    """A sim that got halfway through should get roughly half the penalty."""
    exp_df = _exp_df()
    inner = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    h = PartialProgressFailureHandler(
        inner_evaluator=inner,
        base_penalty=1e6,
        progress_weight=1.0,
        strain_target=0.01,
    )
    # Partial sim: only first 25 rows (strain 0..0.005, half of 0.01).
    t = np.linspace(0, 0.5, 25)
    strain = 0.01 * t
    stress = _voce_stress(strain, 200, 2000, 50)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": stress,
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + strain})
    partial = _make_result_set(stress_df, dg_df)

    v = h.on_failure(CaseContext(0, 0, 0), "timeout", partial)
    # Roughly 50% of base_penalty since progress_fraction ~ 0.5.
    # Plus a small contribution from the inner error (close to 0 for
    # matching parameters).
    assert 4e5 < v < 6e5


def test_failure_handlers_protocol_conformance():
    exp_df = _exp_df()
    inner = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(),
    )
    assert isinstance(InfinityFailureHandler(), FailureHandler)
    assert isinstance(ConstantPenaltyFailureHandler(), FailureHandler)
    assert isinstance(
        PartialProgressFailureHandler(inner_evaluator=inner), FailureHandler
    )


def test_extractor_supports_non_exaconstit_column_convention():
    """The framework lives in ExaConstit and defaults to ExaConstit's
    column convention (Szz / F33 / Time). Users calibrating a different
    FEM code whose output uses a different convention (s11 / F11 / time,
    or anything else) can still drive the extractor by supplying
    explicit column names. This test verifies that override path stays
    working — the framework provides ExaConstit defaults but doesn't
    hard-code them anywhere in the logic.
    """
    t = np.linspace(0, 1, 10)
    # Fake a non-ExaConstit code's output: lowercase "time", "s11" as
    # the axial stress, no Volume column.
    stress_df = pd.DataFrame({
        "time": t,
        "s11": _voce_stress(t * 0.01, 200, 2000, 50),
    })
    dg_df = pd.DataFrame({"time": t, "F11": 1.0 + 0.01 * t})

    extractor = StressStrainExtractor(
        stress_column="s11",
        strain_source_column="F11",
        time_column="time",
        # strain_source left at default ("biot") — applies the X-1
        # formula to whatever strain_source_column names.
    )
    strain, stress = extractor.extract(_make_result_set(stress_df, dg_df))
    assert np.allclose(strain, 0.01 * t)
    assert len(stress) == len(t)
    # Sanity: default-construction against the same non-ExaConstit
    # DataFrame fails loudly (ExaConstit default Szz not in columns).
    default_extractor = StressStrainExtractor()
    with pytest.raises(KeyError, match="Szz"):
        default_extractor.extract(_make_result_set(stress_df, dg_df))





def test_extractor_direct_strain_reads_column_verbatim():
    """``strain_source='direct'`` reads a strain measure already on disk.

    Use case: the simulation outputs Lagrange/Euler/Biot strain in
    its own file, and the user wants to consume it directly rather
    than recomputing from F. The extractor's job becomes "pull the
    column" with no math applied.
    """
    t = np.linspace(0, 1, 10)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": _voce_stress(t * 0.05, 200, 2000, 50),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    # Simulate a Lagrange-strain output file with E33 already
    # computed by the sim.
    e33 = 0.5 * ((1.0 + 0.05 * t) ** 2 - 1.0)
    strain_df = pd.DataFrame({"Time": t, "E33": e33})
    extractor = StressStrainExtractor(
        strain_source="direct",
        strain_source_output="avg_lagrangian_strain",
        strain_source_column="E33",
    )
    rs = _make_result_set(stress_df, strain_df,
                          strain_output_name="avg_lagrangian_strain")
    strain, stress = extractor.extract(rs)
    # No transformation — strain comes straight from the column.
    assert np.allclose(strain, e33)
    assert len(stress) == len(t)


def test_extractor_window_crops_to_strain_interval():
    """``window=(lo, hi)`` keeps only points where lo <= |strain| <= hi.

    This is the optimization-restriction feature: skip the elastic
    regime, ignore the elastic-plastic transition, focus the
    optimizer on the plastic portion.
    """
    t = np.linspace(0, 1, 21)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": _voce_stress(t * 0.1, 200, 2000, 50),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    # Strain ramps 0 -> 0.1 (Biot).
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + 0.1 * t})
    extractor = StressStrainExtractor(window=(0.02, 0.08))
    strain, stress = extractor.extract(_make_result_set(stress_df, dg_df))
    # Every kept point lies in the interval.
    assert np.all(strain >= 0.02 - 1e-12)
    assert np.all(strain <= 0.08 + 1e-12)
    # Same length on both sides.
    assert len(strain) == len(stress)
    # Original had 21 points; cropping to [0.02, 0.08] keeps roughly
    # the middle 13 points (indices 4..16 inclusive on a uniform grid).
    assert 10 <= len(strain) <= 14


def test_extractor_window_works_for_compression_via_abs():
    """Window comparison uses ``|strain|`` so a compression run with
    negative strain values gets the same windowing as tension.
    """
    t = np.linspace(0, 1, 11)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": -1.0 * _voce_stress(t * 0.1, 200, 2000, 50),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    # Compression: F33 < 1, strain < 0.
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 - 0.1 * t})
    extractor = StressStrainExtractor(window=(0.02, 0.08))
    strain, _ = extractor.extract(_make_result_set(stress_df, dg_df))
    # Returned strain is still negative (window doesn't change sign);
    # only the |strain| satisfies the bounds.
    assert np.all(strain <= 0.0)
    assert np.all(np.abs(strain) >= 0.02 - 1e-12)
    assert np.all(np.abs(strain) <= 0.08 + 1e-12)


def test_extractor_window_excludes_all_raises():
    """A window with no overlap with the actual strain range fails loudly."""
    t = np.linspace(0, 1, 5)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.zeros_like(t),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + 0.01 * t})
    # Strain range is [0, 0.01]; window at [10, 100] is way out.
    extractor = StressStrainExtractor(window=(10.0, 100.0))
    with pytest.raises(ValueError, match="excluded every data point"):
        extractor.extract(_make_result_set(stress_df, dg_df))


def test_extractor_window_bad_bounds_raises():
    """``window=(hi, lo)`` is operator error; flag at extract time."""
    t = np.linspace(0, 1, 5)
    stress_df = pd.DataFrame({
        "Time": t, "Szz": np.zeros_like(t),
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })
    dg_df = pd.DataFrame({"Time": t, "F33": 1.0 + 0.01 * t})
    extractor = StressStrainExtractor(window=(0.05, 0.01))  # inverted
    with pytest.raises(ValueError, match="lower bound .* exceeds"):
        extractor.extract(_make_result_set(stress_df, dg_df))


# --- StressStrainExtractor JSON serialization ---------------------------


def test_extractor_to_dict_round_trips_through_json():
    """to_dict -> json.dumps -> json.loads -> from_dict is identity.

    Used by the archive to persist run-time extractor configs so
    plotters reconstruct exactly what the optimizer scored against.
    Tuples MUST round-trip back to tuples (not lists) since the
    dataclass declares window: Tuple.
    """
    import json
    e = StressStrainExtractor(
        stress_column="Sxx",
        strain_source="time_rate",
        strain_rate=1.5e-3,
        time_column="t",
        window=(0.005, 0.13),
    )
    d = e.to_dict()
    serialized = json.dumps(d)
    restored = StressStrainExtractor.from_dict(json.loads(serialized))
    assert restored == e
    # Window must be a tuple, not a list — dataclass type contract.
    assert isinstance(restored.window, tuple)


def test_extractor_from_dict_tolerates_missing_keys():
    """An older config (lacking newer fields) should still load by
    falling back to dataclass defaults, so a forward-compat plotter
    can read older archives."""
    # Just stress info — most of the dataclass uses defaults.
    e = StressStrainExtractor.from_dict({
        "strain_source": "time_rate",
        "strain_rate": 1e-3,
    })
    assert e.strain_source == "time_rate"
    assert e.strain_rate == 1e-3
    # Defaulted fields.
    assert e.stress_column == "Szz"
    assert e.window is None


def test_extractor_from_dict_ignores_unknown_keys():
    """Forward-compat the other way: a newer archive with extra
    fields should still load on older code that doesn't know about
    them."""
    e = StressStrainExtractor.from_dict({
        "strain_source": "biot",
        "strain_source_column": "F33",
        "future_field_not_yet_invented": "ignored",
    })
    assert e.strain_source == "biot"
    assert e.strain_source_column == "F33"


def test_extractor_to_dict_round_trips_window_none():
    """window=None must JSON-serialize as null and round-trip back to None."""
    import json
    e = StressStrainExtractor(window=None)
    d = e.to_dict()
    assert d["window"] is None
    assert json.loads(json.dumps(d))["window"] is None
    assert StressStrainExtractor.from_dict(d).window is None




# --- Example evaluator alignment direction ------------------------------


def test_std_normalized_stress_evaluator_aligns_exp_to_sim_via_pchip():
    """The example's stress evaluator interpolates EXPERIMENTAL data
    onto the SIMULATION's strain grid via PchipSmoother — the
    direction the original ExaConstit code used. Verify the score
    matches what we'd compute by hand using PchipSmoother.sample_at,
    NOT what np.interp(sim_at_exp_grid) would produce.

    Pins Robert's correction: 'It was interpolating the experimental
    data to the simulation data. It's the entire reason you have
    workflow/smoothing.py.'
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    import nsga3_calibration

    from workflow_common.smoothing import PchipSmoother

    # Sim and exp on different strain grids.
    sim_strain = np.linspace(0.0, 0.1, 50)
    sim_stress = 200.0 + 1900.0 * (1 - np.exp(-48.0 * sim_strain))
    exp_strain = np.linspace(0.0, 0.1, 25)
    exp_stress = 210.0 + 1900.0 * (1 - np.exp(-48.0 * exp_strain))
    exp_df = pd.DataFrame({"strain": exp_strain, "stress": exp_stress})

    class _FakeExtractor:
        def extract(self, results):
            return sim_strain, sim_stress

    ev = nsga3_calibration._StdNormalizedStressEvaluator(
        experimental=exp_df,
        extractor=_FakeExtractor(),
    )
    score = ev.evaluate(None, None)

    # Hand-compute the expected score: align exp onto sim's strain
    # grid via PchipSmoother (same tool the evaluator uses), then
    # RMSE / std on that grid.
    smoother = PchipSmoother(strict_monotonic=False)
    exp_at_sim = smoother.sample_at(
        exp_strain, exp_stress, sim_strain,
    ).y
    residual = sim_stress - exp_at_sim
    expected = float(np.sqrt(np.mean(residual ** 2)) / np.std(exp_at_sim))
    assert score == pytest.approx(expected, rel=1e-9)


def test_std_normalized_slope_evaluator_aligns_exp_to_sim_via_pchip():
    """Slope evaluator's analogue of the stress test."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    import nsga3_calibration

    from workflow_common.smoothing import PchipSmoother

    sim_strain = np.linspace(0.0, 0.1, 50)
    sim_stress = 200.0 + 1900.0 * (1 - np.exp(-48.0 * sim_strain))
    exp_strain = np.linspace(0.0, 0.1, 25)
    exp_stress = 210.0 + 1900.0 * (1 - np.exp(-48.0 * exp_strain))
    exp_df = pd.DataFrame({"strain": exp_strain, "stress": exp_stress})

    class _FakeExtractor:
        def extract(self, results):
            return sim_strain, sim_stress

    ev = nsga3_calibration._StdNormalizedSlopeEvaluator(
        experimental=exp_df,
        extractor=_FakeExtractor(),
    )
    score = ev.evaluate(None, None)

    smoother = PchipSmoother(strict_monotonic=False)
    exp_at_sim = smoother.sample_at(
        exp_strain, exp_stress, sim_strain,
    ).y
    diff_strain = np.diff(sim_strain)
    sim_slope = np.diff(sim_stress) / diff_strain
    exp_slope = np.diff(exp_at_sim) / diff_strain
    residual = sim_slope - exp_slope
    expected = float(np.sqrt(np.mean(residual ** 2)) / np.std(exp_slope))
    assert score == pytest.approx(expected, rel=1e-9)
