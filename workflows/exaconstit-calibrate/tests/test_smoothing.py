"""
Unit tests for :mod:`workflow_common.smoothing`.

Covers the three smoothers (:class:`PchipSmoother`,
:class:`ArcLengthSmoother`, :class:`LegacyLinearSmoother`), the
arc-length helper, monotonicity detection, and the auto_smoother
selector. Both happy paths and the degenerate cases that real
experimental data produces.
"""
from __future__ import annotations

import numpy as np
import pytest

from workflow_common.smoothing import (
    ArcLengthSmoother,
    LegacyLinearSmoother,
    PchipSmoother,
    SmoothedCurve,
    Smoother,
    arc_length,
    auto_smoother,
    is_monotonic,
)


# --- Helpers used across multiple tests ----------------------------------


def _voce_curve(n: int = 50, sat: float = 2000.0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic saturating stress-strain curve for testing.

    Returns (strain, stress) where strain is linear 0..1 and stress
    follows a Voce-law saturating response. Monotonic by construction.
    """
    eps = np.linspace(0.0, 1.0, n)
    sig = 200.0 + sat * (1.0 - np.exp(-5.0 * eps))
    return eps, sig


def _snapback_curve(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic non-monotonic curve simulating snap-back behavior.

    A spiral-like path in (x, y) space; x is decidedly non-monotonic.
    """
    t = np.linspace(0, 2 * np.pi, n)
    x = np.cos(t) * (1.0 - 0.1 * t)
    y = np.sin(t) * (1.0 - 0.1 * t)
    return x, y


# --- is_monotonic --------------------------------------------------------


def test_is_monotonic_strictly_increasing():
    assert is_monotonic(np.array([1.0, 2.0, 3.0, 4.0]))


def test_is_monotonic_with_plateau():
    """Equal consecutive values are tolerated (non-strict definition)."""
    assert is_monotonic(np.array([1.0, 2.0, 2.0, 3.0]))


def test_is_monotonic_rejects_decrease():
    assert not is_monotonic(np.array([1.0, 2.0, 1.5, 3.0]))


def test_is_monotonic_tolerance_accepts_noise():
    """Small backward steps within tol are allowed."""
    x = np.array([1.0, 2.0, 1.9999999, 3.0])
    assert is_monotonic(x, tol=1e-6)
    assert not is_monotonic(x, tol=0.0)


def test_is_monotonic_trivially_true_on_short_input():
    """Arrays of length 0 or 1 are monotonic by convention."""
    assert is_monotonic(np.array([]))
    assert is_monotonic(np.array([5.0]))


# --- arc_length ----------------------------------------------------------


def test_arc_length_starts_at_zero():
    x, y = _voce_curve(10)
    s = arc_length(x, y)
    assert s[0] == 0.0
    assert s[-1] > 0.0


def test_arc_length_monotonic_increasing():
    x, y = _snapback_curve(50)
    s = arc_length(x, y)
    assert np.all(np.diff(s) > 0)


def test_arc_length_straight_line_in_unit_box():
    """A diagonal line from (0,0) to (1,1) has normalized arc length sqrt(2)."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    s = arc_length(x, y, x_scale=1.0, y_scale=1.0)
    assert s[-1] == pytest.approx(np.sqrt(2.0))


def test_arc_length_explicit_scales_invariance():
    """Scaling x or y by a factor should cancel out when the scale is passed."""
    x = np.array([0.0, 2000.0])
    y = np.array([0.0, 2000.0])
    s = arc_length(x, y, x_scale=2000.0, y_scale=2000.0)
    assert s[-1] == pytest.approx(np.sqrt(2.0))


def test_arc_length_zero_scale_raises():
    x = np.array([1.0, 1.0, 1.0])  # constant
    with pytest.raises(ValueError, match="positive"):
        arc_length(x, x)  # x_scale and y_scale both zero


def test_arc_length_shape_mismatch_raises():
    with pytest.raises(ValueError):
        arc_length(np.array([1, 2, 3]), np.array([1, 2]))


# --- PchipSmoother -------------------------------------------------------


def test_pchip_reproduces_endpoints():
    """PCHIP on a smooth curve should hit the endpoints exactly."""
    x, y = _voce_curve(20)
    s = PchipSmoother(n_samples=100).smooth(x, y)
    assert s.x[0] == pytest.approx(x[0])
    assert s.x[-1] == pytest.approx(x[-1])
    assert s.y[0] == pytest.approx(y[0])
    assert s.y[-1] == pytest.approx(y[-1])


def test_pchip_does_not_overshoot():
    """The defining property of PCHIP: no overshoot beyond source extrema."""
    # Construct a curve with a sharp plateau that a naive cubic would overshoot.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    s = PchipSmoother(n_samples=200).smooth(x, y)
    # Output must stay within the input y range.
    assert s.y.max() <= y.max() + 1e-12
    assert s.y.min() >= y.min() - 1e-12


def test_pchip_sample_at_user_targets():
    x, y = _voce_curve(20)
    smoother = PchipSmoother(n_samples=100)
    targets = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    s = smoother.sample_at(x, y, targets)
    assert np.allclose(s.x, targets)
    assert s.y.shape == targets.shape


def test_pchip_strict_mono_rejects_backwards():
    """Strict mode refuses non-monotonic input."""
    x = np.array([0.0, 1.0, 0.5, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="monotonic"):
        PchipSmoother(strict_monotonic=True).smooth(x, y)


def test_pchip_non_strict_auto_sorts_mild_noise():
    """Non-strict mode handles tiny backward steps by sorting silently."""
    x = np.array([0.0, 0.1, 0.0999999, 0.2, 0.3])
    y = np.array([0.0, 0.1, 0.1, 0.2, 0.3])
    s = PchipSmoother(strict_monotonic=False, n_samples=50).smooth(x, y)
    assert len(s.x) == 50
    # Result should still be monotonic.
    assert np.all(np.diff(s.y) >= -1e-9)


def test_pchip_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        PchipSmoother().smooth(np.array([1.0]), np.array([1.0]))


def test_pchip_rejects_bad_n_samples():
    with pytest.raises(ValueError):
        PchipSmoother(n_samples=1)


def test_pchip_output_s_is_none():
    """PchipSmoother doesn't compute arc length, so SmoothedCurve.s is None."""
    x, y = _voce_curve()
    s = PchipSmoother().smooth(x, y)
    assert s.s is None


# --- ArcLengthSmoother ---------------------------------------------------


def test_arclength_handles_non_monotonic():
    """Arc-length smoother must not crash on spiral-like curves."""
    x, y = _snapback_curve(60)
    s = ArcLengthSmoother(n_samples=150).smooth(x, y)
    assert s.x.shape == (150,)
    assert s.y.shape == (150,)
    assert s.s is not None
    assert np.all(np.diff(s.s) > 0)


def test_arclength_s_spans_input_range():
    """Smoothed s range should match the input arc-length range."""
    x, y = _snapback_curve(60)
    smoother = ArcLengthSmoother(n_samples=100)
    smoothed = smoother.smooth(x, y)
    src_s = arc_length(x, y, x_scale=x.max()-x.min(), y_scale=y.max()-y.min())
    # Default behavior: x_scale/y_scale come from source data range, so
    # src_s and smoothed.s should cover the same interval.
    assert smoothed.s[0] == pytest.approx(src_s[0])
    assert smoothed.s[-1] == pytest.approx(src_s[-1])


def test_arclength_explicit_scales_matter():
    """Different explicit scales produce different arc-length totals."""
    x, y = _snapback_curve(60)
    a = ArcLengthSmoother(n_samples=50, x_scale=1.0, y_scale=1.0).smooth(x, y)
    b = ArcLengthSmoother(n_samples=50, x_scale=10.0, y_scale=10.0).smooth(x, y)
    # Larger scales -> smaller normalized arc length.
    assert a.s[-1] > b.s[-1]


def test_arclength_drops_duplicate_points():
    """Exact duplicates should be collapsed, not crash the PCHIP build."""
    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])  # one exact duplicate
    y = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    # Should not raise.
    s = ArcLengthSmoother(n_samples=50).smooth(x, y)
    assert s.s[0] == 0.0
    assert s.s[-1] > 0.0


def test_arclength_sample_at_s_respects_target():
    x, y = _snapback_curve(40)
    smoother = ArcLengthSmoother()
    full = smoother.smooth(x, y)  # populates internal s range
    # Sample at s values matching the default smooth range.
    targets = np.linspace(full.s[0], full.s[-1], 10)
    out = smoother.sample_at_s(x, y, targets)
    assert np.allclose(out.s, targets)


def test_arclength_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        ArcLengthSmoother().smooth(np.array([1.0]), np.array([1.0]))


# --- LegacyLinearSmoother -----------------------------------------------


def test_legacy_matches_numpy_interp_on_monotonic():
    """LegacyLinearSmoother must agree exactly with np.interp on monotonic data."""
    x, y = _voce_curve(30)
    ls = LegacyLinearSmoother(n_samples=100)
    out = ls.smooth(x, y)
    expected_x = np.linspace(x.min(), x.max(), 100)
    expected_y = np.interp(expected_x, x, y)
    assert np.allclose(out.x, expected_x)
    assert np.allclose(out.y, expected_y)


def test_legacy_sorts_unsorted_input():
    """Unsorted input is handled by sorting first."""
    x = np.array([2.0, 0.0, 1.0, 3.0])
    y = np.array([4.0, 0.0, 1.0, 9.0])
    out = LegacyLinearSmoother(n_samples=5).smooth(x, y)
    assert out.x[0] == 0.0
    assert out.x[-1] == 3.0
    # Endpoints should match the sorted endpoints of y.
    assert out.y[0] == pytest.approx(0.0)
    assert out.y[-1] == pytest.approx(9.0)


# --- Migration / equivalence: legacy vs PCHIP ----------------------------


def test_legacy_vs_pchip_close_on_dense_monotonic():
    """On dense monotonic data, legacy linear and PCHIP agree well.

    This is the key migration-safety test: before we replace the
    legacy smoother in a production optimization, we want to verify
    that PCHIP does not introduce large artifacts on data similar
    to what production encounters.
    """
    x, y = _voce_curve(100)  # dense
    legacy = LegacyLinearSmoother(n_samples=200).smooth(x, y)
    pchip = PchipSmoother(n_samples=200).smooth(x, y)
    assert np.allclose(legacy.x, pchip.x)
    # Max absolute difference in y should be small compared to the
    # signal range. On smooth Voce-law data, differences are tiny.
    max_abs_diff = float(np.max(np.abs(legacy.y - pchip.y)))
    signal_range = float(y.max() - y.min())
    assert max_abs_diff / signal_range < 1e-3


# --- auto_smoother -------------------------------------------------------


def test_auto_picks_pchip_for_monotonic():
    x, y = _voce_curve()
    s = auto_smoother(x, y)
    assert isinstance(s, PchipSmoother)


def test_auto_picks_arclength_for_non_monotonic():
    x, y = _snapback_curve()
    s = auto_smoother(x, y)
    assert isinstance(s, ArcLengthSmoother)


def test_auto_smoother_respects_n_samples():
    x, y = _voce_curve()
    s = auto_smoother(x, y, n_samples=50)
    out = s.smooth(x, y)
    assert len(out.x) == 50


# --- Protocol conformance ------------------------------------------------


def test_all_smoothers_satisfy_protocol():
    assert isinstance(PchipSmoother(), Smoother)
    assert isinstance(ArcLengthSmoother(), Smoother)
    assert isinstance(LegacyLinearSmoother(), Smoother)


def test_smoothed_curve_dataclass():
    c = SmoothedCurve(x=np.array([0, 1]), y=np.array([2, 3]))
    assert c.s is None
    c2 = SmoothedCurve(x=np.array([0, 1]), y=np.array([2, 3]), s=np.array([0, 1]))
    assert c2.s is not None
