"""
Smoothing and resampling of stress-strain (and other) curves.

Why this module exists
----------------------
Before an objective function can compare a simulated curve to an
experimental one, the two curves must be sampled at the same points.
Real data rarely cooperates:

* Simulation and experiment are sampled at different times.
* Experimental data is noisy and can be irregularly spaced.
* Some material responses are not monotonic in strain - necking,
  snap-back, and strain-softening all produce curves where strain
  decreases partway through the loading history.

This module provides a small toolkit for turning a raw ``(x, y)``
curve into a clean, uniformly-sampled curve suitable for error-metric
computation.

The core abstraction is the :class:`Smoother` Protocol: anything
with a ``smooth(x, y)`` method that returns a :class:`SmoothedCurve`
is acceptable. Three concrete implementations are shipped:

* :class:`PchipSmoother` - monotonic cubic Hermite interpolation.
  The right default for well-behaved stress-strain data.
* :class:`ArcLengthSmoother` - parameterizes the curve by arc length
  so non-monotonic shapes can be resampled without ambiguity.
* :class:`LegacyLinearSmoother` - simple numpy linear interpolation,
  matching the behavior of the pre-refactor code. Kept so that
  migration can be verified to produce identical results on
  monotonic data before switching to PCHIP.

Why PCHIP and not cubic splines?
--------------------------------
A natural cubic spline will happily overshoot the data it was given
- if three samples are ``(0, 0), (1, 1), (2, 1)`` the cubic fit
produces y > 1 between x=1 and x=2, introducing stress values that
never existed in the source data. That's a silent lie and a real
source of bugs in error metrics.

PCHIP (Piecewise Cubic Hermite Interpolating Polynomial, Fritsch-
Carlson 1980) uses local slopes chosen to preserve monotonicity:
if the input is monotonically non-decreasing, so is the output.
It cannot overshoot. The interpolant is C^1 continuous (smooth
first derivatives) but not C^2, which is fine for engineering
use - the second derivative is rarely what drives an objective.

Why arc-length parameterization for non-monotonic data?
-------------------------------------------------------
When x is not a single-valued function of t - i.e., the curve folds
back on itself - you cannot ask "what is y at x=0.5?" because there
may be multiple y values at that x. Linear interpolation of y vs x
fails silently for these curves, either crashing or returning only
one branch.

The fix is to parameterize by arc length ``s`` along the curve::

    s(t) = integral from 0 to t of sqrt(dx/dt^2 + dy/dt^2)

Then both ``x(s)`` and ``y(s)`` are single-valued functions of
``s``, regardless of what shape the curve traces in (x, y) space.
PCHIP interpolation of x vs s and y vs s gives a faithful,
resamplable curve.

Normalization matters for arc length
------------------------------------
Arc length has units of whichever axes it sums over, so a raw
``(strain, stress)`` curve has the stress axis dominating the arc
length because stress is typically 10^8 - 10^9 Pa while strain is
0-1. A resampling uniform in that arc length concentrates almost
all samples along the stress axis and starves the strain axis.

The fix is to divide x and y by reference scales before computing
arc length::

    s = integral of sqrt((dx/x_scale)^2 + (dy/y_scale)^2)

A sensible default scale is the range of the source data itself
(``x.max() - x.min()``), which maps the raw curve into a unit box
and makes arc length dimensionless and reference-free. When
comparing simulation to experiment, however, it's better to use
the EXPERIMENTAL scales for both - otherwise the sim and exp get
smoothed with different parameterizations and become incomparable.
Pass matching scales to both smoothers in that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

from .logging_utils import get_logger

logger = get_logger(__name__)


# --- Result container ----------------------------------------------------


@dataclass
class SmoothedCurve:
    """A smoothed / resampled curve in its final, analysis-ready form.

    Fields:
        x: Independent variable values at the resampled points. For
            :class:`PchipSmoother` these are user-chosen or uniform
            in x-range. For :class:`ArcLengthSmoother` they are the
            x-values at uniformly-spaced arc-length points, which
            means they are NOT uniformly spaced in x.
        y: Dependent variable values at the same sample points,
            aligned element-wise with ``x``.
        s: Optional arc-length values at each sample point, in the
            same (possibly normalized) units the smoother used
            internally. Populated by :class:`ArcLengthSmoother`;
            ``None`` for smoothers that do not compute arc length.

    Example:
        Typical use in an error-metric computation::

            smoother = PchipSmoother(n_samples=200)
            sim = smoother.smooth(sim_strain, sim_stress)
            exp = smoother.sample_at(exp_strain_orig, exp_stress_orig, sim.x)
            rmse = float(np.sqrt(((sim.y - exp.y)**2).mean()))
    """

    x: np.ndarray
    y: np.ndarray
    s: Optional[np.ndarray] = None


# --- Protocol ------------------------------------------------------------


@runtime_checkable
class Smoother(Protocol):
    """Protocol: anything that smooths and resamples an ``(x, y)`` curve.

    Any class with a ``smooth(x, y)`` method returning a
    :class:`SmoothedCurve` can be used as the framework's smoother.
    This means users can swap in their own implementations (spline
    fits, Savitzky-Golay filters, kernel smoothers, etc.) without
    modifying the framework.

    Methods:
        smooth(x, y):
            Given raw arrays of x and y values, return a smoothed
            and resampled :class:`SmoothedCurve`. The number of
            sample points and the sampling strategy (uniform in x,
            uniform in arc length, at user-specified x-values, etc.)
            are implementation choices baked into the Smoother at
            construction time.
    """

    def smooth(self, x: np.ndarray, y: np.ndarray) -> SmoothedCurve: ...


# --- Helpers -------------------------------------------------------------


def is_monotonic(x: np.ndarray, *, tol: float = 0.0) -> bool:
    """Return True if ``x`` is monotonically non-decreasing.

    "Monotonic" here means each value is greater than or equal to
    the previous, within ``tol``. A strictly-increasing check would
    reject typical experimental data, where noise can produce tiny
    backward steps that do not indicate real non-monotonicity. The
    default ``tol=0.0`` accepts plateaus (equal consecutive values)
    but rejects any backward motion; raising ``tol`` is appropriate
    for noisy input.

    Args:
        x: 1-D array of values.
        tol: Permitted backward step size. Values ``x[i] < x[i-1] -
            tol`` are considered a real decrease; anything within
            ``tol`` is tolerated. Must be non-negative.

    Returns:
        True if the sequence is monotonically non-decreasing within
        ``tol``, False otherwise.

    Example:
        Decide which smoother to use based on the shape of the data::

            if is_monotonic(strain, tol=1e-8):
                smoother = PchipSmoother(n_samples=200)
            else:
                smoother = ArcLengthSmoother(n_samples=200)
    """
    x = np.asarray(x)
    if x.size < 2:
        # A single point is vacuously monotonic; an empty array too.
        return True
    # diff < -tol means a real decrease beyond the noise tolerance.
    return bool(np.all(np.diff(x) >= -tol))


def arc_length(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_scale: Optional[float] = None,
    y_scale: Optional[float] = None,
) -> np.ndarray:
    """Compute the cumulative arc length along an ``(x, y)`` polyline.

    Arc length is the running sum of Euclidean segment lengths
    along the curve, starting at 0 for the first point. Optionally,
    x and y are divided by ``x_scale`` / ``y_scale`` before summing,
    which gives a dimensionless arc length suitable for resampling
    curves whose axes have wildly different units.

    Defaults (both scales ``None``) map both axes onto their own
    ranges, producing an arc length in the unit box [0, 1] x [0, 1].
    Pass explicit scales when you need two curves (e.g. sim and
    exp) to share the same arc-length parameterization.

    Args:
        x: 1-D array of x-values.
        y: 1-D array of y-values. Must have the same length as x.
        x_scale: Divisor applied to x before computing segment
            lengths. ``None`` (default) uses ``x.max() - x.min()``.
            Must be positive.
        y_scale: Divisor applied to y before computing segment
            lengths. ``None`` (default) uses ``y.max() - y.min()``.
            Must be positive.

    Returns:
        A 1-D array the same length as x. The first element is
        always 0, the last is the total normalized arc length.

    Raises:
        ValueError: If x and y have different lengths, if either
            scale is zero or negative, or if auto-detected scales
            would be zero (constant series).

    Example:
        ::

            s = arc_length(strain, stress)
            # s ranges from 0 at the start of loading to some value
            # reflecting the total path length in the normalized box.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have same shape, got {x.shape} vs {y.shape}"
        )

    # Default scale: data range. This is the natural choice because
    # it is reference-free and makes the resulting arc length
    # dimensionless and comparable across unit systems.
    if x_scale is None:
        x_scale = float(x.max() - x.min())
    if y_scale is None:
        y_scale = float(y.max() - y.min())
    if x_scale <= 0 or y_scale <= 0:
        raise ValueError(
            f"x_scale and y_scale must be positive; got {x_scale=}, {y_scale=}. "
            f"A zero range typically means a constant series that cannot be "
            f"arc-length parameterized; supply explicit positive scales."
        )

    dx = np.diff(x) / x_scale
    dy = np.diff(y) / y_scale
    seg = np.sqrt(dx * dx + dy * dy)
    # cumsum gives running total; prepend 0 so the array has the same
    # length as the input points (s[i] is the arc length from point 0
    # to point i, so s[0] == 0).
    return np.concatenate(([0.0], np.cumsum(seg)))


# --- PchipSmoother -------------------------------------------------------


class PchipSmoother:
    """Monotonic cubic Hermite (PCHIP) smoother for monotonic-x data.

    Uses ``scipy.interpolate.PchipInterpolator`` to build a smooth,
    non-overshooting interpolant of y vs x, then resamples it at a
    uniform grid (by default) or at user-specified target x values.

    When to use
        * The x-axis is monotonic (strain rarely decreases in a
          standard tensile test, time is always monotonic).
        * You want a smooth interpolant but refuse to let it
          introduce spurious extrema.

    When NOT to use
        * The x-axis has reversals. Use :class:`ArcLengthSmoother`
          instead - PCHIP will either crash or drop branches when
          x is not single-valued.

    Args:
        n_samples: Number of uniformly-spaced x-values to sample at
            when ``smooth()`` is called with no explicit target. The
            resulting x-grid spans ``[x.min(), x.max()]``. Default
            is 200, which is typically plenty for smooth mechanical
            responses.
        strict_monotonic: If True, raise ``ValueError`` when the
            input x is not monotonically non-decreasing. If False
            (default), allow near-monotonic input and let PCHIP
            handle any tiny violations gracefully via its internal
            sort.

    Example:
        Smooth a sim stress-strain curve onto 500 uniform strain
        points and then sample at the experimental points for an
        error metric::

            smoother = PchipSmoother(n_samples=500)
            sim = smoother.smooth(sim_eps, sim_sigma)
            exp_on_sim_grid = smoother.sample_at(
                exp_eps, exp_sigma, target_x=sim.x,
            )
    """

    def __init__(self, *, n_samples: int = 200, strict_monotonic: bool = False):
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        self._n_samples = n_samples
        self._strict = strict_monotonic

    def smooth(self, x: np.ndarray, y: np.ndarray) -> SmoothedCurve:
        """Smooth and resample on a uniform x-grid.

        Args:
            x: 1-D array of independent variable values. Must be
                monotonically non-decreasing (see
                ``strict_monotonic`` in the constructor).
            y: 1-D array of dependent variable values, same length
                as x.

        Returns:
            A :class:`SmoothedCurve` with ``n_samples`` points
            uniformly spaced between ``x.min()`` and ``x.max()``.
            The ``s`` field is ``None`` for PCHIP output.

        Raises:
            ValueError: If x is not monotonic and
                ``strict_monotonic`` was True; or if input arrays
                have mismatched or insufficient length.
        """
        x, y = self._coerce(x, y)
        # Uniform grid from x_min to x_max, inclusive of both
        # endpoints. linspace is stable at the boundaries so the
        # PCHIP interpolant is evaluated at exact input points (not
        # just "close to"), avoiding tiny numerical drift at the
        # endpoints.
        x_targets = np.linspace(x.min(), x.max(), self._n_samples)
        return self.sample_at(x, y, x_targets)

    def sample_at(
        self, x: np.ndarray, y: np.ndarray, x_targets: np.ndarray
    ) -> SmoothedCurve:
        """Build a PCHIP interpolant from (x, y) and evaluate at x_targets.

        Args:
            x: Source x-values (monotonic non-decreasing).
            y: Source y-values, aligned with x.
            x_targets: 1-D array of x-values at which to evaluate
                the interpolant. Should lie within
                ``[x.min(), x.max()]``; out-of-range values are
                extrapolated by scipy's PCHIP, which is well-defined
                but whose behavior should not be relied on far from
                the data.

        Returns:
            A :class:`SmoothedCurve` whose ``x`` is exactly
            ``x_targets``.
        """
        x, y = self._coerce(x, y)
        x_targets = np.asarray(x_targets, dtype=float)
        # PchipInterpolator constructs the interpolant once; evaluating
        # it at many points is O(log n) per point, so this is cheap
        # even for millions of target values.
        interp = PchipInterpolator(x, y, extrapolate=True)
        return SmoothedCurve(x=x_targets, y=interp(x_targets))

    def _coerce(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and normalize the input arrays.

        Private helper shared by ``smooth`` and ``sample_at``.
        Ensures the arrays are 1-D float numpy arrays of matching
        length, checks monotonicity if required, and performs a
        lexicographic sort if monotonicity is almost-but-not-quite
        satisfied (tiny noise) so that PchipInterpolator does not
        reject the input.

        Args:
            x: Raw input x-values.
            y: Raw input y-values.

        Returns:
            A tuple ``(x, y)`` of float arrays, sorted on x if a
            sort was needed.

        Raises:
            ValueError: On shape mismatch, too-few-points, or
                strict-monotonicity violation.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have same shape, got {x.shape} vs {y.shape}"
            )
        if x.size < 2:
            raise ValueError(f"need at least 2 points, got {x.size}")

        if not is_monotonic(x):
            if self._strict:
                raise ValueError(
                    "PchipSmoother requires monotonic non-decreasing x. "
                    "Set strict_monotonic=False to allow auto-sort, or use "
                    "ArcLengthSmoother for genuinely non-monotonic data."
                )
            # Auto-sort handles mild noise that produces tiny backward
            # steps in what should be a monotonic series. This is NOT
            # a substitute for using ArcLengthSmoother on real
            # non-monotonic data - sorting destroys branch information.
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]

        return x, y


# --- ArcLengthSmoother ---------------------------------------------------


class ArcLengthSmoother:
    """PCHIP smoother parameterized by normalized arc length.

    For curves where x is not single-valued (necking, snap-back,
    load-unload cycles), traditional y-vs-x interpolation breaks
    down. This smoother instead computes the cumulative arc length
    ``s`` along the curve, then PCHIP-interpolates both x and y as
    functions of s. The result can be resampled at uniform s-values
    to produce a visually-smooth, analytically-well-defined
    reparameterization of the original path.

    Normalization
        Arc length is computed after dividing x by ``x_scale`` and y
        by ``y_scale``. If scales are omitted, the data's own
        ranges are used, mapping the curve into a unit box. When
        comparing sim to experiment, supply the SAME explicit
        scales to both smoothers - typically the experimental
        ranges - so the parameterizations are commensurable.

    Args:
        n_samples: Number of uniformly-spaced arc-length values to
            sample at. Default 200.
        x_scale: Divisor for x in the arc-length integrand. ``None``
            uses the data range at smooth time.
        y_scale: Divisor for y in the arc-length integrand. ``None``
            uses the data range at smooth time.

    Example:
        A necking curve with strain decrease after the peak::

            smoother = ArcLengthSmoother(
                n_samples=300,
                x_scale=exp_eps.max() - exp_eps.min(),
                y_scale=exp_sig.max() - exp_sig.min(),
            )
            sim = smoother.smooth(sim_eps, sim_sig)
            exp = smoother.smooth(exp_eps, exp_sig)
            # sim.s and exp.s are now on compatible scales; compare
            # at matching arc-length indices.
    """

    def __init__(
        self,
        *,
        n_samples: int = 200,
        x_scale: Optional[float] = None,
        y_scale: Optional[float] = None,
    ):
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        self._n_samples = n_samples
        self._x_scale = x_scale
        self._y_scale = y_scale

    def smooth(self, x: np.ndarray, y: np.ndarray) -> SmoothedCurve:
        """Parameterize by arc length and resample uniformly in s.

        Args:
            x: 1-D array of x-values.
            y: 1-D array of y-values, same length as x.

        Returns:
            A :class:`SmoothedCurve` with ``n_samples`` points
            uniformly spaced in s. The ``s`` field holds the
            arc-length values at each sample point.

        Raises:
            ValueError: On shape mismatch or fewer than 2 points.
        """
        x, y, s = self._prepare(x, y)
        s_targets = np.linspace(s[0], s[-1], self._n_samples)
        return self.sample_at_s(x, y, s_targets)

    def sample_at_s(
        self, x: np.ndarray, y: np.ndarray, s_targets: np.ndarray
    ) -> SmoothedCurve:
        """Evaluate the arc-length parameterization at specific s-values.

        Useful for aligning two arc-length-smoothed curves so their
        i-th sample point corresponds to the same fraction of each
        curve's total path length. Pass e.g. ``np.linspace(0, 1,
        N)`` to sample at N equally-spaced fractions of total arc.

        Args:
            x: Source x-values.
            y: Source y-values.
            s_targets: 1-D array of arc-length values at which to
                evaluate. Should lie within ``[0, s_total]`` where
                s_total is the curve's total arc length.

        Returns:
            A :class:`SmoothedCurve` whose ``s`` is ``s_targets``.
        """
        x, y, s = self._prepare(x, y)
        s_targets = np.asarray(s_targets, dtype=float)
        # Two interpolants: one for x(s), one for y(s). Both are PCHIP
        # so neither overshoots - crucial for the x interpolant since
        # we are effectively reconstructing a (possibly non-monotonic)
        # x-series and spurious extrema would introduce fake load/unload
        # reversals.
        x_interp = PchipInterpolator(s, x, extrapolate=True)
        y_interp = PchipInterpolator(s, y, extrapolate=True)
        return SmoothedCurve(
            x=x_interp(s_targets),
            y=y_interp(s_targets),
            s=s_targets,
        )

    def _prepare(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate input and compute normalized cumulative arc length.

        Args:
            x: Raw x-values.
            y: Raw y-values.

        Returns:
            Tuple ``(x, y, s)`` as float arrays; ``s`` is the
            cumulative arc length at each (x, y) point.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have same shape, got {x.shape} vs {y.shape}"
            )
        if x.size < 2:
            raise ValueError(f"need at least 2 points, got {x.size}")

        # Arc length must be monotonic for PCHIP to accept it. In
        # principle the raw-geometry arc length is always monotonic,
        # but duplicate points (x[i] == x[i-1] and y[i] == y[i-1])
        # produce zero-length segments which some PCHIP versions
        # reject as "non-unique abscissae". Deduplicate here for
        # robustness against noisy or repeated samples.
        s = arc_length(x, y, x_scale=self._x_scale, y_scale=self._y_scale)
        unique = np.concatenate(([True], np.diff(s) > 0))
        if not unique.all():
            # Drop the duplicates but log if it was a lot - that
            # usually indicates a data quality problem worth flagging.
            dropped = int((~unique).sum())
            if dropped > x.size // 10:
                logger.warning(
                    "ArcLengthSmoother: dropped %d/%d duplicate points",
                    dropped, x.size,
                )
            x, y, s = x[unique], y[unique], s[unique]

        return x, y, s


# --- LegacyLinearSmoother ------------------------------------------------


class LegacyLinearSmoother:
    """Backwards-compatible piecewise-linear smoother.

    Reproduces the behavior of the pre-refactor
    ``smoothening_ss_data_fcn.py``: sort the input by x, then
    linearly interpolate onto a uniform grid between x.min() and
    x.max(). No spline, no shape preservation, no arc length.

    Kept for one specific purpose: regression testing. When the new
    PCHIP-based smoothers are dropped into an existing optimization,
    we want to first verify that migrating the rest of the stack
    (templates, paths, manifest, backend) leaves results
    numerically identical on monotonic data. Switching the smoother
    in the same migration would confound any differences. Once
    equivalence is established, the smoother can be swapped to
    PCHIP as a separate, reviewable change.

    Args:
        n_samples: Number of uniformly-spaced x-values to sample at.
            Default 200 matches the old code.

    Example:
        A/B test on a known dataset::

            legacy = LegacyLinearSmoother(n_samples=200)
            new = PchipSmoother(n_samples=200)
            a = legacy.smooth(x, y)
            b = new.smooth(x, y)
            # Differences between a.y and b.y should be small for
            # smooth, monotonic, dense data - larger at sparse or
            # corner regions where PCHIP's local slopes deviate
            # from the naive linear connect-the-dots.
    """

    def __init__(self, *, n_samples: int = 200):
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        self._n_samples = n_samples

    def smooth(self, x: np.ndarray, y: np.ndarray) -> SmoothedCurve:
        """Sort and linearly interpolate onto a uniform x-grid.

        Args:
            x: Raw x-values.
            y: Raw y-values, same length.

        Returns:
            A :class:`SmoothedCurve` on a uniform grid.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(
                f"x and y must have same shape, got {x.shape} vs {y.shape}"
            )
        if x.size < 2:
            raise ValueError(f"need at least 2 points, got {x.size}")

        # Stable sort keeps the original order of exactly-equal
        # x-values, which matters for some edge cases in the old code
        # where consecutive duplicate x's at different y's were
        # treated as a step.
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]

        x_targets = np.linspace(x.min(), x.max(), self._n_samples)
        # numpy.interp is exactly what the old code used - purely
        # piecewise linear with clipped extrapolation at the ends.
        return SmoothedCurve(x=x_targets, y=np.interp(x_targets, x, y))


# --- Convenience: auto-select smoother ------------------------------------


def auto_smoother(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int = 200,
    monotonic_tol: float = 0.0,
    x_scale: Optional[float] = None,
    y_scale: Optional[float] = None,
) -> Smoother:
    """Return a :class:`PchipSmoother` or :class:`ArcLengthSmoother` based on x.

    A small convenience for callers who want the framework to pick
    the right tool without building their own is-monotonic logic.
    Tests whether x is monotonically non-decreasing (within
    ``monotonic_tol``); if so, returns PCHIP, otherwise arc-length.

    Args:
        x: The x-values of the curve to smooth. Used only to inspect
            monotonicity.
        y: The y-values. Ignored for the monotonicity check but kept
            in the signature so callers can pass the same data they
            will later smooth.
        n_samples: How many points to sample at. Passed through to
            the chosen smoother. Default 200.
        monotonic_tol: Tolerance for the monotonicity check; see
            :func:`is_monotonic`.
        x_scale: If arc-length smoothing is chosen, use this as the
            x-scale. ``None`` defaults to ``x.max() - x.min()``.
        y_scale: Same, for y.

    Returns:
        A ready-to-use :class:`Smoother`.

    Example:
        ::

            smoother = auto_smoother(sim_eps, sim_sig, n_samples=300)
            sim = smoother.smooth(sim_eps, sim_sig)
    """
    if is_monotonic(x, tol=monotonic_tol):
        return PchipSmoother(n_samples=n_samples)
    return ArcLengthSmoother(
        n_samples=n_samples, x_scale=x_scale, y_scale=y_scale
    )
