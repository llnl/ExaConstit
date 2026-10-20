"""
Objective evaluation: turning simulation output into a number the
optimizer can minimize, and deciding what to do when a simulation fails.

Why this module exists
----------------------
The last stage of a workflow case converts the raw simulation output
(a DataFrame of stress vs. time) into a single scalar error value
the optimizer can compare against other genes. In the legacy
:class:`ExaProb`, this conversion was tangled up with directory
handling, result reading, and failure policy. Separating it into
small, independently-testable pieces makes each one clearer and
lets users swap in custom behavior without subclassing the whole
machinery.

Three abstractions:

* :class:`StressStrainExtractor` — knows how to derive a
  ``(strain, stress)`` pair from a :class:`CaseResultSet`. This is
  the one place in the pipeline that understands the simulation
  code's output conventions (is strain in the def-grad file? is it
  strain_rate * time? is it log-strain?). Swap it out for different
  codes or different loading conventions.

* :class:`ObjectiveEvaluator` (Protocol) — turns a
  :class:`CaseResultSet` into a single ``float``. The framework
  ships one concrete implementation, :class:`StressStrainObjective`,
  which handles the common case of "compare sim stress-strain to
  experimental stress-strain via RMSE". Users with bespoke metrics
  write their own class satisfying the Protocol.

* :class:`FailureHandler` (Protocol) — decides what objective value
  to return when a simulation fails. Three default policies ship:
  infinity, constant penalty, and partial-progress penalty. Which
  one you want depends on whether your optimizer can handle
  infinities (some can't) and whether you want to give it gradient
  information about how far the failed case got.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    Literal,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .paths import CaseContext
from .results import CaseResultSet
from .smoothing import PchipSmoother, Smoother

logger = get_logger(__name__)


# --- StressStrainExtractor ------------------------------------------------

# Literal type for strain derivation modes. Each name describes the
# *formula* applied to the configured strain-source column.
StrainSource = Literal[
    "biot",        # strain = X - 1, where X = column[strain_source_column]
                   #   The Biot strain measure for 1-D loading. Default.
    "log",         # strain = log(X). Hencky / logarithmic / true strain.
    "time_rate",   # strain = strain_rate * t. No strain-source file
                   #   needed; reads time from the stress file instead.
    "direct",      # strain = column itself. Use when the simulation
                   #   already writes a strain measure (Biot / Lagrange /
                   #   Euler / log) and you want the column verbatim.
]


@dataclass
class StressStrainExtractor:
    """Pulls ``(strain, stress)`` arrays out of a :class:`CaseResultSet`.

    ExaConstit (and similar FEM codes) writes several output files
    per case: ``avg_stress.txt`` for Cauchy-stress components,
    ``avg_def_grad.txt`` for deformation-gradient components, and
    optionally ``avg_lagrangian_strain.txt`` /
    ``avg_euler_strain.txt`` if the user enabled those calc types.
    Different users derive strain from these in different ways
    depending on the loading case AND on what their sim binary
    actually writes:

    * From a deformation gradient column ``X``:

      - Biot strain (small-strain or default uniaxial):
        ``strain = X - 1``  (``strain_source="biot"``)
      - Hencky / logarithmic strain (large-strain uniaxial):
        ``strain = log(X)`` (``strain_source="log"``)

    * From a constant-rate load with no def-grad on disk:
      ``strain = strain_rate * time`` (``strain_source="time_rate"``)

    * From a strain measure already on disk (Biot, Lagrange, Euler,
      Hencky — whichever the sim wrote):
      ``strain = X`` (``strain_source="direct"``)

    Default column names
    --------------------
    Defaults match ExaConstit's on-disk convention for the most
    common case: read ``F33`` from ``avg_def_grad.txt`` and apply
    Biot strain (``X - 1``). For other loading directions, override
    ``strain_source_column`` to ``F11``/``F22``. For the "direct
    strain" path, set ``strain_source_output`` to whichever strain
    file the sim wrote (e.g. ``"avg_lagrangian_strain"``) and
    ``strain_source_column`` to the relevant component
    (e.g. ``"E33"``).

    Stress columns: ExaConstit uses ``Sxx``/``Syy``/``Szz``/``Sxy``
    /``Sxz``/``Syz``. See
    ``ExaConstit/src/postprocessing/postprocessing_file_manager.hpp``
    :: ``GetVolumeAverageHeader`` for the authoritative list.

    Optional window
    ---------------
    ``window=(min, max)`` crops the returned ``(strain, stress)``
    arrays to a strain interval. This is the right place to encode
    "fit only the plastic regime" or "ignore the elastic-plastic
    transition" — the optimizer scoring against the cropped arrays
    will only see the user-relevant part of the curve. Without a
    window, the optimizer is at the mercy of whatever shape
    dominates the full curve (often the elastic regime, which
    isn't even what the gene parameters control).

    Pass an absolute-value interval; the extractor compares against
    ``|strain|`` so the window works for both compression and
    tension loadings without the user having to think about sign.

    Fields:
        stress_output: Logical output name (as registered on the
            :class:`PathResolver`) where the Cauchy-stress table
            lives. Default ``"avg_stress"``.
        stress_column: Column name within the stress DataFrame
            whose values are the stress of interest. Default
            ``"Szz"`` — z-axis normal stress.
        strain_source: Which strain-derivation method to use.
            Default ``"biot"`` (``X - 1``).
        strain_rate: Required when ``strain_source="time_rate"``;
            ignored otherwise.
        strain_source_output: Logical output name for the file
            containing the strain-source column. Default
            ``"avg_def_grad"`` because the def-grad path is the
            most common. For ``strain_source="direct"`` set this
            to whatever strain file the sim wrote
            (e.g. ``"avg_lagrangian_strain"``); for
            ``strain_source="time_rate"`` this field is unused.
        strain_source_column: Column name within the strain-source
            file. Default ``"F33"`` (matches ``stress_column="Szz"``
            for z-axis loading). Override for other loading
            directions or for ``direct`` mode.
        time_column: Column name that holds time values in the
            stress DataFrame. Used only by ``time_rate``. Default
            ``"Time"``.
        window: Optional ``(min, max)`` strain interval to crop to.
            Default ``None`` (no cropping). Compared against
            ``|strain|`` so the same value works for tension and
            compression. ``min=0.0`` is fine; the elastic region
            is typically below ~0.002 strain so a window of
            ``(0.005, 0.1)`` skips the elastic regime entirely.

    Example:
        ExaConstit z-axis uniaxial, default Biot strain::

            extractor = StressStrainExtractor()
            strain, stress = extractor.extract(results)

        Hencky / logarithmic strain::

            extractor = StressStrainExtractor(strain_source="log")

        Constant-rate load, no def-grad output::

            extractor = StressStrainExtractor(
                strain_source="time_rate", strain_rate=1e-3,
            )

        Direct strain from an Euler strain output::

            extractor = StressStrainExtractor(
                strain_source="direct",
                strain_source_output="avg_euler_strain",
                strain_source_column="E33",
            )

        Constrain optimization to plastic regime only::

            extractor = StressStrainExtractor(
                window=sim_case.case_data["minmax_strain"],
            )
    """

    stress_output: str = "avg_stress"
    stress_column: str = "Szz"
    strain_source: StrainSource = "biot"
    strain_rate: Optional[float] = None
    strain_source_output: str = "avg_def_grad"
    strain_source_column: str = "F33"
    time_column: str = "Time"
    window: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-friendly dict for archive storage.

        Tuples are turned into lists (the only common JSON-unfriendly
        thing in the dataclass); :meth:`from_dict` reverses that on
        load. Used by the driver to record the extractor config
        alongside experimental data, so post-run plotters can
        reconstruct (strain, stress) curves the same way the
        optimizer scored them.
        """
        out: Dict[str, object] = {
            "stress_output": self.stress_output,
            "stress_column": self.stress_column,
            "strain_source": self.strain_source,
            "strain_rate": self.strain_rate,
            "strain_source_output": self.strain_source_output,
            "strain_source_column": self.strain_source_column,
            "time_column": self.time_column,
            "window": (
                None if self.window is None
                else [self.window[0], self.window[1]]
            ),
        }
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "StressStrainExtractor":
        """Reconstruct from a :meth:`to_dict` round-trip.

        Tolerant of extra keys (silently ignored) and missing
        keys (filled from defaults), so an archive written by an
        older code version with fewer fields still loads cleanly
        on a newer plotter.
        """
        # Filter to known fields so a future schema addition on the
        # archive doesn't crash an older code's from_dict.
        known = {
            "stress_output", "stress_column", "strain_source",
            "strain_rate", "strain_source_output",
            "strain_source_column", "time_column", "window",
        }
        kwargs = {k: v for k, v in d.items() if k in known}
        # Window comes back as a list from JSON; convert to the
        # tuple shape the dataclass declares.
        if kwargs.get("window") is not None:
            w = kwargs["window"]
            kwargs["window"] = (w[0], w[1])
        return cls(**kwargs)

    def extract(self, results: CaseResultSet) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(strain, stress)`` arrays for one case.

        Args:
            results: The :class:`CaseResultSet` returned by a
                :class:`~workflow_common.results.ResultReader`.

        Returns:
            A tuple ``(strain, stress)`` of 1-D float arrays with
            matching length. Strain is derived per the configured
            ``strain_source``. If a ``window`` is configured, the
            arrays are cropped to the strain interval (compared
            against ``|strain|``) before return.

        Raises:
            KeyError: If a configured output / column is missing
                from the result set.
            ValueError: For misconfigured strain sources, non-finite
                values, or a window that clips out every data point.
        """
        # Stress is always read from stress_output / stress_column,
        # regardless of which strain source we use.
        stress_df = results.df(self.stress_output)
        if self.stress_column not in stress_df.columns:
            raise KeyError(
                f"stress column {self.stress_column!r} not in "
                f"{self.stress_output!r} (columns: {list(stress_df.columns)})"
            )
        stress = stress_df[self.stress_column].to_numpy()

        # Strain derivation branches on strain_source. Every branch
        # ends with ``strain`` populated and the same length as
        # ``stress``.
        source = self.strain_source
        if source == "time_rate":
            # Constant-rate load — read time from the stress file.
            # Only valid if the sim ran a constant rate from t=0;
            # any ramp-up / hold makes this measure inaccurate.
            if self.strain_rate is None:
                raise ValueError(
                    "strain_source='time_rate' requires strain_rate "
                    "to be set on the extractor"
                )
            if self.time_column not in stress_df.columns:
                raise KeyError(
                    f"time column {self.time_column!r} not in "
                    f"{self.stress_output!r} "
                    f"(columns: {list(stress_df.columns)})"
                )
            strain = self.strain_rate * stress_df[self.time_column].to_numpy()
        elif source in ("biot", "log", "direct"):
            # All three read a column from strain_source_output.
            # Behavior differs only in the formula applied below.
            if self.strain_source_output not in results:
                raise KeyError(
                    f"strain_source={source!r} needs the "
                    f"{self.strain_source_output!r} output, but it "
                    f"is not in the result set. Either supply it as "
                    f"required on the reader, or pick a different "
                    f"strain_source."
                )
            src_df = results.df(self.strain_source_output)
            if self.strain_source_column not in src_df.columns:
                raise KeyError(
                    f"strain-source column {self.strain_source_column!r} "
                    f"not in {self.strain_source_output!r} "
                    f"(columns: {list(src_df.columns)})"
                )
            col = src_df[self.strain_source_column].to_numpy()
            if source == "biot":
                # Biot strain (a.k.a. engineering strain in 1D):
                # E_B = X - 1 where X is an axial stretch.
                strain = col - 1.0
            elif source == "log":
                # Hencky / true / logarithmic strain. Stretch must
                # be > 0 everywhere; a zero/negative stretch is
                # nonsensical for a physical deformation.
                if np.any(col <= 0):
                    raise ValueError(
                        f"strain_source='log' requires "
                        f"{self.strain_source_column} > 0 everywhere, "
                        f"but found min({self.strain_source_column})="
                        f"{col.min()}"
                    )
                strain = np.log(col)
            else:  # source == "direct"
                # The column already IS a strain measure (Biot,
                # Lagrange, Euler, Hencky — whichever the sim wrote).
                # Use verbatim.
                strain = col.copy()
        else:
            raise ValueError(
                f"unknown strain_source {self.strain_source!r}. "
                f"Valid options: 'biot', 'log', 'time_rate', 'direct'."
            )

        if not np.all(np.isfinite(strain)) or not np.all(np.isfinite(stress)):
            # Non-finite values almost always mean the sim diverged
            # or wrote truncated data. Fail loudly so the caller can
            # invoke the FailureHandler path rather than silently
            # propagating NaNs into the optimizer.
            raise ValueError(
                "extracted strain or stress contains NaN/inf - "
                "simulation likely diverged or output was truncated"
            )

        # Apply optional window. Compare against |strain| so the
        # window applies symmetrically to tension (positive strain)
        # and compression (negative strain) without the user having
        # to think about sign.
        if self.window is not None:
            lo, hi = self.window
            if lo > hi:
                raise ValueError(
                    f"window lower bound {lo} exceeds upper bound {hi}"
                )
            mask = (np.abs(strain) >= lo) & (np.abs(strain) <= hi)
            if not np.any(mask):
                raise ValueError(
                    f"window {self.window} excluded every data point "
                    f"(extracted |strain| range was "
                    f"[{float(np.abs(strain).min())}, "
                    f"{float(np.abs(strain).max())}])"
                )
            strain = strain[mask]
            stress = stress[mask]

        return strain, stress


# --- ObjectiveEvaluator --------------------------------------------------


@runtime_checkable
class ObjectiveEvaluator(Protocol):
    """Protocol: anything that turns a :class:`CaseResultSet` into a scalar.

    The optimizer only knows about numbers. An evaluator is the
    bridge: given the parsed output of one simulation run and the
    context describing which case it was, produce one ``float`` the
    optimizer can minimize.

    Methods:
        evaluate(results, ctx):
            Compute and return the objective value for this case.
            May raise on unrecoverable errors; the
            :class:`FailureHandler` path should be preferred for
            expected failures so that the optimizer still gets a
            meaningful value.
    """

    def evaluate(
        self, results: CaseResultSet, ctx: CaseContext
    ) -> float: ...


# --- Error metrics for StressStrainObjective -----------------------------


# Callable protocol for user-supplied metrics: takes two same-length
# arrays (sim, exp) already aligned onto a common grid, returns a
# scalar. Kept as a plain Callable type rather than a named Protocol
# because the one-method signature is simple enough that explicit
# naming only adds friction.
ErrorMetric = Callable[[np.ndarray, np.ndarray], float]


def rmse(sim: np.ndarray, exp: np.ndarray) -> float:
    """Root-mean-square error between two aligned arrays.

    Args:
        sim: Simulated values.
        exp: Experimental values. Must be the same shape as sim.

    Returns:
        ``sqrt(mean((sim - exp)^2))`` as a plain Python float.
    """
    residual = sim - exp
    return float(np.sqrt(np.mean(residual * residual)))


def mae(sim: np.ndarray, exp: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(sim - exp)))


def max_abs_error(sim: np.ndarray, exp: np.ndarray) -> float:
    """Maximum absolute error (L-infinity norm of residuals)."""
    return float(np.max(np.abs(sim - exp)))


# Table mapping the shorthand string names to the functions above.
# Exposed as a module constant so tests can iterate over all supported
# metrics, and so users can register their own by mutating the dict.
ERROR_METRICS: "dict[str, ErrorMetric]" = {
    "rmse": rmse,
    "mae": mae,
    "max_abs": max_abs_error,
}


# --- StressStrainObjective -----------------------------------------------


@dataclass
class StressStrainObjective:
    """Concrete :class:`ObjectiveEvaluator` for stress-strain comparisons.

    Pipeline, per evaluation:

    1. Extract ``(sim_strain, sim_stress)`` from the result set
       using the configured :class:`StressStrainExtractor`.
    2. Clip both sim and experimental curves to their common strain
       range so neither is extrapolated.
    3. Smooth both curves onto a common uniform strain grid using
       the configured :class:`Smoother`.
    4. Compute an error via the configured metric and return it.

    Fields:
        experimental: Pre-loaded experimental reference data as a
            :class:`pandas.DataFrame`. Must contain columns
            identified by ``experimental_strain_col`` and
            ``experimental_stress_col``.
        extractor: :class:`StressStrainExtractor` describing how to
            pull strain and stress from the simulation's output.
        experimental_strain_col: Strain column name in the exp data.
            Default ``"strain"``.
        experimental_stress_col: Stress column name in the exp data.
            Default ``"stress"``.
        smoother: A :class:`Smoother`. Defaults to a
            :class:`PchipSmoother` with ``n_samples=200``. For
            non-monotonic data (necking, snap-back), pass an
            :class:`~workflow_common.smoothing.ArcLengthSmoother`
            with explicit ``x_scale`` / ``y_scale``.
        n_samples: Number of points on the common grid. Only used
            when the smoother exposes a ``sample_at`` method (which
            both PCHIP and legacy smoothers do). Default 200.
        metric: Either one of the string names in
            :data:`ERROR_METRICS` (``"rmse"``, ``"mae"``,
            ``"max_abs"``) or a custom callable taking
            ``(sim_array, exp_array)`` and returning a float.

    Example:
        Default RMSE objective::

            evaluator = StressStrainObjective(
                experimental=load_experimental_csv(
                    "exp.txt", columns=["strain", "stress"],
                ),
                extractor=StressStrainExtractor(),
            )
            err = evaluator.evaluate(results, ctx)

        Custom metric, e.g. RMSE normalized by the peak exp stress::

            def normalized_rmse(sim, exp):
                peak = float(np.max(np.abs(exp)))
                return rmse(sim, exp) / peak if peak > 0 else 0.0

            evaluator = StressStrainObjective(
                experimental=exp_df,
                extractor=extractor,
                metric=normalized_rmse,
            )
    """

    experimental: pd.DataFrame
    extractor: StressStrainExtractor
    experimental_strain_col: str = "strain"
    experimental_stress_col: str = "stress"
    smoother: Smoother = field(default_factory=lambda: PchipSmoother(n_samples=200))
    n_samples: int = 200
    metric: "str | ErrorMetric" = "rmse"

    def evaluate(
        self, results: CaseResultSet, ctx: CaseContext
    ) -> float:
        """Compute the stress-strain error for one case.

        Args:
            results: The simulation's parsed outputs.
            ctx: The case's context (unused by this evaluator but
                part of the Protocol signature; subclasses may
                use it to log or branch on case identity).

        Returns:
            The error value according to the configured metric.

        Raises:
            ValueError: If sim and exp have no strain overlap, or
                if the chosen metric string is unrecognized.
            KeyError: On missing columns in sim or exp data.
        """
        # Step 1: extract.
        sim_strain, sim_stress = self.extractor.extract(results)

        # Step 2: load experimental columns into plain arrays for
        # speed and to sidestep pandas indexing subtleties.
        if self.experimental_strain_col not in self.experimental.columns:
            raise KeyError(
                f"strain column {self.experimental_strain_col!r} not in "
                f"experimental DataFrame "
                f"(columns: {list(self.experimental.columns)})"
            )
        if self.experimental_stress_col not in self.experimental.columns:
            raise KeyError(
                f"stress column {self.experimental_stress_col!r} not in "
                f"experimental DataFrame "
                f"(columns: {list(self.experimental.columns)})"
            )
        exp_strain = self.experimental[self.experimental_strain_col].to_numpy()
        exp_stress = self.experimental[self.experimental_stress_col].to_numpy()

        # Step 3: compute common strain range. We clip to the
        # overlap so that neither side is extrapolated. Extrapolating
        # a PCHIP (or anything) far beyond its source data is almost
        # always a silent source of bad objective values.
        strain_min = max(float(sim_strain.min()), float(exp_strain.min()))
        strain_max = min(float(sim_strain.max()), float(exp_strain.max()))
        if strain_min >= strain_max:
            raise ValueError(
                f"sim and exp have no strain overlap: "
                f"sim=[{sim_strain.min()}, {sim_strain.max()}], "
                f"exp=[{exp_strain.min()}, {exp_strain.max()}]"
            )

        common = np.linspace(strain_min, strain_max, self.n_samples)

        # Step 4: smooth both curves at the common grid. Using
        # sample_at if the smoother supports it (PCHIP and Legacy
        # both do); otherwise we fall back to smoothing onto the
        # default uniform grid, which may not exactly match common
        # and produces approximate alignment. A more principled
        # approach would be to interpolate the smoothed output onto
        # common - left as a refinement when a real arc-length use
        # case shows up.
        sim_smooth_fn = getattr(self.smoother, "sample_at", None)
        if callable(sim_smooth_fn):
            sim_curve = sim_smooth_fn(sim_strain, sim_stress, common)
            exp_curve = sim_smooth_fn(exp_strain, exp_stress, common)
            sim_y = sim_curve.y
            exp_y = exp_curve.y
        else:
            sim_y = self.smoother.smooth(sim_strain, sim_stress).y
            exp_y = self.smoother.smooth(exp_strain, exp_stress).y
            if sim_y.shape != exp_y.shape:
                # Should not happen in practice because smoothers emit
                # their configured n_samples, but guard anyway.
                raise ValueError(
                    f"smoother produced mismatched shapes "
                    f"{sim_y.shape} vs {exp_y.shape}"
                )

        # Step 5: compute the metric.
        metric_fn = self._resolve_metric()
        return metric_fn(sim_y, exp_y)

    def _resolve_metric(self) -> ErrorMetric:
        """Turn a string shortcut or callable into the actual function.

        Private helper so the resolution logic is co-located with
        the one place it matters.
        """
        if callable(self.metric):
            return self.metric
        try:
            return ERROR_METRICS[self.metric]
        except KeyError:
            raise ValueError(
                f"unknown metric {self.metric!r}; known: {sorted(ERROR_METRICS)}"
            ) from None


# --- FailureHandler ------------------------------------------------------


@runtime_checkable
class FailureHandler(Protocol):
    """Protocol: decides what objective value a failed case should produce.

    When a simulation fails, the optimizer still needs a number to
    rank the gene. Different strategies make sense for different
    optimizers:

    * **Infinity** - signals "don't ever pick this gene" to
      optimizers that handle inf cleanly. Many genetic algorithms
      do, most gradient-based optimizers do not.
    * **Constant penalty** - a large finite value. Safe for
      optimizers that choke on inf, but treats all failures the
      same regardless of how bad they were.
    * **Partial-progress penalty** - if the sim produced some output
      before dying, read the partial data and compute an error
      scaled by how far the sim got. Gives the optimizer gradient
      information to move away from broken regions of parameter
      space.

    Methods:
        on_failure(ctx, reason, partial_results):
            Called by the :class:`Problem` when a case fails.
            ``partial_results`` is a best-effort
            :class:`CaseResultSet` that may be None if the
            simulation produced no readable outputs at all.
            Returns the objective value to record.
    """

    def on_failure(
        self,
        ctx: CaseContext,
        reason: str,
        partial_results: Optional[CaseResultSet],
    ) -> float: ...


@dataclass
class InfinityFailureHandler:
    """Return :data:`numpy.inf` for every failure.

    The simplest policy. Suitable for any optimizer that can handle
    infinity (NSGA-III can) - it marks the gene as unambiguously
    worse than any successful one without introducing scale-dependent
    penalties.

    Args:
        value: The value returned on failure. Defaults to
            ``numpy.inf`` but callers can pass ``1e18`` or similar
            if they want a "functionally infinite" finite number.
    """

    value: float = float("inf")

    def on_failure(
        self,
        ctx: CaseContext,
        reason: str,
        partial_results: Optional[CaseResultSet],
    ) -> float:
        logger.info(
            "InfinityFailureHandler: gen=%d gene=%d obj=%d returning %g (%s)",
            ctx.generation, ctx.gene, ctx.obj, self.value, reason,
        )
        return self.value


@dataclass
class ConstantPenaltyFailureHandler:
    """Return a fixed large value on failure.

    Good for optimizers that cannot handle infinity or for runs
    where you want to keep failed genes in the visible range of
    the objective values (e.g. for plotting).

    Args:
        penalty: The fixed value returned on failure. Choose
            something well above the worst successful-case value
            you expect, so failures are always ranked worse.
    """

    penalty: float = 1.0e9

    def on_failure(
        self,
        ctx: CaseContext,
        reason: str,
        partial_results: Optional[CaseResultSet],
    ) -> float:
        logger.info(
            "ConstantPenaltyFailureHandler: gen=%d gene=%d obj=%d returning %g (%s)",
            ctx.generation, ctx.gene, ctx.obj, self.penalty, reason,
        )
        return self.penalty


@dataclass
class PartialProgressFailureHandler:
    """Compute a penalty proportional to how far the simulation got.

    If the sim produced at least some output before dying, this
    handler reads the partial data, works out how far along the
    expected strain range it made it, and returns a penalty that
    gets worse for earlier failures.

    The motivation: optimizers with no gradient information (pure GA)
    benefit from being able to tell "this gene failed at strain 0.3"
    vs "this gene failed at strain 0.01". The former is closer to a
    working configuration than the latter, so it should rank better
    even though both failed.

    Args:
        inner_evaluator: An evaluator to run on the partial data
            when available. The inner result is added to the
            progress-based penalty so genes that got most of the
            way and had small residual error rank better than
            genes that got most of the way but were also wrong.
        base_penalty: The cost of total failure (sim produced no
            usable output). Typical value: several times the worst
            successful error you expect.
        progress_weight: How strongly progress-shortage is
            penalized. The final value is:
            ``base_penalty * (1 - progress_fraction) * progress_weight
            + inner_err * progress_fraction``
            where progress_fraction is in [0, 1].
        strain_target: Expected final strain. ``None`` (default)
            asks the inner evaluator's extractor for the
            experimental max strain if available; otherwise falls
            back to 1.0 as a harmless scale.

    Example:
        Use with a StressStrainObjective as the inner evaluator::

            handler = PartialProgressFailureHandler(
                inner_evaluator=evaluator,
                base_penalty=1.0e6,
                strain_target=1.0,
            )

        A gene that made it to strain 0.7 with inner error 50k
        receives:
        ``1e6 * (1 - 0.7) * 1.0 + 50_000 * 0.7 = 335_000``.
        A gene that made it only to strain 0.1 with inner error
        50k receives:
        ``1e6 * 0.9 * 1.0 + 50_000 * 0.1 = 905_000``.
        The further-along gene ranks better.
    """

    inner_evaluator: ObjectiveEvaluator
    base_penalty: float = 1.0e9
    progress_weight: float = 1.0
    strain_target: Optional[float] = None

    def on_failure(
        self,
        ctx: CaseContext,
        reason: str,
        partial_results: Optional[CaseResultSet],
    ) -> float:
        # Without partial results, we have no progress information.
        # Emit a full penalty - same cost as the worst possible
        # progress-weighted case.
        if partial_results is None or not partial_results.tables:
            logger.info(
                "PartialProgressFailureHandler: gen=%d gene=%d obj=%d "
                "no partial output; penalty=%g",
                ctx.generation, ctx.gene, ctx.obj, self.base_penalty,
            )
            return self.base_penalty

        # Try to extract. If extraction itself fails (truncated def-grad
        # file, missing columns), we cannot compute progress - treat as
        # total failure.
        try:
            # Attribute access; evaluators are duck-typed so we do not
            # insist on a specific class. Most evaluators have an
            # "extractor" attribute; if yours does not, supply a
            # custom handler.
            extractor = getattr(self.inner_evaluator, "extractor", None)
            if extractor is None:
                logger.warning(
                    "PartialProgressFailureHandler: inner evaluator has no "
                    "'extractor' attribute; cannot estimate progress"
                )
                return self.base_penalty
            sim_strain, _ = extractor.extract(partial_results)
        except (KeyError, ValueError) as e:
            logger.info(
                "PartialProgressFailureHandler: extraction from partial "
                "results failed (%s); penalty=%g", e, self.base_penalty,
            )
            return self.base_penalty

        # Determine the target strain. Prefer experimental data's max
        # strain so we compare against the full intended range.
        target = self.strain_target
        if target is None:
            exp = getattr(self.inner_evaluator, "experimental", None)
            strain_col = getattr(
                self.inner_evaluator, "experimental_strain_col", "strain"
            )
            if exp is not None and strain_col in exp.columns:
                target = float(exp[strain_col].max())
            else:
                target = 1.0  # harmless fallback

        if target <= 0:
            # Cannot compute a fraction from a nonsensical target.
            return self.base_penalty

        sim_max = float(sim_strain.max())
        fraction = max(0.0, min(1.0, sim_max / target))

        # Try computing an inner error on whatever we have. If the
        # evaluator's comparison logic needs more strain range than
        # we have, it will raise; fall back to pure progress penalty.
        try:
            inner_err = float(self.inner_evaluator.evaluate(partial_results, ctx))
        except Exception as e:
            logger.debug(
                "PartialProgressFailureHandler: inner evaluator raised "
                "on partial data (%s); using pure progress penalty", e,
            )
            inner_err = 0.0

        penalty = (
            self.base_penalty * (1.0 - fraction) * self.progress_weight
            + inner_err * fraction
        )
        logger.info(
            "PartialProgressFailureHandler: gen=%d gene=%d obj=%d "
            "progress=%.2f penalty=%g (%s)",
            ctx.generation, ctx.gene, ctx.obj, fraction, penalty, reason,
        )
        return float(penalty)
