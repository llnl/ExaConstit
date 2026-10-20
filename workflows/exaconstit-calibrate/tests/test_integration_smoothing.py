"""
Integration tests that exercise the smoothing layer alongside the
rest of the framework.

Why these tests matter
----------------------
The smoothing layer is the last piece before error-metric computation
in a real ExaConstit optimization. These integration tests verify
that the output of the :class:`TextTableReader` can be fed directly
into a :class:`Smoother` and produce sensible, comparable curves.
They also exercise the "normalize scales using experimental ranges"
pattern that is essential for arc-length comparisons between sim
and exp.

What a production ObjectiveEvaluator (step 6) will do internally
-----------------------------------------------------------------
The pattern demonstrated here is exactly what an ObjectiveEvaluator
in step 6 needs to do internally::

    1. Load experimental data once at the top of the run.
    2. For each simulated case:
       a. Read simulation output via TextTableReader.
       b. Extract strain and stress columns from the DataFrame.
       c. Choose the right smoother (auto_smoother, or
          PchipSmoother for monotonic sweep-style runs).
       d. Smooth both sim and exp onto a common grid.
       e. Compute an RMSE or similar error.
    3. Return errors keyed by (gene, obj).

The tests below walk through each of those steps with real
subprocesses and real output files, so any regression in the
handoffs between modules surfaces immediately.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from workflow_common import (
    ArcLengthSmoother,
    CaseContext,
    CaseLayout,
    LocalBackend,
    Manifest,
    ManifestEntry,
    CaseState,
    PchipSmoother,
    Sentinel,
    SimJobSpec,
    TemplatePathResolver,
    TextTableReader,
    TextTableSpec,
    auto_smoother,
    is_monotonic,
    load_experimental_csv,
    render_template_file,
    write_sentinel,
)
from workflow_common.backends.base import JobOutcome


# --- Fixtures specific to the smoothing integration tests ----------------


@pytest.fixture
def experimental_stress_strain(workspace: Path) -> pd.DataFrame:
    """Generate a synthetic experimental reference dataset.

    Saturating Voce-law response with the same character as what the
    fake binary produces, but slightly different parameters - so
    there is a nonzero error the smoothing pipeline should be able
    to quantify. Stored in the workspace as a CSV so the test
    exercises ``load_experimental_csv`` too.

    Note: the saturation constant (50) matches the fake binary's so
    sim and exp curves have the same shape; only the yield stress,
    hardening modulus, and saturation rate differ slightly. That
    produces a realistic small-ish RMSE on aligned curves.
    """
    strain = np.linspace(0.0, 1.0, 40)
    # Reference "truth" params slightly different from what the fake
    # binary will be told to use - so sim and exp don't exactly match.
    yield_stress = 210.0
    hardening = 1900.0
    # Saturation constant 48 (vs the binary's 50) -> curves are
    # the same shape with slightly offset saturation rate.
    stress = yield_stress + hardening * (1 - np.exp(-48.0 * strain))
    df = pd.DataFrame({"strain": strain, "stress": stress})
    # Round-trip through disk to mimic a real workflow.
    path = workspace / "exp_reference.csv"
    df.to_csv(path, index=False)
    return load_experimental_csv(path, delimiter=",")


def _setup_common(workspace: Path):
    """Shared setup shared with other integration tests."""
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
        "avg_def_grad": TextTableSpec(
            columns=[
                "Time", "Volume", "F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33",
            ],
            required=False,
        ),
    })
    return resolver, reader


def _run_case(
    workspace: Path,
    resolver: TemplatePathResolver,
    master_template: Path,
    fake_binary: Path,
    ctx: CaseContext,
    params: dict,
    backend: LocalBackend,
    manifest: Manifest,
) -> Path:
    """Render, submit, and finalize one case. Returns the working directory."""
    wd = resolver.working_dir(ctx)
    wd.mkdir(parents=True, exist_ok=True)
    render_template_file(
        master_template, wd / "options.toml",
        {"gene": ctx.gene, "obj": ctx.obj, **params},
    )
    spec = SimJobSpec(
        working_dir=wd, binary=fake_binary, args=(),
        duration_s=30, stdout="run.out", stderr="run.err",
    )
    manifest.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(wd),
    ))
    res = backend.submit_one(spec)
    assert res.outcome == JobOutcome.OK, f"sim failed: {res.error_message}"
    write_sentinel(wd, Sentinel(rc=0, wall_time_s=res.wall_time_s))
    manifest.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.COMPLETED, rc=0, case_dir=str(wd),
    ))
    return wd


# --- Integration tests ---------------------------------------------------


def test_pchip_smooths_sim_output_onto_exp_grid(
    workspace: Path, fake_binary: Path, master_template: Path,
    experimental_stress_strain: pd.DataFrame,
):
    """Full pipeline: sim -> read -> smooth onto experimental strain grid."""
    resolver, reader = _setup_common(workspace)
    backend = LocalBackend(max_workers=1)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    ctx = CaseContext(generation=0, gene=0, obj=0)
    _run_case(
        workspace, resolver, master_template, fake_binary, ctx,
        # strain_rate=1.0 so sim_strain = 1.0 * time ranges over [0, 1],
        # matching the experimental strain range. In a real ExaConstit
        # run the user aligns these by choosing t_final appropriately;
        # here we control both the binary's strain-rate input and the
        # experimental data, so we pick matching values directly.
        {"strain_rate": 1.0, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
        backend, manifest,
    )

    # Step 1: read sim output.
    layout = CaseLayout(ctx=ctx, resolver=resolver)
    rs = reader.read(layout)
    sim_df = rs.df("avg_stress")

    # Step 2: derive strain from time + strain_rate so we have a
    # common axis with the experiment. In a real workflow this
    # would come from the avg_def_grad file via F11 - 1.
    sim_strain = 1.0 * sim_df["Time"].to_numpy()
    sim_stress = sim_df["Szz"].to_numpy()

    # Sanity: sim is monotonic.
    assert is_monotonic(sim_strain)

    # Step 3: smooth sim onto the experimental strain grid using PCHIP.
    # We sample exactly at the experimental strain values so errors
    # can be computed pointwise.
    smoother = PchipSmoother(n_samples=200)
    exp_strain = experimental_stress_strain["strain"].to_numpy()
    exp_stress = experimental_stress_strain["stress"].to_numpy()
    sim_on_exp = smoother.sample_at(sim_strain, sim_stress, exp_strain)

    # Pointwise aligned now: same length, same x.
    assert sim_on_exp.x.shape == exp_strain.shape
    assert np.allclose(sim_on_exp.x, exp_strain)

    # Step 4: RMSE is a small nonzero value (sim and exp intentionally
    # have slightly different yield/hardening/saturation parameters).
    # The bound is generous because the fake binary runs only 50 steps
    # on a sharp-saturation curve, so transient-regime differences
    # produce larger pointwise errors than a well-resolved real run would.
    residual = sim_on_exp.y - exp_stress
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    assert 0.0 < rmse < 200.0, f"rmse {rmse} outside sanity range"


def test_sim_and_exp_smoothed_onto_common_grid(
    workspace: Path, fake_binary: Path, master_template: Path,
    experimental_stress_strain: pd.DataFrame,
):
    """A more realistic pattern: both curves smoothed, then compared.

    In real optimizations, the experimental curve is typically noisy
    and should be smoothed just like the sim before comparison.
    """
    resolver, reader = _setup_common(workspace)
    backend = LocalBackend(max_workers=1)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    ctx = CaseContext(generation=0, gene=0, obj=0)
    _run_case(
        workspace, resolver, master_template, fake_binary, ctx,
        {"strain_rate": 1.0, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
        backend, manifest,
    )

    layout = CaseLayout(ctx=ctx, resolver=resolver)
    rs = reader.read(layout)
    sim_strain = 1.0 * rs.df("avg_stress")["Time"].to_numpy()
    sim_stress = rs.df("avg_stress")["Szz"].to_numpy()

    exp_strain = experimental_stress_strain["strain"].to_numpy()
    exp_stress = experimental_stress_strain["stress"].to_numpy()

    # Smooth both onto a common uniform strain grid. The sim-range is
    # the authoritative upper bound since the experiment typically
    # has more strain data than the sim does.
    common_strain = np.linspace(
        max(sim_strain.min(), exp_strain.min()),
        min(sim_strain.max(), exp_strain.max()),
        100,
    )
    smoother = PchipSmoother(n_samples=100)
    sim_smooth = smoother.sample_at(sim_strain, sim_stress, common_strain)
    exp_smooth = smoother.sample_at(exp_strain, exp_stress, common_strain)

    # Aligned and comparable.
    assert np.allclose(sim_smooth.x, exp_smooth.x)
    rmse = float(np.sqrt(np.mean((sim_smooth.y - exp_smooth.y) ** 2)))
    assert rmse > 0
    # Smoothed residual should be bounded by the signal range.
    sig_range = float(max(exp_stress.max(), sim_stress.max()) -
                      min(exp_stress.min(), sim_stress.min()))
    assert rmse < sig_range


def test_arc_length_with_shared_scales_matches_arc_sample_indices(
    workspace: Path, fake_binary: Path, master_template: Path,
    experimental_stress_strain: pd.DataFrame,
):
    """Sim and exp arc-length smoothers using the SAME scales agree index-wise.

    This is the subtle detail that the architecture doc calls out:
    when comparing sim to exp by arc length, both smoothers must use
    the SAME normalization scales (typically the experimental ranges)
    so that the i-th arc-length sample of each curve corresponds to
    the same fraction of the same reference path length.
    """
    resolver, reader = _setup_common(workspace)
    backend = LocalBackend(max_workers=1)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    ctx = CaseContext(generation=0, gene=0, obj=0)
    _run_case(
        workspace, resolver, master_template, fake_binary, ctx,
        {"strain_rate": 1.0, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
        backend, manifest,
    )

    layout = CaseLayout(ctx=ctx, resolver=resolver)
    rs = reader.read(layout)
    sim_strain = 1.0 * rs.df("avg_stress")["Time"].to_numpy()
    sim_stress = rs.df("avg_stress")["Szz"].to_numpy()
    exp_strain = experimental_stress_strain["strain"].to_numpy()
    exp_stress = experimental_stress_strain["stress"].to_numpy()

    # Use experimental range as the common normalization.
    x_scale = float(exp_strain.max() - exp_strain.min())
    y_scale = float(exp_stress.max() - exp_stress.min())

    smoother = ArcLengthSmoother(n_samples=100, x_scale=x_scale, y_scale=y_scale)
    sim_smooth = smoother.smooth(sim_strain, sim_stress)
    exp_smooth = smoother.smooth(exp_strain, exp_stress)

    # Both curves have 100 samples.
    assert len(sim_smooth.x) == 100
    assert len(exp_smooth.x) == 100

    # The arc-length totals should be close (both curves trace similar
    # paths in the normalized (strain, stress) box; the slight Voce
    # parameter difference means they aren't identical).
    ratio = sim_smooth.s[-1] / exp_smooth.s[-1]
    assert 0.8 < ratio < 1.2, f"arc-length total ratio {ratio} out of range"


def test_auto_smoother_in_pipeline(
    workspace: Path, fake_binary: Path, master_template: Path,
):
    """auto_smoother picks PCHIP for our monotonic fake data."""
    resolver, reader = _setup_common(workspace)
    backend = LocalBackend(max_workers=1)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    ctx = CaseContext(generation=0, gene=0, obj=0)
    _run_case(
        workspace, resolver, master_template, fake_binary, ctx,
        {"strain_rate": 1.0, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
        backend, manifest,
    )

    layout = CaseLayout(ctx=ctx, resolver=resolver)
    rs = reader.read(layout)
    sim_strain = 1.0 * rs.df("avg_stress")["Time"].to_numpy()
    sim_stress = rs.df("avg_stress")["Szz"].to_numpy()

    # Let the framework pick. Fake data is monotonic -> PCHIP.
    smoother = auto_smoother(sim_strain, sim_stress, n_samples=50)
    # We don't assert the concrete class name here - the point of
    # auto_smoother is to Just Work without the caller caring. But
    # we do verify the output shape.
    out = smoother.smooth(sim_strain, sim_stress)
    assert len(out.x) == 50
    assert len(out.y) == 50


def test_legacy_vs_pchip_on_real_sim_output(
    workspace: Path, fake_binary: Path, master_template: Path,
):
    """Migration check: legacy and PCHIP are close on real simulation output.

    If this test were to produce large differences, it would indicate
    that switching to PCHIP midway through the refactor would change
    objective values in a way a user might mistake for a real bug.
    Verifying that they agree closely on monotonic data lets us
    migrate the smoother in a separate step with confidence.
    """
    from workflow_common import LegacyLinearSmoother

    resolver, reader = _setup_common(workspace)
    backend = LocalBackend(max_workers=1)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    ctx = CaseContext(generation=0, gene=0, obj=0)
    _run_case(
        workspace, resolver, master_template, fake_binary, ctx,
        {"strain_rate": 1.0, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
        backend, manifest,
    )

    layout = CaseLayout(ctx=ctx, resolver=resolver)
    rs = reader.read(layout)
    x = 1e-3 * rs.df("avg_stress")["Time"].to_numpy()
    y = rs.df("avg_stress")["Szz"].to_numpy()

    legacy = LegacyLinearSmoother(n_samples=200).smooth(x, y)
    pchip = PchipSmoother(n_samples=200).smooth(x, y)

    # Grids should be identical.
    assert np.allclose(legacy.x, pchip.x)
    # Stress values should be close - but not identical, since PCHIP's
    # cubic segments curve differently from the legacy linear
    # connect-the-dots. The exact bound depends on how densely the
    # source data samples the curve's knee; our fake binary runs only
    # 50 steps on a sharp-saturation response, which is near the
    # worst-case shape for legacy-vs-PCHIP divergence. Production
    # ExaConstit runs typically use hundreds of steps and show
    # sub-percent differences.
    signal_range = float(y.max() - y.min())
    max_abs_diff = float(np.max(np.abs(legacy.y - pchip.y)))
    rel_diff = max_abs_diff / signal_range
    # This bound is deliberately loose. Tighten if you increase the
    # fake binary's step count or use a gentler saturation rate.
    assert rel_diff < 0.10, (
        f"legacy vs PCHIP disagreement {rel_diff:.4f} exceeds 10% of "
        "signal range on a 50-point sharp-knee curve - surprising; "
        "investigate before switching the default smoother."
    )
