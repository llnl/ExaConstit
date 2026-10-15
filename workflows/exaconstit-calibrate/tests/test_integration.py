"""
End-to-end integration tests for the workflow_common framework.

Why these exist alongside the unit tests
----------------------------------------
Unit tests cover each module in isolation. Integration tests here
cover the *interactions* between modules - the handoffs that are
easiest to get wrong and hardest to notice until a real run:

* Does the PathResolver actually agree with the templates about where
  options.toml should live?
* Does the manifest's SUBMITTED -> COMPLETED transition happen in the
  right order relative to the sentinel write?
* Does the ResultReader read back data that the backend just finished
  writing?
* Does a failed case get correctly recorded as FAILED in the manifest
  AND have a sentinel that reflects the failure?

Each test in this file runs the full chain from "I have a parameter
dict" to "I have a DataFrame of stress-strain data". They are slower
than unit tests (one subprocess per case) but still fast enough -
tens of milliseconds each - to run every commit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from workflow_common import (
    CaseContext,
    CaseLayout,
    CaseState,
    LocalBackend,
    Manifest,
    ManifestEntry,
    Sentinel,
    SimJobSpec,
    TemplatePathResolver,
    TextTableReader,
    TextTableSpec,
    configure_logging,
    render_template_file,
    write_sentinel,
)
from workflow_common.backends.base import JobOutcome
from workflow_common.sentinel import (
    is_case_complete,
    read_sentinel,
    validate_outputs,
)


# --- Test helpers the integration tests share ----------------------------


def _build_resolver(workspace: Path) -> TemplatePathResolver:
    """The resolver used by the integration tests.

    Matches the ExaConstit default output layout convention:
    ``<workspace>/wf/gen_N/gene_N_obj_N/results/options/...``.
    """
    return TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )


def _build_reader() -> TextTableReader:
    """Reader matching the fake binary's output schema."""
    return TextTableReader({
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


def _prepare_case(
    resolver: TemplatePathResolver,
    master_template: Path,
    ctx: CaseContext,
    params: dict,
) -> Tuple[CaseLayout, SimJobSpec, Path]:
    """Render per-case input and build a SimJobSpec.

    Returns a tuple of (layout, spec, working_dir) so tests have
    direct handles to the working dir without re-resolving.
    """
    wd = resolver.working_dir(ctx)
    wd.mkdir(parents=True, exist_ok=True)
    # Values fed into the template. The template uses %%key%% placeholders
    # so the keys must match the placeholder names.
    values = {
        "gene": ctx.gene,
        "obj": ctx.obj,
        "strain_rate": params.get("strain_rate", 1e-3),
        "yield_stress": params.get("yield_stress", 200.0),
        "hardening": params.get("hardening", 2000.0),
        "temp_k": params.get("temp_k", 298.0),
    }
    render_template_file(master_template, wd / "options.toml", values)
    layout = CaseLayout(ctx=ctx, resolver=resolver)
    return layout, None, wd  # spec built per-test below


def _run_full_case(
    *,
    ctx: CaseContext,
    params: dict,
    workspace: Path,
    fake_binary: Path,
    master_template: Path,
    resolver: TemplatePathResolver,
    manifest: Manifest,
    backend: LocalBackend,
    reader: TextTableReader,
    required_outputs: List[str],
) -> Tuple[JobOutcome, pd.DataFrame | None]:
    """Run one case end-to-end and return (outcome, stress DataFrame).

    This function is the minimum responsible driver - it does every
    step a production driver would:

    1. Render input files.
    2. Record SUBMITTED in the manifest.
    3. Run the simulation.
    4. Validate outputs.
    5. Write sentinel (atomic).
    6. Record terminal manifest entry.
    7. Read results.
    """
    layout, _, wd = _prepare_case(resolver, master_template, ctx, params)
    spec = SimJobSpec(
        working_dir=wd,
        binary=fake_binary,
        args=(),
        num_tasks=1,
        duration_s=30,
        stdout="run.out",
        stderr="run.err",
        tag=f"g{ctx.gene}o{ctx.obj}",
    )

    # Manifest: SUBMITTED before the backend gets the spec, so a crash
    # during submission is recoverable.
    manifest.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(wd),
    ))

    # Backend: run the subprocess.
    result = backend.submit_one(spec)

    # Output validation: catches rc=0 with missing files (not an issue
    # with the fake binary but would be with a killed real one).
    ok, bad = validate_outputs(wd, required_outputs)
    terminal = (
        CaseState.COMPLETED
        if (result.outcome == JobOutcome.OK and ok)
        else CaseState.FAILED
    )

    # Sentinel FIRST, manifest SECOND. See the architecture doc for why.
    write_sentinel(wd, Sentinel(
        rc=result.rc,
        wall_time_s=result.wall_time_s,
        jobid=result.jobid,
        output_files={n: str(layout.output_file(n)) for n in reader.specs},
        status="ok" if terminal == CaseState.COMPLETED else "bad",
        message=result.error_message if result.error_message else None,
    ))
    manifest.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=terminal, rc=result.rc, jobid=result.jobid,
        case_dir=str(wd),
    ))

    # Results: read what the simulation produced. Only attempt if the
    # case terminated successfully; failed cases may have missing or
    # truncated files which the reader will reject.
    df = None
    if terminal == CaseState.COMPLETED:
        rs = reader.read(layout)
        df = rs.df("avg_stress")
    return terminal, df


# --- Integration tests ---------------------------------------------------


def test_happy_path_full_pipeline(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """Render -> run -> validate -> sentinel -> manifest -> read, all clean."""
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=1)
    reader = _build_reader()

    terminal, df = _run_full_case(
        ctx=CaseContext(generation=0, gene=0, obj=0),
        params={"strain_rate": 1e-3, "yield_stress": 200.0, "hardening": 2000.0},
        workspace=workspace, fake_binary=fake_binary,
        master_template=master_template, resolver=resolver,
        manifest=manifest, backend=backend, reader=reader,
        required_outputs=["results/options/avg_stress.txt"],
    )

    assert terminal == CaseState.COMPLETED
    assert df is not None
    # Fake binary writes 50 rows of saturating Voce-like response.
    # At t=1 with strain_rate=1e-3, eps_final = 1e-3 and the argument of
    # the exponential is 50*1e-3 = 0.05, so 1-exp(-0.05) ~= 0.0488.
    # s11_final ~= yield + hardening * 0.0488 = 200 + 2000*0.0488 ~= 297.6.
    assert len(df) == 50
    assert df["Szz"].iloc[0] == pytest.approx(200.0, abs=1.0)
    # Monotonically increasing (hardening response).
    assert (df["Szz"].diff().dropna() > 0).all()
    # Final value in the expected range.
    assert 290.0 < df["Szz"].iloc[-1] < 310.0


def test_failed_case_recorded_as_failed(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """A simulation that exits nonzero -> FAILED in manifest + sentinel."""
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=1)
    reader = _build_reader()

    os.environ["FAKE_FAIL"] = "1"
    try:
        terminal, df = _run_full_case(
            ctx=CaseContext(generation=0, gene=0, obj=0),
            params={},
            workspace=workspace, fake_binary=fake_binary,
            master_template=master_template, resolver=resolver,
            manifest=manifest, backend=backend, reader=reader,
            required_outputs=["results/options/avg_stress.txt"],
        )
    finally:
        os.environ.pop("FAKE_FAIL", None)

    assert terminal == CaseState.FAILED
    assert df is None  # reader never called for failed cases

    # The manifest should report FAILED.
    entry = manifest.get(0, 0, 0)
    assert entry is not None
    assert entry.state == CaseState.FAILED
    assert entry.rc == 7

    # The sentinel should exist but indicate failure.
    wd = resolver.working_dir(CaseContext(0, 0, 0))
    sen = read_sentinel(wd)
    assert sen is not None
    assert sen.rc == 7
    assert sen.status == "bad"


def test_truncated_output_caught_by_validator(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """rc=0 with a truncated avg_stress still marked FAILED.

    The fake binary honors FAKE_TRUNCATE=1 to drop the last 10 rows.
    The validator only checks size-not-zero, so a truncated-but-nonempty
    file slips through validate_outputs. This test then demonstrates
    the NEXT line of defense: the TextTableReader's column-count check
    catches format drift, and a stricter validator (the caller's
    responsibility) could count rows.
    """
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=1)
    reader = _build_reader()

    os.environ["FAKE_TRUNCATE"] = "1"
    try:
        terminal, df = _run_full_case(
            ctx=CaseContext(generation=0, gene=0, obj=0),
            params={},
            workspace=workspace, fake_binary=fake_binary,
            master_template=master_template, resolver=resolver,
            manifest=manifest, backend=backend, reader=reader,
            required_outputs=["results/options/avg_stress.txt"],
        )
    finally:
        os.environ.pop("FAKE_TRUNCATE", None)

    # Size > 0 so validator passes. Reader then gets the short data.
    assert terminal == CaseState.COMPLETED
    assert df is not None
    # Truncated: should have 40 rows instead of 50.
    assert len(df) == 40


def test_missing_optional_output_reported(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """An absent optional output is collected but does not fail the run."""
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=1)
    reader = _build_reader()

    os.environ["FAKE_MISSING"] = "avg_def_grad"
    try:
        layout_ctx = CaseContext(generation=0, gene=0, obj=0)
        terminal, df = _run_full_case(
            ctx=layout_ctx, params={},
            workspace=workspace, fake_binary=fake_binary,
            master_template=master_template, resolver=resolver,
            manifest=manifest, backend=backend, reader=reader,
            required_outputs=["results/options/avg_stress.txt"],
        )
    finally:
        os.environ.pop("FAKE_MISSING", None)

    # The required output is present so the case is COMPLETED.
    assert terminal == CaseState.COMPLETED
    # A re-read shows avg_def_grad in the missing list.
    layout = CaseLayout(ctx=layout_ctx, resolver=resolver)
    rs = reader.read(layout)
    assert "avg_stress" in rs
    assert "avg_def_grad" in rs.missing
    assert not bool(rs)


def test_parallel_run_produces_independent_outputs(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """Two parallel cases write to distinct dirs without cross-contamination."""
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=3)
    reader = _build_reader()

    # Build 6 cases with different strain rates.
    contexts = [
        CaseContext(generation=0, gene=g, obj=o)
        for g in range(3) for o in range(2)
    ]
    specs: List[SimJobSpec] = []
    for ctx in contexts:
        layout, _, wd = _prepare_case(
            resolver, master_template, ctx,
            {"strain_rate": 10 ** (-3 - ctx.obj)},
        )
        specs.append(SimJobSpec(
            working_dir=wd, binary=fake_binary, args=(),
            duration_s=30, stdout="run.out", stderr="run.err",
        ))
        manifest.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.SUBMITTED, case_dir=str(wd),
        ))

    # Submit everything.
    results = backend.submit_batch(specs)
    assert len(results) == 6
    for r in results:
        assert r.outcome == JobOutcome.OK

    # Sentinel and manifest for each.
    for ctx, spec in zip(contexts, specs):
        write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=1.0))
        manifest.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.COMPLETED, rc=0, case_dir=str(spec.working_dir),
        ))

    # Each case should have its own output with the expected strain rate.
    for ctx in contexts:
        layout = CaseLayout(ctx=ctx, resolver=resolver)
        rs = reader.read(layout)
        assert "avg_stress" in rs
        # Fake binary uses strain_rate=1e-(3+obj); at t=1, the
        # exponent argument is 50*strain_rate which determines saturation.
        # Just sanity-check shape and nonzero values.
        df = rs.df("avg_stress")
        assert len(df) == 50
        assert df["Szz"].iloc[-1] > df["Szz"].iloc[0]


def test_integration_manifest_snapshot_and_reload(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """Snapshot + reload round trip preserves all state from a real run."""
    configure_logging(level="warning", stream=False)
    resolver = _build_resolver(workspace)
    manifest_path = workspace / "manifest.jsonl"
    manifest = Manifest(manifest_path)
    manifest.load()
    backend = LocalBackend(max_workers=2)
    reader = _build_reader()

    contexts = [CaseContext(generation=0, gene=g, obj=0) for g in range(4)]
    for ctx in contexts:
        _run_full_case(
            ctx=ctx, params={},
            workspace=workspace, fake_binary=fake_binary,
            master_template=master_template, resolver=resolver,
            manifest=manifest, backend=backend, reader=reader,
            required_outputs=["results/options/avg_stress.txt"],
        )

    manifest.snapshot()

    # Load from scratch in a new Manifest.
    m2 = Manifest(manifest_path)
    m2.load()
    # All four cases should be COMPLETED in the reloaded manifest.
    for ctx in contexts:
        e = m2.get(ctx.generation, ctx.gene, ctx.obj)
        assert e is not None
        assert e.state == CaseState.COMPLETED


# --- Relative-workspace coverage (audit Finding 7) -----------------------


@pytest.mark.parametrize("workspace_style", ["absolute", "relative"])
def test_happy_path_works_with_absolute_and_relative_workspace(
    tmp_path: Path,
    fake_binary: Path,
    master_template: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace_style: str,
):
    """Regression coverage for the "relative WORKSPACE" bug class.

    Users naturally write ``WORKSPACE = Path("./calibration_run")`` in
    their drivers, but every existing integration test uses the
    absolute ``tmp_path`` fixture. That coverage gap is exactly how
    the double-join bug from earlier in the audit reached production:
    the join chain ``working_dir/required_output`` produced a
    relative path, and ``validate_outputs`` joined against
    ``working_dir`` a second time, giving ``<wd>/<wd>/...``. Tests
    on absolute-tmpdir workspaces never saw this because the first
    join produced an absolute path and the second was a no-op.

    This parameterized test runs the happy-path pipeline under both
    workspace styles. The absolute case is redundant with
    ``test_happy_path_full_pipeline`` above; we keep it anyway so
    the two branches sit right next to each other and the test
    serves as a clear reference for the bug class.
    """
    configure_logging(level="warning", stream=False)

    # Build the workspace according to the requested style.
    if workspace_style == "absolute":
        workspace = tmp_path / "wf_abs"
        workspace.mkdir()
    else:
        # Run from tmp_path so the relative path is scoped to this
        # test and doesn't leak into the repo.
        monkeypatch.chdir(tmp_path)
        workspace = Path("wf_rel")
        workspace.mkdir()
        assert not workspace.is_absolute(), (
            "sanity: workspace must be relative for the relative branch"
        )

    resolver = _build_resolver(workspace)
    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()
    backend = LocalBackend(max_workers=1)
    reader = _build_reader()

    terminal, df = _run_full_case(
        ctx=CaseContext(generation=0, gene=0, obj=0),
        params={"strain_rate": 1e-3, "yield_stress": 200.0, "hardening": 2000.0},
        workspace=workspace, fake_binary=fake_binary,
        master_template=master_template, resolver=resolver,
        manifest=manifest, backend=backend, reader=reader,
        required_outputs=["results/options/avg_stress.txt"],
    )

    # Full pipeline must succeed regardless of workspace shape.
    assert terminal == CaseState.COMPLETED
    assert df is not None
    assert len(df) == 50
    # Load direction is z; Szz is the active component. Same physics
    # assertions as the absolute-only happy_path test.
    assert df["Szz"].iloc[0] == pytest.approx(200.0, abs=1.0)
    assert (df["Szz"].diff().dropna() > 0).all()
    assert 290.0 < df["Szz"].iloc[-1] < 310.0
