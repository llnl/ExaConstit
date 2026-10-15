"""
Restart-scenario integration tests.

These simulate the workflow the framework is specifically designed
for: a long-running optimization where the HPC allocation dies in
the middle, and the user restarts the driver. The restart must:

* Skip cases that completed successfully (sentinel present).
* Rerun cases that were in-flight when the driver died (SUBMITTED
  with no terminal transition -> INTERRUPTED on reload).
* Not re-attempt cases that are terminal FAILED by policy (unless
  the caller's policy says to - here we demonstrate the default
  "leave failed cases alone").

Each test below emulates a crash by simply abandoning a Manifest
object partway through a run, then spinning up a new Manifest and
driving it forward. No actual process kill is necessary - the
crash-safety guarantees live entirely in the on-disk state (JSONL
manifest + per-case sentinels).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

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
    render_template_file,
    write_sentinel,
)
from workflow_common.backends.base import JobOutcome
from workflow_common.sentinel import is_case_complete, read_sentinel


def _setup(workspace: Path):
    """Build the resolver, reader, and manifest path used across tests."""
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
        },
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
    })
    manifest_path = workspace / "manifest.jsonl"
    return resolver, reader, manifest_path


def _render_and_spec(
    ctx: CaseContext,
    resolver: TemplatePathResolver,
    master_template: Path,
    fake_binary: Path,
) -> SimJobSpec:
    wd = resolver.working_dir(ctx)
    wd.mkdir(parents=True, exist_ok=True)
    render_template_file(
        master_template, wd / "options.toml",
        {"gene": ctx.gene, "obj": ctx.obj,
         "strain_rate": 1e-3, "yield_stress": 200.0,
         "hardening": 2000.0, "temp_k": 298.0},
    )
    return SimJobSpec(
        working_dir=wd, binary=fake_binary, args=(),
        duration_s=30, stdout="run.out", stderr="run.err",
        tag=f"g{ctx.gene}o{ctx.obj}",
    )


def test_restart_skips_completed_cases(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """Cases with a valid sentinel on disk are skipped on restart."""
    resolver, reader, manifest_path = _setup(workspace)
    backend = LocalBackend(max_workers=2)

    # --- First "allocation": complete two cases, then "crash". ---------
    m1 = Manifest(manifest_path)
    m1.load()

    contexts = [CaseContext(generation=0, gene=i, obj=0) for i in range(4)]
    # Run cases 0 and 1 fully through.
    for ctx in contexts[:2]:
        spec = _render_and_spec(ctx, resolver, master_template, fake_binary)
        m1.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
        ))
        res = backend.submit_one(spec)
        assert res.outcome == JobOutcome.OK
        write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=res.wall_time_s))
        m1.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.COMPLETED, rc=0, case_dir=str(spec.working_dir),
        ))

    # Case 2 is submitted but "dies" before producing a sentinel. We
    # simulate this by recording SUBMITTED but not actually running.
    ctx = contexts[2]
    spec = _render_and_spec(ctx, resolver, master_template, fake_binary)
    m1.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
    ))
    # Case 3 was never even submitted.

    # --- "Allocation dies." Drop m1, spin up m2 fresh. -----------------
    m2 = Manifest(manifest_path)
    m2.load()
    n_interrupted = m2.mark_submitted_as_interrupted()
    # Exactly one case was stuck in SUBMITTED (case 2).
    assert n_interrupted == 1

    # --- Restart logic: for each context, skip if sentinel present. ----
    skipped: List[int] = []
    rerun: List[int] = []
    for ctx in contexts:
        wd = resolver.working_dir(ctx)
        if is_case_complete(wd):
            skipped.append(ctx.gene)
            continue
        rerun.append(ctx.gene)
        spec = _render_and_spec(ctx, resolver, master_template, fake_binary)
        m2.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
        ))
        res = backend.submit_one(spec)
        assert res.outcome == JobOutcome.OK
        write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=res.wall_time_s))
        m2.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.COMPLETED, rc=0, case_dir=str(spec.working_dir),
        ))

    # Cases 0 and 1 were completed; cases 2 and 3 needed to run now.
    assert skipped == [0, 1]
    assert rerun == [2, 3]

    # Final manifest: every case should be COMPLETED.
    for ctx in contexts:
        e = m2.get(ctx.generation, ctx.gene, ctx.obj)
        assert e is not None
        assert e.state == CaseState.COMPLETED


def test_crash_between_sentinel_and_manifest_is_safe(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """The sentinel-first-then-manifest ordering is crash-safe.

    Simulate: case finishes, sentinel is written, BUT the driver dies
    before it can record the terminal manifest entry. On restart,
    the sentinel is the authoritative signal - the case is treated
    as complete even though the manifest still says SUBMITTED.
    """
    resolver, reader, manifest_path = _setup(workspace)
    backend = LocalBackend(max_workers=1)

    ctx = CaseContext(generation=0, gene=0, obj=0)
    spec = _render_and_spec(ctx, resolver, master_template, fake_binary)

    m1 = Manifest(manifest_path)
    m1.load()
    m1.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
    ))
    res = backend.submit_one(spec)
    assert res.outcome == JobOutcome.OK

    # Write sentinel, but "die" before recording COMPLETED.
    write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=res.wall_time_s))

    # --- Restart: sentinel present, manifest says SUBMITTED. ----------
    m2 = Manifest(manifest_path)
    m2.load()
    n = m2.mark_submitted_as_interrupted()
    # Because we did NOT record COMPLETED, the manifest entry is still
    # SUBMITTED, so it gets promoted to INTERRUPTED. But the sentinel
    # overrides this - our skip logic relies on sentinel presence, not
    # manifest state.
    assert n == 1
    assert m2.get(0, 0, 0).state == CaseState.INTERRUPTED

    # The correct restart policy: trust the sentinel.
    assert is_case_complete(spec.working_dir)
    sen = read_sentinel(spec.working_dir)
    assert sen.rc == 0


def test_interrupted_case_can_be_rerun_cleanly(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """A case marked INTERRUPTED can be cleared and rerun from scratch."""
    resolver, reader, manifest_path = _setup(workspace)
    backend = LocalBackend(max_workers=1)

    ctx = CaseContext(generation=0, gene=0, obj=0)
    spec = _render_and_spec(ctx, resolver, master_template, fake_binary)
    layout = CaseLayout(ctx=ctx, resolver=resolver)

    # Simulate a partial run: the SUBMITTED entry exists but no sentinel.
    m1 = Manifest(manifest_path)
    m1.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
    ))

    # Restart: promote to INTERRUPTED.
    m2 = Manifest(manifest_path)
    m2.load()
    m2.mark_submitted_as_interrupted()
    assert m2.get(0, 0, 0).state == CaseState.INTERRUPTED

    # Re-run logic: clear any partial outputs, then submit fresh.
    layout.clear_outputs()  # idempotent even if none exist
    m2.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
    ))
    res = backend.submit_one(spec)
    assert res.outcome == JobOutcome.OK
    write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=res.wall_time_s))
    m2.record(ManifestEntry(
        generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
        state=CaseState.COMPLETED, rc=0, case_dir=str(spec.working_dir),
    ))

    # Now the case is complete. Read its outputs to verify.
    rs = reader.read(layout)
    assert "avg_stress" in rs
    assert len(rs.df("avg_stress")) == 50


def test_snapshot_survives_simulated_crash(
    workspace: Path, fake_binary: Path, master_template: Path
):
    """A snapshot written before the crash is honored on reload."""
    resolver, reader, manifest_path = _setup(workspace)
    backend = LocalBackend(max_workers=1)

    m1 = Manifest(manifest_path)
    m1.load()
    contexts = [CaseContext(generation=0, gene=i, obj=0) for i in range(3)]

    # Complete every case, take snapshot.
    for ctx in contexts:
        spec = _render_and_spec(ctx, resolver, master_template, fake_binary)
        m1.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.SUBMITTED, case_dir=str(spec.working_dir),
        ))
        res = backend.submit_one(spec)
        write_sentinel(spec.working_dir, Sentinel(rc=0, wall_time_s=res.wall_time_s))
        m1.record(ManifestEntry(
            generation=ctx.generation, gene=ctx.gene, obj=ctx.obj,
            state=CaseState.COMPLETED, rc=0, case_dir=str(spec.working_dir),
        ))
    m1.snapshot()

    # Simulate a torn-write on the JSONL log by appending garbage.
    # This is what a kill-mid-write would produce.
    with open(manifest_path, "a") as f:
        f.write('{"generation": 0, "gene": 99, "obj": 0, "state": "sub')  # truncated

    # Reload: the snapshot is authoritative, the torn line is dropped.
    m2 = Manifest(manifest_path)
    m2.load()
    for ctx in contexts:
        e = m2.get(ctx.generation, ctx.gene, ctx.obj)
        assert e is not None
        assert e.state == CaseState.COMPLETED
    # The garbage entry did not materialize into a phantom case.
    assert m2.get(0, 99, 0) is None
