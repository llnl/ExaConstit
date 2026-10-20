"""
Unit tests for :class:`workflow_common.backends.local.LocalBackend`.

These use the ``fake_binary`` fixture from conftest to exercise the
real subprocess code path. We deliberately do NOT mock out
``subprocess.Popen`` because the subtleties the backend is trying to
handle - stdio redirection, timeouts, bad-binary-paths - all live
inside the subprocess module and are only visible with a real
invocation.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import pytest

from workflow_common.backends.base import JobOutcome, JobResult, SimJobSpec
from workflow_common.backends.local import LocalBackend


def _spec(
    workspace: Path,
    fake_binary: Path,
    *,
    tag: str = "t",
    duration_s: int = 30,
    capture: bool = True,
) -> SimJobSpec:
    """Build a minimal SimJobSpec aimed at the fake binary."""
    wd = workspace / f"case_{tag}"
    wd.mkdir()
    # The fake binary reads options.toml; give it a minimal one.
    (wd / "options.toml").write_text(
        'basename = "options"\n'
        "strain_rate = 1e-3\n"
        "yield_stress = 200.0\n"
        "hardening = 2000.0\n"
    )
    return SimJobSpec(
        working_dir=wd,
        binary=Path(fake_binary),
        args=(),
        num_tasks=1,
        duration_s=duration_s,
        stdout="run.out" if capture else None,
        stderr="run.err" if capture else None,
        tag=tag,
    )


def test_local_serial_happy_path(workspace: Path, fake_binary: Path):
    """max_workers=1 runs cases strictly in order and succeeds."""
    specs = [_spec(workspace, fake_binary, tag=f"{i}") for i in range(3)]
    backend = LocalBackend(max_workers=1)
    results = backend.submit_batch(specs)
    assert len(results) == 3
    for r in results:
        assert r.outcome == JobOutcome.OK
        assert r.rc == 0
    # Output files should be present inside each working dir.
    for spec in specs:
        assert (spec.working_dir / "results" / "options" / "avg_stress.txt").exists()


def test_local_parallel_happy_path(workspace: Path, fake_binary: Path):
    """max_workers>1 completes all cases and preserves submission-order output."""
    specs = [_spec(workspace, fake_binary, tag=f"{i}") for i in range(5)]
    backend = LocalBackend(max_workers=3)
    results = backend.submit_batch(specs)
    assert len(results) == 5
    # submit_batch must return results in submission order even though
    # they complete in arbitrary order internally.
    for spec, result in zip(specs, results):
        assert result.spec is spec
        assert result.outcome == JobOutcome.OK


def test_local_stream_batch_yields_as_completed(workspace: Path, fake_binary: Path):
    """stream_batch yields one JobResult per spec; each points to its input."""
    specs = [_spec(workspace, fake_binary, tag=f"{i}") for i in range(3)]
    backend = LocalBackend(max_workers=2)
    seen: List[JobResult] = list(backend.stream_batch(specs))
    assert len(seen) == 3
    # The set of returned specs must equal the set submitted.
    assert {id(r.spec) for r in seen} == {id(s) for s in specs}


def test_local_failing_case_reports_failed(workspace: Path, fake_binary: Path):
    """rc=7 from the fake binary -> JobOutcome.FAILED, rc preserved."""
    spec = _spec(workspace, fake_binary, tag="fail")
    os.environ["FAKE_FAIL"] = "1"
    try:
        backend = LocalBackend(max_workers=1)
        result = backend.submit_one(spec)
    finally:
        os.environ.pop("FAKE_FAIL", None)
    assert result.outcome == JobOutcome.FAILED
    assert result.rc == 7


def test_local_bad_binary_reports_submit_error(workspace: Path):
    """Nonexistent binary path -> SUBMIT_ERROR, not FAILED."""
    wd = workspace / "case"
    wd.mkdir()
    # Fake options.toml so the working directory has something.
    (wd / "options.toml").write_text("")
    spec = SimJobSpec(
        working_dir=wd,
        binary=Path("/no/such/binary"),
        args=(),
        duration_s=10,
    )
    backend = LocalBackend(max_workers=1)
    result = backend.submit_one(spec)
    assert result.outcome == JobOutcome.SUBMIT_ERROR
    assert result.error_message is not None


def test_local_timeout_kills_process(workspace: Path, fake_binary: Path):
    """A spec with duration_s shorter than FAKE_SLEEP yields TIMEOUT."""
    spec = _spec(workspace, fake_binary, tag="timeout", duration_s=1)
    os.environ["FAKE_SLEEP"] = "5"
    try:
        t0 = time.monotonic()
        backend = LocalBackend(max_workers=1, kill_on_timeout=True)
        result = backend.submit_one(spec)
        elapsed = time.monotonic() - t0
    finally:
        os.environ.pop("FAKE_SLEEP", None)

    assert result.outcome == JobOutcome.TIMEOUT
    # Kill-on-timeout should interrupt well before the full 5s sleep.
    # Give it a generous upper bound to avoid flakiness on slow CI.
    assert elapsed < 4.5


def test_local_stdio_files_are_written(workspace: Path, fake_binary: Path):
    """stdout/stderr files requested in the spec actually get written."""
    spec = _spec(workspace, fake_binary, tag="stdio")
    backend = LocalBackend(max_workers=1)
    result = backend.submit_one(spec)
    assert result.outcome == JobOutcome.OK
    # The fake binary prints "fake sim ok" to stdout on success via our
    # default. It doesn't actually do that currently; the stdout file
    # should at least exist (possibly empty).
    assert (spec.working_dir / "run.out").exists()
    assert (spec.working_dir / "run.err").exists()


def test_local_rejects_invalid_workers():
    """max_workers < 1 is nonsensical and must raise."""
    with pytest.raises(ValueError):
        LocalBackend(max_workers=0)


def test_local_empty_batch_is_noop(workspace: Path):
    """submit_batch([]) returns [] without spinning up a pool."""
    backend = LocalBackend(max_workers=4)
    assert backend.submit_batch([]) == []
    assert list(backend.stream_batch([])) == []


def test_local_refuses_multi_rank_without_launcher(workspace: Path, fake_binary: Path):
    """Requesting num_tasks > 1 with no MPI launcher is a user bug
    (would silently drop ranks); the backend refuses at submit time
    so the problem surfaces at the first case instead of after a
    multi-day run produces single-rank results.
    """
    backend = LocalBackend(max_workers=1)  # no mpi_launcher
    spec = SimJobSpec(
        working_dir=workspace / "c",
        binary=fake_binary,
        args=(),
        num_tasks=4,
        duration_s=30,
    )
    with pytest.raises(ValueError, match="num_tasks=4"):
        backend.submit_one(spec)


def test_local_ranks_silent_allows_dropped_ranks(workspace: Path, fake_binary: Path):
    """ranks_silent=True opt-in preserves the old behavior for users
    who legitimately want to run a serial binary despite num_tasks > 1
    (e.g. using num_tasks only for bookkeeping).
    """
    backend = LocalBackend(max_workers=1, ranks_silent=True)
    spec = _spec(workspace, fake_binary, tag="serial_with_tasks")
    # Override to request multi-rank while leaving everything else alone.
    spec = SimJobSpec(
        working_dir=spec.working_dir,
        binary=spec.binary,
        args=spec.args,
        num_tasks=4,           # would normally raise
        duration_s=spec.duration_s,
        stdout=spec.stdout,
        stderr=spec.stderr,
        tag=spec.tag,
    )
    # Should not raise; the binary runs single-rank.
    result = backend.submit_one(spec)
    assert result.outcome == JobOutcome.OK


def test_local_uses_mpi_launcher_for_multi_rank(workspace: Path, tmp_path):
    """When mpi_launcher is configured, the launcher is prepended
    with the configured ntasks flag and the task count.
    """
    backend = LocalBackend(mpi_launcher="mpirun")
    spec = SimJobSpec(
        working_dir=workspace / "c",
        binary=Path("/usr/bin/echo"),
        args=("hello",),
        num_tasks=4,
    )
    cmd = backend._build_cmd(spec)
    assert cmd[:3] == ["mpirun", "-n", "4"]
    assert cmd[3] == "/usr/bin/echo"


def test_local_custom_ntasks_flag(workspace: Path):
    """mpi_launcher_ntasks_flag overrides the default '-n' for
    launchers that need different syntax (jsrun '--nrs', lrun '-T').
    """
    backend = LocalBackend(
        mpi_launcher="jsrun", mpi_launcher_ntasks_flag="--nrs",
    )
    spec = SimJobSpec(
        working_dir=workspace / "c",
        binary=Path("/bin/true"),
        args=(),
        num_tasks=8,
    )
    cmd = backend._build_cmd(spec)
    assert cmd[:3] == ["jsrun", "--nrs", "8"]


def test_local_poll_stats_zero_before_any_jobs(workspace: Path):
    """Fresh backend with no work: zero running, cores_total from OS."""
    backend = LocalBackend(max_workers=4)
    stats = backend.poll_stats()
    assert stats.running_jobs == 0
    assert stats.max_concurrent == 4
    assert stats.cores_in_use == 0
    # cores_total may be 0 on exotic kernels; >= 1 everywhere a test runs.
    assert stats.cores_total >= 1


def test_local_poll_stats_reports_running_jobs(workspace: Path, fake_binary: Path):
    """poll_stats reflects in-flight jobs while stream_batch is pumping."""
    import threading
    backend = LocalBackend(max_workers=2)
    # Build four serial specs, each with a small sleep so poll_stats
    # has a real window to observe in-flight work.
    base_specs = [_spec(workspace, fake_binary, tag=f"p{i}") for i in range(4)]
    specs = [
        SimJobSpec(
            working_dir=s.working_dir,
            binary=s.binary,
            args=s.args,
            env={"FAKE_SLEEP": "0.3"},
            num_tasks=1,
            duration_s=s.duration_s,
            stdout=s.stdout,
            stderr=s.stderr,
            tag=s.tag,
        )
        for s in base_specs
    ]

    seen_running = []

    def poll():
        # Poll a few times during the batch run.
        for _ in range(6):
            seen_running.append(backend.poll_stats().running_jobs)
            import time as _t; _t.sleep(0.1)

    t = threading.Thread(target=poll)
    t.start()
    results = list(backend.stream_batch(specs))
    t.join()
    assert len(results) == 4
    # At some point during the run, at least one job was in flight.
    assert max(seen_running) >= 1
    # After completion, everything is cleared.
    assert backend.poll_stats().running_jobs == 0
