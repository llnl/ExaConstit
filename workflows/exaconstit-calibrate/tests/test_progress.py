"""
Unit tests for :class:`workflow_common.progress.ProgressReporter`.
"""
from __future__ import annotations

import io

from workflow_common.backends.base import BackendStats
from workflow_common.progress import ProgressReporter, _fmt_duration


class _FakeBackend:
    """Stub backend that just returns a pre-configured BackendStats."""

    def __init__(self, stats: BackendStats):
        self._stats = stats

    def poll_stats(self) -> BackendStats:
        return self._stats


def test_fmt_duration_seconds_only():
    assert _fmt_duration(0) == "0s"
    assert _fmt_duration(5.9) == "5s"       # truncation, not rounding
    assert _fmt_duration(59) == "59s"


def test_fmt_duration_minutes():
    assert _fmt_duration(60) == "1m00s"
    assert _fmt_duration(3599) == "59m59s"


def test_fmt_duration_hours():
    assert _fmt_duration(3600) == "1h00m"
    assert _fmt_duration(7323) == "2h02m"


def test_fmt_duration_negative_clamped_to_zero():
    assert _fmt_duration(-5) == "0s"


def test_progress_reports_completed_and_total():
    buf = io.StringIO()
    r = ProgressReporter(
        total=4, label="gen 1/10", stream=buf,
        min_update_interval_s=0.0,  # render every tick for determinism
        tty_override=False,
    )
    r.tick()
    r.tick()
    text = buf.getvalue()
    assert "gen 1/10" in text
    # Non-TTY mode: each tick becomes a line; after 2 ticks we expect
    # at least one line showing "2/4 (...%)".
    assert "2/4" in text
    assert "50.0%" in text


def test_progress_no_total_renders_count_only():
    buf = io.StringIO()
    r = ProgressReporter(
        total=0, stream=buf,
        min_update_interval_s=0.0, tty_override=False,
    )
    r.tick()
    assert buf.getvalue()      # something got written
    # Should NOT include a percent sign when total=0.
    assert "%" not in buf.getvalue()


def test_progress_includes_backend_stats_when_backend_supplied():
    stats = BackendStats(
        running_jobs=3, max_concurrent=4,
        cores_in_use=12, cores_total=16,
    )
    buf = io.StringIO()
    r = ProgressReporter(
        total=10, stream=buf, backend=_FakeBackend(stats),
        min_update_interval_s=0.0, tty_override=False,
    )
    r.tick()
    out = buf.getvalue()
    assert "running 3/4" in out
    assert "cores 12/16" in out


def test_progress_includes_queued_jobs_when_backend_reports_them():
    stats = BackendStats(
        running_jobs=28, max_concurrent=28,
        cores_in_use=112, cores_total=112,
        queued_jobs=11,
    )
    buf = io.StringIO()
    r = ProgressReporter(
        total=40, stream=buf, backend=_FakeBackend(stats),
        min_update_interval_s=0.0, tty_override=False,
    )
    r.tick()
    out = buf.getvalue()
    assert "running 28/28" in out
    assert "queued 11" in out
    assert "cores 112/112" in out


def test_progress_tolerates_backend_poll_failure():
    class _Broken:
        def poll_stats(self):
            raise RuntimeError("sched is down")
    buf = io.StringIO()
    r = ProgressReporter(
        total=5, stream=buf, backend=_Broken(),
        min_update_interval_s=0.0, tty_override=False,
    )
    # Must not raise — we explicitly catch backend failures so a
    # monitoring-only feature can't take down a real run.
    r.tick()
    assert "1/5" in buf.getvalue()


def test_progress_rate_limits_updates():
    """With min_update_interval_s > 0, rapid ticks produce fewer lines."""
    buf = io.StringIO()
    r = ProgressReporter(
        total=100, stream=buf,
        min_update_interval_s=10.0,    # effectively suppress auto-render
        tty_override=False,
    )
    for _ in range(50):
        r.tick()
    # At most the very first tick renders; subsequent ticks inside the
    # interval are silent. completed count is still accurate.
    assert r.completed == 50
    line_count = buf.getvalue().count("\n")
    assert line_count <= 1


def test_progress_finish_forces_final_render_and_newline():
    buf = io.StringIO()
    r = ProgressReporter(
        total=2, stream=buf,
        min_update_interval_s=10.0,
        tty_override=False,
    )
    r.tick()
    r.tick()
    r.finish()
    out = buf.getvalue()
    # finish() must have emitted the final 2/2 (100%) line, and
    # the output must end with a newline for grep-friendliness.
    assert "2/2" in out
    assert out.endswith("\n")


def test_progress_tty_mode_uses_carriage_return():
    buf = io.StringIO()
    r = ProgressReporter(
        total=3, stream=buf,
        min_update_interval_s=0.0, tty_override=True,   # force TTY
    )
    r.tick()
    r.tick()
    out = buf.getvalue()
    # TTY mode rewrites the line in place: '\r' appears, and the
    # per-tick lines do NOT have trailing newlines (until finish()).
    assert "\r" in out
    assert out.count("\n") == 0
    r.finish()
    assert buf.getvalue().endswith("\n")
