"""
Progress reporter for long-running simulation batches.

Why this module exists
----------------------
A multi-generation NSGA-III calibration routinely runs for hours or
days, producing thousands of simulation cases. Without live feedback
the user cannot tell the difference between "healthy and running" and
"stuck in a way that will never finish." A progress line that ticks
every second or two gives that feedback cheaply.

Two features matter enough to build in:

1. **Concurrency visibility.** Showing "3 of 4 workers busy" catches
   launcher misconfigurations early. The pre-refactor ``ExaConstit_NSGA3``
   had no way to surface that `num_tasks=4` was silently running one
   rank per case — the whole batch reported the same wall time
   whether you got 1x4-rank or 4x1-rank behavior. This reporter
   polls the backend's :meth:`poll_stats` and surfaces
   running-job + core-usage numbers, so the discrepancy would have
   been visible on the first generation.

2. **Works on non-interactive TTYs.** HPC job scripts redirect stdout
   to a log file; tqdm and curses-based progress bars produce garbage
   output there because they rely on carriage returns. This module
   detects ``isatty()`` and switches to a one-line-per-update mode
   that is grep-friendly in batch logs. Interactive runs get the
   in-place carriage-return update.

Why not tqdm
------------
tqdm is a great library and a perfectly reasonable dependency. We
don't use it here for two reasons: (a) avoiding a hard runtime dep
is worth the ~100 lines of code in this file; (b) tqdm's
pretty-printing does misbehave in some HPC environments (stdout
being captured into a file, then tail'd in real time, produces a
mix of carriage-returned and appended lines that are hard to read).
Rolling our own is small enough not to regret.

Not thread-safe
---------------
``ProgressReporter`` assumes a single caller updating it. The NSGA-III
driver calls ``reporter.tick()`` from its main evaluation loop; the
backend itself is not expected to touch the reporter. If you need
live updates from a background thread, wrap the reporter's ``tick``
call in an ``asyncio.Lock`` or ``threading.Lock`` in your driver
code — not here.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional, TextIO


@dataclass
class ProgressReporter:
    """Live progress reporter for a single batch of simulation cases.

    A reporter tracks three counts:

    * ``total`` — number of cases the batch will produce.
    * ``completed`` — cases finished so far (any outcome).
    * live backend stats from :meth:`JobBackend.poll_stats` — how many
      workers are busy right now and how many cores they're using.

    One line of output per tick. In interactive mode (TTY) the line
    is rewritten in place via carriage return; in non-interactive
    mode (log file, CI capture, job scheduler stdout) each tick
    appends a fresh line so the log is grep-friendly.

    The reporter does NOT start a background thread. Call :meth:`tick`
    yourself from whatever loop drives completion; the reporter
    rate-limits its output internally so calling ``tick`` a thousand
    times per second is cheap.

    Args:
        total: Total cases in this batch. Used for the "% complete"
            readout; must be > 0 or the reporter degrades gracefully
            to "no percent" output.
        label: Prefix string displayed at the start of each line.
            Typical values: "gen 3/100" or "calibration". Keep it
            short — long labels eat the useful screen real estate.
        backend: Optional backend whose :meth:`poll_stats` will be
            called on every tick to fetch concurrent-job / core-usage
            numbers. If ``None``, that part of the line is omitted.
        min_update_interval_s: Don't redraw the line more often than
            this. Default 0.5s is fast enough to feel live, slow
            enough to keep overhead negligible. Set to 0 to render
            on every tick (useful for tests).
        stream: Output stream. Default ``sys.stderr`` because that
            is what interactive progress should use — stdout is
            often captured by downstream tooling. Set to
            ``sys.stdout`` if your workflow prefers it there.
        tty_override: Force TTY / non-TTY behavior instead of
            auto-detecting. ``None`` (default) means detect from
            ``stream.isatty()``. Useful for tests and for forcing
            clean append-mode output when the driver knows it is
            inside a job scheduler.

    Example:
        ::

            reporter = ProgressReporter(
                total=len(specs), label="gen 3/100",
                backend=backend,
            )
            for result in backend.stream_batch(specs):
                reporter.tick()
                # ... process result ...
            reporter.finish()
    """

    total: int
    label: str = ""
    backend: Optional[Any] = None
    min_update_interval_s: float = 0.5
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    tty_override: Optional[bool] = None

    completed: int = field(default=0, init=False)
    _last_render_t: float = field(default=0.0, init=False)
    _start_t: float = field(default_factory=time.monotonic, init=False)
    _last_line_len: int = field(default=0, init=False)

    @property
    def _is_tty(self) -> bool:
        """Interactive detection; overridable."""
        if self.tty_override is not None:
            return self.tty_override
        try:
            return self.stream.isatty()
        except (AttributeError, ValueError):
            # ValueError if stream was closed; treat as non-TTY.
            return False

    def tick(self, *, force: bool = False) -> None:
        """Record one case completion and possibly redraw the line.

        Args:
            force: If True, bypass ``min_update_interval_s`` and
                render immediately. Used internally by
                :meth:`finish` and by tests; ordinary callers
                should pass ``False`` (the default) so rate-limiting
                applies.
        """
        self.completed += 1
        now = time.monotonic()
        if not force and (now - self._last_render_t) < self.min_update_interval_s:
            return
        self._last_render_t = now
        self._render()

    def finish(self) -> None:
        """Redraw the line one final time and close with a newline.

        Always render on finish (``force=True``) so the final
        counts aren't lost to rate-limiting. On TTY, moves the cursor
        past the progress line so subsequent prints don't get
        overwritten. On non-TTY, just writes a trailing newline.
        """
        self._render()
        # Move past the live progress line.
        self.stream.write("\n")
        try:
            self.stream.flush()
        except (AttributeError, ValueError):
            pass

    def _render(self) -> None:
        """Assemble and write the current progress line."""
        parts = []
        if self.label:
            parts.append(self.label)

        if self.total > 0:
            pct = 100.0 * self.completed / self.total
            parts.append(f"{self.completed}/{self.total} ({pct:5.1f}%)")
        else:
            parts.append(f"{self.completed}")

        # Backend-derived live state: running jobs and cores in use.
        if self.backend is not None:
            try:
                stats = self.backend.poll_stats()
            except Exception:
                # A failing poll_stats must never kill a real run;
                # degrade to no stats and keep going.
                stats = None
            if stats is not None and stats.max_concurrent > 0:
                parts.append(
                    f"running {stats.running_jobs}/{stats.max_concurrent}"
                )
            if (
                stats is not None
                and getattr(stats, "queued_jobs", 0) > 0
            ):
                parts.append(f"queued {stats.queued_jobs}")
            if stats is not None and stats.cores_total > 0:
                parts.append(
                    f"cores {stats.cores_in_use}/{stats.cores_total}"
                )

        elapsed = time.monotonic() - self._start_t
        parts.append(f"elapsed {_fmt_duration(elapsed)}")

        # Rough ETA if we have enough signal. "Enough signal" =
        # at least one completion, so we have a nonzero rate.
        if self.completed > 0 and self.total > self.completed:
            rate = self.completed / max(elapsed, 1e-6)
            remaining = (self.total - self.completed) / max(rate, 1e-9)
            parts.append(f"ETA {_fmt_duration(remaining)}")

        line = " | ".join(parts)

        if self._is_tty:
            # Carriage-return-and-overwrite. Pad with spaces to erase
            # any trailing characters from a previous (longer) line;
            # this keeps the display clean when the text shrinks.
            pad = max(0, self._last_line_len - len(line))
            self.stream.write("\r" + line + (" " * pad))
            self._last_line_len = len(line)
        else:
            # One-line-per-update. No \r, no padding, ends with a
            # newline so tail -f reads it as a complete line.
            self.stream.write(line + "\n")

        try:
            self.stream.flush()
        except (AttributeError, ValueError):
            pass


def _fmt_duration(seconds: float) -> str:
    """Format a seconds count as '1h23m', '5m17s', or '45s'.

    Intentionally compact: the progress line has limited width and
    seconds-level precision isn't useful past a minute. Fractional
    seconds are truncated, not rounded, to avoid the "59s -> 1m0s"
    flicker on a live display.
    """
    seconds = int(max(0.0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"
