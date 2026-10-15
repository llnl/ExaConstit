"""
Local subprocess backend.

When to use
-----------
``LocalBackend`` runs simulations as ordinary subprocesses on the
machine where the driver is running. It is the right choice for:

* Development and unit tests - no scheduler or cluster required, the
  framework can be exercised on a laptop with the real code paths.
* Small desktop runs where submitting cases one-at-a-time (or a
  handful at a time via a thread pool) is acceptable.
* CI pipelines where the "simulation" is replaced by a fake script
  that writes expected output files, so the rest of the framework
  can be tested without building the real simulation code.

When NOT to use
---------------
For real HPC runs that need multi-node MPI jobs, use
:class:`FluxBackend` (or a future SLURM-step backend). This class can
invoke an MPI launcher if you supply one, but it does not know about
schedulers, node allocations, or cluster topology.

Concurrency model
-----------------
Concurrency is controlled by ``max_workers``.

* ``max_workers=1`` runs cases strictly serially. This is the fastest
  path when only one case at a time makes sense (e.g. for debugging)
  and it avoids the overhead of spawning a thread pool.
* ``max_workers > 1`` runs cases in a ``ThreadPoolExecutor``. Each
  simulation is a subprocess, so there is no Python GIL contention -
  the threads just happen to be waiting on subprocess completion.
  This maps well to a desktop machine with, say, 4-8 cores where you
  want to run a handful of small cases in parallel.

Timeouts
--------
Each spec has a ``duration_s``. The local backend enforces this by
waiting on the subprocess with a timeout and killing the process if
the timeout fires. Set ``kill_on_timeout=False`` to skip the kill -
you will still get a TIMEOUT outcome but the subprocess will continue
running in the background, which is almost never what you want.
"""
from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from ..logging_utils import get_logger
from .base import BackendStats, BaseBackend, JobOutcome, JobResult, SimJobSpec

logger = get_logger(__name__)


class LocalBackend(BaseBackend):
    """Run :class:`SimJobSpec` values as local subprocesses.

    Args:
        max_workers: Maximum number of concurrent simulations. Must
            be >= 1. A value of 1 runs cases serially on the calling
            thread, avoiding any threadpool overhead.
        mpi_launcher: Optional path or name of an MPI launcher
            (``"mpirun"``, ``"srun"``, ``"jsrun"``, ...). If set and
            a spec has ``num_tasks > 1``, the command is built as
            ``<launcher> -n <tasks> <binary> <args>``. If ``None``,
            the binary is launched directly — valid only when every
            spec runs single-rank (``num_tasks == 1``). A
            ``num_tasks > 1`` spec combined with
            ``mpi_launcher=None`` raises :class:`ValueError` at
            submit time rather than silently dropping ranks.
        mpi_launcher_ntasks_flag: Flag prefix for the rank count,
            default ``"-n"``. ``"srun"`` uses the same flag;
            ``"jsrun"`` uses ``"--nrs"``; IBM's ``lrun`` uses
            ``"-T"``. Set this if your launcher differs.
        ranks_silent: If True, skip the safety check described
            above. Old behavior: ``num_tasks > 1`` specs with no
            launcher run single-rank and the extra ranks are
            silently dropped. The default is False because that
            behavior has caused production runs to waste days
            producing single-rank results when four-rank was
            requested. Set to True only for workflows that
            intentionally use ``num_tasks`` for bookkeeping while
            running serial binaries.
        kill_on_timeout: If True (default), a spec whose runtime
            exceeds ``duration_s`` is killed with ``proc.kill()``.
            If False, the process is allowed to continue running;
            the TIMEOUT outcome is still reported but the caller is
            responsible for any further cleanup.

    Raises:
        ValueError: If ``max_workers`` is less than 1.

    Example:
        Run four cases in parallel on a desktop, each using 4 MPI
        ranks via mpirun::

            backend = LocalBackend(
                max_workers=4, mpi_launcher="mpirun",
            )

        Serial run with srun::

            backend = LocalBackend(
                max_workers=1, mpi_launcher="srun",
            )

        IBM Spectrum / jsrun (as used on LLNL Sierra/Lassen)::

            backend = LocalBackend(
                max_workers=2, mpi_launcher="jsrun",
                mpi_launcher_ntasks_flag="--nrs",
            )
    """

    def __init__(
        self,
        *,
        max_workers: int = 1,
        mpi_launcher: Optional[str] = None,
        mpi_launcher_ntasks_flag: str = "-n",
        ranks_silent: bool = False,
        kill_on_timeout: bool = True,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._mpi_launcher = mpi_launcher
        self._mpi_launcher_ntasks_flag = mpi_launcher_ntasks_flag
        self._ranks_silent = ranks_silent
        self._kill_on_timeout = kill_on_timeout
        # Running-jobs bookkeeping used by :meth:`poll_stats`. We
        # accept the tiny cost of a lock for the tiny duration of
        # mutating this dict; progress reporters poll it roughly
        # once a second so the contention is negligible.
        import threading
        self._running_lock = threading.Lock()
        self._running: Dict[int, SimJobSpec] = {}

    def _build_cmd(self, spec: SimJobSpec) -> List[str]:
        """Construct the argv list to pass to ``subprocess.Popen``.

        Inserts the MPI launcher prefix when one is configured and
        the spec requests more than one rank. When ``num_tasks > 1``
        and no launcher is configured, raises :class:`ValueError`
        rather than silently running the binary single-rank — that
        failure mode has burned real production runs where a user
        requested ``num_tasks=4`` and got 4x single-rank results
        instead of 4-rank parallel results. Set
        ``ranks_silent=True`` on the backend to opt out of the check.

        Args:
            spec: The spec being launched.

        Returns:
            A list of strings suitable for ``subprocess.Popen``.

        Raises:
            ValueError: If ``spec.num_tasks > 1`` and no
                ``mpi_launcher`` is configured (unless
                ``ranks_silent=True``).
        """
        if spec.num_tasks > 1:
            if self._mpi_launcher is None and not self._ranks_silent:
                raise ValueError(
                    f"LocalBackend: spec requests num_tasks="
                    f"{spec.num_tasks} but no mpi_launcher is "
                    f"configured. Fix one of:\n"
                    f"  * Pass mpi_launcher='mpirun' (or 'srun', "
                    f"'jsrun', 'lrun', ...) when constructing "
                    f"LocalBackend.\n"
                    f"  * Set the SimCase's num_tasks=1 if this "
                    f"is a serial binary.\n"
                    f"  * Pass ranks_silent=True to LocalBackend "
                    f"to suppress this check (not recommended)."
                )
            if self._mpi_launcher is not None:
                return [
                    self._mpi_launcher,
                    self._mpi_launcher_ntasks_flag,
                    str(spec.num_tasks),
                    str(spec.binary),
                    *spec.args,
                ]
        return [str(spec.binary), *spec.args]

    def _run_one(self, spec: SimJobSpec) -> JobResult:
        """Run a single spec to completion and return its result.

        This is the worker function used by both the serial path and
        the threadpool path. Bookkeeping registers the spec as
        "running" before launch and removes it on completion so
        :meth:`poll_stats` can report accurate concurrent-job and
        in-use-core counts. The bookkeeping uses object ``id()`` as
        the key because SimJobSpec is frozen — equality-based keys
        would conflate two distinct specs that happen to share field
        values.

        Handles the same failure modes as before: bad binary path,
        permission / resource error, nonzero exit, duration_s
        overrun, kill-does-not-reap. See class docstring for the
        outcome mapping.

        Args:
            spec: The spec to run.

        Returns:
            A :class:`JobResult` describing the outcome.
        """
        # Register as running before any work. The finally block
        # below removes the entry regardless of outcome.
        with self._running_lock:
            self._running[id(spec)] = spec
        try:
            return self._run_one_inner(spec)
        finally:
            with self._running_lock:
                self._running.pop(id(spec), None)

    def _run_one_inner(self, spec: SimJobSpec) -> JobResult:
        """Actual work. See :meth:`_run_one`; this is the innermost
        body separated only so the outer method can wrap it with
        running-jobs bookkeeping without growing the indentation of
        the real code.
        """
        # --- Step 1: prepare the filesystem -------------------------------
        # The working directory usually already exists (the caller
        # creates it before calling the backend) but mkdir with
        # exist_ok=True is cheap insurance. Similarly, we resolve
        # stdout/stderr paths against the working directory here so
        # the file paths in the JobResult are absolute and meaningful.
        spec.working_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = spec.resolved_stdout()
        stderr_path = spec.resolved_stderr()

        # --- Step 2: open capture files ----------------------------------
        # subprocess.DEVNULL is a sentinel, not a real file object, so
        # we use it when the user did not request capture. For real
        # capture paths, we open in truncate mode: each case overwrites
        # any previous run's captured output in the same directory.
        stdout_f = open(stdout_path, "w") if stdout_path else subprocess.DEVNULL
        stderr_f = open(stderr_path, "w") if stderr_path else subprocess.DEVNULL

        # --- Step 3: timing + default outcome ----------------------------
        # We measure wall time from here rather than from Popen()
        # construction so the timing includes any OS-level overhead
        # (exec, dynamic-linker, etc.). For HPC simulations this is
        # negligible but we do not want tests to show mysteriously
        # negative differences.
        t0 = time.monotonic()
        outcome = JobOutcome.OK
        rc: int = -1
        err_msg: Optional[str] = None

        try:
            # --- Step 4: build and launch --------------------------------
            # _build_cmd handles MPI launcher prefixing based on the
            # configured launcher and the spec's num_tasks. The
            # subprocess inherits the configured environment and runs
            # in the spec's working directory so relative paths
            # (e.g. "options.toml") resolve correctly.
            cmd = self._build_cmd(spec)
            logger.debug(
                "LocalBackend running %s (cwd=%s)", cmd, spec.working_dir
            )
            proc = subprocess.Popen(
                cmd,
                cwd=str(spec.working_dir),
                env=spec.resolved_env(),
                stdout=stdout_f if stdout_path else subprocess.DEVNULL,
                stderr=stderr_f if stderr_path else subprocess.DEVNULL,
            )

            # --- Step 5: wait with timeout -------------------------------
            # Popen.wait with a timeout is the standard way to enforce
            # a wall-time cap. It raises TimeoutExpired on overrun;
            # any other kind of failure propagates out to the outer
            # try/except below.
            try:
                rc = proc.wait(timeout=spec.duration_s)
            except subprocess.TimeoutExpired:
                # The simulation ran past its allotted wall time. We
                # classify this as TIMEOUT (distinct from FAILED) so
                # driver code can retry with a longer limit if it
                # wants.
                outcome = JobOutcome.TIMEOUT
                err_msg = f"exceeded duration_s={spec.duration_s}"
                if self._kill_on_timeout:
                    proc.kill()
                    try:
                        # Give the kernel a moment to reap the killed
                        # process. If it fails to die even after kill
                        # (extremely rare, usually a zombie), we note
                        # that in the error message but proceed; the
                        # caller cannot productively wait forever.
                        rc = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        err_msg = err_msg + "; process did not exit after kill"
                        # -9 is the conventional "killed by SIGKILL"
                        # indicator in Unix shells; use it here so
                        # the rc alone hints at what happened.
                        rc = -9

        except FileNotFoundError as e:
            # Typical cause: the binary path is wrong. This is
            # user error, not a simulation failure, so we report
            # SUBMIT_ERROR to distinguish. The driver can decide
            # whether to retry or abort.
            outcome = JobOutcome.SUBMIT_ERROR
            err_msg = str(e)
        except OSError as e:
            # Other launch-time problems (bad permissions, too many
            # open files, etc). Same category as FileNotFoundError.
            outcome = JobOutcome.SUBMIT_ERROR
            err_msg = str(e)
        finally:
            # --- Step 6: close capture files -----------------------------
            # Always close the real file handles we opened, even on
            # exception paths. subprocess.DEVNULL is a special sentinel
            # and must NOT be closed - so we branch on whether we
            # opened a real path.
            if stdout_path:
                stdout_f.close()
            if stderr_path:
                stderr_f.close()

        # --- Step 7: build the result -----------------------------------
        wall = time.monotonic() - t0
        # Promote rc != 0 to FAILED unless we have already classified
        # this as something more specific (TIMEOUT, SUBMIT_ERROR).
        # Order matters here: we want the most-specific categorization
        # to stick.
        if outcome == JobOutcome.OK and rc != 0:
            outcome = JobOutcome.FAILED

        return JobResult(
            spec=spec,
            outcome=outcome,
            rc=rc,
            wall_time_s=wall,
            # There is no real "job ID" for a local subprocess, but
            # including the driver PID makes the identifier at least
            # marginally useful when correlating logs across processes.
            jobid=f"local-{os.getpid()}",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error_message=err_msg,
        )

    def poll_stats(self) -> "BackendStats":
        """Return a snapshot of the backend's current resource usage.

        Safe to call from a non-worker thread while the backend is
        busy; accesses running-jobs state under a lock. The snapshot
        is best-effort — between the moment the stats are captured
        and the moment the caller reads them, jobs may have
        started or completed.

        Returns:
            :class:`BackendStats` describing concurrent jobs and
            cores in use. ``cores_total`` reports
            :func:`os.cpu_count`; on cgroups-constrained systems
            (containers, HPC compute nodes with cpuset cgroups) this
            may overreport the cores the process can actually use,
            but Python's stdlib has no portable cgroup-aware cpu
            count, so we report the kernel-level value and leave
            cgroup-awareness to the user.
        """
        with self._running_lock:
            running = list(self._running.values())
        cores_in_use = sum(s.num_tasks * s.cores_per_task for s in running)
        return BackendStats(
            running_jobs=len(running),
            max_concurrent=self._max_workers,
            cores_in_use=cores_in_use,
            cores_total=os.cpu_count() or 0,
        )

    def stream_batch(
        self, specs: Sequence[SimJobSpec]
    ) -> Iterator[JobResult]:
        """Submit all specs and yield results as cases complete.

        For ``max_workers == 1`` the execution is strictly serial:
        each spec is run to completion, its result yielded, then the
        next one starts. This path has no threadpool overhead and
        is easiest to debug under ``pdb``.

        For ``max_workers > 1`` the specs are handed to a
        ``ThreadPoolExecutor``. Results come out in completion order
        via ``concurrent.futures.as_completed``.

        Args:
            specs: Sequence of :class:`SimJobSpec` to run. May be
                empty, in which case the iterator yields nothing.

        Yields:
            One :class:`JobResult` per spec. Completion order, not
            submission order.
        """
        if not specs:
            # Empty input - return empty iterator without starting
            # a threadpool we will not use.
            return

        if self._max_workers == 1:
            # Strictly serial path. Single-threaded execution is
            # fastest for small batches and it makes stack traces
            # much easier to read when something goes wrong.
            for s in specs:
                yield self._run_one(s)
            return

        # Parallel path. ThreadPoolExecutor is fine here because each
        # worker spends almost all its time blocked on subprocess I/O,
        # not doing Python-level work, so the GIL does not hurt us.
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(self._run_one, s): s for s in specs}
            for fut in as_completed(futures):
                yield fut.result()
