"""
Flux-based job backend using ``flux.job.FluxExecutor``.

Why FluxExecutor
----------------
Flux's python bindings provide two ways to submit and manage jobs:

* The low-level way: call ``flux.job.submit`` to get a jobid, then
  track jobs yourself through ``flux.job.wait``, ``flux.job.JobList``,
  ``flux.job.stats``, etc. This is what the original ``flux_map.py``
  did; the code is verbose and easy to get wrong because you are
  juggling several parallel data structures.
* The high-level way: use ``FluxExecutor``, which wraps all of that
  into a ``concurrent.futures``-style interface. Submissions return
  futures, completion arrives via ``as_completed``, and the executor
  shutdown handles cleanup. Much simpler code.

This backend uses the high-level interface. The old hand-rolled
``submit`` / ``wait`` / ``JobList`` bookkeeping is gone, which removes
several classes of bug (forgotten jobids, out-of-order waits) and
cuts the code roughly in half.

Requirements
------------
* Flux python bindings installed and importable.
* The driver must be running inside an active Flux instance - either
  started automatically by a SLURM/LSF allocation script that does
  ``flux start python driver.py``, or started interactively with
  ``flux start``. The backend does not start a Flux instance itself.

Restart caveat (important)
--------------------------
When the enclosing HPC allocation is killed, the Flux instance dies
with it. Any in-flight Flux jobids are unrecoverable afterwards: you
cannot query ``flux.job.wait(old_jobid)`` from a new Flux instance to
learn the fate of a job from the previous instance. Restart must
therefore be driven by filesystem state (the manifest plus per-case
sentinel files), NOT by Flux. This backend does not attempt any
Flux-state persistence. See :mod:`workflow_common.manifest` and
:mod:`workflow_common.sentinel` for the restart machinery.

Live status
-----------
:meth:`FluxBackend.poll_stats` returns a snapshot of the current
batch's active futures split into estimated running jobs and queued
jobs, plus Flux's allocated-core count. Useful for progress logging
from a separate thread while ``stream_batch`` runs. Safe to call
concurrently with submission.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

# Import flux lazily-ish: we import at module load so the module is
# usable as-imported, but we catch the ImportError so the workflow_common
# top-level package can still load on a machine without flux bindings.
# Callers that explicitly want this backend will catch the ImportError
# themselves.
try:
    import flux  # noqa: F401
    import flux.job
    from flux.job import FluxExecutor, JobspecV1
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FluxBackend requires the 'flux' python bindings. Install flux "
        "and make sure its python module is on PYTHONPATH, or use "
        "LocalBackend for off-cluster development."
    ) from e

from ..logging_utils import get_logger
from ..platform_detect import is_spectrum_machine
from .base import (
    BackendStats,
    BaseBackend,
    JobOutcome,
    JobResult,
    SimJobSpec,
)

logger = get_logger(__name__)


class FluxBackend(BaseBackend):
    """JobBackend backed by ``flux.job.FluxExecutor``.

    Args:
        spectrum_mpi: Whether to set the ``mpi=spectrum`` flux shell
            option on every job. ``None`` (the default) means
            auto-detect via :func:`is_spectrum_machine`. Override
            with a bool if you have a reason to force one way or
            the other - typically only needed in tests or for
            unusual hostnames the auto-detector does not recognize.
        gpu_affinity: Value for the ``gpu-affinity`` flux shell
            option. Default ``"per-task"`` matches the existing
            ExaConstit convention and is what almost every workflow
            wants. Set to empty string or ``None`` to skip setting
            the option entirely.
        cpu_affinity: Value for the ``cpu-affinity`` flux shell
            option. Same conventions as ``gpu_affinity``.
        threads: Worker thread count for the FluxExecutor itself.
            ``None`` (default) lets flux pick. Usually fine to leave
            alone; only tune if you see the executor becoming a
            bottleneck for very large batches.
        extra_shell_options: Optional mapping of additional flux
            shell option names to values. Applied to every jobspec
            after the built-in affinity options. Use for bespoke
            settings like ``"rlimit"`` or ``"pty"`` that specific
            workflows need.

    Example:
        Default construction - auto-detects Spectrum MPI, uses
        per-task affinity::

            backend = FluxBackend()
            results = backend.submit_batch(specs)

        Explicit overrides (useful on unusual hosts)::

            backend = FluxBackend(
                spectrum_mpi=False,
                gpu_affinity="per-task",
                cpu_affinity="off",
                extra_shell_options={"pty": False},
            )
    """

    def __init__(
        self,
        *,
        spectrum_mpi: Optional[bool] = None,
        gpu_affinity: str = "per-task",
        cpu_affinity: str = "per-task",
        threads: Optional[int] = None,
        extra_shell_options: Optional[Dict[str, object]] = None,
    ):
        # None means "auto-detect"; explicit True/False lets tests
        # force a value or lets callers override the detection on
        # hosts we don't recognize.
        self._spectrum = (
            is_spectrum_machine() if spectrum_mpi is None else spectrum_mpi
        )
        self._gpu_affinity = gpu_affinity
        self._cpu_affinity = cpu_affinity
        self._threads = threads
        self._extra_shell_options = dict(extra_shell_options or {})
        # Keep a best-effort view of this backend instance's submitted
        # but unfinished specs. Flux may queue some of these until
        # resources become available, so poll_stats() separates
        # "running" from "queued" using Flux's allocated-core count.
        import threading
        self._running_lock = threading.Lock()
        self._active: Dict[int, SimJobSpec] = {}
        self._last_batch_min_cores = 0

    # --- jobspec construction --------------------------------------------

    def _build_jobspec(self, spec: SimJobSpec) -> JobspecV1:
        """Translate a :class:`SimJobSpec` into a Flux ``JobspecV1``.

        Flux has its own jobspec datatype; this is where we map
        framework-agnostic fields into Flux-specific ones. The bulk
        of the work is setting shell options for MPI flavor and
        CPU/GPU affinity.

        What's a "shell option"?
            When a Flux job launches, a small program called the "job
            shell" runs on each allocated node and is responsible for
            bringing up the simulation - setting up MPI bootstrap info,
            binding ranks to cores and GPUs, opening stdio redirects,
            and finally exec'ing the user command. Shell options are
            key/value pairs attached to the jobspec that the job shell
            reads on startup to configure itself. They are the
            narrow-waist customization point for things that are not
            part of the resource request itself: "use Spectrum MPI",
            "bind each rank to its own GPU", and so on.

        Args:
            spec: The framework spec to translate.

        Returns:
            A fully-populated ``JobspecV1`` ready for submission.

        Shell options set by this method (each described below):
            * ``mpi=spectrum`` (conditional on ``self._spectrum``)
            * ``gpu-affinity=<value>`` (conditional on
              ``self._gpu_affinity``)
            * ``cpu-affinity=<value>`` (conditional on
              ``self._cpu_affinity``)
            * Anything in ``self._extra_shell_options``, applied last
              so user-supplied values override the defaults if they
              collide on key name.
        """
        # --- Step 1: normalize GPU count --------------------------------
        # Flux treats ``gpus_per_task=None`` as "no GPUs" and omits the
        # GPU resource request from the generated jobspec entirely.
        # Passing ``0`` explicitly is NOT the same thing on some flux
        # releases - it can cause the GPU resource request to be
        # emitted as zeros, which has been observed to confuse the
        # scheduler into allocating jobs to the wrong queue. Normalize
        # to None here so we only emit GPU requests when we actually
        # want them.
        gpt = spec.gpus_per_task if spec.gpus_per_task > 0 else None

        # --- Step 2: build the base jobspec -----------------------------
        # JobspecV1.from_command is the canonical entry point for "I
        # have a command and a resource request". It produces a fully
        # valid jobspec with the command, resource layout, and default
        # attributes already populated; we then layer customization on
        # top via attribute assignment.
        js = JobspecV1.from_command(
            [str(spec.binary), *spec.args],
            num_nodes=spec.num_nodes,
            num_tasks=spec.num_tasks,
            cores_per_task=spec.cores_per_task,
            gpus_per_task=gpt,
        )

        # --- Step 3: filesystem and I/O attributes ----------------------
        # cwd: the working directory the job shell ``cd``s into before
        # exec'ing the command. Relative argv paths resolve against it.
        # We always send Flux an absolute path here — Flux's behavior
        # with a relative cwd is version-dependent (some versions
        # ``os.chdir`` from the broker's cwd, which isn't the driver's
        # cwd in a submit-only workflow). Resolving to absolute up
        # front removes that ambiguity.
        js.cwd = str(Path(spec.working_dir).resolve())
        # stdout / stderr: if set, the job shell redirects the command's
        # stdio into these files on each node. If unset (left at
        # default), flux captures them into its KVS and makes them
        # available via ``flux job attach`` - handy for interactive
        # debugging but not durable across broker restarts. We pass
        # the resolved (absolute) paths rather than the user's raw
        # string so the resolution logic matches LocalBackend's and
        # the file lands in the same place regardless of Flux version.
        resolved_out = spec.resolved_stdout()
        resolved_err = spec.resolved_stderr()
        if resolved_out is not None:
            js.stdout = str(resolved_out)
        if resolved_err is not None:
            js.stderr = str(resolved_err)
        # environment: the job shell exports these before exec. We use
        # ``spec.resolved_env()`` which inherits the parent env when
        # spec.env is None, matching what almost every workflow expects.
        js.environment = spec.resolved_env()
        # duration: wall-time limit in seconds. Flux will SIGTERM the
        # job shell when this expires; the shell then forwards SIGTERM
        # to the command. If the command does not die promptly flux
        # eventually escalates to SIGKILL.
        js.duration = int(spec.duration_s)

        # --- Step 4: shell options --------------------------------------
        # Each block below is conditional on the resolver's config; a
        # caller can opt out of any by passing an empty-string or None
        # to the constructor. The extra_shell_options loop runs last so
        # user-specified values override anything we set above if the
        # keys collide.

        # 4a. MPI flavor. On IBM Spectrum MPI systems (Lassen, Sierra,
        # Summit, etc.) flux needs to know to use the Spectrum-provided
        # PMIX shim rather than its own built-in PMI. Without this, MPI
        # ranks fail to wire themselves up and ``MPI_Init`` hangs or
        # aborts. On OpenMPI / MPICH systems the built-in PMI works
        # fine and this option should NOT be set - doing so would
        # actually break things. ``self._spectrum`` defaults to an
        # auto-detection of the hostname in ``__init__``.
        if self._spectrum:
            js.setattr_shell_option("mpi", "spectrum")

        # 4b. GPU affinity. Controls how GPUs are bound to ranks.
        #   "per-task"  - each rank gets its own GPU slice (the usual
        #                 choice for domain-decomposed GPU codes like
        #                 ExaConstit where each MPI rank owns one GPU).
        #   "off"       - no binding; ranks see all GPUs on the node.
        #                 Use for single-rank-per-node codes that want
        #                 to manage CUDA devices explicitly themselves.
        # If the value is a falsy string, we skip the option entirely,
        # which leaves flux at its compiled-in default.
        if self._gpu_affinity:
            js.setattr_shell_option("gpu-affinity", self._gpu_affinity)

        # 4c. CPU affinity. Same idea for CPU cores.
        #   "per-task"  - each rank is pinned to its allocated core(s).
        #                 This is what you want for OMP+MPI hybrid codes
        #                 so threads stay on the right socket.
        #   "off"       - no pinning; the OS scheduler is free to move
        #                 ranks around. Useful when running multiple
        #                 cases per node where pinning would collide.
        if self._cpu_affinity:
            js.setattr_shell_option("cpu-affinity", self._cpu_affinity)

        # 4d. Any additional shell options the caller wants, applied
        # last so they override our defaults on key collision. Example
        # use cases:
        #   {"pty": False}              - disable the interactive PTY
        #                                 shell plugin, recommended for
        #                                 batch runs.
        #   {"rlimit": "core=0"}        - zero the core-file size limit.
        #   {"stop-tasks-in-exec": 1}   - stop ranks right after exec
        #                                 for debugger attach.
        for k, v in self._extra_shell_options.items():
            js.setattr_shell_option(k, v)

        return js

    # --- main submission path --------------------------------------------

    def stream_batch(
        self, specs: Sequence[SimJobSpec]
    ) -> Iterator[JobResult]:
        """Submit all specs and yield results as cases complete.

        The flow is:

        1. Enter a ``FluxExecutor`` context manager. On exit the
           executor blocks for any in-flight work and releases
           resources, which protects against leaking Flux handles
           if the caller abandons iteration midway.
        2. Submit every spec, collecting futures and stamping each
           with a submission timestamp (used for wall-time
           measurement).
        3. Consume futures via ``concurrent.futures.as_completed``.
           For each completed future, translate it to a
           :class:`JobResult` and yield it.

        Args:
            specs: Sequence of :class:`SimJobSpec` to run. Empty
                sequences are handled as a no-op.

        Yields:
            One :class:`JobResult` per spec, in completion order.

        FluxExecutor lifecycle, narrated
        --------------------------------
        The ``FluxExecutor`` object manages a pool of worker threads
        that talk to the local Flux broker. Its job is to turn each
        ``submit()`` call into a broker-level submission and return a
        future that completes when the corresponding job finishes.
        The lifecycle stages that matter here are::

            [enter context]        <-- creates broker handle, spins up
                                       worker threads, starts processing
                                       submissions

                ... submit() ...   <-- each call sends a jobspec to the
                                       broker; returns a Future almost
                                       immediately. The job itself may
                                       sit in the broker's pending
                                       queue for a while.

                ... future fires   <-- when flux detects the job has
                                       exited, it fulfills the future
                                       with the rc (or sets an exception
                                       if something went wrong at the
                                       submission/broker level).

            [exit context]         <-- drains in-flight futures, joins
                                       worker threads, closes the broker
                                       handle. This is the critical
                                       cleanup step: without it,
                                       abandoning the iterator partway
                                       would leave dangling threads and
                                       an open broker connection.

        We deliberately yield *inside* the ``with`` block so the
        executor stays alive until the caller has consumed every
        result. If we yielded outside, the executor would be torn down
        at the end of the generator body while futures might still be
        in progress, which would race.

        Subtlety: context-manager drain behavior
        ----------------------------------------
        On recent flux releases the ``FluxExecutor.__exit__`` call
        waits for pending futures to complete before returning - i.e.
        it behaves like ``shutdown(wait=True)``. If a user code path
        raises partway through consuming results, we still want to
        let the broker finish any jobs it has already launched so
        they do not get orphaned with their output files
        half-written. The context-manager exit handles this
        automatically, which is why we prefer it over an explicit
        ``executor.shutdown()`` call.

        On very old flux releases (pre-0.45ish) the context-manager
        exit was ``shutdown(wait=False)``, which would let in-flight
        jobs die alongside the executor. If you observe orphaned
        Flux jobs on a particular machine, check the flux version
        first.
        """
        if not specs:
            # Empty input - return empty iterator without starting
            # a broker handle we would only immediately tear down.
            return

        # --- Step 0: build executor kwargs ------------------------------
        # FluxExecutor accepts a 'threads' kwarg only in recent
        # releases. We pass it conditionally so older flux versions
        # keep working without us having to probe their API shape.
        executor_kwargs: Dict[str, object] = {}
        if self._threads is not None:
            executor_kwargs["threads"] = self._threads

        # --- Step 0.5: per-future bookkeeping ---------------------------
        # We need two lookups when a future completes:
        #   * which spec produced it, so the JobResult can reference
        #     the originating spec (callers correlate results back to
        #     specs via identity);
        #   * when the submission happened, so the JobResult can
        #     compute wall-clock runtime.
        # Using the future object as a dict key is safe because
        # FluxExecutorFuture instances are hashable (by identity).
        submission_ts: Dict[object, float] = {}
        future_to_spec: Dict[object, SimJobSpec] = {}

        with FluxExecutor(**executor_kwargs) as executor:
            batch_min_cores = min(
                max(1, spec.num_tasks * spec.cores_per_task)
                for spec in specs
            )
            with self._running_lock:
                self._last_batch_min_cores = batch_min_cores
            # --- Phase 1: submit everything -----------------------------
            # We fire every submission up-front rather than streaming
            # them in. The broker will queue any that do not fit
            # immediately in the current allocation; from our side,
            # we get a future for each one right away and can move on
            # to consuming results.
            #
            # Creating working directories on demand here means the
            # caller does not have to pre-create them. We also log a
            # per-submission debug line for traceability - on a run
            # with 1000+ cases these are invaluable when something
            # misbehaves.
            futures = []
            for spec in specs:
                spec.working_dir.mkdir(parents=True, exist_ok=True)
                js = self._build_jobspec(spec)
                t_sub = time.monotonic()
                fut = executor.submit(js)
                futures.append(fut)
                submission_ts[fut] = t_sub
                future_to_spec[fut] = spec
                with self._running_lock:
                    self._active[id(spec)] = spec
                logger.debug(
                    "FluxBackend submitted %s (nodes=%d, tasks=%d, gpt=%d)",
                    spec.tag or spec.working_dir.name,
                    spec.num_nodes,
                    spec.num_tasks,
                    spec.gpus_per_task,
                )

            # --- Phase 2: consume as completed --------------------------
            # as_completed yields futures in the order they fire, not
            # the order they were submitted. That is the whole point:
            # if gene 7 finishes before gene 3, the caller sees the
            # gene 7 result first and can log / react without waiting.
            #
            # Yielding inside the ``with`` block keeps the executor
            # alive until the caller has consumed every result. Do
            # NOT move this outside the block - the executor's
            # context-manager exit would then fire before the caller
            # has picked up all the results, potentially orphaning
            # in-flight jobs.
            for fut in _as_completed_flux(futures):
                spec = future_to_spec[fut]
                t_sub = submission_ts[fut]
                with self._running_lock:
                    self._active.pop(id(spec), None)
                yield _future_to_result(fut, spec, t_sub)
        with self._running_lock:
            self._active.clear()

    # --- live status polling --------------------------------------------

    def poll_stats(self) -> BackendStats:
        """Return a snapshot of the Flux instance's current usage.

        Intended for use by a separate progress-logging thread that
        periodically prints "pending=X running=Y completed=Z" while
        the main thread iterates over ``stream_batch``. Safe to call
        concurrently with submission.

        Returns:
            A :class:`BackendStats` snapshot. ``cores_total`` and
            ``cores_in_use`` come from Flux's scheduler resource view:
            total allocation cores and currently allocated cores,
            respectively. ``running_jobs`` is estimated from allocated
            cores divided by the smallest per-job core request in the
            current batch. ``queued_jobs`` is the remainder of this
            backend's unfinished submitted futures. ``max_concurrent``
            is estimated as ``cores_total // per_job_cores`` using the
            same per-job core request. On any error, returns an all-zero
            snapshot and logs a warning rather than raising, so the
            caller's progress thread keeps running.

        Example:
            Progress printer running in a background thread::

                import threading, time
                def progress():
                    while not done.is_set():
                        s = backend.poll_stats()
                        logger.info(
                            "pending=%d running=%d done=%d",
                            0,
                            s.running_jobs,
                            0,
                        )
                        time.sleep(30)
                threading.Thread(target=progress, daemon=True).start()
        """
        try:
            handle = flux.Flux()
            resources = flux.resource.list.resource_list(handle).get()
            cores_total = int(resources.all.ncores)
            cores_in_use = int(resources.allocated.ncores)

            with self._running_lock:
                active_specs = list(self._active.values())
                batch_min_cores = self._last_batch_min_cores

            max_concurrent = 0
            if cores_total > 0 and batch_min_cores > 0:
                max_concurrent = max(1, cores_total // batch_min_cores)

            running = 0
            queued = 0
            if active_specs and batch_min_cores > 0:
                running = min(
                    len(active_specs),
                    max(0, cores_in_use) // batch_min_cores,
                )
                queued = max(0, len(active_specs) - running)

            return BackendStats(
                running_jobs=running,
                max_concurrent=max_concurrent,
                cores_in_use=cores_in_use,
                cores_total=cores_total,
                queued_jobs=queued,
            )
        except Exception as e:  # pragma: no cover
            logger.warning("FluxBackend.poll_stats failed: %s", e)
            return BackendStats(
                running_jobs=0,
                max_concurrent=0,
                cores_in_use=0,
                cores_total=0,
            )


# --- module-level helpers -------------------------------------------------

def _as_completed_flux(futures: List[object]) -> Iterator[object]:
    """Yield flux futures in completion order.

    ``FluxExecutorFuture`` is duck-compatible with
    ``concurrent.futures.Future`` for the attributes ``as_completed``
    cares about, so we can just delegate. Factoring this into its own
    function keeps the submission loop tidy and makes it trivial to
    swap in a different ordering strategy in the future (for example,
    yielding in submission order rather than completion order).

    Args:
        futures: List of flux futures to monitor.

    Yields:
        Each future, as soon as it completes.
    """
    from concurrent.futures import as_completed

    yield from as_completed(futures)


def _future_to_result(
    fut: object, spec: SimJobSpec, submit_ts: float
) -> JobResult:
    """Translate a completed flux future into a :class:`JobResult`.

    Handles three outcomes:

    1. ``fut.result()`` raises. Something went wrong at the flux
       submission or broker level - the simulation never ran or
       flux lost track of it. Reported as :class:`JobOutcome.SUBMIT_ERROR`.
    2. ``fut.result()`` returns 0. Success.
    3. ``fut.result()`` returns nonzero. The simulation ran and
       exited with an error. Reported as :class:`JobOutcome.FAILED`,
       with the exact rc preserved on the result so callers can
       inspect it if they care (e.g. rc == 143 means SIGTERM).

    We also try to capture the flux jobid for logging, but swallow
    errors from ``fut.jobid()`` since it may not be available on
    every flux version.

    Args:
        fut: The completed flux future.
        spec: The :class:`SimJobSpec` that produced the future.
        submit_ts: ``time.monotonic()`` value at submission, for
            wall-time computation.

    Returns:
        A :class:`JobResult`.
    """
    wall = time.monotonic() - submit_ts
    try:
        rc = int(fut.result())  # type: ignore[attr-defined]
    except Exception as e:
        # Broker error, bad jobspec, or some other submission-level
        # failure. The simulation never got a chance to run.
        return JobResult(
            spec=spec,
            outcome=JobOutcome.SUBMIT_ERROR,
            rc=-1,
            wall_time_s=wall,
            error_message=str(e),
        )

    # Best-effort jobid retrieval. The timeout=0 form is non-blocking
    # and returns immediately with whatever the broker has cached. On
    # older flux releases it may raise; we treat that as "no jobid
    # available" and carry on.
    try:
        jobid = str(fut.jobid(timeout=0))  # type: ignore[attr-defined]
    except Exception:
        jobid = None

    if rc == 0:
        outcome = JobOutcome.OK
    else:
        # Note: flux surfaces "killed by signal" as rc == 128+signo
        # on most releases but as a negative number on some. We do
        # not try to distinguish; callers that need to know the
        # signal should inspect rc themselves.
        outcome = JobOutcome.FAILED

    return JobResult(
        spec=spec,
        outcome=outcome,
        rc=rc,
        wall_time_s=wall,
        jobid=jobid,
        stdout_path=spec.resolved_stdout(),
        stderr_path=spec.resolved_stderr(),
    )
