"""
Core types for the job-execution abstraction.

This module defines the vocabulary that backends and driver code use
to talk about simulation runs:

* :class:`SimJobSpec` - the input to a backend. An immutable record
  describing *what* to run and *how much* hardware it needs.
* :class:`JobResult` - the output from a backend. An immutable record
  describing *what happened* when the simulation ran.
* :class:`JobOutcome` - a small enum of terminal status categories.
* :class:`JobBackend` - the Protocol that every backend implements.
* :class:`BaseBackend` - a mixin that concrete backends can inherit
  to get default ``submit_batch`` / ``submit_one`` implementations
  for free.

The whole abstraction fits in this one file on purpose. Everything
else in the backends subpackage is a concrete implementation.

Data flow through a backend
---------------------------
::

    driver code                     backend                    simulation
    -----------                     -------                    ----------

    build SimJobSpec     ------>    build launcher cmd
                                    (argv for LocalBackend;
                                     JobspecV1 for FluxBackend)
                                              |
                                              v
                                     launch subprocess   ----->  simulation
                                                                   runs
                                              |                    |
                                              v                    v
                                     wait for exit      <-----  exit + rc
                                              |
                                              v
                                    build JobResult
                                              |
    JobResult            <------    return / yield
    - correlate to spec
    - validate outputs
    - record terminal
      state in manifest
    - write sentinel

The driver only ever sees the ``SimJobSpec`` -> ``JobResult`` pair.
Everything in between is the backend's responsibility and differs
across implementations.

Concepts briefly, for readers without a software-engineering background
-----------------------------------------------------------------------
A few Python-specific patterns come up repeatedly below; this is the
minimum-jargon version of each.

* **Dataclass** - a class defined by listing its fields as annotated
  attributes. Python generates ``__init__``, ``__repr__``, and
  equality automatically. Marking one ``frozen=True`` makes its
  instances immutable, which is what we want for values like specs
  and results that get logged, cached, and compared.
* **Protocol** - a pure interface description: "any object with these
  method names and signatures is acceptable here". Unlike classical
  inheritance, no ``class X(Protocol)`` subclass declaration is
  required on the implementer. This is called *structural* typing
  and it makes backends easy to swap out, and easy to test against
  with minimal fakes.
* **Future** - an object representing work that will eventually
  produce a result. Code can submit work, keep a reference to the
  future, and either wait on it (``fut.result()``) or iterate over
  a batch of futures as they complete (``as_completed``). Both
  backends use futures internally - the thread pool in the local
  backend, and ``FluxExecutor`` in the Flux backend.
* **Iterator** - an object you loop over with ``for``. A function
  that *yields* values instead of returning a list produces an
  iterator; callers can consume values as soon as they are available
  without waiting for the whole batch. ``stream_batch`` uses this
  pattern so the driver can log or react to each completed case
  as soon as it finishes.

Design notes
------------
``SimJobSpec`` is frozen (an immutable dataclass). Specs represent
pure values and are often retried, logged, or reconstructed from a
manifest. Allowing them to mutate would make those workflows fragile.

``JobResult`` carries a reference to the originating ``SimJobSpec``.
This lets driver code correlate async results back to specs without
needing separate bookkeeping dicts. It also means a logged result
fully describes the case that produced it.

``JobBackend`` has three methods, but a concrete backend typically
only needs to implement one of them (``stream_batch``). The other
two fall out of :class:`BaseBackend`'s default implementations.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


class JobOutcome(str, Enum):
    """Terminal status categories for a single simulation submission.

    Inherits from ``str`` so values serialize transparently in JSON
    and are readable when logged. The categories are intended to be
    mutually exclusive: one result gets one outcome.

    Members:
        OK: The simulation exited with rc == 0 and any sanity checks
            the backend performs internally also passed. Note that
            further validation (output-file sanity, physical
            plausibility) happens outside the backend and may still
            flip an OK result to a workflow-level failure.
        FAILED: The simulation exited with a nonzero rc. The backend
            does not try to distinguish "crashed" from "exited
            cleanly with error code"; rc alone is the signal.
        TIMEOUT: The backend killed the simulation because it
            exceeded its ``duration_s``. Distinct from FAILED so
            driver code can apply different policy (e.g. retry with
            a longer timeout).
        SUBMIT_ERROR: The backend was unable to launch the simulation
            at all - missing binary, bad jobspec, broker unreachable,
            etc. The simulation never actually ran.
        CANCELLED: The submission was cancelled by user action or a
            signal. Retained for future use; current backends do not
            produce this outcome but Ctrl-C handling may eventually.
    """

    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SUBMIT_ERROR = "submit_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SimJobSpec:
    """Everything a backend needs to launch one simulation run.

    Specs are immutable (``frozen=True``). If you need to modify a
    spec, call ``dataclasses.replace`` to get a new one with the
    updated fields. Doing it this way keeps specs safe to pass around
    and log.

    Fields:
        working_dir: Directory the simulation process runs in. Any
            relative ``stdout`` / ``stderr`` paths are resolved
            against this directory. The directory is typically
            created by the caller before the backend is invoked,
            though backends may also create it defensively.
        binary: Path to the executable. May be absolute or relative
            to PATH; backends pass it through to their respective
            launch mechanism unchanged.
        args: Command-line arguments to pass after ``binary``. The
            binary itself is NOT duplicated in this tuple. Stored as
            a tuple rather than a list so the dataclass stays
            hashable (lists are not hashable).
        num_nodes: Number of nodes to allocate. For single-node runs,
            set to 1.
        num_tasks: Number of MPI ranks. For non-MPI runs, set to 1.
        cores_per_task: Per-rank core count. Usually 1 for pure-MPI
            codes; higher values make sense for MPI+OpenMP hybrid
            codes. For GPU-only codes, leave at 1 regardless.
        gpus_per_task: GPUs per rank. 0 means CPU-only.
        duration_s: Maximum wall time in seconds. Backends may kill
            the simulation if it exceeds this. A reasonable default
            for quick debugging is one hour (3600).
        stdout: Optional filename or absolute path for captured
            stdout. ``None`` discards stdout (routed to /dev/null
            on the local backend, inherited by the scheduler on
            Flux). Relative paths resolve against ``working_dir``.
        stderr: Same as ``stdout`` but for stderr.
        env: Environment variable mapping. ``None`` means inherit the
            parent process's environment. Supply a fresh dict to run
            in a clean environment (note: flux backends may still
            inject their own vars on top).
        tag: Free-form label used for logging and tracing. Backends
            do not interpret its value; pass something that uniquely
            identifies the case in your logs, such as
            ``"gen0_gene3_obj1"``.
        extra: Backend-specific hints. Each backend documents the
            keys it recognizes; unknown keys are ignored. Use this
            for one-off per-case overrides rather than polluting
            the core spec.

    Example:
        A minimal CPU-only, single-rank spec::

            spec = SimJobSpec(
                working_dir=Path("wf/gen_0/gene_3"),
                binary=Path("/opt/bin/mechanics"),
                args=("-opt", "options.toml"),
                num_tasks=1,
                duration_s=600,
                stdout="run.out",
                stderr="run.err",
                tag="gen0_gene3",
            )

        A GPU-enabled 4-rank spec on 1 node::

            spec = SimJobSpec(
                working_dir=case_dir,
                binary=mechanics_path,
                args=("-opt", "options.toml"),
                num_nodes=1,
                num_tasks=4,
                gpus_per_task=1,
                duration_s=3600,
            )
    """

    working_dir: Path
    binary: Path
    args: Tuple[str, ...] = ()
    num_nodes: int = 1
    num_tasks: int = 1
    cores_per_task: int = 1
    gpus_per_task: int = 0
    duration_s: int = 3600
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    env: Optional[Mapping[str, str]] = None
    tag: Optional[str] = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def resolved_stdout(self) -> Optional[Path]:
        """Return the absolute stdout path, or ``None`` if stdout is unset.

        If ``stdout`` is a relative path, it is resolved against
        ``working_dir`` so the file lands next to the simulation's
        other outputs. Backends use this method rather than reading
        the raw ``stdout`` field so the resolution logic lives in
        one place.

        Returns:
            A ``Path`` if ``stdout`` is set, else ``None``.
        """
        if self.stdout is None:
            return None
        p = Path(self.stdout)
        return p if p.is_absolute() else (self.working_dir / p)

    def resolved_stderr(self) -> Optional[Path]:
        """Return the absolute stderr path, or ``None`` if stderr is unset.

        See :meth:`resolved_stdout`; the behavior is identical.
        """
        if self.stderr is None:
            return None
        p = Path(self.stderr)
        return p if p.is_absolute() else (self.working_dir / p)

    def resolved_env(self) -> Dict[str, str]:
        """Return the environment dict to hand to the simulation process.

        If ``env`` is ``None`` (the default), the current process's
        environment is inherited wholesale. Callers that need a
        clean or modified environment should build a dict and pass
        it in explicitly.

        Returns:
            A plain ``dict`` of environment variables. A copy is
            returned in both branches so callers mutating the
            result cannot corrupt ``os.environ``.
        """
        if self.env is None:
            return dict(os.environ)
        return dict(self.env)


@dataclass(frozen=True)
class JobResult:
    """Outcome of running one :class:`SimJobSpec`.

    Results are immutable. The original spec is included so that
    async consumers (``stream_batch``) can correlate the result back
    to its input without maintaining external bookkeeping.

    Fields:
        spec: The :class:`SimJobSpec` that produced this result.
        outcome: High-level category (see :class:`JobOutcome`).
        rc: Return code from the simulation. Negative values are
            sometimes used by backends to indicate "the process was
            killed by a signal" or "we never got one"; driver code
            that cares about the distinction should inspect
            ``outcome`` first.
        wall_time_s: Wall-clock runtime in seconds, measured from
            submission (not from start of execution, which may
            differ when jobs queue). Useful for cost tracking.
        jobid: Optional backend-assigned identifier, as a string.
            Flux encodes this as a short hash-like token; the local
            backend uses ``"local-<pid>"``.
        stdout_path: Resolved path to the captured stdout file, if
            any. ``None`` when the spec did not set ``stdout``.
        stderr_path: Resolved path to the captured stderr file, if
            any.
        error_message: Free-form explanation when the outcome is
            something other than OK. Content depends on the failure
            mode: Python exception repr for SUBMIT_ERROR, a short
            "exceeded duration_s" for TIMEOUT, etc.
        submitted_ts: Unix timestamp of submission. Defaults to the
            construction time of the result, which is the finish
            time of the job - so by default this represents
            "finished at" unless a backend overrides it.
    """

    spec: SimJobSpec
    outcome: JobOutcome
    rc: int
    wall_time_s: float
    jobid: Optional[str] = None
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    error_message: Optional[str] = None
    submitted_ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class BackendStats:
    """Snapshot of a backend's current concurrent-job state.

    Returned by :meth:`JobBackend.poll_stats` for progress reporters
    that want to display live in-use/available resource counts. All
    fields are best-effort: the numbers are accurate at the moment
    :meth:`poll_stats` captured them but jobs may have started or
    completed by the time the caller reads them.

    Fields:
        running_jobs: Number of simulations currently executing.
        queued_jobs: Number of simulations submitted to the backend but
            waiting for resources. Backends that do not distinguish
            queued from running jobs leave this as 0.
        max_concurrent: Upper bound on ``running_jobs`` — typically
            the backend's worker-pool size. A separate field because
            ``running_jobs < max_concurrent`` is informative (the
            pool isn't saturated — likely waiting on the driver to
            submit more work, or the batch is finishing up).
        cores_in_use: Sum of ``num_tasks * cores_per_task`` across
            running jobs. For a typical pure-MPI setup where
            ``cores_per_task = 1`` this is just total rank count.
        cores_total: Cores the backend can schedule across.
            :class:`LocalBackend` reports :func:`os.cpu_count`;
            :class:`FluxBackend` reports the allocation's core count.
            May be 0 on backends that don't track this (a progress
            reporter should then display "cores: N/A").
    """

    running_jobs: int
    max_concurrent: int
    cores_in_use: int
    cores_total: int
    queued_jobs: int = 0


@runtime_checkable
class JobBackend(Protocol):
    """Protocol any job-execution backend must satisfy.

    A minimal backend implementation needs only to provide
    :meth:`stream_batch`. The :class:`BaseBackend` mixin provides
    :meth:`submit_batch` and :meth:`submit_one` for free in terms
    of ``stream_batch``.

    Methods:
        stream_batch(specs):
            Submit every spec. Yield :class:`JobResult` objects as
            individual cases complete, in completion order. Callers
            use this when they want to log or react to cases as
            soon as they finish.
        submit_batch(specs):
            Submit every spec. Return a list of results aligned to
            the input order (so ``result_list[i]`` corresponds to
            ``specs[i]``). Blocks until all cases finish. Callers
            use this when they just want all results at once.
        submit_one(spec):
            Convenience wrapper for submitting a single spec.

    Note on ``@runtime_checkable``: this decorator lets
    ``isinstance(obj, JobBackend)`` succeed for any object with the
    right methods, not just explicit subclasses. Useful for driver
    code that wants to accept any duck-typed backend.
    """

    def stream_batch(
        self, specs: Sequence[SimJobSpec]
    ) -> Iterator[JobResult]: ...

    def submit_batch(
        self, specs: Sequence[SimJobSpec]
    ) -> List[JobResult]: ...

    def submit_one(self, spec: SimJobSpec) -> JobResult: ...

    def poll_stats(self) -> BackendStats: ...


class BaseBackend:
    """Mixin providing ``submit_batch`` / ``submit_one`` from ``stream_batch``.

    Inheriting from this class means a concrete backend only needs to
    implement :meth:`stream_batch` - the other Protocol methods
    follow automatically. This eliminates duplicated "iterate and
    collect" plumbing across backends.

    Subclasses can override the default implementations if they
    have a more efficient native path for batch submission.
    """

    def submit_batch(
        self, specs: Sequence[SimJobSpec]
    ) -> List[JobResult]:
        """Submit all specs and return results in submission order.

        Blocks until every case finishes. Results are returned in
        the order their specs were given, so ``result[i].spec is
        specs[i]`` (identity match). This is stricter than just
        ``==`` because it leaves no ambiguity if two specs happened
        to compare equal.

        Args:
            specs: Sequence of :class:`SimJobSpec` to run.

        Returns:
            A list of :class:`JobResult` with length equal to
            ``len(specs)``, aligned to submission order.

        Raises:
            RuntimeError: If ``stream_batch`` does not yield a
                result for every spec. This indicates a bug in the
                concrete backend.
        """
        # We identify specs by their Python object id because two
        # distinct specs with identical field values would otherwise
        # be confused by a naive equality-based mapping. id() is
        # safe here because specs are frozen and the dict only lives
        # for the duration of this call.
        pos: Dict[int, int] = {id(s): i for i, s in enumerate(specs)}
        out: List[Optional[JobResult]] = [None] * len(specs)
        for r in self.stream_batch(specs):  # type: ignore[attr-defined]
            i = pos[id(r.spec)]
            out[i] = r
        # Defensive check: make sure the backend produced a result
        # for every spec. A missing entry almost certainly indicates
        # a backend bug rather than a user error.
        if any(r is None for r in out):
            missing = [i for i, r in enumerate(out) if r is None]
            raise RuntimeError(
                f"Backend did not produce results for specs {missing}"
            )
        return out  # type: ignore[return-value]

    def submit_one(self, spec: SimJobSpec) -> JobResult:
        """Submit a single spec and return its result.

        Thin convenience wrapper over :meth:`submit_batch`. Present
        so callers writing one-off retry logic do not have to wrap
        each spec in a single-element list themselves.

        Args:
            spec: The :class:`SimJobSpec` to run.

        Returns:
            The corresponding :class:`JobResult`.
        """
        results = self.submit_batch([spec])
        return results[0]

    def poll_stats(self) -> BackendStats:
        """Default no-info stats snapshot.

        Subclasses that track concurrent-job state should override
        this. The default returns zeros so progress reporters built
        on top of it still function — they just display "N/A"-like
        placeholders for what they cannot know.
        """
        return BackendStats(
            running_jobs=0, max_concurrent=0,
            cores_in_use=0, cores_total=0,
        )
