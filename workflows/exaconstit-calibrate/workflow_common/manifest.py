"""
JSONL manifest for crash-safe tracking of simulation cases.

Why this file exists
--------------------
Optimization runs on HPC frequently last longer than the SLURM or LSF
allocation they were launched inside. When the allocation expires,
every process belonging to that job is killed immediately - including
any running simulations and the optimizer itself. On restart the next
day, we need to answer three questions:

1. Which cases finished successfully? (don't rerun them)
2. Which cases had been submitted but did not finish? (likely killed;
   rerun, but clean up any partial output files first)
3. Which cases never made it to submission at all? (run them now)

This module stores the information needed to answer those questions in
a durable on-disk manifest. The :class:`.sentinel` module stores the
corresponding per-case completion marker. Together they make restart
deterministic.

Why JSONL rather than SQLite or some database
---------------------------------------------
On HPC systems, shared parallel filesystems (Lustre, GPFS, sometimes
NFS) have quirks around POSIX advisory locking. SQLite, which relies
on those locks for concurrency control, is documented as unsafe or
unreliable on several of them - WAL mode over NFS is especially
fraught. Spinning up a real RDBMS (MySQL, PostgreSQL) adds an
operational burden that is painful on machines the user does not
administer.

JSONL (JSON Lines) is a simple text format: one JSON object per line.
Appending a line with ``open("a")`` on POSIX is atomic for payloads
under ``PIPE_BUF`` (4 KiB on Linux), which easily covers one
``ManifestEntry``. Because we have a single writer (the optimizer
process) and no contention to worry about, this is safe across every
filesystem of interest. It is also trivially greppable, diffable, and
human-readable during debugging, which has proven valuable more than
once when reasoning about what went wrong overnight.

Snapshots
---------
Long runs can accumulate thousands of manifest entries. We do not want
to replay the entire file every time the driver starts. The manifest
supports periodic compaction: :meth:`Manifest.snapshot` writes the
current reduced state (one entry per case key, holding the latest
transition) to a companion snapshot file using an atomic rename. On
:meth:`Manifest.load`, the snapshot is loaded first and the JSONL log
is replayed only from the snapshot's timestamp forward. This keeps
restart cost bounded.

Case lifecycle diagram
----------------------
Every case walks through this small state machine during a run::

                      (case enters the plan)
                              |
                              v
                         +---------+
                         | PENDING |
                         +---------+
                              |
                       record SUBMITTED
                              |
                              v
                       +-----------+
                       | SUBMITTED |
                       +-----------+
                        /    |     \\
                       /     |      \\
              backend  |  backend   |  driver dies /
              says ok  |  says bad  |  allocation ends
                       v            v        v
                 +-----------+  +--------+  +-------------+
                 | COMPLETED |  | FAILED |  | INTERRUPTED |
                 +-----------+  +--------+  +-------------+
                  (terminal)   (terminal)   (reconsidered
                                             on next run)

Only COMPLETED and FAILED are terminal. A case in INTERRUPTED is
reconsidered on the next run: it may be resubmitted (default policy)
or skipped (user policy).

Manifest vs sentinel: who is the source of truth?
-------------------------------------------------
``Manifest`` and :mod:`workflow_common.sentinel` overlap in purpose,
but play complementary roles::

    Manifest (global, in-memory + JSONL log)
        Answers: "what do I know about every case in this run?"
        Written by: the driver, on every state transition.
        Read by:   the driver at startup for restart planning.

    Sentinel (local, one .done file per case directory)
        Answers: "is the work in THIS directory finished?"
        Written by: the driver, after validating outputs.
        Read by:   the driver's skip logic, by analysts poking
                   around output directories, by downstream tools.

The driver writes the sentinel FIRST, then records a terminal manifest
entry. If a crash happens in between, restart still sees the sentinel
and correctly treats the case as complete; the manifest catches up on
the next record. The reverse ordering would leave a window in which
the manifest believes a case is done but no sentinel exists, creating
an ambiguous "did the outputs really get validated?" state.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

from ._fs import atomic_write_text, ensure_dir


class CaseState(str, Enum):
    """Lifecycle states for a single workflow case.

    Cases transition through these states as the optimization
    progresses. The framework records every transition so that restart
    logic can reconstruct the state of the world from disk.

    Inheriting from ``str`` makes the values transparent in JSON
    (serialized as plain strings) and greppable in the on-disk log
    file, which helps when debugging.

    Members:
        PENDING: The case is known to the optimizer (it's in the plan)
            but has not yet been submitted to a backend. Most cases
            pass through this state briefly.
        SUBMITTED: Handed to a backend. This is an intermediate state
            - no terminal status has been recorded yet. If the driver
            dies while a case is in this state, restart logic will
            transition it to INTERRUPTED.
        COMPLETED: Finished successfully. Return code 0 and any
            output-file sanity checks passed. The case's sentinel
            file will also be present on disk.
        FAILED: Finished unsuccessfully. Either the simulation exited
            with a non-zero return code, or the output files were
            missing or empty. Still "terminal" in the sense that we
            will not automatically rerun it without explicit policy.
        INTERRUPTED: Was in SUBMITTED state when the driver restarted.
            The simulation was presumably killed with its allocation.
            Downstream logic decides whether to resubmit or skip.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @property
    def is_terminal(self) -> bool:
        """Whether this state represents the end of a case's lifecycle.

        Returns:
            True for COMPLETED and FAILED. False for PENDING,
            SUBMITTED, and INTERRUPTED (which represent work that
            either has not started or has not finished).
        """
        return self in (CaseState.COMPLETED, CaseState.FAILED)


# Type alias for the (generation, gene, obj) triple that uniquely
# identifies a case. Defined once here so internal typing stays
# readable.
Key = Tuple[int, int, int]


@dataclass
class ManifestEntry:
    """One recorded transition for a single case.

    Each time a case changes state (PENDING -> SUBMITTED,
    SUBMITTED -> COMPLETED, etc.), a fresh ``ManifestEntry`` is
    appended to the JSONL manifest. The ``(generation, gene, obj)``
    triple is the key; everything else is metadata about the
    transition.

    Fields:
        generation: GA generation index.
        gene: Individual index within the generation.
        obj: Objective index for multi-objective runs.
        state: The :class:`CaseState` being recorded.
        ts: Unix timestamp at which this transition was recorded.
            Defaults to the time ``ManifestEntry`` is constructed.
        jobid: Optional backend-assigned ID (e.g. a Flux jobid encoded
            as a string). Absent for transitions that predate
            submission.
        rc: Optional return code from the simulation. Absent for
            transitions that happen before the process exits.
        message: Optional free-form human-readable note. Useful for
            explaining INTERRUPTED / FAILED transitions.
        case_dir: Optional working directory of the case. Helpful for
            restart logic that needs to clean up partial outputs.
    """

    generation: int
    gene: int
    obj: int
    state: CaseState
    ts: float = field(default_factory=time.time)
    jobid: Optional[str] = None
    rc: Optional[int] = None
    message: Optional[str] = None
    case_dir: Optional[str] = None

    @property
    def key(self) -> Key:
        """Return the ``(generation, gene, obj)`` identifier triple."""
        return (self.generation, self.gene, self.obj)

    def to_json_line(self) -> str:
        """Serialize to a single JSON-encoded string for JSONL append.

        The output has no trailing newline; the caller appends one.
        Keys are written without extra whitespace to keep each line
        short (the manifest may contain thousands of entries).

        Returns:
            A single-line JSON string.
        """
        d = asdict(self)
        # Enum values inherit from str, but `asdict` preserves the enum
        # type on some Python versions. Coerce explicitly so the JSON
        # output is always a plain string regardless of version.
        d["state"] = self.state.value
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestEntry":
        """Rehydrate a ``ManifestEntry`` from a parsed JSON object.

        Intentionally strict about the required fields
        (``generation``, ``gene``, ``obj``, ``state``) and lenient
        about optional fields, so older manifest formats with fewer
        fields remain readable.

        Args:
            d: A dict as produced by ``json.loads`` of one line of
               the JSONL manifest.

        Returns:
            A ``ManifestEntry`` populated from the dict.

        Raises:
            KeyError / ValueError: If required fields are missing or
                have unexpected types. Callers that want to skip
                malformed lines should catch these broadly.
        """
        return cls(
            generation=int(d["generation"]),
            gene=int(d["gene"]),
            obj=int(d["obj"]),
            state=CaseState(d["state"]),
            ts=float(d.get("ts", 0.0)),
            jobid=d.get("jobid"),
            rc=d.get("rc"),
            message=d.get("message"),
            case_dir=d.get("case_dir"),
        )


class Manifest:
    """Append-only JSONL manifest with snapshot-based compaction.

    Typical lifecycle::

        # At startup
        m = Manifest(Path("wf_files/manifest.jsonl"))
        m.load()                          # read snapshot + replay log
        m.mark_submitted_as_interrupted() # restart hygiene

        # During the run, on every state transition
        m.record(ManifestEntry(
            generation=0, gene=3, obj=1,
            state=CaseState.SUBMITTED,
            jobid="f1A2B3",
            case_dir="wf_files/gen_0/gene_3_obj_1",
        ))

        # Periodically, e.g. at the end of each generation
        m.snapshot()

    Concurrency:
        The manifest assumes a single writer process. If your workflow
        fans writes out across multiple processes, give each process
        its own manifest file and merge them on read. An internal
        threading lock protects against multi-threaded writes within
        one process; it does NOT protect against inter-process races.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        snapshot_path: Union[str, Path, None] = None,
    ):
        """Initialize a ``Manifest`` bound to on-disk files.

        The object is created in an "unloaded" state; call
        :meth:`load` before reading state.

        Args:
            path: Path to the JSONL log file. Created if missing on
                first :meth:`record` call.
            snapshot_path: Optional path to a companion snapshot file.
                Defaults to ``<path>.snapshot`` (i.e. ``manifest.jsonl``
                gets a sibling ``manifest.jsonl.snapshot``). Snapshot
                files are written with atomic rename, so a kill
                during snapshot cannot corrupt the previous snapshot.
        """
        self._path = Path(path)
        # Default snapshot file lives right next to the JSONL log with
        # a recognizable suffix. Easy to spot, easy to archive.
        self._snapshot_path = (
            Path(snapshot_path)
            if snapshot_path is not None
            else self._path.with_suffix(self._path.suffix + ".snapshot")
        )
        # In-memory reduced state: exactly one entry per case key,
        # holding the most recent transition. Updated on load() and
        # on every record() call.
        self._state: Dict[Key, ManifestEntry] = {}
        # Timestamp of the last snapshot. On load(), only JSONL lines
        # with ts >= _snapshot_ts are replayed, which keeps restart
        # cost proportional to the work done since the last snapshot.
        self._snapshot_ts: float = 0.0
        # Guards multi-threaded record() calls from one process. Does
        # NOT help across processes - see the class docstring on
        # concurrency.
        self._write_lock = Lock()

    # --- state access -----------------------------------------------------

    def get(self, generation: int, gene: int, obj: int) -> Optional[ManifestEntry]:
        """Return the most recent entry for a case, or ``None`` if none.

        Args:
            generation: GA generation index.
            gene: Individual index within the generation.
            obj: Objective index.

        Returns:
            The last ``ManifestEntry`` recorded for the given
            ``(generation, gene, obj)`` triple, or ``None`` if no
            entry has ever been recorded for it.
        """
        return self._state.get((generation, gene, obj))

    def __contains__(self, key: Key) -> bool:
        """Support ``(g, gene, obj) in manifest`` membership tests."""
        return key in self._state

    def all_entries(self) -> Iterator[ManifestEntry]:
        """Iterate over the most recent entry of every known case.

        Order is unspecified. If you need sorted output, sort the
        result yourself on whatever key you prefer.

        Yields:
            One :class:`ManifestEntry` per case.
        """
        return iter(self._state.values())

    def filter_state(self, state: CaseState) -> Iterator[ManifestEntry]:
        """Iterate over entries currently in the given state.

        Args:
            state: The :class:`CaseState` to filter on.

        Yields:
            Each matching :class:`ManifestEntry`.

        Example:
            Find cases that were submitted but never finished::

                for e in manifest.filter_state(CaseState.INTERRUPTED):
                    print(e.case_dir, e.message)
        """
        return (e for e in self._state.values() if e.state == state)

    # --- load / replay ----------------------------------------------------

    def load(self) -> None:
        """Load the on-disk manifest into memory.

        The procedure is:

        1. Load the snapshot file if it exists. The snapshot is a
           single JSON document containing a list of the latest
           entries as of the snapshot's timestamp.
        2. Replay the JSONL log, taking only entries with a timestamp
           at or later than the snapshot's timestamp. Each replayed
           entry overwrites any earlier entry for the same case key.
        3. Malformed trailing lines (for example a half-written final
           line from a kill-mid-write) are silently skipped. Earlier
           complete lines are kept. If a case's most recent entry was
           lost this way, it remains in SUBMITTED state and will be
           handled by :meth:`mark_submitted_as_interrupted`.

        Safe to call more than once. Each call completely replaces
        the in-memory state.

        Reading time complexity:
            O(S + L) where S is the number of entries in the snapshot
            and L is the number of JSONL lines written *since* the
            snapshot. Before any snapshot is taken, L equals the full
            run history; after each ``snapshot()`` call, L resets to
            (effectively) zero. This is why periodic snapshots keep
            restart fast for long-running optimizations.

        Timeline diagram::

            t=0    .............. snapshot .............. now
            log:   [entry1][entry2][entry3][entry4][entry5][entry6]
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          replayed on load()
            snap:                   [entry1, entry2, entry3]
                                     ^^^^^^^^^^^^^^^^^^^^^^
                                     loaded first from snap

        Entries 4, 5, 6 were written after the snapshot; they are
        replayed in order from the JSONL log and supersede anything
        the snapshot said about the same case keys.
        """
        # Step 0: reset in-memory state so a second load() call is a
        # clean reload, not an incremental merge.
        self._state.clear()
        self._snapshot_ts = 0.0

        # Step 1: seed from snapshot if one is present.
        #
        # The snapshot is a single JSON file mapping each (gen, gene,
        # obj) triple to its most recent entry at the time the
        # snapshot was written. Loading it is much faster than
        # replaying thousands of JSONL lines from the start of the run.
        if self._snapshot_path.exists():
            try:
                with open(self._snapshot_path, "r", encoding="utf-8") as f:
                    snap = json.load(f)
                # Remember the snapshot's timestamp so we know where
                # to start replaying the JSONL log.
                self._snapshot_ts = float(snap.get("ts", 0.0))
                for d in snap.get("entries", []):
                    e = ManifestEntry.from_dict(d)
                    # Keyed assignment; no merging needed because the
                    # snapshot already contains at most one entry per
                    # key (it is the compacted view).
                    self._state[e.key] = e
            except (OSError, json.JSONDecodeError):
                # A corrupt snapshot is treated as if it did not exist.
                # The JSONL replay below will rebuild everything from
                # scratch in that case, which is slow but always
                # correct. Snapshots are an optimization, not the
                # source of truth.
                self._state.clear()
                self._snapshot_ts = 0.0

        # Step 2: replay JSONL entries newer than the snapshot.
        #
        # We open the log read-only and process it line by line. Each
        # line is supposed to be a complete JSON object followed by a
        # newline, but we do not rely on that - any line that fails
        # to parse is silently dropped. The typical cause is a
        # "torn write": the process was killed partway through
        # flushing the final line to disk, so the last line ends
        # mid-token. Everything before that line is intact.
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        # Blank lines can appear if the log was
                        # hand-edited or concatenated. Ignore them.
                        continue
                    try:
                        d = json.loads(line)
                        e = ManifestEntry.from_dict(d)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Malformed line: almost always indicates a
                        # torn write from a kill signal. Drop it and
                        # continue. Earlier well-formed lines in the
                        # same file are preserved.
                        continue
                    # Only apply entries at or newer than the snapshot.
                    # Older entries are already represented in the
                    # snapshot and re-applying them would be a waste of
                    # work, though not incorrect (they would overwrite
                    # themselves).
                    if e.ts >= self._snapshot_ts:
                        self._state[e.key] = e

    # --- record / append --------------------------------------------------

    def record(self, entry: ManifestEntry) -> None:
        """Append a state transition to the manifest.

        Writes a single JSON line to the JSONL log file and updates
        the in-memory state. The write is an ``open("a")`` each time,
        which is fast and crash-safe: under ``PIPE_BUF`` a single line
        is atomic on POSIX filesystems, so a kill during the write
        cannot interleave bytes from this record with bytes from
        another.

        Args:
            entry: The :class:`ManifestEntry` to record. Typically
                freshly constructed at the transition site.

        Raises:
            OSError: If the log file cannot be opened or written.

        Example:
            Record that a case has been submitted::

                manifest.record(ManifestEntry(
                    generation=gen, gene=i, obj=j,
                    state=CaseState.SUBMITTED,
                    jobid=str(result.jobid),
                    case_dir=str(working_dir),
                ))
        """
        ensure_dir(self._path.parent)
        line = entry.to_json_line() + "\n"
        with self._write_lock:
            # Open-write-close each call. This is slower than holding
            # a long-lived file handle, but it is easier to reason
            # about (no flush-ordering concerns) and the cost is
            # trivial compared to the simulations themselves.
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                # Best-effort fsync. On shared filesystems this can be
                # slow but it is the main durability guarantee against
                # node-level power loss. Swallow OSError because not
                # every file-like object supports fsync.
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            # In-memory state update happens inside the lock so other
            # threads reading through __contains__ / get cannot see a
            # state that is not yet in the log.
            self._state[entry.key] = entry

    def record_many(self, entries: Iterable[ManifestEntry]) -> None:
        """Record a sequence of entries by calling :meth:`record` in order.

        Args:
            entries: Iterable of :class:`ManifestEntry` to record.

        Note:
            This is a convenience wrapper. Each entry is written
            independently - there is no batching. If you need
            transactional semantics across many entries, write them
            all and then call :meth:`snapshot`.
        """
        for e in entries:
            self.record(e)

    # --- snapshot / compaction --------------------------------------------

    def snapshot(self) -> None:
        """Write an atomic snapshot of the current reduced state.

        The snapshot contains one entry per case key - the most recent
        transition - plus the timestamp at which the snapshot was
        taken. Future ``load()`` calls use this snapshot as the
        starting point and only replay JSONL entries newer than the
        snapshot timestamp, keeping restart cost bounded.

        Does NOT truncate the JSONL log. The log is kept for
        auditability; users can archive or rotate it manually if it
        becomes inconveniently large. A future version of this module
        may add automatic log rotation tied to snapshot events.

        Safe to call as often as you like; a common cadence is once
        per GA generation.
        """
        now = time.time()
        # Build a plain dict-of-lists payload so it round-trips through
        # json.load cleanly regardless of Python version.
        payload = {
            "ts": now,
            "entries": [
                {
                    "generation": e.generation,
                    "gene": e.gene,
                    "obj": e.obj,
                    "state": e.state.value,
                    "ts": e.ts,
                    "jobid": e.jobid,
                    "rc": e.rc,
                    "message": e.message,
                    "case_dir": e.case_dir,
                }
                for e in self._state.values()
            ],
        }
        # atomic_write_text gives us tempfile-plus-rename, so any
        # process that was reading an older snapshot keeps seeing the
        # old file until the rename completes. No partial snapshot
        # can ever be observed on disk.
        atomic_write_text(
            self._snapshot_path,
            json.dumps(payload, separators=(",", ":")),
        )
        self._snapshot_ts = now

    # --- restart helpers --------------------------------------------------

    def mark_submitted_as_interrupted(self) -> int:
        """Tag any SUBMITTED cases as INTERRUPTED.

        Called once immediately after :meth:`load` on restart. The
        assumption is that any case recorded as SUBMITTED with no
        subsequent terminal transition was killed alongside the old
        allocation. Converting them to INTERRUPTED makes that
        explicit, so downstream logic can decide whether to rerun or
        skip.

        Has no effect on the first run of a workflow (no SUBMITTED
        entries yet). Idempotent: calling it twice in a row finds
        nothing to do the second time.

        Returns:
            The number of cases that were transitioned to
            INTERRUPTED.

        Example:
            Standard restart prelude::

                manifest.load()
                n = manifest.mark_submitted_as_interrupted()
                if n:
                    logger.warning(
                        "detected %d interrupted cases from previous run",
                        n,
                    )
        """
        count = 0
        # Iterate over a snapshot of (key, entry) pairs because record()
        # will mutate self._state as we go.
        for key, entry in list(self._state.items()):
            if entry.state == CaseState.SUBMITTED:
                self.record(
                    ManifestEntry(
                        generation=entry.generation,
                        gene=entry.gene,
                        obj=entry.obj,
                        state=CaseState.INTERRUPTED,
                        jobid=entry.jobid,
                        case_dir=entry.case_dir,
                        message="detected as SUBMITTED on restart",
                    )
                )
                count += 1
        return count
