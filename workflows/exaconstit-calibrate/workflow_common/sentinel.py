"""
Per-case ``.done`` sentinel files.

Purpose
-------
A sentinel file is a small JSON document written into a case's working
directory after the simulation finishes AND its output files have been
validated. Its mere existence on disk is the authoritative signal that
a case is complete; if no sentinel is present, the case either never
ran or was killed partway through.

Sentinel files are complementary to the manifest in
:mod:`workflow_common.manifest`. The manifest records every state
transition and is the source of truth for global restart planning.
The sentinel records the completion of one specific case and is the
source of truth for "can I reuse this case's outputs?"

Why both?
---------
A kill between sentinel-write and manifest-write (in that order) is
safe: on restart the sentinel is found, so the case is correctly
treated as complete, and the manifest catches up on the next record.
A kill in the opposite order would be dangerous - the manifest would
claim the case was done but the sentinel would be missing, leaving
downstream code unsure which signal to trust. The convention in this
framework is therefore **sentinel first, then manifest update**.

Why is the sentinel separate from the simulation code's own output?
-------------------------------------------------------------------
Because output files by themselves are not trustworthy. A killed
process can leave a truncated ``avg_stress.txt`` that parses as a
valid float array but is silently missing the last several rows.
The sentinel is written by the outer workflow harness AFTER it has
validated the outputs. The writer is not the simulation binary, so
killing the binary cannot produce a false-positive sentinel.

Atomic write semantics
----------------------
Writes go through :func:`atomic_write_text` (tempfile + rename). No
reader can ever observe a partially-written sentinel: either the
file is fully present and parseable, or it is absent entirely. That
invariant is what lets restart logic rely on sentinel presence as a
boolean completion signal.

Example ``.done`` file on disk
------------------------------
A successful case's sentinel looks like this (reformatted from the
compact JSON we actually write, for readability)::

    {
      "rc": 0,
      "wall_time_s": 142.37,
      "finished_ts": 1711930421.88,
      "jobid": "f1A2B3",
      "output_files": {
        "avg_stress": "wf/gen_0/gene_3/results/options/avg_stress.txt",
        "avg_def_grad": "wf/gen_0/gene_3/results/options/avg_def_grad.txt"
      },
      "status": "ok",
      "message": null
    }

A failed case's sentinel has a nonzero ``rc``, a tag like
``"bad_strain"`` or ``"output_missing"`` in ``status``, and often a
human-readable explanation in ``message``. The rest of the schema is
identical - the sentinel's job is to record "this case finished",
not to pass or fail judgement.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from ._fs import atomic_write_text

# Filename used for per-case sentinel files. Dotfile so it does not
# clutter directory listings, and consistently named so restart logic
# can check for it without parsing a manifest.
SENTINEL_FILENAME = ".done"


@dataclass
class Sentinel:
    """The contents of a per-case ``.done`` sentinel file.

    Kept small on purpose. The sentinel is read frequently during
    restart (once per case in the plan) and any unnecessary bulk
    slows that down. Large payloads belong in the manifest or in
    per-case log files, not here.

    Fields:
        rc: The simulation's return code. 0 indicates success; any
            nonzero value indicates the simulation failed but the
            framework still wrote a sentinel (so restart does not
            retry) because the caller decided to persist the
            failure rather than treat it as interrupted.
        wall_time_s: Wall-clock runtime of the simulation, in seconds.
            Used for cost-tracking and performance logs.
        finished_ts: Unix timestamp at which the sentinel was written.
            Defaults to construction time.
        jobid: Optional backend-assigned ID (Flux jobid string, etc.)
            for cross-referencing with external logs.
        output_files: Mapping from logical output name (e.g.
            ``"avg_stress"``) to the on-disk path where it was found.
            Populated by the validation step that preceded the
            sentinel write.
        status: A short free-form tag summarizing how the case ended.
            Common values: ``"ok"``, ``"bad_strain"``, ``"timeout"``,
            ``"output_missing"``. Framework code does not interpret
            this string; it is surfaced to user-written postprocessors.
        message: Optional longer human-readable explanation, mostly
            useful for failed cases where the tag alone is not
            enough to diagnose.
    """

    rc: int
    wall_time_s: float
    finished_ts: float = field(default_factory=time.time)
    jobid: Optional[str] = None
    output_files: Dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    message: Optional[str] = None


def sentinel_path(case_dir: Union[str, Path]) -> Path:
    """Return the on-disk path where the sentinel for ``case_dir`` lives.

    Thin helper that exists so callers do not have to remember the
    filename convention. If you ever want to change the sentinel
    filename (don't), this is the one place to edit.

    Args:
        case_dir: The case's working directory.

    Returns:
        ``case_dir/.done`` as a ``Path``.
    """
    return Path(case_dir) / SENTINEL_FILENAME


def write_sentinel(case_dir: Union[str, Path], sentinel: Sentinel) -> Path:
    """Atomically write a sentinel file into ``case_dir``.

    Uses :func:`atomic_write_text` so a crash mid-write cannot leave
    a half-written sentinel behind. Either the sentinel is fully on
    disk after this call, or it is absent entirely.

    Args:
        case_dir: The case's working directory. Created implicitly
            by :func:`atomic_write_text` if missing (though by this
            point it should already exist, since the simulation has
            just run inside it).
        sentinel: A :class:`Sentinel` describing the completion
            status to persist.

    Returns:
        The path of the written sentinel file, mostly for logging.

    Example:
        ::

            write_sentinel(
                "wf/gen_0/gene_3",
                Sentinel(
                    rc=0,
                    wall_time_s=142.3,
                    jobid="f1A2B3",
                    output_files={
                        "avg_stress":
                            "wf/gen_0/gene_3/results/options/avg_stress.txt",
                    },
                    status="ok",
                ),
            )
    """
    path = sentinel_path(case_dir)
    atomic_write_text(path, json.dumps(asdict(sentinel), separators=(",", ":")))
    return path


def read_sentinel(case_dir: Union[str, Path]) -> Optional[Sentinel]:
    """Load the sentinel from ``case_dir`` if one is present.

    A missing sentinel or an unparseable one both return ``None``.
    Treating unparseable as absent is the safe default: if we cannot
    read the sentinel, we should not trust outputs in that directory,
    and the correct recovery is to resubmit the case. This matches
    what the manifest does with malformed JSONL lines.

    Args:
        case_dir: The case's working directory.

    Returns:
        A :class:`Sentinel` instance if the file exists and parses,
        otherwise ``None``.

    Example:
        Restart-time skip check::

            if read_sentinel(case_dir) is not None:
                # Case already completed on a previous run; don't rerun.
                ...
    """
    path = sentinel_path(case_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Reconstruct the dataclass explicitly rather than using
        # `Sentinel(**d)` so we have per-field error handling and can
        # tolerate older sentinel files missing optional fields.
        return Sentinel(
            rc=int(d["rc"]),
            wall_time_s=float(d.get("wall_time_s", 0.0)),
            finished_ts=float(d.get("finished_ts", 0.0)),
            jobid=d.get("jobid"),
            output_files=dict(d.get("output_files", {})),
            status=d.get("status", "ok"),
            message=d.get("message"),
        )
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        # Treat any failure to parse as "no sentinel". This is the
        # conservative choice - a subtle parse issue should cause a
        # rerun, not a silent acceptance of possibly-bad outputs.
        return None


def is_case_complete(case_dir: Union[str, Path]) -> bool:
    """Return True iff a readable sentinel exists for the given case.

    Convenience wrapper for restart logic that only needs a yes/no
    answer to "should I skip this case?". For cases where the sentinel
    contents matter (e.g. distinguishing rc=0 from rc=7), call
    :func:`read_sentinel` directly and inspect the returned object.

    Args:
        case_dir: The case's working directory.

    Returns:
        True if a sentinel file exists AND parses successfully,
        False otherwise.
    """
    return read_sentinel(case_dir) is not None


def clear_sentinel(case_dir: Union[str, Path]) -> None:
    """Remove the sentinel file for ``case_dir`` if one exists.

    Used when deliberately re-running a case (e.g. because its
    outputs are known to be stale or because user policy calls for
    it). Silently no-ops if no sentinel is present, so the call site
    does not need to check first.

    Args:
        case_dir: The case's working directory.
    """
    p = sentinel_path(case_dir)
    try:
        p.unlink()
    except FileNotFoundError:
        # Already gone - nothing to do. This is the expected path
        # when rerunning a case that was never completed in the
        # first place.
        pass


def validate_outputs(
    case_dir: Union[str, Path],
    required_files: List[Union[str, Path]],
    *,
    min_size_bytes: int = 1,
) -> "tuple[bool, list[str]]":
    """Check that required output files exist and are not empty.

    This is the last line of defense against a killed simulation that
    happened to exit with rc=0 but left truncated output files. The
    check is deliberately minimal: each named file must exist and
    have at least ``min_size_bytes`` bytes. Users with format-specific
    sanity requirements (for example, "the second column must have
    exactly N rows") should add their own validation on top of this
    one before writing a sentinel.

    Args:
        case_dir: Directory to resolve relative ``required_files``
            against. Absolute entries in ``required_files`` are
            used as-is.
        required_files: List of file paths that must exist. Relative
            paths are resolved against ``case_dir``; absolute paths
            are used directly.
        min_size_bytes: Minimum acceptable size for each file, in
            bytes. Default is 1, which catches truly zero-byte
            outputs from catastrophic writes. Increase if your code
            always writes a nontrivial amount (e.g. a CSV header).

    Returns:
        A tuple ``(ok, bad)`` where:

        * ``ok`` is True iff every required file passes the checks.
        * ``bad`` is a list of file paths (as strings) that failed
          the checks. Empty when ``ok`` is True.

    Example:
        Post-simulation validation step::

            required = [
                "results/options/avg_stress.txt",
                "results/options/avg_def_grad.txt",
            ]
            ok, missing = validate_outputs(case_dir, required)
            if not ok:
                logger.warning("outputs incomplete: %s", missing)
    """
    case_dir = Path(case_dir)
    bad: List[str] = []
    for rel in required_files:
        p = Path(rel)
        # Only resolve against case_dir if the path is relative. This
        # lets callers mix-and-match: some outputs in the case dir,
        # some at absolute paths (e.g. shared postprocessing results).
        if not p.is_absolute():
            p = case_dir / p
        try:
            size = p.stat().st_size
        except OSError:
            bad.append(str(p))
            continue
        if size < min_size_bytes:
            bad.append(str(p))
    return (len(bad) == 0, bad)
