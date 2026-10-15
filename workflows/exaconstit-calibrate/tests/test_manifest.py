"""
Unit tests for :mod:`workflow_common.manifest` and
:mod:`workflow_common.sentinel`.

These exercise the crash-safe state tracking that underpins restart.
Key scenarios:

* Round-tripping state through the JSONL log.
* Snapshot compaction and partial replay.
* Survival of torn (malformed) trailing lines.
* Correct promotion of SUBMITTED -> INTERRUPTED on restart.
* Sentinel read/write/validation.

"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflow_common.manifest import (
    CaseState,
    Manifest,
    ManifestEntry,
)
from workflow_common.sentinel import (
    Sentinel,
    clear_sentinel,
    is_case_complete,
    read_sentinel,
    sentinel_path,
    validate_outputs,
    write_sentinel,
)


# --- Manifest -------------------------------------------------------------


def test_manifest_record_and_get(workspace: Path):
    """Recording an entry puts it in memory and on disk."""
    m = Manifest(workspace / "m.jsonl")
    e = ManifestEntry(
        generation=0, gene=1, obj=0, state=CaseState.SUBMITTED
    )
    m.record(e)
    retrieved = m.get(0, 1, 0)
    assert retrieved is not None
    assert retrieved.state == CaseState.SUBMITTED


def test_manifest_persists_across_reload(workspace: Path):
    """A fresh Manifest instance loads the prior one's state."""
    path = workspace / "m.jsonl"
    m = Manifest(path)
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.SUBMITTED))
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.COMPLETED))

    m2 = Manifest(path)
    m2.load()
    got = m2.get(0, 0, 0)
    assert got is not None
    assert got.state == CaseState.COMPLETED  # latest entry wins


def test_manifest_snapshot_plus_new_entries(workspace: Path):
    """Snapshot + subsequent appends reload correctly after restart."""
    path = workspace / "m.jsonl"
    m = Manifest(path)
    # Pre-snapshot work:
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.COMPLETED))
    m.record(ManifestEntry(generation=0, gene=1, obj=0, state=CaseState.COMPLETED))
    m.snapshot()
    # Post-snapshot work:
    m.record(ManifestEntry(generation=1, gene=0, obj=0, state=CaseState.SUBMITTED))

    m2 = Manifest(path)
    m2.load()
    # All three cases should be present, snapshot entries plus post-snapshot.
    assert m2.get(0, 0, 0).state == CaseState.COMPLETED
    assert m2.get(0, 1, 0).state == CaseState.COMPLETED
    assert m2.get(1, 0, 0).state == CaseState.SUBMITTED


def test_manifest_tolerates_torn_trailing_line(workspace: Path):
    """A malformed final line should be dropped, not crash the load."""
    path = workspace / "m.jsonl"
    m = Manifest(path)
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.COMPLETED))
    # Append a half-written line, mimicking a kill mid-write.
    with open(path, "a") as f:
        f.write('{"generation": 0, "gene": 1, "obj": 0, "sta')  # no newline, truncated

    m2 = Manifest(path)
    m2.load()
    # The complete earlier entry must still be there.
    assert m2.get(0, 0, 0).state == CaseState.COMPLETED
    # The torn entry must be absent.
    assert m2.get(0, 1, 0) is None


def test_mark_submitted_as_interrupted(workspace: Path):
    """SUBMITTED entries without a terminal transition become INTERRUPTED."""
    m = Manifest(workspace / "m.jsonl")
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.SUBMITTED))
    m.record(ManifestEntry(generation=0, gene=1, obj=0, state=CaseState.COMPLETED))

    # New Manifest -> load -> mark: the submitted-but-no-terminal gets
    # promoted; the completed one is left alone.
    m2 = Manifest(workspace / "m.jsonl")
    m2.load()
    n = m2.mark_submitted_as_interrupted()
    assert n == 1
    assert m2.get(0, 0, 0).state == CaseState.INTERRUPTED
    assert m2.get(0, 1, 0).state == CaseState.COMPLETED


def test_mark_submitted_as_interrupted_idempotent(workspace: Path):
    """Calling mark_submitted_as_interrupted twice finds nothing new the 2nd time."""
    m = Manifest(workspace / "m.jsonl")
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.SUBMITTED))
    m.load()  # Not strictly needed here but mirrors restart
    assert m.mark_submitted_as_interrupted() == 1
    # Second call: no SUBMITTED entries remain.
    assert m.mark_submitted_as_interrupted() == 0


def test_manifest_filter_state(workspace: Path):
    """filter_state yields only entries matching the requested state."""
    m = Manifest(workspace / "m.jsonl")
    m.record(ManifestEntry(generation=0, gene=0, obj=0, state=CaseState.COMPLETED))
    m.record(ManifestEntry(generation=0, gene=1, obj=0, state=CaseState.FAILED))
    m.record(ManifestEntry(generation=0, gene=2, obj=0, state=CaseState.COMPLETED))

    completed = list(m.filter_state(CaseState.COMPLETED))
    failed = list(m.filter_state(CaseState.FAILED))
    assert len(completed) == 2
    assert len(failed) == 1


def test_case_state_terminal_property():
    """is_terminal should be True only for COMPLETED and FAILED."""
    assert CaseState.COMPLETED.is_terminal
    assert CaseState.FAILED.is_terminal
    assert not CaseState.PENDING.is_terminal
    assert not CaseState.SUBMITTED.is_terminal
    assert not CaseState.INTERRUPTED.is_terminal


def test_manifest_entry_json_round_trip():
    """to_json_line followed by from_dict(json.loads(...)) is an identity."""
    e = ManifestEntry(
        generation=2, gene=3, obj=1,
        state=CaseState.COMPLETED,
        rc=0, jobid="f1A2B3", message="ok", case_dir="/tmp/x",
    )
    serialized = e.to_json_line()
    e2 = ManifestEntry.from_dict(json.loads(serialized))
    assert e2.generation == e.generation
    assert e2.gene == e.gene
    assert e2.obj == e.obj
    assert e2.state == e.state
    assert e2.rc == e.rc
    assert e2.jobid == e.jobid


# --- Sentinel -------------------------------------------------------------


def test_sentinel_write_read_round_trip(workspace: Path):
    """Writing then reading should recover every field."""
    case_dir = workspace / "case"
    case_dir.mkdir()
    original = Sentinel(
        rc=0, wall_time_s=3.14, jobid="abc",
        output_files={"avg_stress": "results/options/avg_stress.txt"},
        status="ok",
    )
    write_sentinel(case_dir, original)
    got = read_sentinel(case_dir)
    assert got is not None
    assert got.rc == 0
    assert got.wall_time_s == 3.14
    assert got.jobid == "abc"
    assert got.status == "ok"
    assert got.output_files == {"avg_stress": "results/options/avg_stress.txt"}


def test_sentinel_missing_returns_none(workspace: Path):
    """No sentinel -> read_sentinel returns None, is_case_complete returns False."""
    assert read_sentinel(workspace) is None
    assert not is_case_complete(workspace)


def test_sentinel_malformed_returns_none(workspace: Path):
    """Garbage in the sentinel file should be treated as no sentinel at all."""
    (workspace / ".done").write_text("{not valid json")
    assert read_sentinel(workspace) is None
    assert not is_case_complete(workspace)


def test_sentinel_clear(workspace: Path):
    """clear_sentinel removes the file if present and no-ops otherwise."""
    case_dir = workspace / "case"
    case_dir.mkdir()
    write_sentinel(case_dir, Sentinel(rc=0, wall_time_s=1.0))
    assert is_case_complete(case_dir)
    clear_sentinel(case_dir)
    assert not is_case_complete(case_dir)
    # Second call must not raise on missing file.
    clear_sentinel(case_dir)


def test_validate_outputs_happy_path(workspace: Path):
    """All required files present and nonempty -> ok=True, bad=[]."""
    case_dir = workspace / "case"
    case_dir.mkdir()
    (case_dir / "a.txt").write_text("xxx")
    (case_dir / "b.txt").write_text("yy")
    ok, bad = validate_outputs(case_dir, ["a.txt", "b.txt"])
    assert ok
    assert bad == []


def test_validate_outputs_flags_missing_and_empty(workspace: Path):
    """Missing files and empty files both show up in ``bad``."""
    case_dir = workspace / "case"
    case_dir.mkdir()
    (case_dir / "empty.txt").write_text("")
    (case_dir / "full.txt").write_text("x")
    ok, bad = validate_outputs(case_dir, ["empty.txt", "full.txt", "missing.txt"])
    assert not ok
    bad_names = [Path(b).name for b in bad]
    assert set(bad_names) == {"empty.txt", "missing.txt"}


def test_sentinel_path_convention(workspace: Path):
    """The sentinel file lives at ``<case_dir>/.done``."""
    assert sentinel_path(workspace).name == ".done"
    assert sentinel_path(workspace).parent == workspace
