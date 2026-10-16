"""Tests for :mod:`workflows.optimization.regenerate_logbook_files`.

Covers:
  * Happy path — checkpoint with logbooks produces two populated
    .log files in DEAP tab-delimited format.
  * Default output dir = checkpoint's parent dir.
  * --output-dir override.
  * Existing stale log files get overwritten.
  * Bad checkpoint shape (missing logbook keys) fails cleanly.
  * Non-existent checkpoint path fails cleanly.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import pytest
from deap import tools

from workflows.optimization import regenerate_logbook_files


def _make_checkpoint(
    tmp_path: Path, *, stats_records: int = 3, solutions_records: int = 12,
) -> Path:
    """Build a small synthetic checkpoint pickle with known content.

    Returns the pickle's path. The logbook shapes here match what
    the real driver writes: logbook1 carries one row per
    generation (stats), logbook2 carries one row per individual
    per generation (solutions).
    """
    lb1 = tools.Logbook()
    lb1.header = "gen", "avg", "min", "max"
    for g in range(stats_records):
        lb1.record(
            gen=g, avg=1.0 - 0.1 * g,
            min=0.5 - 0.05 * g, max=1.5 - 0.1 * g,
        )

    lb2 = tools.Logbook()
    lb2.header = "gen", "fitness", "solutions"
    # Distribute `solutions_records` rows across `stats_records`
    # generations so the math is symmetric (default 12 / 3 = 4 per gen).
    per_gen = max(1, solutions_records // max(1, stats_records))
    for g in range(stats_records):
        for i in range(per_gen):
            lb2.record(
                gen=g, fitness=[1.0 - 0.01 * i],
                solutions=[100.0 + i],
            )

    ckp = {
        "logbook1": lb1,
        "logbook2": lb2,
        "pop_library": [],
        "iter_tot": 0, "fail_count": 0, "stop_count": 0,
        "generation": stats_records - 1,
        "rndstate": random.getstate(),
        "archive_run_id": "test-run",
    }
    ckpt_path = tmp_path / "checkpoint_gen_2.pkl"
    with ckpt_path.open("wb") as f:
        pickle.dump(ckp, f)
    return ckpt_path


def test_regenerate_produces_populated_log_files(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Happy path: checkpoint + empty output dir → two populated logs."""
    ckpt = _make_checkpoint(tmp_path)
    rc = regenerate_logbook_files.main([str(ckpt)])
    assert rc == 0

    stats = (tmp_path / "logbook1_stats.log").read_text()
    sols = (tmp_path / "logbook2_solutions.log").read_text()

    # Shape checks: both files have a header row + body rows. Body
    # rows start with a digit (generation index). DEAP's exact
    # column-width formatting isn't worth pinning, but the row
    # count is stable.
    stats_body = [ln for ln in stats.splitlines()
                  if ln.strip() and ln.strip().split()[0].isdigit()]
    sols_body = [ln for ln in sols.splitlines()
                 if ln.strip() and ln.strip().split()[0].isdigit()]
    assert len(stats_body) == 3
    assert len(sols_body) == 12

    # DEAP always prepends the header row to .stream output — the
    # regenerated files should contain the column labels too.
    assert "gen" in stats
    assert "fitness" in sols


def test_regenerate_default_output_dir_is_checkpoint_parent(
    tmp_path: Path,
):
    """Without --output-dir, files land next to the pickle."""
    nested_dir = tmp_path / "checkpoint_files"
    nested_dir.mkdir()
    ckpt = _make_checkpoint(nested_dir)

    rc = regenerate_logbook_files.main([str(ckpt)])
    assert rc == 0
    # Landed next to the pickle — not in tmp_path's root.
    assert (nested_dir / "logbook1_stats.log").is_file()
    assert (nested_dir / "logbook2_solutions.log").is_file()
    assert not (tmp_path / "logbook1_stats.log").exists()


def test_regenerate_honors_output_dir_flag(tmp_path: Path):
    """--output-dir redirects the files elsewhere, creating the dir if needed."""
    ckpt = _make_checkpoint(tmp_path)
    alt = tmp_path / "elsewhere" / "deeper"  # doesn't exist yet
    rc = regenerate_logbook_files.main(
        [str(ckpt), "--output-dir", str(alt)],
    )
    assert rc == 0
    assert (alt / "logbook1_stats.log").is_file()
    assert (alt / "logbook2_solutions.log").is_file()
    # Pickle's parent dir should NOT have them in this case.
    assert not (tmp_path / "logbook1_stats.log").exists()


def test_regenerate_overwrites_stale_log_files(tmp_path: Path):
    """Existing files in the output dir get replaced — that's the point.

    This is the exact scenario Robert hit: stale/truncated .log
    files in the checkpoint dir, regenerate should wipe them and
    write the full pickled history in their place.
    """
    ckpt = _make_checkpoint(tmp_path)
    (tmp_path / "logbook1_stats.log").write_text("STALE STUFF\n")
    (tmp_path / "logbook2_solutions.log").write_text("GARBAGE\n")

    rc = regenerate_logbook_files.main([str(ckpt)])
    assert rc == 0
    assert "STALE" not in (tmp_path / "logbook1_stats.log").read_text()
    assert "GARBAGE" not in (tmp_path / "logbook2_solutions.log").read_text()


def test_regenerate_rejects_missing_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Clear error when the checkpoint path doesn't exist."""
    rc = regenerate_logbook_files.main(
        [str(tmp_path / "does_not_exist.pkl")],
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_regenerate_rejects_non_driver_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Pickle that doesn't have the expected logbook keys → exit 1.

    Prevents a confusing stack trace deep in DEAP when the user
    accidentally points the script at some other pickle.
    """
    bogus = tmp_path / "not-a-real-checkpoint.pkl"
    with bogus.open("wb") as f:
        pickle.dump({"unrelated": "data"}, f)

    rc = regenerate_logbook_files.main([str(bogus)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "logbook1" in err  # names the missing key
