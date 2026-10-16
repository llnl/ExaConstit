"""
Unit tests for :mod:`workflow_common._fs`.

These cover the low-level filesystem helpers: the ``cd`` context
manager and :func:`atomic_write_text`. Focus is on the invariants
the rest of the framework relies on - exception safety for ``cd``,
and crash-resilience for ``atomic_write_text``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from workflow_common._fs import (
    atomic_replace,
    atomic_write_text,
    cd,
    ensure_dir,
)


def test_cd_changes_and_restores(workspace: Path):
    """``cd`` should enter the directory and restore cwd on exit."""
    original = Path.cwd()
    target = workspace / "subdir"
    target.mkdir()
    with cd(target) as resolved:
        assert Path.cwd() == target
        assert resolved == target
    assert Path.cwd() == original


def test_cd_restores_even_on_exception(workspace: Path):
    """Exceptions inside the ``with`` block must not leak cwd changes."""
    original = Path.cwd()
    target = workspace / "subdir"
    target.mkdir()
    with pytest.raises(RuntimeError, match="boom"):
        with cd(target):
            raise RuntimeError("boom")
    assert Path.cwd() == original


def test_cd_raises_when_target_missing(workspace: Path):
    """A nonexistent target should raise FileNotFoundError."""
    missing = workspace / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        with cd(missing):
            pass


def test_atomic_write_text_basic(workspace: Path):
    """Happy path: file appears with exactly the requested contents."""
    target = workspace / "out.txt"
    atomic_write_text(target, "hello world")
    assert target.read_text() == "hello world"


def test_atomic_write_text_creates_parents(workspace: Path):
    """Parent directories are created if missing (``mkdir -p`` semantics)."""
    target = workspace / "nested" / "deeply" / "out.txt"
    atomic_write_text(target, "ok")
    assert target.read_text() == "ok"


def test_atomic_write_text_overwrites_existing(workspace: Path):
    """Writes are fully replacing, not appending."""
    target = workspace / "out.txt"
    target.write_text("old contents with more bytes")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_text_leaves_no_tempfile_on_success(workspace: Path):
    """The tempfile-plus-rename scheme must not leak tempfiles."""
    target = workspace / "out.txt"
    atomic_write_text(target, "x")
    # After a successful write, only the target should remain; no
    # leftover ".out.txt.<suffix>.tmp" should be visible.
    leftovers = [p for p in workspace.iterdir() if p.name.startswith(".out.txt.")]
    assert leftovers == [], f"unexpected tempfiles: {leftovers}"


def test_atomic_write_text_unicode(workspace: Path):
    """Default UTF-8 encoding handles unicode correctly."""
    target = workspace / "out.txt"
    atomic_write_text(target, "resumé café 你好")
    assert target.read_text(encoding="utf-8") == "resumé café 你好"


def test_atomic_replace(workspace: Path):
    """atomic_replace should move src over dst, replacing any existing file."""
    src = workspace / "src.txt"
    dst = workspace / "dst.txt"
    src.write_text("NEW")
    dst.write_text("OLD")
    atomic_replace(src, dst)
    assert not src.exists()
    assert dst.read_text() == "NEW"


def test_ensure_dir_creates_and_idempotent(workspace: Path):
    """ensure_dir creates missing dirs and no-ops on existing ones."""
    target = workspace / "a" / "b" / "c"
    out = ensure_dir(target)
    assert target.is_dir()
    assert out == target
    # Second call must not raise
    ensure_dir(target)
    assert target.is_dir()
