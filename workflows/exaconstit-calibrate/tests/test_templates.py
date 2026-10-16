"""
Unit tests for :mod:`workflow_common.templates`.

Covers ``%%key%%`` placeholder substitution: happy-path rendering,
strict vs non-strict unknown-key handling, multi-pass no-recursion
guarantees, and the ``render_template_file`` / ``extract_placeholders``
convenience wrappers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from workflow_common.templates import (
    UnresolvedPlaceholderError,
    extract_placeholders,
    render_template,
    render_template_file,
)


def test_render_basic():
    """Single placeholder replaced with its value."""
    assert render_template("%%name%%", {"name": "alice"}) == "alice"


def test_render_multiple_and_repeated():
    """Multiple different placeholders and repeats both work in one pass."""
    tmpl = "A=%%a%% B=%%b%% A_again=%%a%%"
    assert render_template(tmpl, {"a": 1, "b": 2}) == "A=1 B=2 A_again=1"


def test_render_numeric_values_stringified():
    """Non-string values get ``str()``'d."""
    result = render_template("rate=%%r%%", {"r": 1e-3})
    # The exact string form depends on Python's float->str behavior
    # but it must start with the right mantissa characters.
    assert result.startswith("rate=0.001") or result == "rate=0.001"


def test_render_strict_unknown_raises():
    """In strict mode, an unknown placeholder is an error."""
    with pytest.raises(UnresolvedPlaceholderError) as excinfo:
        render_template("hello %%who%%", {})
    # The error message should name the missing key so typos are easy
    # to find.
    assert "who" in str(excinfo.value)


def test_render_non_strict_passes_through():
    """With strict=False, unknown placeholders survive unchanged."""
    out = render_template("%%a%% and %%b%%", {"a": "yes"}, strict=False)
    assert out == "yes and %%b%%"


def test_render_no_recursion():
    """A value containing %%X%% should NOT trigger further substitution."""
    out = render_template("%%a%%", {"a": "%%b%%", "b": "should_not_appear"})
    assert out == "%%b%%"


def test_render_placeholder_must_be_identifier():
    """Non-identifier placeholders are not matched (documented behavior)."""
    # Anything that's not [A-Za-z_][A-Za-z0-9_]* inside %% %% is passed
    # through verbatim. This is intentional - prevents accidental matches
    # on literal content that happens to contain percent signs.
    tmpl = "%%123%% %%has space%% %%ok_key%%"
    out = render_template(tmpl, {"ok_key": "X", "123": "Y", "has space": "Z"},
                          strict=False)
    assert out == "%%123%% %%has space%% X"


def test_render_template_file_round_trip(workspace: Path):
    """render_template_file reads, renders, and writes atomically."""
    src = workspace / "in.toml"
    dst = workspace / "nested" / "out.toml"
    src.write_text("rate = %%r%%\nname = \"%%n%%\"\n")
    render_template_file(src, dst, {"r": 1e-3, "n": "foo"})
    content = dst.read_text()
    assert "%%r%%" not in content
    assert "name = \"foo\"" in content


def test_render_template_file_strict_raises(workspace: Path):
    """render_template_file defers strictness to render_template."""
    src = workspace / "in.toml"
    src.write_text("rate = %%missing%%\n")
    with pytest.raises(UnresolvedPlaceholderError):
        render_template_file(src, workspace / "out.toml", {})


def test_extract_placeholders_deduplicates():
    """Placeholders appearing multiple times should show up once."""
    tmpl = "%%a%% %%b%% %%a%% %%c%% %%b%%"
    assert extract_placeholders(tmpl) == {"a", "b", "c"}


def test_extract_placeholders_empty():
    """Empty template -> empty set; non-template text -> empty set."""
    assert extract_placeholders("") == set()
    assert extract_placeholders("no placeholders here") == set()
