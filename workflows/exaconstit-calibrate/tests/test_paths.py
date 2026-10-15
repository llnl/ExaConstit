"""
Unit tests for :mod:`workflow_common.paths`.

Covers the :class:`TemplatePathResolver` - the default
:class:`PathResolver` implementation. Exercises the format-string
substitution for working directories and output files, the
``{working_dir}`` special key, missing-key error reporting, and the
``defaults`` / ``extra`` precedence rules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from workflow_common.paths import (
    CaseContext,
    PathResolver,
    TemplatePathResolver,
)


def test_case_context_as_format_mapping_basic():
    """Top-level fields and the gen alias both appear in the mapping."""
    ctx = CaseContext(generation=4, gene=7, obj=1)
    m = ctx.as_format_mapping()
    assert m["generation"] == 4
    assert m["gen"] == 4  # alias
    assert m["gene"] == 7
    assert m["obj"] == 1


def test_case_context_extra_merged():
    """Entries in ``extra`` appear in the format mapping too."""
    ctx = CaseContext(
        generation=0, gene=0, obj=0, extra={"rve_name": "grain_32"}
    )
    assert ctx.as_format_mapping()["rve_name"] == "grain_32"


def test_case_context_extra_cannot_shadow_core_fields():
    """Top-level fields win over collisions in ``extra``."""
    # Users sometimes pass "generation" or "gen" in extra by mistake;
    # the top-level value must take precedence.
    ctx = CaseContext(
        generation=99, gene=0, obj=0, extra={"generation": 1, "gen": 2}
    )
    m = ctx.as_format_mapping()
    assert m["generation"] == 99
    assert m["gen"] == 99


def test_working_dir_basic(workspace: Path):
    """Simple pattern produces the expected path."""
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}",
        output_file_patterns={},
        root=workspace,
    )
    ctx = CaseContext(generation=2, gene=5, obj=0)
    assert resolver.working_dir(ctx) == workspace / "wf" / "gen_2" / "gene_5"


def test_working_dir_absolute_pattern_ignores_root(workspace: Path):
    """Absolute patterns bypass the root prefix."""
    resolver = TemplatePathResolver(
        working_dir_pattern="/absolute/path/gen_{generation}",
        output_file_patterns={},
        root=workspace,
    )
    ctx = CaseContext(generation=3, gene=0, obj=0)
    assert resolver.working_dir(ctx) == Path("/absolute/path/gen_3")


def test_working_dir_missing_key_raises_helpfully(workspace: Path):
    """Missing placeholder -> KeyError whose message lists available keys."""
    resolver = TemplatePathResolver(
        working_dir_pattern="{missing_key}/gen_{generation}",
        output_file_patterns={},
        root=workspace,
    )
    ctx = CaseContext(generation=0, gene=0, obj=0)
    with pytest.raises(KeyError) as excinfo:
        resolver.working_dir(ctx)
    # The error must mention the key name AND the available fallbacks,
    # so typos are easy to debug.
    assert "missing_key" in str(excinfo.value)
    assert "generation" in str(excinfo.value)


def test_output_file_with_working_dir_placeholder(workspace: Path):
    """``{working_dir}`` in an output pattern expands to the case's working dir."""
    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}",
        output_file_patterns={
            "stress": "{working_dir}/results/{basename}/avg_stress.txt",
        },
        defaults={"basename": "options"},
        root=workspace,
    )
    ctx = CaseContext(generation=0, gene=1, obj=0)
    expected = workspace / "gen_0" / "gene_1" / "results" / "options" / "avg_stress.txt"
    assert resolver.output_file("stress", ctx) == expected


def test_output_file_unknown_logical_name_raises(workspace: Path):
    """Unknown output names get a descriptive KeyError."""
    resolver = TemplatePathResolver(
        working_dir_pattern="x",
        output_file_patterns={"known": "{working_dir}/a.txt"},
        root=workspace,
    )
    ctx = CaseContext(generation=0, gene=0, obj=0)
    with pytest.raises(KeyError) as excinfo:
        resolver.output_file("not_a_thing", ctx)
    assert "not_a_thing" in str(excinfo.value)
    assert "known" in str(excinfo.value)  # lists the valid ones


def test_defaults_are_overridden_by_extra(workspace: Path):
    """``ctx.extra`` entries override the resolver's defaults on name collision."""
    resolver = TemplatePathResolver(
        working_dir_pattern="{basename}/gene_{gene}",
        output_file_patterns={},
        defaults={"basename": "options"},
        root=workspace,
    )
    ctx = CaseContext(
        generation=0, gene=0, obj=0, extra={"basename": "override"}
    )
    assert resolver.working_dir(ctx) == workspace / "override" / "gene_0"


def test_known_outputs_returns_registered_names(workspace: Path):
    """known_outputs reflects the patterns passed at construction."""
    resolver = TemplatePathResolver(
        working_dir_pattern="x",
        output_file_patterns={"a": "x/a", "b": "x/b", "c": "x/c"},
        root=workspace,
    )
    assert set(resolver.known_outputs()) == {"a", "b", "c"}


def test_template_resolver_satisfies_protocol(workspace: Path):
    """TemplatePathResolver should pass runtime_checkable isinstance check."""
    resolver = TemplatePathResolver(
        working_dir_pattern="x",
        output_file_patterns={"a": "x/a"},
        root=workspace,
    )
    assert isinstance(resolver, PathResolver)


def test_output_file_auto_prepends_working_dir_for_bare_relative_pattern():
    """Regression: output patterns WITHOUT an explicit ``{working_dir}``
    marker used to return bare relative paths, which then resolved
    against the driver's cwd at read-time and silently missed the case
    dir. Now they auto-prepend working_dir, matching how
    ``working_dir_pattern`` auto-prepends ``root``.
    """
    from pathlib import Path

    from workflow_common import TemplatePathResolver, CaseContext

    r = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "bare":         "results/options/avg_stress.txt",
            "with_marker":  "{working_dir}/results/options/avg_stress.txt",
            "abs":          "/tmp/external.txt",
        },
        root=Path("calibration_run"),
    )
    ctx = CaseContext(generation=0, gene=3, obj=1)
    wd = r.working_dir(ctx)

    # Bare relative: auto-prepended working_dir.
    assert r.output_file("bare", ctx) == wd / "results/options/avg_stress.txt"
    # Explicit marker: unchanged — user opted in, we don't double up.
    assert r.output_file("with_marker", ctx) == wd / "results/options/avg_stress.txt"
    # Absolute: honored exactly as-is, no prepend.
    assert r.output_file("abs", ctx) == Path("/tmp/external.txt")


def test_output_file_bare_relative_matches_working_dir_regardless_of_root_type(tmp_path):
    """The auto-prepend must work for both absolute and relative
    workspace roots — relative was the case that hurt in production.
    """
    from pathlib import Path

    from workflow_common import TemplatePathResolver, CaseContext

    for root in (Path("relative_workspace"), tmp_path / "absolute_workspace"):
        r = TemplatePathResolver(
            working_dir_pattern="gen_{generation}/gene_{gene}",
            output_file_patterns={"s": "results/avg_stress.txt"},
            root=root,
        )
        ctx = CaseContext(generation=0, gene=0, obj=0)
        expected = r.working_dir(ctx) / "results/avg_stress.txt"
        assert r.output_file("s", ctx) == expected
