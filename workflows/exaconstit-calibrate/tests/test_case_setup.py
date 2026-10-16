"""
Unit tests for :mod:`workflow_common.case_setup`.

Covers :class:`CaseTemplater` and the two shipped
:class:`PropertyWriter` implementations. Each test writes a small
template fixture to the workspace, exercises the writer, and
asserts the resulting file has the expected content.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from workflow_common import (
    CallablePropertyWriter,
    CaseContext,
    CaseLayout,
    CaseTemplater,
    DelimitedPropertyWriter,
    PropertyWriter,
    TemplatePathResolver,
    TemplatePropertyWriter,
    TemplateTarget,
)


def _make_layout(workspace: Path) -> CaseLayout:
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={},
        root=workspace,
    )
    return CaseLayout(
        ctx=CaseContext(generation=0, gene=3, obj=1),
        resolver=resolver,
    )


# --- CaseTemplater -------------------------------------------------------


def test_templater_renders_single_target(workspace: Path):
    """One template, one output. Substitution works."""
    src = workspace / "master.toml"
    src.write_text("rate = %%rate%%\nname = %%name%%\n")
    templater = CaseTemplater([TemplateTarget(source=src, dest="options.toml")])

    layout = _make_layout(workspace)
    written = templater.render(layout, {"rate": 1e-3, "name": "hello"})

    assert len(written) == 1
    content = written[0].read_text()
    assert "rate = 0.001" in content
    assert "name = hello" in content
    assert "%%" not in content


def test_templater_renders_multiple_targets(workspace: Path):
    """Multiple targets all rendered in one render() call."""
    src1 = workspace / "opts.toml"
    src2 = workspace / "mesh.toml"
    src1.write_text("rate = %%rate%%\n")
    src2.write_text("mesh_size = %%mesh%%\n")
    templater = CaseTemplater([
        TemplateTarget(source=src1, dest="options.toml"),
        TemplateTarget(source=src2, dest="mesh.toml"),
    ])

    layout = _make_layout(workspace)
    written = templater.render(layout, {"rate": 1e-3, "mesh": "fine"})

    assert len(written) == 2
    assert "rate = 0.001" in written[0].read_text()
    assert "mesh_size = fine" in written[1].read_text()


def test_templater_creates_parent_directories(workspace: Path):
    """Destination with nested subpath creates parents."""
    src = workspace / "master.toml"
    src.write_text("value = %%v%%\n")
    templater = CaseTemplater([
        TemplateTarget(source=src, dest="cfg/sub/options.toml"),
    ])

    layout = _make_layout(workspace)
    written = templater.render(layout, {"v": 42})

    assert written[0].exists()
    assert written[0].parent.is_dir()


def test_templater_passes_through_extras(workspace: Path):
    """Values not used by the template are not an error."""
    src = workspace / "master.toml"
    src.write_text("only_this = %%a%%\n")
    templater = CaseTemplater([TemplateTarget(source=src, dest="out.toml")])

    layout = _make_layout(workspace)
    # "unused_b" is not referenced by the template; should not raise.
    written = templater.render(layout, {"a": 1, "unused_b": 99})
    assert "only_this = 1" in written[0].read_text()


def test_templater_strict_raises_on_missing_key(workspace: Path):
    """A strict target with an unresolved placeholder raises."""
    src = workspace / "master.toml"
    src.write_text("value = %%missing%%\n")
    templater = CaseTemplater([
        TemplateTarget(source=src, dest="out.toml", strict=True),
    ])

    layout = _make_layout(workspace)
    with pytest.raises(KeyError):
        templater.render(layout, {"other": 1})


def test_templater_non_strict_passes_through_unknown(workspace: Path):
    """A non-strict target leaves unknown placeholders intact."""
    src = workspace / "master.toml"
    src.write_text("a = %%a%%, b = %%b%%\n")
    templater = CaseTemplater([
        TemplateTarget(source=src, dest="out.toml", strict=False),
    ])

    layout = _make_layout(workspace)
    written = templater.render(layout, {"a": 1})
    content = written[0].read_text()
    assert "a = 1" in content
    assert "%%b%%" in content


def test_templater_accepts_empty_targets(workspace: Path):
    """Empty targets is a valid no-op: some workflows do all per-case
    file writing inside a PropertyWriter and need no template
    rendering at all. render() must succeed silently in that case.
    """
    from workflow_common.paths import CaseContext
    from workflow_common.results import CaseLayout

    templater = CaseTemplater([])
    assert templater.targets == ()
    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}",
        output_file_patterns={},
        root=workspace,
    )
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0),
        resolver=resolver,
    )
    # render() should not raise even with no targets.
    templater.render(layout, {"any": "values"})


def test_templater_substitute_false_copies_bytes_verbatim(workspace: Path):
    """substitute=False bypasses %%key%% substitution entirely so
    binary or binary-safe files can be staged alongside rendered
    text files.
    """
    # A file whose bytes would be corrupted by a naive text-mode
    # read/write round trip. The %%looks_like_a_placeholder%% text
    # is included deliberately - substitute=False must NOT try to
    # resolve it.
    src = workspace / "binary_ish.ori"
    raw_bytes = b"\x00\x01\x02%%looks_like_a_placeholder%%\xffquat\n"
    src.write_bytes(raw_bytes)

    templater = CaseTemplater([
        TemplateTarget(source=src, dest="staged.ori", substitute=False),
    ])
    layout = _make_layout(workspace)
    written = templater.render(layout, values={})
    assert len(written) == 1
    # Bytes are preserved exactly - no decoding, no placeholder
    # substitution, no trailing-newline fiddling.
    assert written[0].read_bytes() == raw_bytes


def test_templater_substitute_false_ignores_missing_values(workspace: Path):
    """A substitute=False target must not raise
    UnresolvedPlaceholderError even if the source happens to contain
    text that looks like a %%placeholder%%. The whole point of the
    flag is to skip the placeholder pass.
    """
    src = workspace / "raw.bin"
    src.write_text("this has %%unresolved%% text in it\n")

    # strict=True would normally blow up on %%unresolved%% during a
    # substituting render. With substitute=False, strict is ignored.
    templater = CaseTemplater([
        TemplateTarget(
            source=src, dest="out.bin",
            substitute=False, strict=True,
        ),
    ])
    layout = _make_layout(workspace)
    # Empty values mapping: proves we never consulted values at all.
    written = templater.render(layout, values={})
    assert written[0].read_text() == "this has %%unresolved%% text in it\n"


def test_templater_mixed_substituting_and_raw_targets(workspace: Path):
    """A single templater can render some targets and copy others
    verbatim in one call. Order is preserved in the returned list.
    """
    tmpl_src = workspace / "master.toml"
    tmpl_src.write_text("rate = %%rate%%\n")
    raw_src = workspace / "static.bin"
    raw_src.write_bytes(b"\x00\x01static\n")

    templater = CaseTemplater([
        TemplateTarget(source=tmpl_src, dest="options.toml"),
        TemplateTarget(source=raw_src, dest="static.bin", substitute=False),
    ])
    layout = _make_layout(workspace)
    written = templater.render(layout, values={"rate": 1.5})
    assert len(written) == 2
    assert written[0].read_text() == "rate = 1.5\n"
    assert written[1].read_bytes() == b"\x00\x01static\n"


# --- TemplatePropertyWriter ---------------------------------------------


def test_template_property_writer_basic(workspace: Path):
    """Gene values substituted into a template file."""
    tmpl = workspace / "master_props.toml"
    tmpl.write_text(
        "[Voce]\n"
        "yield = %%yield%%\n"
        "hardening = %%hardening%%\n"
    )
    writer = TemplatePropertyWriter(
        template_path=tmpl, dest="properties.toml",
    )

    layout = _make_layout(workspace)
    path = writer.write(
        layout,
        gene=[200.0, 2000.0],
        param_names=["yield", "hardening"],
    )
    content = path.read_text()
    assert "yield = 200" in content
    assert "hardening = 2000" in content


def test_template_property_writer_mismatch_raises(workspace: Path):
    """Gene and param_names must have same length."""
    tmpl = workspace / "p.toml"
    tmpl.write_text("x = %%x%%\n")
    writer = TemplatePropertyWriter(template_path=tmpl)

    layout = _make_layout(workspace)
    with pytest.raises(ValueError, match="entries"):
        writer.write(layout, gene=[1.0, 2.0], param_names=["x"])


def test_template_property_writer_extra_values(workspace: Path):
    """extra_values are applied, gene values win on collision."""
    tmpl = workspace / "p.toml"
    tmpl.write_text(
        "gene_val = %%yield%%\n"
        "extra_val = %%constant%%\n"
    )
    writer = TemplatePropertyWriter(
        template_path=tmpl,
        extra_values={"constant": "fixed", "yield": "should_be_overridden"},
    )
    layout = _make_layout(workspace)
    path = writer.write(
        layout, gene=[250.0], param_names=["yield"],
    )
    content = path.read_text()
    assert "gene_val = 250" in content       # gene wins
    assert "extra_val = fixed" in content    # extra still applied


def test_template_property_writer_protocol_conformance(workspace: Path):
    """TemplatePropertyWriter satisfies the PropertyWriter Protocol."""
    assert isinstance(
        TemplatePropertyWriter(template_path=Path("x")), PropertyWriter
    )


# --- DelimitedPropertyWriter --------------------------------------------


def test_delimited_writer_default_one_per_line(workspace: Path):
    """Default newline delimiter produces one value per line."""
    writer = DelimitedPropertyWriter(dest="props.txt")
    layout = _make_layout(workspace)
    path = writer.write(
        layout, gene=[200.0, 2000.0, 5.0],
        param_names=["a", "b", "c"],
    )
    text = path.read_text()
    lines = text.rstrip("\n").split("\n")
    assert lines == ["200", "2000", "5"]


def test_delimited_writer_csv_with_header(workspace: Path):
    writer = DelimitedPropertyWriter(
        dest="p.csv", delimiter=",", include_names=True,
    )
    layout = _make_layout(workspace)
    path = writer.write(
        layout, gene=[1.0, 2.0], param_names=["a", "b"],
    )
    lines = path.read_text().rstrip("\n").split("\n")
    assert lines[0] == "a,b"
    assert lines[1] == "1,2"


def test_delimited_writer_include_names_requires_alignment(workspace: Path):
    writer = DelimitedPropertyWriter(
        dest="p.txt", include_names=True,
    )
    layout = _make_layout(workspace)
    with pytest.raises(ValueError):
        writer.write(layout, gene=[1.0, 2.0], param_names=["only_one"])


def test_delimited_writer_custom_fmt(workspace: Path):
    """fmt controls per-value formatting."""
    writer = DelimitedPropertyWriter(dest="p.txt", fmt="%.4e")
    layout = _make_layout(workspace)
    path = writer.write(layout, gene=[0.001234], param_names=["x"])
    assert path.read_text().rstrip("\n") == "1.2340e-03"


def test_delimited_writer_protocol_conformance():
    """DelimitedPropertyWriter satisfies the PropertyWriter Protocol."""
    assert isinstance(DelimitedPropertyWriter(), PropertyWriter)


# --- CallablePropertyWriter --------------------------------------------


def test_callable_property_writer_invokes_user_function(workspace: Path):
    """The user callable receives (case_dir, gene, names, sim_case) and
    its returned path is echoed back to the framework.
    """
    captured = {}

    def write_props(case_dir, gene, names, sim_case):
        captured["case_dir"] = case_dir
        captured["gene"] = list(gene)
        captured["names"] = list(names)
        captured["sim_case"] = sim_case
        path = case_dir / "mat.props"
        case_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# gene={list(gene)}\n")
        return path

    writer = CallablePropertyWriter(func=write_props)
    layout = _make_layout(workspace)
    out = writer.write(
        layout, gene=[1.5, 2.5], param_names=["a", "b"],
    )
    assert out == layout.working_dir / "mat.props"
    assert out.read_text() == "# gene=[1.5, 2.5]\n"
    assert captured["case_dir"] == layout.working_dir
    assert captured["gene"] == [1.5, 2.5]
    assert captured["names"] == ["a", "b"]
    # No Problem dispatched this, so ctx.sim_case stays None.
    assert captured["sim_case"] is None


def test_callable_property_writer_receives_sim_case(workspace: Path):
    """When ctx.sim_case is set (as Problem does in production),
    the callable gets the SimCase object and can reach its
    case_data / label for per-experiment logic.
    """
    from workflow_common import SimCase

    captured_sim_case = {}

    def write_props(case_dir, gene, names, sim_case):
        captured_sim_case["obj"] = sim_case
        p = case_dir / "p.txt"
        case_dir.mkdir(parents=True, exist_ok=True)
        p.write_text("ok\n")
        return p

    writer = CallablePropertyWriter(func=write_props)

    sc = SimCase(
        case_data={"strain_rate": 1e-3, "ori_file": "grains.ori"},
        label="quasi_static",
    )
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={},
        root=workspace,
    )
    ctx = CaseContext(generation=0, gene=0, obj=0, sim_case=sc)
    layout = CaseLayout(ctx=ctx, resolver=resolver)
    writer.write(layout, gene=[1.0], param_names=["x"])

    assert captured_sim_case["obj"] is sc
    assert captured_sim_case["obj"].case_data["ori_file"] == "grains.ori"
    assert captured_sim_case["obj"].label == "quasi_static"


def test_callable_property_writer_rejects_non_path_return(workspace: Path):
    """A callable that forgets to return a Path gets a clear TypeError."""
    def bad_writer(case_dir, gene, names, sim_case):
        return "/tmp/some/str/path"  # str instead of Path

    writer = CallablePropertyWriter(func=bad_writer)
    layout = _make_layout(workspace)
    with pytest.raises(TypeError, match="pathlib.Path"):
        writer.write(layout, gene=[1.0], param_names=["x"])


def test_callable_property_writer_propagates_exceptions(workspace: Path):
    """User errors inside the callable surface unchanged so the
    configuration-vs-sim-failure distinction stays clear.
    """
    class _ConfigError(Exception):
        pass

    def raise_config_error(case_dir, gene, names, sim_case):
        raise _ConfigError("gene out of bounds")

    writer = CallablePropertyWriter(func=raise_config_error)
    layout = _make_layout(workspace)
    with pytest.raises(_ConfigError, match="out of bounds"):
        writer.write(layout, gene=[1.0], param_names=["x"])


def test_callable_property_writer_protocol_conformance():
    """Structurally satisfies PropertyWriter."""
    def noop(case_dir, gene, names, sim_case):
        p = case_dir / "x"
        case_dir.mkdir(parents=True, exist_ok=True)
        p.touch()
        return p

    assert isinstance(CallablePropertyWriter(func=noop), PropertyWriter)
