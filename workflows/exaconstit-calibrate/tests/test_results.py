"""
Unit tests for :mod:`workflow_common.results`.

These exercise the step-4 additions: :class:`CaseLayout`,
:class:`TextTableReader`, experimental-data loading, and the
time-alignment helpers. Each test writes a small in-memory test
fixture to the workspace and asserts the reader produces the
expected structure.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common.paths import CaseContext, TemplatePathResolver
from workflow_common.results import (
    CaseLayout,
    ResultReader,
    TabularResult,
    TextTableReader,
    TextTableSpec,
    common_time_range,
    interpolate_to,
    load_experimental_csv,
)


def _resolver(workspace: Path) -> TemplatePathResolver:
    """Standard resolver used across several tests.

    Layout pattern mirrors ExaConstit's actual output structure:
    ``<workspace>/wf/gen_N/gene_N_obj_N/results/options/...``.
    """
    return TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
            "avg_plastic_work": "{working_dir}/results/options/avg_plastic_work.txt",
        },
        root=workspace,
    )


def _write_table(path: Path, rows: list[list[float]]) -> None:
    """Write a whitespace-delimited numeric table with ``mkdir -p`` semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{v}" for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


# --- CaseLayout ----------------------------------------------------------


def test_layout_working_dir_and_output_file(workspace: Path):
    """Layout delegates to the resolver and exposes case-scoped paths."""
    resolver = _resolver(workspace)
    ctx = CaseContext(generation=0, gene=3, obj=1)
    layout = CaseLayout(ctx=ctx, resolver=resolver)

    assert layout.working_dir == workspace / "wf" / "gen_0" / "gene_3_obj_1"
    assert layout.output_file("avg_stress") == (
        layout.working_dir / "results" / "options" / "avg_stress.txt"
    )


def test_layout_known_outputs_reflects_resolver(workspace: Path):
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    assert set(layout.known_outputs()) == {
        "avg_stress", "avg_def_grad", "avg_plastic_work"
    }


def test_layout_exists_and_present_outputs(workspace: Path):
    """exists() and present_outputs() reflect what is really on disk."""
    resolver = _resolver(workspace)
    ctx = CaseContext(generation=0, gene=0, obj=0)
    layout = CaseLayout(ctx=ctx, resolver=resolver)

    assert not layout.exists("avg_stress")
    assert layout.present_outputs() == []

    _write_table(layout.output_file("avg_stress"), [[0.0, 100.0]])
    assert layout.exists("avg_stress")
    assert layout.present_outputs() == ["avg_stress"]


def test_layout_clear_outputs_removes_only_specified(workspace: Path):
    """clear_outputs selectively deletes only files we ask for."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _write_table(layout.output_file("avg_stress"), [[0.0]])
    _write_table(layout.output_file("avg_def_grad"), [[0.0]])

    removed = layout.clear_outputs(["avg_stress"])
    assert len(removed) == 1
    assert not layout.exists("avg_stress")
    assert layout.exists("avg_def_grad")  # untouched


def test_layout_clear_outputs_all_when_none_given(workspace: Path):
    """clear_outputs with no arg clears every known output that exists."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _write_table(layout.output_file("avg_stress"), [[0.0]])
    _write_table(layout.output_file("avg_def_grad"), [[0.0]])

    removed = layout.clear_outputs()
    assert len(removed) == 2
    assert layout.present_outputs() == []


def test_layout_clear_outputs_tolerates_missing(workspace: Path):
    """Clearing an already-absent file is a no-op, not an error."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    # No files on disk yet.
    removed = layout.clear_outputs()
    assert removed == []


# --- TextTableReader -----------------------------------------------------


def _stress_spec(required: bool = True) -> TextTableSpec:
    return TextTableSpec(
        columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
        required=required,
    )


def _def_grad_spec(required: bool = False) -> TextTableSpec:
    return TextTableSpec(
        columns=[
            "Time", "Volume",
            "F11", "F12", "F13",
            "F21", "F22", "F23",
            "F31", "F32", "F33",
        ],
        required=required,
    )


def _make_avg_stress(layout: CaseLayout, n: int = 10) -> None:
    """Write a plausible avg_stress.txt with ``n`` rows matching
    ExaConstit's Time + Volume + 6-Cauchy layout. The test asserts
    Szz[0] == 100 (load along z), so Szz is the active component;
    Sxx/Syy/shears stay zero.
    """
    rows = []
    for i in range(n):
        t = 0.1 * i
        szz = 100.0 + 200.0 * (1 - 2 ** (-t))
        # Layout: Time, Volume, Sxx, Syy, Szz, Sxy, Sxz, Syz
        rows.append([t, 1.0, 0.0, 0.0, szz, 0.0, 0.0, 0.0])
    _write_table(layout.output_file("avg_stress"), rows)


def test_reader_reads_required_output(workspace: Path):
    """A present, well-formed required file becomes a TabularResult."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _make_avg_stress(layout)

    reader = TextTableReader({"avg_stress": _stress_spec(required=True)})
    rs = reader.read(layout)

    assert "avg_stress" in rs
    df = rs.df("avg_stress")
    assert list(df.columns) == ["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"]
    assert len(df) == 10
    assert df["Szz"].iloc[0] == pytest.approx(100.0)


def test_reader_missing_required_raises(workspace: Path):
    """FileNotFoundError when a required output is absent."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    # No file on disk.
    reader = TextTableReader({"avg_stress": _stress_spec(required=True)})
    with pytest.raises(FileNotFoundError):
        reader.read(layout)


def test_reader_missing_optional_recorded(workspace: Path):
    """Missing optional outputs show up in rs.missing and do not abort."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _make_avg_stress(layout)

    reader = TextTableReader({
        "avg_stress": _stress_spec(required=True),
        "avg_def_grad": _def_grad_spec(required=False),
    })
    rs = reader.read(layout)

    assert "avg_stress" in rs
    assert "avg_def_grad" not in rs.tables
    assert rs.missing == ["avg_def_grad"]
    # Truthiness reflects "complete" status.
    assert not bool(rs)


def test_reader_column_count_mismatch_raises(workspace: Path):
    """A file with the wrong number of columns should raise a clear error."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    # 3 columns but the spec expects 8 (Time+Volume+6 stress).
    _write_table(layout.output_file("avg_stress"), [[0.0, 1.0, 2.0]])

    reader = TextTableReader({"avg_stress": _stress_spec(required=True)})
    with pytest.raises(ValueError, match="expected 8 columns"):
        reader.read(layout)


def test_reader_all_ok_is_truthy(workspace: Path):
    """CaseResultSet.__bool__ is True when nothing is missing."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _make_avg_stress(layout)

    reader = TextTableReader({"avg_stress": _stress_spec(required=True)})
    rs = reader.read(layout)
    assert bool(rs)


def test_reader_protocol_conformance():
    """TextTableReader satisfies the ResultReader Protocol."""
    r = TextTableReader({})
    assert isinstance(r, ResultReader)


def test_tabular_result_preserves_source_path(workspace: Path):
    """The TabularResult carries the source path for downstream diagnostics."""
    resolver = _resolver(workspace)
    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0), resolver=resolver
    )
    _make_avg_stress(layout)
    reader = TextTableReader({"avg_stress": _stress_spec()})
    rs = reader.read(layout)
    assert isinstance(rs["avg_stress"], TabularResult)
    assert rs["avg_stress"].source_path == layout.output_file("avg_stress")


# --- Experimental loader -------------------------------------------------


def test_load_experimental_csv_whitespace(workspace: Path):
    """Whitespace-delimited file, header-free, explicit columns."""
    path = workspace / "exp.txt"
    path.write_text(
        "# This is a comment\n"
        "0.000 0.0\n"
        "0.001 150.0\n"
        "0.002 200.0\n"
    )
    df = load_experimental_csv(path, columns=["strain", "stress"])
    assert list(df.columns) == ["strain", "stress"]
    assert len(df) == 3
    assert df["stress"].iloc[-1] == pytest.approx(200.0)


def test_load_experimental_csv_commas(workspace: Path):
    """Comma-separated with a header row (default: first row is header)."""
    path = workspace / "exp.csv"
    path.write_text("strain,stress\n0.0,0.0\n0.001,100.0\n")
    df = load_experimental_csv(path, delimiter=",")
    assert list(df.columns) == ["strain", "stress"]
    assert len(df) == 2


# --- Time-alignment helpers ---------------------------------------------


def test_common_time_range_basic():
    """Simple overlap of two DataFrames."""
    a = pd.DataFrame({"Time": [0.0, 1.0, 2.0, 3.0], "x": [0, 1, 2, 3]})
    b = pd.DataFrame({"Time": [0.5, 1.5, 2.5, 3.5], "x": [0, 1, 2, 3]})
    tmin, tmax = common_time_range(a, b)
    assert tmin == 0.5
    assert tmax == 3.0


def test_common_time_range_empty_intersection_raises():
    a = pd.DataFrame({"Time": [0.0, 1.0]})
    b = pd.DataFrame({"Time": [5.0, 6.0]})
    with pytest.raises(ValueError):
        common_time_range(a, b)


def test_common_time_range_missing_column_raises():
    a = pd.DataFrame({"Time": [0.0, 1.0]})
    b = pd.DataFrame({"t": [0.0, 1.0]})  # wrong name
    with pytest.raises(ValueError, match="Time"):
        common_time_range(a, b)


def test_interpolate_to_linear_exact():
    """On exactly-matching target times, interpolate is identity (up to fp)."""
    df = pd.DataFrame({"Time": [0.0, 1.0, 2.0], "x": [0.0, 10.0, 20.0]})
    out = interpolate_to(df, np.array([0.5, 1.5]))
    assert out["x"].tolist() == pytest.approx([5.0, 15.0])
    assert list(out["Time"]) == [0.5, 1.5]


def test_interpolate_to_refuses_extrapolation():
    """Extrapolating beyond the source data must raise."""
    df = pd.DataFrame({"Time": [0.0, 1.0], "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="outside"):
        interpolate_to(df, np.array([0.0, 2.0]))


def test_text_reader_parses_indented_commented_header(workspace, tmp_path):
    """ExaConstit's ``avg_stress_global.txt`` writes a header line that
    is indented AND commented (``"      # Time  Volume  Sxx ..."``),
    followed by data rows that are also indented with whitespace.
    The pandas C engine mishandles this exact shape, raising
    ``EmptyDataError: No columns to parse from file``. The reader
    auto-switches to the python engine whenever ``comment`` is set,
    so this regression test verifies the auto-switch actually works
    end-to-end for an ExaConstit-formatted file.
    """
    from pathlib import Path

    from workflow_common import (
        TemplatePathResolver, TextTableReader, TextTableSpec,
        CaseContext, CaseLayout,
    )

    # Build a file whose shape matches what ExaConstit writes, byte
    # for byte (modulo values). Indentation and comment placement are
    # the bug-relevant parts.
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "avg_stress.txt").write_text(
        "      # Time            Volume             Sxx               Syy               Szz               Sxy               Sxz               Syz        \n"
        "    1.00000000e-03    9.99999687e-01   -1.37887089e-10   -1.38705682e-10   -1.28596065e-04    1.25773677e-11    4.52148129e-11    4.09139718e-12\n"
        "    4.12500000e-03    9.99998710e-01   -4.66103996e-15   -4.73576816e-15   -5.30460051e-04    4.72689040e-16    8.44011981e-16    5.07378188e-17\n"
    )

    resolver = TemplatePathResolver(
        working_dir_pattern="case",
        output_file_patterns={
            "avg_stress": "avg_stress.txt",   # bare relative; auto-prepends working_dir
        },
        root=tmp_path,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
    })

    layout = CaseLayout(
        ctx=CaseContext(generation=0, gene=0, obj=0),
        resolver=resolver,
    )
    rs = reader.read(layout)

    # The whole point of the test: we get a real DataFrame, not an
    # EmptyDataError and not a missing-file report.
    assert "avg_stress" in rs.tables
    df = rs.df("avg_stress")
    assert df.shape == (2, 8)
    # Load direction column should carry the expected values.
    assert df["Szz"].tolist() == [-1.28596065e-04, -5.30460051e-04]
    # And the Time column survives the parse too.
    assert df["Time"].tolist() == [1e-3, 4.125e-3]
