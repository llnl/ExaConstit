"""Tests for the :mod:`workflows.optimization.inspect_archive` CLI.

Exercises each view (--runs, --gens, --genes) and each output format
against a small fake archive built by hand. The tests check that the
CLI produces output in the right SHAPE (header + rows, CSV with the
right column count, parseable JSON) rather than exact bytes, because
column widths and formatting nuances shouldn't pin the test.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from workflow_common.archive import ArchiveDB, GeneRecord
from workflows.optimization import inspect_archive


@pytest.fixture
def tiny_archive(tmp_path: Path) -> Path:
    """A three-generation, four-individual archive with known ranks.

    Two runs so --run selection can be tested. ``run-old`` has one
    gen and earlier timestamp; ``run-new`` has three gens and is
    what --runs defaults to when ``--run`` is omitted.
    """
    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        # Old run first so its started_at is earlier.
        old_id = a.start_run(
            run_id="run-old", seed=1,
            param_names=["p1"], objective_labels=["obj"],
        )
        a.record_generation(
            old_id, gen_idx=0,
            genes=[GeneRecord(
                run_id=old_id, gen_idx=0, pop_idx=0,
                birth_gen=0, birth_gene=0,
                gene_vector=np.array([1.0]),
                fitness=(0.1,), rank=0,
            )],
            stats={"avg": [0.1]},
        )
        a.end_run(old_id)

        new_id = a.start_run(
            run_id="run-new", seed=2,
            param_names=["yield_stress", "hardening"],
            objective_labels=["rmse"],
        )
        for gen_idx in range(3):
            genes = [
                GeneRecord(
                    run_id=new_id, gen_idx=gen_idx, pop_idx=i,
                    birth_gen=gen_idx, birth_gene=i,
                    gene_vector=np.array([200.0 + i, 2000.0 + 10 * i]),
                    fitness=(1.5 - 0.1 * gen_idx + 0.01 * i,),
                    # First two individuals per gen are rank 0.
                    rank=0 if i < 2 else 1,
                )
                for i in range(4)
            ]
            a.record_generation(
                new_id, gen_idx=gen_idx, genes=genes,
                stats={
                    "avg": [1.5 - 0.1 * gen_idx],
                    "min": [1.4 - 0.1 * gen_idx],
                    "max": [1.6 - 0.1 * gen_idx],
                },
            )
        a.end_run(new_id)
    return tmp_path


def test_inspect_runs_lists_every_run(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--runs dumps one row per run; both runs appear; headers present."""
    rc = inspect_archive.main([str(tiny_archive), "--runs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_id" in out                # header present
    assert "run-old" in out
    assert "run-new" in out


def test_inspect_gens_defaults_to_latest_run(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--gens (no --run) picks the most recent run automatically."""
    rc = inspect_archive.main([str(tiny_archive), "--gens"])
    assert rc == 0
    out = capsys.readouterr().out
    # Three gens in the new run, so three body lines.
    body = [
        line for line in out.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(body) == 3


def test_inspect_gens_respects_explicit_run(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """Passing --run forces that run even when newer ones exist."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--gens", "--run", "run-old"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    body = [
        line for line in out.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(body) == 1  # only one gen in run-old


def test_inspect_gens_unknown_run_errors(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """Unknown run_id fails cleanly with exit code 1."""
    with pytest.raises(SystemExit) as excinfo:
        inspect_archive.main(
            [str(tiny_archive), "--gens", "--run", "nope"],
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "nope" in err


def test_inspect_genes_at_specific_gen(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--genes --gen 2 yields NPOP rows with real parameter names."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--genes", "--gen", "2"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    # Header has the real param names (not 'p0', 'p1').
    assert "yield_stress" in out
    assert "hardening" in out
    assert "rmse" in out
    body = [
        line for line in out.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(body) == 4


def test_inspect_genes_pareto_only(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only filters to rank-0 individuals."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--genes", "--gen", "2", "--pareto-only"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    body = [
        line for line in out.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    # Fixture put two rank-0 individuals per gen.
    assert len(body) == 2


def test_inspect_genes_csv_is_parseable(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--format csv produces a CSV that csv.DictReader can parse back."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--genes", "--gen", "2", "--format", "csv"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == 4
    # Each row has the parameter columns filled with numeric-looking
    # strings.
    for r in rows:
        assert r["yield_stress"]
        assert r["hardening"]
        assert r["rmse"]


def test_inspect_json_is_valid_json(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--format json produces a valid JSON array of objects."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--genes", "--gen", "2", "--format", "json"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 4
    assert all("yield_stress" in item for item in parsed)


def test_inspect_no_header(
    tiny_archive: Path, capsys: pytest.CaptureFixture,
):
    """--no-header suppresses the header row in CSV output."""
    rc = inspect_archive.main(
        [str(tiny_archive), "--genes", "--gen", "2",
         "--format", "csv", "--no-header"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    # First line shouldn't contain the word "yield_stress" (which
    # would imply the header printed).
    first_line = out.splitlines()[0]
    assert "yield_stress" not in first_line


def test_inspect_missing_archive_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Pointing at a directory with no *.db files at all errors cleanly.

    Used to rely on the old ``--archive-name`` path; now routes
    through ``_resolve_archive_path`` which raises ``SystemExit(1)``
    — argparse convention. The error message names the directory
    and the default filename so the user knows what was looked for.
    """
    with pytest.raises(SystemExit) as excinfo:
        inspect_archive.main([str(tmp_path), "--runs"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "archive.db" in err
    assert "no " in err


# --- Path-resolution behaviors (direct file / auto-pick / prompt) -------


def test_resolve_accepts_file_path_directly(tmp_path: Path):
    """Behavior 1: positional arg is a ``.db`` file, used as-is.

    The user reported `./calibration_run/calibration.db` failing
    because the old CLI blindly appended `archive.db` to anything.
    The fix is: if the path IS a file, return it verbatim.
    """
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    # Custom filename — not 'archive.db'.
    custom = tmp_path / "calibration.db"
    with ArchiveDB(custom) as a:
        a.start_run(
            run_id="r0", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        a.end_run("r0")

    resolved = _resolve_archive_path(custom, "archive.db")
    assert resolved == custom


def test_resolve_prefers_default_name_in_directory(tmp_path: Path):
    """Behavior 2: directory contains ``archive.db``; that wins."""
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    default = tmp_path / "archive.db"
    other = tmp_path / "other.db"
    for p in (default, other):
        with ArchiveDB(p) as a:
            a.start_run(
                run_id="r", seed=1,
                param_names=["p"], objective_labels=["o"],
            )
            a.end_run("r")

    # Even though there's also "other.db", the default wins silently.
    resolved = _resolve_archive_path(tmp_path, "archive.db")
    assert resolved == default


def test_resolve_auto_picks_sole_db_file_and_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Behavior 3a: no archive.db, exactly one *.db → use it, warn on stderr.

    The stderr notice is the audit trail: the user sees which file
    was inferred so they can't be surprised later.
    """
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    sole = tmp_path / "calibration.db"
    with ArchiveDB(sole) as a:
        a.start_run(
            run_id="r", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        a.end_run("r")

    resolved = _resolve_archive_path(tmp_path, "archive.db")
    assert resolved == sole
    err = capsys.readouterr().err
    assert "calibration.db" in err


def test_resolve_prompts_when_multiple_db_files(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Behavior 3b: multiple *.db files, interactive → prompt for a pick.

    Uses injected ``input_fn`` and ``isatty_fn`` to drive the
    prompt without touching real stdin. Verifies the user can
    pick one of the candidates by 1-based index.
    """
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    a_path = tmp_path / "a.db"
    b_path = tmp_path / "b.db"
    for p in (a_path, b_path):
        with ArchiveDB(p) as arch:
            arch.start_run(
                run_id="r", seed=1,
                param_names=["p"], objective_labels=["o"],
            )
            arch.end_run("r")

    resolved = _resolve_archive_path(
        tmp_path, "archive.db",
        input_fn=lambda _prompt: "2",        # pick the 2nd (glob is sorted)
        isatty_fn=lambda: True,              # pretend we have a tty
    )
    # Sorted candidates: [a.db, b.db]; index 2 = b.db.
    assert resolved == b_path


def test_resolve_prompt_rejects_bad_input_then_accepts(
    tmp_path: Path,
):
    """Prompt loops on garbage input rather than bailing on first error.

    User types 'banana', then '99', then '1'. Resolver loops
    until it gets a valid index.
    """
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    (tmp_path / "one.db").touch()
    (tmp_path / "two.db").touch()
    with ArchiveDB(tmp_path / "one.db") as a:
        a.start_run(run_id="r", seed=1, param_names=["p"], objective_labels=["o"])
        a.end_run("r")
    with ArchiveDB(tmp_path / "two.db") as a:
        a.start_run(run_id="r", seed=1, param_names=["p"], objective_labels=["o"])
        a.end_run("r")

    replies = iter(["banana", "99", "1"])
    resolved = _resolve_archive_path(
        tmp_path, "archive.db",
        input_fn=lambda _p: next(replies),
        isatty_fn=lambda: True,
    )
    assert resolved == tmp_path / "one.db"


def test_resolve_prompt_quit_exits_nonzero(tmp_path: Path):
    """User typing 'q' at the prompt should exit cleanly with rc=1."""
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    for name in ("a.db", "b.db"):
        (tmp_path / name).touch()
        with ArchiveDB(tmp_path / name) as a:
            a.start_run(
                run_id="r", seed=1,
                param_names=["p"], objective_labels=["o"],
            )
            a.end_run("r")

    with pytest.raises(SystemExit) as excinfo:
        _resolve_archive_path(
            tmp_path, "archive.db",
            input_fn=lambda _p: "q",
            isatty_fn=lambda: True,
        )
    assert excinfo.value.code == 1


def test_resolve_refuses_to_prompt_in_non_interactive_context(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Behavior 3c: multiple *.db and no tty → error, don't hang.

    This is the CI-friendly path. Without this guard, a CI job
    piping data to the tool would deadlock at the prompt.
    """
    from workflow_common.archive import ArchiveDB
    from workflows.optimization.inspect_archive import _resolve_archive_path

    for name in ("a.db", "b.db"):
        with ArchiveDB(tmp_path / name) as a:
            a.start_run(
                run_id="r", seed=1,
                param_names=["p"], objective_labels=["o"],
            )
            a.end_run("r")

    with pytest.raises(SystemExit) as excinfo:
        _resolve_archive_path(
            tmp_path, "archive.db",
            # input_fn would hang if called; verify it isn't.
            input_fn=lambda _p: (_ for _ in ()).throw(
                AssertionError("input_fn must NOT be called on non-tty")
            ),
            isatty_fn=lambda: False,
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "multiple" in err
    assert "a.db" in err and "b.db" in err


def test_resolve_reports_nonexistent_path(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Bad path arg (typo, missing dir) exits 1 with a clear message."""
    from workflows.optimization.inspect_archive import _resolve_archive_path

    with pytest.raises(SystemExit) as excinfo:
        _resolve_archive_path(
            tmp_path / "does_not_exist",
            "archive.db",
        )
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_inspect_accepts_file_path_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Full-CLI round trip: pass a non-default filename directly.

    Regression for the original user report:
        python -m ... inspect_archive ./calibration_run/calibration.db --runs
    used to fail with "archive not found at calibration_run/calibration.db/archive.db".
    """
    from workflow_common.archive import ArchiveDB

    custom = tmp_path / "calibration.db"
    with ArchiveDB(custom) as a:
        a.start_run(
            run_id="calib-run-0", seed=42,
            param_names=["yield_stress"],
            objective_labels=["rmse"],
        )
        a.end_run("calib-run-0")

    rc = inspect_archive.main([str(custom), "--runs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "calib-run-0" in out


# --- Cross-generation --pareto-only ------------------------------------


@pytest.fixture
def cross_gen_archive(tmp_path: Path) -> Path:
    """Archive where each category's winner lives in a different gen.

    Designed so the cross-gen view can be verified by exact
    identity: gen 1 pop 0 is the L2 champion, gen 3 pop 0 is the
    stress_rmse champion, gen 5 pop 0 is the slope_rmse champion.
    Other individuals cluster mid-fitness so they shouldn't appear
    in any top-3 list.
    """
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="cross-gen-test",
            seed=42,
            param_names=["yield_stress", "hardening"],
            objective_labels=["stress_rmse", "slope_rmse"],
        )
        for gen_idx in range(6):
            genes = []
            for pop_idx in range(5):
                # Three "champion" genes scattered across gens:
                if gen_idx == 1 and pop_idx == 0:
                    gv, fit = np.array([250.0, 2000.0]), (0.10, 0.15)
                elif gen_idx == 3 and pop_idx == 0:
                    gv, fit = np.array([260.0, 2200.0]), (0.05, 0.60)
                elif gen_idx == 5 and pop_idx == 0:
                    gv, fit = np.array([230.0, 1800.0]), (0.70, 0.08)
                else:
                    # Mid-pack filler, different gene per slot so
                    # dedup doesn't collapse them.
                    gv = np.array([
                        200.0 + gen_idx * 5 + pop_idx,
                        2000.0 + gen_idx * 20 + pop_idx * 10,
                    ])
                    fit = (0.3 + 0.01 * pop_idx, 0.4 + 0.01 * pop_idx)
                genes.append(GeneRecord(
                    run_id=rid, gen_idx=gen_idx, pop_idx=pop_idx,
                    birth_gen=gen_idx, birth_gene=pop_idx,
                    gene_vector=gv, fitness=fit, rank=pop_idx,
                ))
            a.record_generation(rid, gen_idx=gen_idx, genes=genes, stats={})
        a.end_run(rid)
    return tmp_path


def test_pareto_only_shows_cross_generation_winners(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only alone (no --gen) surfaces the historical bests.

    Each of the three category winners in the fixture should
    appear as row 1 of its respective category: L2 champion at
    the top of the l2 block, stress_rmse winner at the top of its
    block, slope_rmse winner at the top of its.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--genes", "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    out = capsys.readouterr().out
    rows = list(csv.DictReader(io.StringIO(out)))

    l2_rows = [r for r in rows if r["category"] == "l2"]
    stress_rows = [r for r in rows if r["category"] == "obj:stress_rmse"]
    slope_rows = [r for r in rows if r["category"] == "obj:slope_rmse"]

    # Default --top is 3.
    assert len(l2_rows) == 3
    assert len(stress_rows) == 3
    assert len(slope_rows) == 3

    # Champion identities (verified by birth_gen — the fixture
    # places each champion at a known generation).
    assert l2_rows[0]["birth_gen"] == "1"
    assert stress_rows[0]["birth_gen"] == "3"
    assert slope_rows[0]["birth_gen"] == "5"


def test_pareto_only_respects_top_flag(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """--top 5 shows five rows per category instead of three."""
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--genes", "--pareto-only",
         "--top", "5", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    # 5 per category × (1 L2 + 2 objectives) = 15 rows.
    assert len(rows) == 15


def test_pareto_only_with_gen_keeps_old_single_gen_behavior(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only --gen N preserves the pre-cross-gen behavior.

    Users who were scripting against the old "rank-0 of one
    generation" semantics need that mode to still exist. Binding
    it to "pareto-only AND explicit gen" is the compatibility
    story.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--genes", "--gen", "3",
         "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    # Fixture gave pop_idx 0 rank 0; only that row comes back.
    assert len(rows) == 1
    assert rows[0]["gen_idx"] == "3"
    assert rows[0]["rank"] == "0"
    # And the columns are the OLD layout, not the cross-gen one.
    assert "gen_idx" in rows[0]
    assert "category" not in rows[0]


def test_pareto_only_dedups_identical_genes_across_gens(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Elitism copies good genes across gens; dedup keeps each unique
    gene once, attributed to its FIRST-seen (birth) generation.
    Without dedup the top-3 table would be ``[champion, champion,
    champion]`` — all the same individual surviving three
    generations — which tells the user nothing.
    """
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="elitist",
            seed=1,
            param_names=["p"],
            objective_labels=["o"],
        )
        # Same champion at gens 0, 1, 2 — elitism. Filler genes
        # differ per gen so they dedup to three unique records.
        for gen_idx in range(3):
            champion = GeneRecord(
                run_id=rid, gen_idx=gen_idx, pop_idx=0,
                birth_gen=0, birth_gene=0,  # birth stays at gen 0
                gene_vector=np.array([42.0]),
                fitness=(0.01,), rank=0,
            )
            filler = GeneRecord(
                run_id=rid, gen_idx=gen_idx, pop_idx=1,
                birth_gen=gen_idx, birth_gene=1,
                gene_vector=np.array([100.0 + gen_idx]),
                fitness=(1.0 + 0.1 * gen_idx,), rank=1,
            )
            a.record_generation(rid, gen_idx=gen_idx,
                                genes=[champion, filler], stats={})
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--genes", "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    l2_rows = [r for r in rows if r["category"] == "l2"]
    # Champion should appear exactly ONCE despite showing up in
    # every generation.
    champion_appearances = [r for r in l2_rows if r["p"] == "42"]
    assert len(champion_appearances) == 1


def test_pareto_only_skips_nonfinite_fitness(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Failed genes (fitness=inf or nan) must NOT appear in the
    rankings — they'd either dominate the "worst" slot meaninglessly
    or crash sort comparisons depending on which way the
    implementation bent.
    """
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="with-failures", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        a.record_generation(
            rid, gen_idx=0,
            genes=[
                GeneRecord(run_id=rid, gen_idx=0, pop_idx=0,
                           birth_gen=0, birth_gene=0,
                           gene_vector=np.array([1.0]),
                           fitness=(0.5,), rank=0),
                GeneRecord(run_id=rid, gen_idx=0, pop_idx=1,
                           birth_gen=0, birth_gene=1,
                           gene_vector=np.array([2.0]),
                           # Simulate a failure-handler penalty.
                           fitness=(float("inf"),), rank=1),
                GeneRecord(run_id=rid, gen_idx=0, pop_idx=2,
                           birth_gen=0, birth_gene=2,
                           gene_vector=np.array([3.0]),
                           fitness=(0.3,), rank=0),
            ],
            stats={},
        )
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--genes", "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    # Only 2 finite individuals — top-3 request returns what's available.
    l2_rows = [r for r in rows if r["category"] == "l2"]
    assert len(l2_rows) == 2
    # The inf-fitness gene (p=2) must not appear.
    assert not any(r["p"] == "2" for r in l2_rows)


# --- --limit cap on raw --genes output ---------------------------------


def test_limit_caps_raw_genes_output(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Default --limit 200 truncates a 500-row generation with a notice."""
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="big", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        genes = [
            GeneRecord(run_id=rid, gen_idx=0, pop_idx=i,
                       birth_gen=0, birth_gene=i,
                       gene_vector=np.array([float(i)]),
                       fitness=(float(i),), rank=0)
            for i in range(500)
        ]
        a.record_generation(rid, gen_idx=0, genes=genes, stats={})
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--genes", "--format", "csv"],
    )
    assert rc == 0
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    # Default cap is 200.
    assert len(rows) == 200
    # User notified via stderr about the truncation.
    assert "truncated" in captured.err
    assert "500" in captured.err


def test_limit_zero_disables_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """--limit 0 returns every row with no truncation notice."""
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="big", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        genes = [
            GeneRecord(run_id=rid, gen_idx=0, pop_idx=i,
                       birth_gen=0, birth_gene=i,
                       gene_vector=np.array([float(i)]),
                       fitness=(float(i),), rank=0)
            for i in range(350)
        ]
        a.record_generation(rid, gen_idx=0, genes=genes, stats={})
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--genes", "--limit", "0", "--format", "csv"],
    )
    assert rc == 0
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert len(rows) == 350
    assert "truncated" not in captured.err


def test_limit_ignored_for_pareto_only(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only output is already bounded; --limit doesn't touch it.

    A 600-gene cross-gen view at top 3 returns at most 3*(1+M) rows
    per run. Setting --limit 1 must not truncate this further.
    """
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="p", seed=1,
            param_names=["p"], objective_labels=["oa", "ob"],
        )
        for gen_idx in range(3):
            genes = [
                GeneRecord(run_id=rid, gen_idx=gen_idx, pop_idx=i,
                           birth_gen=gen_idx, birth_gene=i,
                           gene_vector=np.array([float(10 * gen_idx + i)]),
                           fitness=(float(i), float(i) + 1.0), rank=0)
                for i in range(5)
            ]
            a.record_generation(rid, gen_idx=gen_idx, genes=genes, stats={})
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--genes", "--pareto-only",
         "--limit", "1", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    # Top 3 × (l2 + obj:oa + obj:ob) = 9 rows; --limit 1 must NOT
    # shrink this to 1.
    assert len(rows) == 9


# --- --gens-best convergence view --------------------------------------


def test_gens_best_tracks_running_minimums(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Per-generation best-ever fitness monotonically improves (or holds)."""
    from workflow_common.archive import ArchiveDB, GeneRecord

    # Fixture designed with a visible plateau:
    # gen 0:  best (0.5, 0.5)
    # gen 2:  new champion (0.2, 0.3)
    # gens 3-6: no one beats the gen-2 champion
    # gen 7:  new champion (0.15, 0.25)
    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="conv", seed=1,
            param_names=["p"], objective_labels=["a", "b"],
        )
        for gen_idx in range(10):
            if gen_idx == 2:
                fit = (0.2, 0.3)
            elif gen_idx == 7:
                fit = (0.15, 0.25)
            else:
                fit = (0.5, 0.5)
            a.record_generation(
                rid, gen_idx=gen_idx,
                genes=[GeneRecord(
                    run_id=rid, gen_idx=gen_idx, pop_idx=0,
                    birth_gen=gen_idx, birth_gene=0,
                    gene_vector=np.array([float(gen_idx)]),
                    fitness=fit, rank=0,
                )],
                stats={},
            )
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--gens-best", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 10

    # Running minimums: non-increasing from one generation to the next.
    a_best = [float(r["a_best"]) for r in rows]
    b_best = [float(r["b_best"]) for r in rows]
    for prev, cur in zip(a_best, a_best[1:]):
        assert cur <= prev + 1e-12
    for prev, cur in zip(b_best, b_best[1:]):
        assert cur <= prev + 1e-12

    # Champion-birth plateau: gens 2-6 all pinned to birth_gen 2.
    plateau = [int(r["champion_birth_gen"]) for r in rows[2:7]]
    assert plateau == [2, 2, 2, 2, 2]
    # Gen 7 shows the new champion taking over.
    assert int(rows[7]["champion_birth_gen"]) == 7


def test_gens_best_skips_nonfinite_fitness(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Failed/penalty genes must not peg the running minimums to inf."""
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="with-nan", seed=1,
            param_names=["p"], objective_labels=["a"],
        )
        a.record_generation(
            rid, gen_idx=0,
            genes=[
                GeneRecord(run_id=rid, gen_idx=0, pop_idx=0,
                           birth_gen=0, birth_gene=0,
                           gene_vector=np.array([1.0]),
                           fitness=(float("inf"),), rank=1),
                GeneRecord(run_id=rid, gen_idx=0, pop_idx=1,
                           birth_gen=0, birth_gene=1,
                           gene_vector=np.array([2.0]),
                           fitness=(0.4,), rank=0),
            ],
            stats={},
        )
        a.end_run(rid)

    rc = inspect_archive.main(
        [str(tmp_path), "--gens-best", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == 1
    # a_best must reflect the 0.4 gene, not inf.
    assert float(rows[0]["a_best"]) == 0.4
    # champion_birth_gen should also name the non-inf gene.
    assert int(rows[0]["champion_birth_gen"]) == 0


def test_gens_best_empty_run_returns_no_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """A run with zero generations errors out via _emit's empty-row path.

    Same contract as the other views: an empty result yields a
    non-zero exit code so CI scripts don't silently continue when
    their expected data isn't there.
    """
    from workflow_common.archive import ArchiveDB

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        a.start_run(
            run_id="empty", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        a.end_run("empty")

    with pytest.raises(SystemExit) as excinfo:
        inspect_archive.main([str(tmp_path), "--gens-best"])
    # _emit raises SystemExit(2) on "no rows matched the query".
    assert excinfo.value.code == 2


# --- --pareto-only implies --genes ------------------------------------


def test_pareto_only_alone_implies_genes_view(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """Typing just --pareto-only (without --genes) must produce the
    cross-generation top-N view.

    Before the fix, the view-selector fallback silently promoted
    to --gens, and --pareto-only was dropped. The output was
    per-generation summary stats — completely different from what
    the user asked for.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    # Cross-gen view has a "category" column; --gens would have
    # "gen_idx"/"recorded_at"/... and no "category".
    assert rows, "expected cross-gen rows, got none"
    assert "category" in rows[0]
    # Must contain the L2 + per-objective blocks.
    categories = {r["category"] for r in rows}
    assert "l2" in categories
    assert any(c.startswith("obj:") for c in categories)


def test_pareto_only_combined_with_gens_errors(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only together with a non-gene view is nonsense;
    the tool exits 1 with a clear message rather than silently
    dropping the flag.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--gens", "--pareto-only"],
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--pareto-only" in err
    assert "--genes" in err


def test_pareto_only_combined_with_runs_errors(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """Same rule for --runs + --pareto-only."""
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--runs", "--pareto-only"],
    )
    assert rc == 1


def test_pareto_only_combined_with_gens_best_errors(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """And --gens-best."""
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--gens-best", "--pareto-only"],
    )
    assert rc == 1


# --- l2_norm column visible in cross-gen view -------------------------


def test_cross_gen_view_includes_l2_norm_column(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """Every row in the --pareto-only output carries an l2_norm value.

    The fixture's L2 champion (gen 1 pop 0, fitness (0.10, 0.15))
    has norm sqrt(0.01 + 0.0225) == 0.18027..., which must appear
    as the first L2-category row's l2_norm value.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--genes", "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))

    # Every row has the column populated with a float.
    for r in rows:
        assert "l2_norm" in r
        val = float(r["l2_norm"])
        assert val >= 0.0  # norms are non-negative

    # The fixture's L2 champion should have the smallest norm in the
    # whole table. And the top-L2 row's norm matches the computed
    # value for (0.10, 0.15) -> 0.18027...
    l2_block = [r for r in rows if r["category"] == "l2"]
    assert l2_block, "no L2-category rows"
    top_l2 = float(l2_block[0]["l2_norm"])
    # CSV formatting uses "%g" (6 significant figures) so the
    # round-trip loses sub-microscopic precision. 1e-5 tolerance
    # easily distinguishes the champion's norm from any other
    # in the fixture (next-smallest L2 is 0.5).
    assert abs(top_l2 - float(np.hypot(0.10, 0.15))) < 1e-5


def test_l2_column_norms_sort_ascending_within_l2_block(
    cross_gen_archive: Path, capsys: pytest.CaptureFixture,
):
    """The L2 category is ranked by l2_norm; the column must reflect that.

    Scanning down the L2 block, l2_norm values should be
    non-decreasing. The column is the same quantity the ranking
    uses, so anything else would be a sign the column and the
    sort key got out of sync.
    """
    rc = inspect_archive.main(
        [str(cross_gen_archive), "--genes", "--pareto-only", "--format", "csv"],
    )
    assert rc == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    l2_norms = [
        float(r["l2_norm"]) for r in rows if r["category"] == "l2"
    ]
    assert l2_norms, "no L2-category rows"
    for a, b in zip(l2_norms, l2_norms[1:]):
        assert a <= b + 1e-12


# --- --clean-empty-runs CLI flag ---------------------------------------


def _seed_archive_with_mixed_runs(tmp_path: Path) -> tuple[str, list[str]]:
    """Build an archive with one populated + two empty-completed runs.

    Returns (full_run_id, [empty_ids]) for the tests to assert against.
    """
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    empty_ids = []
    with ArchiveDB(db) as a:
        full_rid = a.start_run(
            run_id="full-run", seed=1,
            param_names=["p"], objective_labels=["o"],
        )
        a.record_generation(
            full_rid, gen_idx=0,
            genes=[GeneRecord(
                run_id=full_rid, gen_idx=0, pop_idx=0,
                birth_gen=0, birth_gene=0,
                gene_vector=np.array([1.0]),
                fitness=(0.1,), rank=0,
            )],
            stats={},
        )
        a.end_run(full_rid)

        for i in range(2):
            rid = a.start_run(
                run_id=f"empty-{i}", seed=10 + i,
                param_names=["p"], objective_labels=["o"],
            )
            a.end_run(rid)
            empty_ids.append(rid)
    return full_rid, empty_ids


def test_clean_empty_runs_deletes_empty_and_keeps_full(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Normal mode: removes every completed-empty run, leaves populated ones."""
    full_rid, empty_ids = _seed_archive_with_mixed_runs(tmp_path)

    rc = inspect_archive.main([str(tmp_path), "--clean-empty-runs"])
    assert rc == 0
    err = capsys.readouterr().err
    # stderr reports the deletion count and names the victims.
    assert "deleted 2 empty run(s)" in err
    for rid in empty_ids:
        assert rid in err

    # Verify: only full-run remains in the archive.
    from workflow_common.archive import ArchiveDB
    with ArchiveDB(tmp_path / "archive.db", readonly=True) as a:
        survivors = [r.run_id for r in a.list_runs()]
    assert survivors == [full_rid]


def test_clean_empty_runs_dry_run_lists_without_deleting(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Dry run shows candidates on stderr but doesn't touch the DB."""
    full_rid, empty_ids = _seed_archive_with_mixed_runs(tmp_path)

    rc = inspect_archive.main(
        [str(tmp_path), "--clean-empty-runs", "--dry-run"],
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "would delete 2 empty run(s)" in err
    assert "dry run: no changes were made" in err

    # DB state unchanged.
    from workflow_common.archive import ArchiveDB
    with ArchiveDB(tmp_path / "archive.db", readonly=True) as a:
        survivors = sorted(r.run_id for r in a.list_runs())
    assert survivors == sorted([full_rid] + empty_ids)


def test_clean_empty_runs_nothing_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """Empty-run-free archive yields a helpful 'nothing to do' message."""
    from workflow_common.archive import ArchiveDB, GeneRecord

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(
            run_id="only-real", seed=0,
            param_names=["p"], objective_labels=["o"],
        )
        a.record_generation(
            rid, gen_idx=0,
            genes=[GeneRecord(
                run_id=rid, gen_idx=0, pop_idx=0,
                birth_gen=0, birth_gene=0,
                gene_vector=np.array([1.0]),
                fitness=(0.1,), rank=0,
            )],
            stats={},
        )
        a.end_run(rid)

    rc = inspect_archive.main([str(tmp_path), "--clean-empty-runs"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "no empty runs eligible" in err
    # Helper line pointing at --age-minutes 0 — discoverable UX.
    assert "--age-minutes 0" in err


def test_clean_empty_runs_respects_age_minutes(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """A young, never-ended empty run is protected unless --age-minutes 0."""
    from workflow_common.archive import ArchiveDB

    db = tmp_path / "archive.db"
    with ArchiveDB(db) as a:
        # Young empty run — not completed, just started.
        a.start_run(run_id="young-empty", seed=1,
                    param_names=["p"], objective_labels=["o"])

    # Default age gate preserves it.
    rc = inspect_archive.main([str(tmp_path), "--clean-empty-runs"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "no empty runs eligible" in err

    # --age-minutes 0 catches it.
    rc = inspect_archive.main(
        [str(tmp_path), "--clean-empty-runs", "--age-minutes", "0"],
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "deleted 1 empty run(s)" in err
    assert "young-empty" in err


def test_clean_empty_runs_rejects_combination_with_pareto_only(
    tmp_path: Path, capsys: pytest.CaptureFixture,
):
    """--pareto-only + --clean-empty-runs is nonsense; error out."""
    _seed_archive_with_mixed_runs(tmp_path)
    rc = inspect_archive.main(
        [str(tmp_path), "--clean-empty-runs", "--pareto-only"],
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--pareto-only" in err
