"""
Unit tests for :mod:`workflow_common.archive`.

What these prove
----------------
1. **Round-trip correctness** — runs, generations, genes, and case
   outputs written to the archive come back bit-identical via the
   query methods. Pickled DataFrames round-trip exactly (dtype,
   index, values).
2. **Key constraints** — duplicate run_ids raise, out-of-bound
   gene records on record_generation raise at construction,
   foreign-key cascades delete dependent rows.
3. **Resume discard** — ``discard_from_generation`` removes
   exactly the rows with ``gen_idx >= K``, cascading into genes
   and case_outputs.
4. **Read-only mode** — opening ``readonly=True`` rejects writes
   and allows queries.
5. **WAL concurrent access** — a reader opened while a writer is
   active sees committed generations without blocking the writer.
6. **Schema version check** — a DB stamped with a future schema
   version raises a clear error at open.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common import ArchiveDB, GeneRecord
from workflow_common.archive import SCHEMA_VERSION
from workflow_common.paths import CaseContext
from workflow_common.results import CaseResultSet, TabularResult


# --- Helpers ------------------------------------------------------------


def _make_case_result_set(
    ctx: CaseContext,
    *,
    avg_stress: pd.DataFrame,
    avg_def_grad: pd.DataFrame = None,
) -> CaseResultSet:
    """Synthesize a CaseResultSet for archive tests."""
    tables = {
        "avg_stress": TabularResult(
            name="avg_stress", df=avg_stress,
            source_path=Path("fake/avg_stress.txt"),
        ),
    }
    if avg_def_grad is not None:
        tables["avg_def_grad"] = TabularResult(
            name="avg_def_grad", df=avg_def_grad,
            source_path=Path("fake/avg_def_grad.txt"),
        )
    return CaseResultSet(ctx=ctx, tables=tables)


def _voce_stress_df(y0=200.0, H=1900.0, k=48.0, n=20) -> pd.DataFrame:
    t = np.linspace(0.0, 1.0, n)
    strain = t
    stress = y0 + H * (1 - np.exp(-k * strain))
    return pd.DataFrame({
        "Time": t, "Szz": stress,
        "Sxx": np.zeros_like(t), "Syy": np.zeros_like(t),
        "Sxy": np.zeros_like(t), "Syz": np.zeros_like(t),
        "Sxz": np.zeros_like(t),
    })


def _example_gene_record(run_id, *, gen_idx, pop_idx, birth_gen, birth_gene,
                         fitness=(0.5,), rank=0):
    return GeneRecord(
        run_id=run_id,
        gen_idx=gen_idx,
        pop_idx=pop_idx,
        birth_gen=birth_gen,
        birth_gene=birth_gene,
        gene_vector=np.array([1.0, 2.0, 3.0]),
        fitness=fitness,
        rank=rank,
    )


# --- Run-level round-trip ---------------------------------------------


def test_start_end_run_round_trip(tmp_path):
    """A run written + ended comes back with matching fields."""
    db_path = tmp_path / "a.db"
    with ArchiveDB(db_path) as a:
        run_id = a.start_run(
            seed=42,
            param_names=["yield_stress", "hardening"],
            objective_labels=["stress_rmse", "slope_rmse"],
            config={"n_gens": 10},
        )
        assert run_id  # non-empty
        a.end_run(run_id)

    # Re-open and query.
    with ArchiveDB(db_path, readonly=True) as a:
        run = a.get_run(run_id)
        assert run.seed == 42
        assert run.param_names == ["yield_stress", "hardening"]
        assert run.objective_labels == ["stress_rmse", "slope_rmse"]
        assert run.completed_at is not None


def test_start_run_generates_uuid_when_none():
    """No run_id -> UUID4 generated."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with ArchiveDB(Path(d) / "a.db") as a:
            rid = a.start_run(seed=0, param_names=["x"])
            # Basic UUID shape: 5 hex groups separated by dashes.
            assert rid.count("-") == 4


def test_duplicate_run_id_raises(tmp_path):
    with ArchiveDB(tmp_path / "a.db") as a:
        a.start_run(run_id="explicit", seed=0, param_names=["x"])
        with pytest.raises(sqlite3.IntegrityError):
            a.start_run(run_id="explicit", seed=1, param_names=["x"])


def test_list_runs_order(tmp_path):
    """list_runs returns runs oldest-first."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid1 = a.start_run(run_id="r1", seed=1, param_names=["x"])
        time.sleep(0.01)
        rid2 = a.start_run(run_id="r2", seed=2, param_names=["x"])
        runs = a.list_runs()
        assert [r.run_id for r in runs] == [rid1, rid2]


# --- Case-level round-trip --------------------------------------------


def test_record_and_load_case_outputs(tmp_path):
    """Pickled DataFrames round-trip bit-identical through the archive."""
    db = tmp_path / "a.db"
    ctx = CaseContext(generation=0, gene=0, obj=0)
    stress_df = _voce_stress_df()
    dg_df = pd.DataFrame({"Time": [0.0, 1.0], "F33": [1.0, 1.5]})
    rs = _make_case_result_set(ctx, avg_stress=stress_df, avg_def_grad=dg_df)

    with ArchiveDB(db) as a:
        rid = a.start_run(seed=0, param_names=["x", "y"])
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0, results=rs,
        )

    with ArchiveDB(db, readonly=True) as a:
        loaded = a.load_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
        assert loaded is not None
        pd.testing.assert_frame_equal(
            loaded.df("avg_stress"), stress_df,
        )
        pd.testing.assert_frame_equal(
            loaded.df("avg_def_grad"), dg_df,
        )


def test_load_case_outputs_missing_returns_none(tmp_path):
    """Unknown coordinates -> None (matches disk-based API)."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x"])
        got = a.load_case_outputs(
            rid, birth_gen=99, birth_gene=99, sim_case_idx=0,
        )
        assert got is None


def test_record_case_outputs_replaces_on_key_collision(tmp_path):
    """INSERT OR REPLACE: re-running a case overwrites the old blob."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x"])
        ctx = CaseContext(0, 0, 0)
        df1 = pd.DataFrame({"Time": [0.0], "Szz": [1.0]})
        df2 = pd.DataFrame({"Time": [0.0], "Szz": [99.0]})
        # Write once, then overwrite.
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            results=_make_case_result_set(ctx, avg_stress=df1),
        )
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            results=_make_case_result_set(ctx, avg_stress=df2),
        )
        loaded = a.load_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
        assert loaded.df("avg_stress")["Szz"].tolist() == [99.0]


# --- Generation-level round-trip --------------------------------------


def test_record_and_load_generation(tmp_path):
    """Genes + stats round-trip via record_generation / load_genes."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x", "y", "z"])
        genes = [
            _example_gene_record(
                rid, gen_idx=5, pop_idx=i, birth_gen=3, birth_gene=i,
                fitness=(0.1 * i, 0.2 * i),
            )
            for i in range(4)
        ]
        a.record_generation(
            rid, gen_idx=5, genes=genes, stats={"min": [0.0, 0.0]},
        )

    with ArchiveDB(tmp_path / "a.db", readonly=True) as a:
        loaded = a.load_genes(rid, gen_idx=5)
        assert len(loaded) == 4
        # Order preserved by pop_idx.
        for i, g in enumerate(loaded):
            assert g.pop_idx == i
            assert g.birth_gen == 3
            assert g.birth_gene == i
            np.testing.assert_array_equal(
                g.gene_vector, [1.0, 2.0, 3.0],
            )
            assert g.fitness == (0.1 * i, 0.2 * i)

        gens = a.list_generations(rid)
        assert len(gens) == 1
        assert gens[0].gen_idx == 5
        assert gens[0].n_pop == 4
        assert gens[0].stats == {"min": [0.0, 0.0]}


def test_record_generation_rejects_mismatched_gen_idx(tmp_path):
    """A gene record with gen_idx != outer must raise."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x"])
        bad = _example_gene_record(
            rid, gen_idx=5, pop_idx=0, birth_gen=0, birth_gene=0,
        )
        with pytest.raises(ValueError, match="gen_idx"):
            a.record_generation(rid, gen_idx=6, genes=[bad])


def test_record_generation_rejects_mismatched_run_id(tmp_path):
    with ArchiveDB(tmp_path / "a.db") as a:
        rid1 = a.start_run(run_id="r1", seed=0, param_names=["x"])
        rid2 = a.start_run(run_id="r2", seed=0, param_names=["x"])
        bad = _example_gene_record(
            rid1, gen_idx=0, pop_idx=0, birth_gen=0, birth_gene=0,
        )
        with pytest.raises(ValueError, match="run_id"):
            a.record_generation(rid2, gen_idx=0, genes=[bad])


def test_record_generation_replaces_partial_prior_write(tmp_path):
    """A re-record of the same gen_idx replaces any partial rows.

    The specific scenario: a crash mid-generation left partial gene
    rows in the archive. On resume the driver re-runs the
    generation, calling record_generation with the full pop. The
    stale partial rows must be gone afterwards.
    """
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x"])
        # Simulate a partial write: just 2 of 4 gene rows.
        partial = [
            _example_gene_record(
                rid, gen_idx=1, pop_idx=i, birth_gen=1, birth_gene=i,
            )
            for i in range(2)
        ]
        a.record_generation(rid, gen_idx=1, genes=partial)

        # Full write on retry.
        full = [
            _example_gene_record(
                rid, gen_idx=1, pop_idx=i, birth_gen=1, birth_gene=i,
            )
            for i in range(4)
        ]
        a.record_generation(rid, gen_idx=1, genes=full)

        loaded = a.load_genes(rid, gen_idx=1)
        # Exactly 4; no leftover duplicates or missing rows.
        assert [g.pop_idx for g in loaded] == [0, 1, 2, 3]


# --- Resume semantics -------------------------------------------------


def test_discard_from_generation_cascades(tmp_path):
    """discard_from_generation(K) removes gens K.. and cascades to genes+case_outputs."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=0, param_names=["x"])
        # Populate 3 generations of stuff.
        for g in range(3):
            genes = [
                _example_gene_record(
                    rid, gen_idx=g, pop_idx=0, birth_gen=g, birth_gene=0,
                ),
            ]
            a.record_generation(rid, gen_idx=g, genes=genes)
            # one case_output per gen too, keyed by birth_gen=g
            ctx = CaseContext(generation=g, gene=0, obj=0)
            a.record_case_outputs(
                rid, birth_gen=g, birth_gene=0, sim_case_idx=0,
                results=_make_case_result_set(
                    ctx, avg_stress=_voce_stress_df(y0=100 + g),
                ),
            )

        # Discard gen 1..
        a.discard_from_generation(rid, gen_idx=1)

        # Gen 0 survives, gens 1..2 gone. load_genes returns [].
        assert len(a.load_genes(rid, 0)) == 1
        assert len(a.load_genes(rid, 1)) == 0
        assert len(a.load_genes(rid, 2)) == 0
        # Cascaded case outputs: gen 0 still has its output, 1..2 don't.
        assert a.load_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        ) is not None
        assert a.load_case_outputs(
            rid, birth_gen=1, birth_gene=0, sim_case_idx=0,
        ) is None
        assert a.load_case_outputs(
            rid, birth_gen=2, birth_gene=0, sim_case_idx=0,
        ) is None

        # generations table also cleaned.
        assert [g.gen_idx for g in a.list_generations(rid)] == [0]


def test_delete_run_removes_run_and_all_dependents(tmp_path):
    """delete_run wipes the target run entirely — row + generations +
    genes + case_outputs — via schema CASCADE, leaving OTHER runs in
    the same archive untouched.

    Exercises the "clean up the residue of a bad invocation" use
    case: two runs in one archive, delete one, the other survives
    in full.
    """
    with ArchiveDB(tmp_path / "a.db") as a:
        keep_id = a.start_run(seed=1, param_names=["x"])
        drop_id = a.start_run(seed=2, param_names=["x"])
        # Two generations in each run, plus one case_output per gene,
        # so we can verify the cascade reaches every dependent table.
        for rid in (keep_id, drop_id):
            for g in range(2):
                a.record_generation(
                    rid, gen_idx=g,
                    genes=[
                        _example_gene_record(
                            rid, gen_idx=g, pop_idx=0,
                            birth_gen=g, birth_gene=0,
                        ),
                    ],
                )
                ctx = CaseContext(generation=g, gene=0, obj=0)
                a.record_case_outputs(
                    rid, birth_gen=g, birth_gene=0, sim_case_idx=0,
                    results=_make_case_result_set(
                        ctx, avg_stress=_voce_stress_df(y0=100 + g),
                    ),
                )

        # Delete one run.
        n = a.delete_run(drop_id)
        assert n == 1  # confirmation signal

        # drop_id must be gone from every table.
        assert drop_id not in [r.run_id for r in a.list_runs()]
        assert a.list_generations(drop_id) == []
        assert a.load_genes(drop_id, 0) == []
        assert a.load_case_outputs(
            drop_id, birth_gen=0, birth_gene=0, sim_case_idx=0,
        ) is None

        # keep_id must be completely unaffected.
        assert keep_id in [r.run_id for r in a.list_runs()]
        kept_gens = [g.gen_idx for g in a.list_generations(keep_id)]
        assert kept_gens == [0, 1]
        assert len(a.load_genes(keep_id, 0)) == 1
        assert len(a.load_genes(keep_id, 1)) == 1
        assert a.load_case_outputs(
            keep_id, birth_gen=0, birth_gene=0, sim_case_idx=0,
        ) is not None


def test_delete_run_raises_on_unknown_id(tmp_path):
    """Typos and already-deleted IDs get a KeyError, not a silent no-op.

    Silent-no-op is a classic trap: caller thinks they cleaned up
    and actually just misspelled the UUID.
    """
    with ArchiveDB(tmp_path / "a.db") as a:
        a.start_run(run_id="real-one", seed=0, param_names=["x"])
        with pytest.raises(KeyError, match="no run with run_id"):
            a.delete_run("not-a-real-run-id")
        # The real run is still there — we didn't partial-delete.
        assert [r.run_id for r in a.list_runs()] == ["real-one"]


def test_delete_run_rejects_on_readonly(tmp_path):
    """Read-only connections must refuse delete_run like every other mutation."""
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        a.start_run(run_id="only", seed=0, param_names=["x"])
    # Reopen read-only.
    with ArchiveDB(db, readonly=True) as a:
        with pytest.raises(RuntimeError, match="read-only"):
            a.delete_run("only")


# --- prune_empty_runs --------------------------------------------------


def test_prune_empty_runs_removes_completed_empty_runs(tmp_path):
    """Runs that called end_run but never wrote a generation are always eligible."""
    with ArchiveDB(tmp_path / "a.db") as a:
        # Three runs: the middle one is the only "real" one with
        # a generation; the other two are empty-and-completed.
        empty1 = a.start_run(seed=1, param_names=["x"])
        a.end_run(empty1)

        full = a.start_run(seed=2, param_names=["x"])
        a.record_generation(
            full, gen_idx=0,
            genes=[_example_gene_record(
                full, gen_idx=0, pop_idx=0, birth_gen=0, birth_gene=0,
            )],
        )
        a.end_run(full)

        empty2 = a.start_run(seed=3, param_names=["x"])
        a.end_run(empty2)

        deleted = a.prune_empty_runs()
        # Both empty runs, oldest first (started_at ASC).
        assert deleted == [empty1, empty2]

        # Only the one real run survives.
        survivors = [r.run_id for r in a.list_runs()]
        assert survivors == [full]


def test_prune_empty_runs_preserves_young_empty_runs_by_default(tmp_path):
    """A run that was just started and is empty must NOT be pruned —
    it might be a live run still initializing. Default age gate of
    60 minutes protects it.
    """
    with ArchiveDB(tmp_path / "a.db") as a:
        # Young empty run — not completed, just started.
        young = a.start_run(seed=1, param_names=["x"])
        # Another young run that's completed — eligible regardless of age.
        completed = a.start_run(seed=2, param_names=["x"])
        a.end_run(completed)

        deleted = a.prune_empty_runs()
        # Only the completed one got pruned; the live-looking young
        # one stays.
        assert deleted == [completed]
        survivors = [r.run_id for r in a.list_runs()]
        assert survivors == [young]


def test_prune_empty_runs_with_age_zero_prunes_young_runs_too(tmp_path):
    """--age-minutes 0 disables the young-run protection.

    For the "I know my live run isn't running" case, the user can
    override the safety default.
    """
    with ArchiveDB(tmp_path / "a.db") as a:
        young = a.start_run(seed=1, param_names=["x"])
        # Don't call end_run — simulates a crashed run that'll never
        # flip completed_at.

        deleted = a.prune_empty_runs(min_age_minutes=0.0)
        assert deleted == [young]
        assert a.list_runs() == []


def test_prune_empty_runs_dry_run_returns_list_without_deleting(tmp_path):
    """dry_run=True shows what WOULD be deleted. Nothing is actually removed."""
    with ArchiveDB(tmp_path / "a.db") as a:
        empty = a.start_run(seed=1, param_names=["x"])
        a.end_run(empty)

        preview = a.prune_empty_runs(dry_run=True)
        assert preview == [empty]
        # Run still there.
        assert [r.run_id for r in a.list_runs()] == [empty]

        # Second dry-run gives same answer (no side effects).
        assert a.prune_empty_runs(dry_run=True) == [empty]

        # Real run confirms the prediction.
        assert a.prune_empty_runs() == [empty]
        assert a.list_runs() == []


def test_prune_empty_runs_preserves_run_with_case_outputs_only(tmp_path):
    """A run with case_outputs but no generations represents a partial
    crash mid-gen-0. Those case outputs are real simulation artifacts
    — don't delete them even though ``generations`` is empty.
    """
    from workflow_common.paths import CaseContext

    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["x"])
        # Case output written, but record_generation never called.
        ctx = CaseContext(generation=0, gene=0, obj=0)
        a.record_case_outputs(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            results=_make_case_result_set(
                ctx, avg_stress=_voce_stress_df(y0=100),
            ),
        )
        a.end_run(rid)  # mark completed to bypass age gate

        deleted = a.prune_empty_runs()
        # Not pruned — the case_outputs row protects it.
        assert deleted == []
        assert [r.run_id for r in a.list_runs()] == [rid]


def test_prune_empty_runs_noop_on_empty_archive(tmp_path):
    """An archive with no runs returns an empty list, no errors."""
    with ArchiveDB(tmp_path / "a.db") as a:
        assert a.prune_empty_runs() == []
        assert a.prune_empty_runs(dry_run=True) == []


def test_prune_empty_runs_rejects_on_readonly(tmp_path):
    """Read-only connections must refuse the prune operation."""
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        rid = a.start_run(run_id="empty", seed=0, param_names=["x"])
        a.end_run(rid)
    with ArchiveDB(db, readonly=True) as a:
        with pytest.raises(RuntimeError, match="read-only"):
            a.prune_empty_runs()


# --- Read-only mode ---------------------------------------------------


def test_readonly_rejects_writes(tmp_path):
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        a.start_run(run_id="ro", seed=0, param_names=["x"])
    # Re-open read-only.
    with ArchiveDB(db, readonly=True) as a:
        # Queries work.
        assert a.get_run("ro").run_id == "ro"
        # Writes do not.
        with pytest.raises(RuntimeError, match="read-only"):
            a.start_run(run_id="ro2", seed=0, param_names=["x"])


# --- WAL concurrent reader --------------------------------------------


def test_writer_and_reader_coexist(tmp_path):
    """WAL mode: a reader opened while a writer is active sees committed data."""
    db = tmp_path / "a.db"
    with ArchiveDB(db) as writer:
        rid = writer.start_run(run_id="r", seed=0, param_names=["x"])

        # Open a read-only connection while writer is alive. Both
        # connections exist simultaneously.
        reader = ArchiveDB(db, readonly=True)
        reader.open()
        try:
            # Reader sees the existing run.
            assert reader.get_run(rid).run_id == rid

            # Writer commits more data; reader sees it on next query.
            writer.record_generation(
                rid, gen_idx=0,
                genes=[_example_gene_record(
                    rid, gen_idx=0, pop_idx=0, birth_gen=0, birth_gene=0,
                )],
            )
            # New connection-level query sees the just-committed row.
            # (SQLite snapshot-at-txn-start semantics mean the same
            # SELECT statement on a live cursor might not; a fresh
            # execute does.)
            assert len(reader.load_genes(rid, 0)) == 1
        finally:
            reader.close()


# --- Schema version ----------------------------------------------------


def test_schema_version_stamped(tmp_path):
    """On first write, the schema_meta table records the current version."""
    db = tmp_path / "a.db"
    with ArchiveDB(db) as a:
        pass  # just create
    # Direct SQL read to verify the stamp.
    conn = sqlite3.connect(db)
    v = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()
    conn.close()
    assert int(v[0]) == SCHEMA_VERSION


def test_future_schema_version_raises(tmp_path):
    """Opening a DB stamped with a newer schema raises at open()."""
    db = tmp_path / "a.db"
    # Create a normal DB first.
    with ArchiveDB(db) as a:
        pass
    # Manually bump the stamp to a future version.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE schema_meta SET value=? WHERE key='schema_version'",
        (str(SCHEMA_VERSION + 5),),
    )
    conn.commit()
    conn.close()
    # Now opening should refuse.
    with pytest.raises(RuntimeError, match="schema version"):
        with ArchiveDB(db) as a:
            pass


# --- Context-manager discipline ---------------------------------------


def test_query_without_open_raises(tmp_path):
    a = ArchiveDB(tmp_path / "a.db")  # not opened
    with pytest.raises(RuntimeError, match="not open"):
        a.list_runs()


def test_close_is_idempotent(tmp_path):
    a = ArchiveDB(tmp_path / "a.db")
    a.open()
    a.close()
    a.close()  # second close: no-op


# --- Experimental data ---------------------------------------------------


def test_record_experiment_round_trips_dataframe(tmp_path):
    """Experimental data round-trips faithfully (column names, dtypes, values)."""
    db = tmp_path / "a.db"
    df = pd.DataFrame({
        "strain": np.linspace(0.0, 0.1, 11),
        "stress": np.linspace(200.0, 350.0, 11),
        "extra_meta": ["a"] * 11,
    })
    with ArchiveDB(db) as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df, label="exp1")
    with ArchiveDB(db, readonly=True) as a:
        result = a.load_experiment(rid, 0)
    assert result is not None
    label, got = result
    assert label == "exp1"
    assert list(got.columns) == ["strain", "stress", "extra_meta"]
    pd.testing.assert_frame_equal(got, df)


def test_record_experiment_replaces_on_collision(tmp_path):
    """Re-recording the same (run_id, sim_case_idx) overwrites — supports resume."""
    df1 = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    df2 = pd.DataFrame({"strain": [0.0, 0.2], "stress": [100.0, 250.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df1, label="v1")
        a.record_experiment(rid, sim_case_idx=0, df=df2, label="v2")
        result = a.load_experiment(rid, 0)
    assert result is not None
    label, got = result
    assert label == "v2"
    pd.testing.assert_frame_equal(got, df2)


def test_load_experiment_returns_none_when_missing(tmp_path):
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        # No record_experiment call.
        assert a.load_experiment(rid, 0) is None
        assert a.load_experiment(rid, 99) is None


def test_list_experiments_returns_sim_case_index_label_pairs(tmp_path):
    """list_experiments enumerates without loading the (potentially big) DataFrames."""
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df, label="exp_quasi")
        a.record_experiment(rid, sim_case_idx=2, df=df, label="exp_dyn")
        # No data for sim_case_idx=1 — ordering should still be ascending,
        # gaps preserved.
        listing = a.list_experiments(rid)
    assert listing == [(0, "exp_quasi"), (2, "exp_dyn")]


def test_record_experiment_cascades_on_run_delete(tmp_path):
    """Deleting a run must remove its experiment rows too (FK cascade)."""
    df = pd.DataFrame({"strain": [0.0], "stress": [100.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df, label="exp")
        a.delete_run(rid)
        # Experiment row should be gone.
        cur = a._conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE run_id=?", (rid,),
        )
        assert cur.fetchone()[0] == 0


def test_record_experiment_rejects_on_readonly(tmp_path):
    db = tmp_path / "a.db"
    df = pd.DataFrame({"strain": [0.0], "stress": [100.0]})
    with ArchiveDB(db) as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
    with ArchiveDB(db, readonly=True) as a:
        with pytest.raises(RuntimeError, match="read-only"):
            a.record_experiment(rid, sim_case_idx=0, df=df, label="exp")


# --- Backward compat for archives lacking the experiments table ---------


def _build_pre_experiments_archive(db_path):
    """Build an archive with the pre-experiments-table schema.

    Used to exercise the load_experiment / list_experiments
    backward-compat path: those methods must return None / [] when
    the table is missing rather than letting sqlite3.OperationalError
    bubble up.
    """
    import sqlite3
    con = sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT
    );
    CREATE TABLE generations (
        run_id TEXT NOT NULL, gen_idx INTEGER NOT NULL,
        recorded_at TEXT NOT NULL, n_pop INTEGER NOT NULL,
        stats_json TEXT, PRIMARY KEY (run_id, gen_idx)
    );
    CREATE TABLE genes (
        run_id TEXT NOT NULL, gen_idx INTEGER NOT NULL,
        pop_idx INTEGER NOT NULL, birth_gen INTEGER NOT NULL,
        birth_gene INTEGER NOT NULL, gene_vector TEXT NOT NULL,
        fitness TEXT NOT NULL, rank INTEGER,
        PRIMARY KEY (run_id, gen_idx, pop_idx)
    );
    CREATE TABLE case_outputs (
        run_id TEXT NOT NULL, birth_gen INTEGER NOT NULL,
        birth_gene INTEGER NOT NULL, sim_case_idx INTEGER NOT NULL,
        output_name TEXT NOT NULL, data_blob BLOB NOT NULL,
        PRIMARY KEY (run_id, birth_gen, birth_gene, sim_case_idx, output_name)
    );
    INSERT INTO runs VALUES ('r1', '2025-01-01T00:00:00', NULL, 1,
        '["p"]', '["o"]', NULL);
    """)
    con.commit()
    con.close()


def test_load_experiment_returns_none_when_table_missing(tmp_path):
    """An archive written before the experiments table existed must
    still be readable. load_experiment returns None instead of
    raising sqlite3.OperationalError.

    This is the bug Robert reported: a real-world archive on his
    machine pre-dated this table, and the plotter crashed with
    "no such table: experiments" before falling back to the user's
    --experimental CSVs.
    """
    db = tmp_path / "old.db"
    _build_pre_experiments_archive(db)
    with ArchiveDB(db, readonly=True) as a:
        # Sanity: the table genuinely isn't there.
        cur = a._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
        )
        assert cur.fetchone() is None
        # The contract: returns None, doesn't raise.
        assert a.load_experiment("r1", 0) is None
        assert a.load_experiment("r1", 99) is None


def test_list_experiments_returns_empty_when_table_missing(tmp_path):
    """Same backward-compat as load_experiment, but for the
    enumerate path used by tools that ask "what experiments exist?"
    before deciding whether to fetch any.
    """
    db = tmp_path / "old.db"
    _build_pre_experiments_archive(db)
    with ArchiveDB(db, readonly=True) as a:
        assert a.list_experiments("r1") == []


def test_writable_open_creates_missing_experiments_table(tmp_path):
    """Opening a pre-experiments archive in WRITE mode must create
    the table on the fly (via ``IF NOT EXISTS`` in the schema script)
    so subsequent writes work without manual migration. Read-only
    opens skip this and stay backward-compat via the load path's
    None return.
    """
    db = tmp_path / "old.db"
    _build_pre_experiments_archive(db)
    df = pd.DataFrame({"strain": [0.0, 0.5], "stress": [100.0, 200.0]})
    # Writable open — schema script runs, experiments table created.
    with ArchiveDB(db) as a:
        a.record_experiment("r1", sim_case_idx=0, df=df, label="x")
        result = a.load_experiment("r1", 0)
    assert result is not None
    label, got = result
    assert label == "x"
    pd.testing.assert_frame_equal(got, df)


# --- minmax_strain on experiments ---------------------------------------


def test_record_experiment_stores_minmax_strain(tmp_path):
    """record_experiment must persist the (lo, hi) window so plotters
    can shade the optimized region against the full curve."""
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df, label="exp",
                            minmax_strain=(0.005, 0.13))
        win = a.load_experiment_window(rid, 0)
    assert win == (0.005, 0.13)


def test_record_experiment_window_supports_unbounded_sides(tmp_path):
    """(None, hi) and (lo, None) round-trip — either side is optional."""
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df,
                            minmax_strain=(None, 0.13))
        a.record_experiment(rid, sim_case_idx=1, df=df,
                            minmax_strain=(0.005, None))
        a.record_experiment(rid, sim_case_idx=2, df=df,
                            minmax_strain=None)
        assert a.load_experiment_window(rid, 0) == (None, 0.13)
        assert a.load_experiment_window(rid, 1) == (0.005, None)
        # No window stored at all → None (NOT (None, None) — that
        # sentinel would mean "stored but unbounded both sides,"
        # which the user didn't say).
        assert a.load_experiment_window(rid, 2) is None


def test_load_experiment_window_returns_none_for_missing_record(tmp_path):
    """Querying a (run_id, sim_case_idx) that was never recorded returns
    None, not an exception. Plotter relies on this for graceful skip."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        assert a.load_experiment_window(rid, 99) is None


def test_load_experiment_window_returns_none_on_archive_without_column(tmp_path):
    """An archive written before the minmax_strain column existed
    should stay readable read-only — the load returns None and the
    plotter falls through to "no shading" cleanly.
    """
    import sqlite3, pickle
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT);
    CREATE TABLE experiments(
        run_id TEXT NOT NULL, sim_case_idx INTEGER NOT NULL,
        label TEXT, data_blob BLOB NOT NULL,
        PRIMARY KEY (run_id, sim_case_idx));
    INSERT INTO runs VALUES('r1', '2025', NULL, 1, '["p"]', '["o"]', NULL);
    """)
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    con.execute("INSERT INTO experiments VALUES('r1', 0, 'old', ?)",
                (pickle.dumps(df),))
    con.commit(); con.close()

    with ArchiveDB(db, readonly=True) as a:
        # The DataFrame still loads — only the new column is missing.
        assert a.load_experiment("r1", 0)[0] == "old"
        # And the window query degrades to None.
        assert a.load_experiment_window("r1", 0) is None


def test_writable_open_migrates_minmax_strain_column(tmp_path):
    """Opening an old archive in writable mode auto-adds the
    minmax_strain column so subsequent record_experiment calls
    can persist windows. No manual migration step required."""
    import sqlite3, pickle
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT);
    CREATE TABLE experiments(
        run_id TEXT NOT NULL, sim_case_idx INTEGER NOT NULL,
        label TEXT, data_blob BLOB NOT NULL,
        PRIMARY KEY (run_id, sim_case_idx));
    INSERT INTO runs VALUES('r1', '2025', NULL, 1, '["p"]', '["o"]', NULL);
    """)
    con.commit(); con.close()

    # Writable open triggers _migrate_add_missing_columns.
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    with ArchiveDB(db) as a:
        # Column is present — record_experiment with minmax_strain
        # works without raising.
        a.record_experiment("r1", sim_case_idx=0, df=df, label="new",
                            minmax_strain=(0.005, 0.10))
        assert a.load_experiment_window("r1", 0) == (0.005, 0.10)

    # Verify by hand: the column genuinely exists in the file.
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute(
        "PRAGMA table_info(experiments)"
    ).fetchall()]
    con.close()
    assert "minmax_strain" in cols


def test_migration_idempotent_on_fresh_archive(tmp_path):
    """Running the migration on a brand-new archive (where the column
    was already created by the schema script) must be a no-op rather
    than raising "duplicate column name."""
    db = tmp_path / "new.db"
    # First open creates fresh schema with the column already present.
    with ArchiveDB(db):
        pass
    # Second open re-runs the migration step. Must not raise.
    with ArchiveDB(db):
        pass
    import sqlite3
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute(
        "PRAGMA table_info(experiments)"
    ).fetchall()]
    con.close()
    # Exactly one minmax_strain column, no duplicates.
    assert cols.count("minmax_strain") == 1


# --- extractor_config on experiments -----------------------------------


def test_record_experiment_persists_extractor_config(tmp_path):
    """Round-trip a full extractor config through record/load."""
    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    cfg = {
        "stress_output": "avg_stress",
        "stress_column": "Szz",
        "strain_source": "time_rate",
        "strain_rate": 1e-3,
        "strain_source_output": "avg_def_grad",
        "strain_source_column": "F33",
        "time_column": "Time",
        "window": [0.005, 0.13],
    }
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_experiment(rid, sim_case_idx=0, df=df,
                            extractor_config=cfg)
        loaded = a.load_extractor_config(rid, 0)
    assert loaded == cfg


def test_load_extractor_config_returns_none_when_no_record(tmp_path):
    """Querying an unrecorded (run, sim_case) returns None, not raise."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        assert a.load_extractor_config(rid, 99) is None


def test_load_extractor_config_returns_none_when_record_has_null_config(tmp_path):
    """A recorded experiment with extractor_config=None reads back as None."""
    df = pd.DataFrame({"strain": [0.0], "stress": [0.0]})
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        # No extractor_config supplied.
        a.record_experiment(rid, sim_case_idx=0, df=df, label="bare")
        assert a.load_extractor_config(rid, 0) is None


def test_load_extractor_config_returns_none_when_column_absent(tmp_path):
    """Old archive without the extractor_config column → None on read,
    no crash. Read-only opens skip the migration so the column stays
    absent."""
    import sqlite3, pickle
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT);
    CREATE TABLE experiments(
        run_id TEXT NOT NULL, sim_case_idx INTEGER NOT NULL,
        label TEXT, data_blob BLOB NOT NULL, minmax_strain TEXT,
        PRIMARY KEY (run_id, sim_case_idx));
    INSERT INTO runs VALUES('r1', '2025', NULL, 1, '["p"]', '["o"]', NULL);
    """)
    df = pd.DataFrame({"strain": [0.0], "stress": [0.0]})
    con.execute("INSERT INTO experiments VALUES('r1', 0, 'old', ?, NULL)",
                (pickle.dumps(df),))
    con.commit(); con.close()

    with ArchiveDB(db, readonly=True) as a:
        # The earlier-introduced fields still load.
        assert a.load_experiment("r1", 0)[0] == "old"
        # And the new column degrades to None gracefully.
        assert a.load_extractor_config("r1", 0) is None


def test_writable_open_migrates_extractor_config_column(tmp_path):
    """Opening an old archive in writable mode auto-adds the new
    column (via _migrate_add_missing_columns), and subsequent
    record_experiment with extractor_config persists correctly.
    """
    import sqlite3, pickle
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT);
    CREATE TABLE experiments(
        run_id TEXT NOT NULL, sim_case_idx INTEGER NOT NULL,
        label TEXT, data_blob BLOB NOT NULL,
        PRIMARY KEY (run_id, sim_case_idx));
    INSERT INTO runs VALUES('r1', '2025', NULL, 1, '["p"]', '["o"]', NULL);
    """)
    con.commit(); con.close()

    df = pd.DataFrame({"strain": [0.0, 0.1], "stress": [100.0, 200.0]})
    cfg = {"strain_source": "time_rate", "strain_rate": 1e-3}
    with ArchiveDB(db) as a:
        a.record_experiment("r1", sim_case_idx=0, df=df,
                            extractor_config=cfg)
        loaded = a.load_extractor_config("r1", 0)
    # Round-trip should match — the from_dict tolerance is for the
    # plotter; the archive stores verbatim what was given.
    assert loaded == cfg

    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute(
        "PRAGMA table_info(experiments)"
    ).fetchall()]
    con.close()
    assert "extractor_config" in cols


# --- case_curves: extracted (independent, dependent) curves --------------


def test_record_case_curve_round_trips_arrays(tmp_path):
    """Stored independent/dependent arrays load back identically."""
    db = tmp_path / "a.db"
    independent = np.linspace(0.0, 0.1, 50)
    dependent = 200.0 + 1900.0 * (1 - np.exp(-48.0 * independent))
    with ArchiveDB(db) as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            independent=independent, dependent=dependent,
            independent_label="strain", dependent_label="stress",
        )
    with ArchiveDB(db, readonly=True) as a:
        result = a.load_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
    assert result is not None
    ind, dep, ind_label, dep_label = result
    np.testing.assert_array_equal(ind, independent)
    np.testing.assert_array_equal(dep, dependent)
    assert ind_label == "strain"
    assert dep_label == "stress"


def test_record_case_curve_validates_shapes(tmp_path):
    """Mismatched lengths or non-1-D arrays raise."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        with pytest.raises(ValueError, match="same length"):
            a.record_case_curve(
                rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
                independent=np.array([0.0, 0.1]),
                dependent=np.array([1.0, 2.0, 3.0]),
            )
        with pytest.raises(ValueError, match="1-D"):
            a.record_case_curve(
                rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
                independent=np.zeros((3, 2)),
                dependent=np.zeros((3, 2)),
            )


def test_load_case_curve_returns_none_for_missing(tmp_path):
    """Unrecorded coordinates return None, not raise."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        assert a.load_case_curve(
            rid, birth_gen=99, birth_gene=99, sim_case_idx=99,
        ) is None


def test_load_case_curve_returns_none_when_table_missing(tmp_path):
    """Old archive without case_curves table reads as None."""
    import sqlite3
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
    CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    INSERT INTO schema_meta VALUES('schema_version', '1');
    CREATE TABLE runs(run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
        completed_at TEXT, seed INTEGER, param_names TEXT NOT NULL,
        objective_labels TEXT, config_json TEXT);
    INSERT INTO runs VALUES('r1', '2025', NULL, 1, '["p"]', '["o"]', NULL);
    """)
    con.commit(); con.close()
    with ArchiveDB(db, readonly=True) as a:
        assert a.load_case_curve(
            "r1", birth_gen=0, birth_gene=0, sim_case_idx=0,
        ) is None


def test_record_case_curve_replaces_on_collision(tmp_path):
    """Re-recording the same coords overwrites — supports retries."""
    a1 = np.array([0.0, 0.1])
    b1 = np.array([100.0, 200.0])
    a2 = np.array([0.0, 0.05, 0.1])
    b2 = np.array([100.0, 150.0, 200.0])
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_case_curve(rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
                            independent=a1, dependent=b1)
        a.record_case_curve(rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
                            independent=a2, dependent=b2)
        ind, dep, _, _ = a.load_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
    np.testing.assert_array_equal(ind, a2)
    np.testing.assert_array_equal(dep, b2)


def test_case_curves_cascade_on_run_delete(tmp_path):
    """delete_run propagates to case_curves via the FK cascade."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        a.record_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
            independent=np.array([0.0, 0.1]),
            dependent=np.array([100.0, 200.0]),
        )
        a.delete_run(rid)
        cur = a._conn.execute(
            "SELECT COUNT(*) FROM case_curves WHERE run_id=?", (rid,),
        )
        assert cur.fetchone()[0] == 0


def test_discard_from_generation_drops_case_curves(tmp_path):
    """discard_from_generation must delete case_curves at gen >= K."""
    with ArchiveDB(tmp_path / "a.db") as a:
        rid = a.start_run(seed=1, param_names=["p"], objective_labels=["o"])
        # Need a generation row first so the FK doesn't get angry.
        a.record_generation(rid, gen_idx=0, genes=[], stats={})
        a.record_generation(rid, gen_idx=1, genes=[], stats={})
        for g in (0, 1):
            a.record_case_curve(
                rid, birth_gen=g, birth_gene=0, sim_case_idx=0,
                independent=np.array([0.0, 0.1]),
                dependent=np.array([100.0, 200.0]),
            )
        a.discard_from_generation(rid, gen_idx=1)
        # gen 0 survives, gen 1 is gone.
        assert a.load_case_curve(
            rid, birth_gen=0, birth_gene=0, sim_case_idx=0,
        ) is not None
        assert a.load_case_curve(
            rid, birth_gen=1, birth_gene=0, sim_case_idx=0,
        ) is None
