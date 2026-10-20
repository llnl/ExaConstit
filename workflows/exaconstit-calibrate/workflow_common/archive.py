"""
SQLite-backed archive for optimization-run data.

Why this module exists
----------------------
A realistic crystal-plasticity optimization produces an enormous
filesystem footprint. With 100 generations × 300 genes × 4
experiments × (several output files + input files + logs per case),
a single run leaves 500k–2M small files behind. Networked
filesystems (Lustre, GPFS, NFS) struggle with that: ``ls`` becomes
slow, ``rm -rf`` takes hours, and inode quotas can be exceeded
before the optimization converges.

The pre-refactor workaround was to stash the stress histories on
DEAP Individuals so the pickle checkpoint carried everything needed
for post-processing. That works but (a) bloats the checkpoint file,
and (b) couples post-processing to DEAP.

This module takes a different approach: after each case completes,
pull its simulation output into a SQLite database; periodically
delete the now-archived case directories from disk. Post-processing
reads from the database. The filesystem holds at most two
generations' worth of case directories at any moment (the current
one and the previous one, for crash safety).

What is stored
--------------
For one run, the archive stores:

* **Run metadata**: run_id, seed, param_names, objective_labels,
  start and end timestamps, config JSON for reproducibility.
* **Per-generation records**: stats dict (logbook compile output),
  number of individuals, timestamp.
* **Per-gene records** (one row per Individual per generation in the
  pop_library): gene vector, fitness tuple, rank, birth generation
  and birth offspring index (for case lookup).
* **Case outputs**: pickled ``pandas.DataFrame`` per simulation
  output per case. Keyed by
  ``(run_id, birth_gen, birth_gene, sim_case_idx, output_name)``.

Why SQLite
----------
It ships with Python. A single file is trivially archivable,
transferable, and versionable. WAL journaling mode lets a
post-processor read the archive while the driver is still writing.
Primary keys + foreign keys catch ID mismatches early.

Why pickle DataFrames in BLOB columns
-------------------------------------
We considered three alternatives:

* **Parquet** — compact and queryable outside Python, but adds a
  pyarrow dependency the rest of the framework doesn't need.
* **CSV in TEXT** — universally readable, but slower, larger, and
  loses column dtypes.
* **JSON in TEXT** — verbose for large arrays, and no native
  numpy-array support.

Pickle is stdlib, fastest, smallest, and round-trips DataFrames
bit-for-bit. The archive is user-owned data (not adversarial input),
so the pickle deserialization concern does not apply here.

Concurrency
-----------
A driver process is the single writer. Post-processors and external
tools can open the archive read-only (``ArchiveDB(path,
readonly=True)``) at the same time. WAL mode allows this without
read/write contention.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .paths import CaseContext
from .results import CaseResultSet, TabularResult

logger = get_logger(__name__)


# --- Schema -------------------------------------------------------------

# Bump SCHEMA_VERSION whenever columns change in a backward-incompatible
# way. The archive records it on first open and refuses to read DBs
# written by a newer version of this code than it knows about.
SCHEMA_VERSION = 1


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    seed INTEGER,
    param_names TEXT NOT NULL,
    objective_labels TEXT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    run_id TEXT NOT NULL,
    gen_idx INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    n_pop INTEGER NOT NULL,
    stats_json TEXT,
    PRIMARY KEY (run_id, gen_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS genes (
    run_id TEXT NOT NULL,
    gen_idx INTEGER NOT NULL,
    pop_idx INTEGER NOT NULL,
    birth_gen INTEGER NOT NULL,
    birth_gene INTEGER NOT NULL,
    gene_vector TEXT NOT NULL,
    fitness TEXT NOT NULL,
    rank INTEGER,
    PRIMARY KEY (run_id, gen_idx, pop_idx),
    FOREIGN KEY (run_id, gen_idx) REFERENCES generations(run_id, gen_idx)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS case_outputs (
    run_id TEXT NOT NULL,
    birth_gen INTEGER NOT NULL,
    birth_gene INTEGER NOT NULL,
    sim_case_idx INTEGER NOT NULL,
    output_name TEXT NOT NULL,
    data_blob BLOB NOT NULL,
    PRIMARY KEY (run_id, birth_gen, birth_gene, sim_case_idx, output_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Experimental reference data per SimCase. Stored as a pickled
-- pandas DataFrame in ``data_blob``. This is what the optimizer
-- compared simulated curves against; saving it here means
-- post-run plotting tools never have to ask the user "which CSV
-- corresponds to which SimCase?" — the archive carries the
-- mapping. ``label`` is the SimCase's human-readable label as
-- supplied by the driver, useful for chart titles. ``minmax_strain``
-- is JSON-encoded ``[lo, hi]`` (or NULL) capturing the strain
-- window used during optimization, so plotting tools can shade
-- the optimized region against the full curve. ``extractor_config``
-- is JSON for the StressStrainExtractor that the optimizer used,
-- so plotting tools can reconstruct (strain, stress) curves
-- matching exactly what the optimizer scored against.
CREATE TABLE IF NOT EXISTS experiments (
    run_id TEXT NOT NULL,
    sim_case_idx INTEGER NOT NULL,
    label TEXT,
    data_blob BLOB NOT NULL,
    minmax_strain TEXT,
    extractor_config TEXT,
    PRIMARY KEY (run_id, sim_case_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Extracted (independent, dependent) curves per simulation case. This
-- is the EXACT pair the objective evaluator scored against the
-- experimental reference, but saved over the FULL extraction range
-- (window stripped before extraction) so users can re-run analyses
-- with different windows / metrics / weighting without re-extracting
-- from raw case_outputs.
--
-- Why store this separately from case_outputs: case_outputs holds
-- raw simulator output tables (avg_stress.txt, avg_def_grad.txt,
-- etc.) which are an input to extraction. The extracted curve is
-- the comparison target the optimizer actually used. Persisting both
-- means: (1) post-run plotters get curves that match the run
-- exactly, (2) re-analysis with a different metric is one query,
-- not a re-run of the extraction pipeline, (3) Bayesian surrogates
-- and other downstream models can train on (gene → curve) pairs
-- directly.
--
-- ``independent_label`` / ``dependent_label`` keep this generic for
-- future extractor types: a thermal-expansion fit might store
-- (temperature, strain), a creep test (time, strain). The
-- StressStrainExtractor's defaults are 'strain' / 'stress'.
CREATE TABLE IF NOT EXISTS case_curves (
    run_id TEXT NOT NULL,
    birth_gen INTEGER NOT NULL,
    birth_gene INTEGER NOT NULL,
    sim_case_idx INTEGER NOT NULL,
    independent_blob BLOB NOT NULL,
    dependent_blob BLOB NOT NULL,
    independent_label TEXT,
    dependent_label TEXT,
    PRIMARY KEY (run_id, birth_gen, birth_gene, sim_case_idx),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_case_curves_by_gen
    ON case_curves(run_id, birth_gen);

CREATE INDEX IF NOT EXISTS idx_case_outputs_by_gen
    ON case_outputs(run_id, birth_gen);
CREATE INDEX IF NOT EXISTS idx_genes_by_gen
    ON genes(run_id, gen_idx);
"""


# --- Record dataclasses ------------------------------------------------


@dataclass
class RunSummary:
    """Top-level metadata for one run.

    Fields:
        run_id: Unique identifier, typically a UUID. Assigned at
            :meth:`ArchiveDB.start_run` time.
        started_at: ISO-8601 UTC timestamp of run start.
        completed_at: ISO-8601 UTC timestamp of run end. ``None``
            if the run never reached :meth:`ArchiveDB.end_run`
            (killed mid-run, for example).
        seed: The GA seed used. Useful for determinism audits.
        param_names: Names of the optimized parameters, in the same
            order as gene vectors in this run's records.
        objective_labels: Labels for the M objectives, in the same
            order as fitness tuples in this run's records. Default
            is ``["obj0", "obj1", ...]`` if none were supplied at
            run start.
        n_generations: Number of generations archived so far (may
            grow during a live run).
    """

    run_id: str
    started_at: str
    completed_at: Optional[str]
    seed: Optional[int]
    param_names: List[str]
    objective_labels: List[str]
    n_generations: int


@dataclass
class GenerationSummary:
    """Summary of one generation in the archive.

    Fields:
        run_id: The owning run's ID.
        gen_idx: Pop-library index (0-based).
        recorded_at: ISO-8601 UTC timestamp of the ``record_generation``
            call.
        n_pop: Population size for this generation.
        stats: Arbitrary JSON-serializable stats dict, typically
            the DEAP ``Statistics.compile`` output.
    """

    run_id: str
    gen_idx: int
    recorded_at: str
    n_pop: int
    stats: Dict[str, Any]


@dataclass
class GeneRecord:
    """One gene's archived record.

    Fields:
        run_id: The owning run.
        gen_idx: Position in the pop_library (generation index).
        pop_idx: Position within that generation's population.
        birth_gen: Generation where this individual was evaluated.
            If selection copied an old individual into a later
            generation, this stays at the original birth gen.
            Used for on-disk / archive case-output lookup.
        birth_gene: Offspring index at birth. Combined with
            ``birth_gen`` uniquely identifies the case directory
            that was written for this gene.
        gene_vector: Parameter values (1-D numpy array).
        fitness: Tuple of objective values. Length equals the
            run's number of objectives.
        rank: DEAP non-domination rank (0 = Pareto front).
            ``None`` if not assigned.
    """

    run_id: str
    gen_idx: int
    pop_idx: int
    birth_gen: int
    birth_gene: int
    gene_vector: np.ndarray
    fitness: Tuple[float, ...]
    rank: Optional[int]


# --- The archive -------------------------------------------------------


class ArchiveDB:
    """SQLite-backed archive of optimization-run data.

    Usage is context-manager style or manual ``open()`` / ``close()``.
    All writes are committed per-call (implicit transactions) because
    typical writes are tiny (one case, one generation). Batching can
    be added later if write contention ever becomes a bottleneck.

    Args:
        path: Filesystem path to the SQLite file. Created if missing
            (unless ``readonly=True``). Parent directory created if
            needed.
        readonly: Open the database in read-only mode using SQLite's
            ``file:PATH?mode=ro`` URI form. Default False. Set True
            for post-processing tools that should not accidentally
            mutate the archive.

    Example:
        Writer-side during a driver run::

            archive = ArchiveDB("opt.db")
            with archive:
                run_id = archive.start_run(
                    seed=42,
                    param_names=["yield_stress", "hardening"],
                    objective_labels=["stress_rmse"],
                )
                # ... driver loop ...
                archive.record_case_outputs(
                    run_id, birth_gen=g, birth_gene=i, sim_case_idx=0,
                    results=case_result_set,
                )
                archive.record_generation(run_id, gen_idx=g, genes=...,
                                         stats={"min": 0.42})
                archive.end_run(run_id)

        Reader-side during post-processing::

            with ArchiveDB("opt.db", readonly=True) as archive:
                runs = archive.list_runs()
                genes = archive.load_genes(runs[-1].run_id, gen_idx=0)
                outputs = archive.load_case_outputs(
                    runs[-1].run_id,
                    birth_gen=0, birth_gene=0, sim_case_idx=0,
                )
    """

    def __init__(self, path: Union[str, Path], *, readonly: bool = False):
        self.path = Path(path)
        self._readonly = readonly
        self._conn: Optional[sqlite3.Connection] = None

    def open(self) -> sqlite3.Connection:
        """Open (and if writable, initialize) the database.

        Idempotent — calling twice is a no-op after the first time.
        Returns the underlying sqlite3 connection in case the caller
        wants to run a custom query.
        """
        if self._conn is not None:
            return self._conn

        if self._readonly:
            # URI form is the only way to force read-only in sqlite3.
            uri = f"file:{self.path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
        else:
            # Create parent dir once, at the first open. If the user
            # passed a nested path that doesn't exist yet, this DTRT.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            # WAL mode lets readers coexist with the single writer.
            # Must be set on a writable connection before any writes.
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA_SQL)
            # Schema migrations for archives written by older
            # versions of this code that pre-date column additions.
            # Each migration here is idempotent and safe on a fresh
            # DB (it'll be a no-op because the column already exists
            # from the executescript above).
            self._migrate_add_missing_columns()
            # Stamp the schema version the first time we touch this
            # file. INSERT OR IGNORE so reopens don't overwrite.
            self._conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self._conn.commit()

        self._check_schema_version()
        # Row access by name is more readable than by index, and
        # doesn't cost anything meaningful for our query volumes.
        self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close the connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _check_schema_version(self) -> None:
        """Fail loudly if the DB was written by a newer schema.

        Defensive: if a future version of this library bumps
        SCHEMA_VERSION and a user opens that DB with an older
        version, we want a clear error rather than silent data
        corruption from missing columns.
        """
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key=?",
            ("schema_version",),
        )
        row = cur.fetchone()
        if row is None:
            # A truly empty DB (readonly=True pointing at a non-
            # existent file, say). Nothing to check.
            return
        stored = int(row[0])
        if stored > SCHEMA_VERSION:
            raise RuntimeError(
                f"archive at {self.path} has schema version {stored}; "
                f"this code knows version {SCHEMA_VERSION}. Upgrade "
                f"workflow_common to read this archive."
            )

    # --- Run-level -----------------------------------------------------

    def start_run(
        self,
        *,
        run_id: Optional[str] = None,
        seed: Optional[int] = None,
        param_names: Sequence[str],
        objective_labels: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new run row and return its ID.

        Called once at the top of a driver invocation. On resume
        from a pickle checkpoint, pass the previous ``run_id`` to
        continue writing to the same archive entry — :meth:`discard_from_generation`
        handles cleanup of half-written generations.

        Args:
            run_id: Optional explicit ID. ``None`` generates a UUID4.
            seed: GA seed. Archived for reproducibility audits;
                ``None`` if unseeded.
            param_names: Ordered list of parameter names. Must
                match the gene vector layout used in subsequent
                ``record_generation`` calls.
            objective_labels: Ordered list of objective labels. If
                ``None``, defaults are generated at query time.
            config: Optional JSON-serializable dict with the full
                run configuration. ``None`` is fine; storing it
                makes the archive self-describing.

        Returns:
            The run_id (generated or echoed).

        Raises:
            RuntimeError: If the archive is read-only.
            sqlite3.IntegrityError: If run_id already exists.
        """
        self._require_writable()
        if run_id is None:
            run_id = str(uuid.uuid4())
        now = _utc_now_iso()
        param_names_list = list(param_names)
        labels_list = list(objective_labels) if objective_labels else None
        config_json = json.dumps(config, default=_json_default) if config else None

        self._conn.execute(
            "INSERT INTO runs("
            "    run_id, started_at, completed_at, seed, "
            "    param_names, objective_labels, config_json"
            ") VALUES (?, ?, NULL, ?, ?, ?, ?)",
            (
                run_id,
                now,
                seed,
                json.dumps(param_names_list),
                json.dumps(labels_list) if labels_list else None,
                config_json,
            ),
        )
        self._conn.commit()
        logger.info("archive: started run %s", run_id)
        return run_id

    def end_run(self, run_id: str) -> None:
        """Record the completion timestamp for a run.

        Call at the end of a successful driver invocation. Not
        strictly required — unclosed runs are still fully queryable
        — but the ``completed_at`` column is useful as an "is this
        run finished" signal when iterating over many archives.
        """
        self._require_writable()
        now = _utc_now_iso()
        self._conn.execute(
            "UPDATE runs SET completed_at=? WHERE run_id=?",
            (now, run_id),
        )
        self._conn.commit()

    def discard_from_generation(self, run_id: str, gen_idx: int) -> None:
        """Delete every record with ``gen_idx >= gen_idx`` for this run.

        Called on resume from a pickle checkpoint: the pickle
        captures the state after generation ``K`` completed, so on
        resume we drop any archive rows for generations > K (which
        would be stale half-writes from the previous crash).

        Foreign-key cascade deletes the dependent genes and
        case_outputs automatically.

        Args:
            run_id: The run being resumed.
            gen_idx: The generation index to discard FROM (inclusive).
        """
        self._require_writable()
        self._conn.execute(
            "DELETE FROM case_outputs WHERE run_id=? AND birth_gen >= ?",
            (run_id, gen_idx),
        )
        # case_curves was added in a later schema revision; delete
        # only if the table is present (writable open with the
        # migration step ensures this on existing archives, but
        # belt-and-suspenders).
        if self._has_table("case_curves"):
            self._conn.execute(
                "DELETE FROM case_curves WHERE run_id=? AND birth_gen >= ?",
                (run_id, gen_idx),
            )
        self._conn.execute(
            "DELETE FROM genes WHERE run_id=? AND gen_idx >= ?",
            (run_id, gen_idx),
        )
        self._conn.execute(
            "DELETE FROM generations WHERE run_id=? AND gen_idx >= ?",
            (run_id, gen_idx),
        )
        self._conn.commit()
        logger.info(
            "archive: discarded run %s records from gen %d onward",
            run_id, gen_idx,
        )

    def delete_run(self, run_id: str) -> int:
        """Delete a run and every dependent record from the archive.

        Cascades through ``generations``, ``genes``, and
        ``case_outputs`` via the schema's ``ON DELETE CASCADE``
        foreign keys (which require ``PRAGMA foreign_keys=ON``,
        set at open time).

        Useful for pruning the residue of failed invocations —
        e.g. an aborted resume that accidentally called
        ``start_run`` and left an empty row behind. Refuses to
        silently no-op on unknown IDs so a typo gets surfaced
        rather than giving the caller a false sense of cleanup.

        Args:
            run_id: The run to delete.

        Returns:
            Number of parent ``runs`` rows deleted — should always
            be 1 on success. Exposed as a cheap confirmation
            signal.

        Raises:
            KeyError: if ``run_id`` is not in the archive.
        """
        self._require_writable()
        # Confirm existence first so the caller gets a clear error
        # for typos rather than a silent 0-row delete.
        cur = self._conn.execute(
            "SELECT 1 FROM runs WHERE run_id=?", (run_id,),
        )
        if cur.fetchone() is None:
            raise KeyError(
                f"delete_run: no run with run_id={run_id!r} in the "
                f"archive. Run ``list_runs()`` to see available IDs."
            )
        cur = self._conn.execute(
            "DELETE FROM runs WHERE run_id=?", (run_id,),
        )
        self._conn.commit()
        logger.info(
            "archive: deleted run %s (cascaded through generations, "
            "genes, case_outputs)", run_id,
        )
        return cur.rowcount

    def prune_empty_runs(
        self,
        *,
        min_age_minutes: float = 60.0,
        dry_run: bool = False,
    ) -> List[str]:
        """Delete runs that have no generations and no case_outputs.

        "Empty" here means truly empty — no ``generations`` rows AND
        no ``case_outputs`` rows for the run. A crash mid-way
        through gen 0 might leave case outputs without a matching
        generation row; those outputs represent real simulation
        work and shouldn't be pruned, so both tables have to be
        clean for a run to qualify.

        Two guards against deleting live work:

        * ``min_age_minutes`` filters by ``started_at``. Runs
          younger than this stay untouched. Default 60 minutes —
          any real run will have filled in its first generation
          within that window, so anything older-and-still-empty
          is dead. Pass ``0.0`` to prune regardless of age.
        * A run that has already called ``end_run`` (``completed_at
          IS NOT NULL``) is always eligible regardless of age —
          if it finished empty, it's by definition not live.

        Args:
            min_age_minutes: Runs started within this many minutes
                are preserved even if empty. Ignored for runs that
                have been explicitly ended.
            dry_run: If True, return the list of run_ids that WOULD
                be deleted without actually deleting them. Useful
                for previewing before committing.

        Returns:
            List of run_ids that were (or would have been) deleted,
            in the order they were considered. Empty list if nothing
            qualified.
        """
        self._require_writable()

        # SQLite-friendly "age in minutes" calculation: ``started_at``
        # is stored as an ISO8601 UTC string with a trailing 'Z', so
        # ``julianday`` handles it directly. julianday(now) - julianday(x)
        # is in days; multiply by 1440 for minutes.
        # Candidate runs: those with zero generations AND zero
        # case_outputs. We compute both counts in subqueries so the
        # caller gets a single-statement filter.
        cur = self._conn.execute(
            """
            SELECT r.run_id,
                   r.started_at,
                   r.completed_at,
                   (julianday('now') - julianday(r.started_at)) * 1440.0
                       AS age_minutes
            FROM runs AS r
            WHERE
                (SELECT COUNT(*) FROM generations g
                 WHERE g.run_id = r.run_id) = 0
            AND
                (SELECT COUNT(*) FROM case_outputs c
                 WHERE c.run_id = r.run_id) = 0
            ORDER BY r.started_at ASC
            """,
        )
        candidates = cur.fetchall()

        eligible: List[str] = []
        for row in candidates:
            run_id = row["run_id"]
            # Completed runs are always eligible — they explicitly
            # called end_run, which means no writer will touch them
            # further. Otherwise require min age.
            completed = row["completed_at"] is not None
            age = float(row["age_minutes"])
            if completed or age >= min_age_minutes:
                eligible.append(run_id)

        if dry_run:
            return eligible

        for run_id in eligible:
            # delete_run cascades and commits per-run. A single
            # batch DELETE-IN would be faster, but per-run delete
            # keeps the log trail clean (one INFO per deletion)
            # and interacts safely with foreign-key cascades in
            # all SQLite versions.
            self.delete_run(run_id)

        return eligible

    # --- Case-level ----------------------------------------------------

    def record_case_outputs(
        self,
        run_id: str,
        *,
        birth_gen: int,
        birth_gene: int,
        sim_case_idx: int,
        results: CaseResultSet,
    ) -> None:
        """Archive every DataFrame in a CaseResultSet.

        Called by :class:`~workflow_common.problem.Problem` after a
        successful case read, so the archive captures exactly what
        the evaluator saw. Uses ``INSERT OR REPLACE`` so re-runs of
        the same case (e.g. manual retry of a previously-failed
        gene) overwrite the old blob.

        Args:
            run_id: The active run.
            birth_gen, birth_gene, sim_case_idx: Case coordinates.
                Must match what the PathResolver produced.
            results: The ``CaseResultSet`` whose DataFrames to
                archive. One row per table is inserted.
        """
        self._require_writable()
        rows = []
        for name, table in results.tables.items():
            df = table.df
            # pickle with the default protocol is fine here: the
            # archive is user-owned and rarely transferred across
            # Python major versions.
            blob = pickle.dumps(df, protocol=pickle.DEFAULT_PROTOCOL)
            rows.append((
                run_id, birth_gen, birth_gene, sim_case_idx, name, blob,
            ))
        self._conn.executemany(
            "INSERT OR REPLACE INTO case_outputs("
            "    run_id, birth_gen, birth_gene, "
            "    sim_case_idx, output_name, data_blob"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def record_case_curve(
        self,
        run_id: str,
        *,
        birth_gen: int,
        birth_gene: int,
        sim_case_idx: int,
        independent: np.ndarray,
        dependent: np.ndarray,
        independent_label: Optional[str] = None,
        dependent_label: Optional[str] = None,
    ) -> None:
        """Archive an extracted ``(independent, dependent)`` curve.

        Stored once per ``(gene, sim_case)`` — the curve the
        objective evaluator scored against the experimental
        reference, but over the FULL extraction range (the user's
        optimization window MUST be stripped from the extractor
        before calling this). Storing the un-windowed range lets
        downstream consumers re-run analyses with different
        metrics, weighting, or windows without re-extracting from
        raw case_outputs.

        Use cases this enables:

        * Post-run re-analysis with a different error metric.
        * Bayesian / Gaussian-process surrogate models trained on
          (gene_vector → curve) mappings.
        * Side-by-side comparison of curves from different runs at
          the same gene.

        Args:
            run_id: The active run.
            birth_gen, birth_gene, sim_case_idx: Case coordinates,
                same convention as case_outputs.
            independent: 1-D float array (typically strain).
            dependent: 1-D float array (typically stress). Must
                share length with ``independent``.
            independent_label: Optional column-name hint
                (default 'strain' if omitted at read time —
                callers can pass other strings for time/temperature
                etc.).
            dependent_label: Optional column-name hint
                (default 'stress' at read time).
        """
        self._require_writable()
        independent = np.asarray(independent, dtype=float)
        dependent = np.asarray(dependent, dtype=float)
        if independent.ndim != 1 or dependent.ndim != 1:
            raise ValueError(
                f"independent/dependent must be 1-D; got shapes "
                f"{independent.shape} / {dependent.shape}"
            )
        if independent.shape[0] != dependent.shape[0]:
            raise ValueError(
                f"independent (n={independent.shape[0]}) and dependent "
                f"(n={dependent.shape[0]}) must have the same length"
            )
        ind_blob = pickle.dumps(independent)
        dep_blob = pickle.dumps(dependent)
        self._conn.execute(
            "INSERT OR REPLACE INTO case_curves("
            "    run_id, birth_gen, birth_gene, sim_case_idx, "
            "    independent_blob, dependent_blob, "
            "    independent_label, dependent_label"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, birth_gen, birth_gene, sim_case_idx,
             ind_blob, dep_blob,
             independent_label, dependent_label),
        )
        self._conn.commit()

    def load_case_curve(
        self, run_id: str,
        *, birth_gen: int, birth_gene: int, sim_case_idx: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[str], Optional[str]]]:
        """Retrieve an archived curve as ``(ind, dep, ind_label, dep_label)``.

        Returns ``None`` if no curve was stored for these
        coordinates (e.g. older archives that pre-date this
        feature, or a case that failed before extraction
        completed). Callers can fall back to extracting from
        ``case_outputs`` in that case — the result will match
        what the optimizer scored against, modulo the extractor's
        determinism.

        Tolerates a missing ``case_curves`` table on read-only
        opens of older archives (returns ``None``).
        """
        self._require_open()
        if not self._has_table("case_curves"):
            return None
        cur = self._conn.execute(
            "SELECT independent_blob, dependent_blob, "
            "       independent_label, dependent_label "
            "FROM case_curves WHERE run_id=? AND birth_gen=? "
            "AND birth_gene=? AND sim_case_idx=?",
            (run_id, birth_gen, birth_gene, sim_case_idx),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (
            pickle.loads(row["independent_blob"]),
            pickle.loads(row["dependent_blob"]),
            row["independent_label"],
            row["dependent_label"],
        )

    # --- Generation-level ----------------------------------------------

    def record_generation(
        self,
        run_id: str,
        *,
        gen_idx: int,
        genes: Sequence[GeneRecord],
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Archive one generation's metadata and every gene in its pop.

        Writes two rows to ``generations`` and N rows to ``genes``
        in one transaction. The driver calls this at the end of
        each generation; rolling forward, the archive always
        reflects a completely-finished generation.

        Args:
            run_id: The active run.
            gen_idx: Pop-library index. Generations numbered from 0.
            genes: One :class:`GeneRecord` per individual in the
                pop_library at ``gen_idx``. The record's ``gen_idx``
                field must match the outer argument; the record's
                ``pop_idx`` places it within the generation.
            stats: Optional JSON-serializable dict of summary stats
                (logbook compile output).
        """
        self._require_writable()
        for g in genes:
            if g.gen_idx != gen_idx:
                raise ValueError(
                    f"gene record has gen_idx={g.gen_idx} but outer "
                    f"gen_idx={gen_idx}"
                )
            if g.run_id != run_id:
                raise ValueError(
                    f"gene record has run_id={g.run_id!r} but outer "
                    f"run_id={run_id!r}"
                )

        now = _utc_now_iso()
        stats_json = json.dumps(stats, default=_json_default) if stats else None
        n_pop = len(genes)
        # Single transaction so a crash mid-write doesn't leave the
        # generations row without its genes. INSERT OR REPLACE on the
        # generations row handles the resume-with-same-id case where
        # the row existed from a previous crash.
        with self._conn:  # context manager = atomic transaction
            self._conn.execute(
                "INSERT OR REPLACE INTO generations("
                "    run_id, gen_idx, recorded_at, n_pop, stats_json"
                ") VALUES (?, ?, ?, ?, ?)",
                (run_id, gen_idx, now, n_pop, stats_json),
            )
            # Remove any leftover gene rows for this (run_id, gen_idx).
            # Resume after a mid-gen crash could otherwise leave
            # partial pop_idx rows.
            self._conn.execute(
                "DELETE FROM genes WHERE run_id=? AND gen_idx=?",
                (run_id, gen_idx),
            )
            self._conn.executemany(
                "INSERT INTO genes("
                "    run_id, gen_idx, pop_idx, birth_gen, birth_gene, "
                "    gene_vector, fitness, rank"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        g.run_id, g.gen_idx, g.pop_idx,
                        g.birth_gen, g.birth_gene,
                        json.dumps(g.gene_vector.tolist()),
                        json.dumps(list(g.fitness)),
                        g.rank,
                    )
                    for g in genes
                ],
            )
        logger.info(
            "archive: recorded generation %d for run %s (%d genes)",
            gen_idx, run_id, n_pop,
        )

    # --- Queries -------------------------------------------------------

    def list_runs(self) -> List[RunSummary]:
        """Enumerate all runs in the archive, oldest first."""
        self._require_open()
        cur = self._conn.execute(
            "SELECT r.run_id, r.started_at, r.completed_at, r.seed, "
            "       r.param_names, r.objective_labels, "
            "       COALESCE("
            "           (SELECT COUNT(*) FROM generations "
            "            WHERE generations.run_id=r.run_id), 0) AS n_gen "
            "FROM runs r "
            "ORDER BY r.started_at ASC"
        )
        return [self._row_to_run_summary(row) for row in cur.fetchall()]

    def get_run(self, run_id: str) -> RunSummary:
        """Fetch one run's summary by ID.

        Raises:
            KeyError: If no such run exists.
        """
        self._require_open()
        cur = self._conn.execute(
            "SELECT r.run_id, r.started_at, r.completed_at, r.seed, "
            "       r.param_names, r.objective_labels, "
            "       COALESCE("
            "           (SELECT COUNT(*) FROM generations "
            "            WHERE generations.run_id=r.run_id), 0) AS n_gen "
            "FROM runs r WHERE r.run_id=?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"no run with id {run_id!r}")
        return self._row_to_run_summary(row)

    def list_generations(self, run_id: str) -> List[GenerationSummary]:
        """Enumerate a run's generations in order."""
        self._require_open()
        cur = self._conn.execute(
            "SELECT run_id, gen_idx, recorded_at, n_pop, stats_json "
            "FROM generations WHERE run_id=? ORDER BY gen_idx ASC",
            (run_id,),
        )
        out = []
        for row in cur.fetchall():
            stats = (
                json.loads(row["stats_json"])
                if row["stats_json"] is not None else {}
            )
            out.append(GenerationSummary(
                run_id=row["run_id"],
                gen_idx=row["gen_idx"],
                recorded_at=row["recorded_at"],
                n_pop=row["n_pop"],
                stats=stats,
            ))
        return out

    def load_genes(self, run_id: str, gen_idx: int) -> List[GeneRecord]:
        """Load every archived gene in one generation, ordered by pop_idx.

        Returns an empty list if the generation isn't archived yet
        (e.g. live run that hasn't finished gen_idx).
        """
        self._require_open()
        cur = self._conn.execute(
            "SELECT run_id, gen_idx, pop_idx, birth_gen, birth_gene, "
            "       gene_vector, fitness, rank "
            "FROM genes WHERE run_id=? AND gen_idx=? "
            "ORDER BY pop_idx ASC",
            (run_id, gen_idx),
        )
        out = []
        for row in cur.fetchall():
            out.append(GeneRecord(
                run_id=row["run_id"],
                gen_idx=row["gen_idx"],
                pop_idx=row["pop_idx"],
                birth_gen=row["birth_gen"],
                birth_gene=row["birth_gene"],
                gene_vector=np.asarray(json.loads(row["gene_vector"]), dtype=float),
                fitness=tuple(json.loads(row["fitness"])),
                rank=row["rank"],
            ))
        return out

    def load_all_genes(self, run_id: str) -> List[GeneRecord]:
        """Load every archived gene for a run in a single query.

        Equivalent to concatenating :meth:`load_genes` over every
        generation, but with one SQLite round-trip instead of one
        per generation. On a 50k-record run this is the difference
        between ~1.5 s and a few hundred ms — the per-query
        overhead dominates at that scale because each generation
        only returns ~100 rows.

        The returned list is ordered by ``(gen_idx ASC, pop_idx ASC)``
        so callers that care about "earliest-seen first" (e.g. dedup
        keeping the oldest birth) can rely on the ordering.

        Returns an empty list if the run has no archived generations.
        """
        self._require_open()
        cur = self._conn.execute(
            "SELECT run_id, gen_idx, pop_idx, birth_gen, birth_gene, "
            "       gene_vector, fitness, rank "
            "FROM genes WHERE run_id=? "
            "ORDER BY gen_idx ASC, pop_idx ASC",
            (run_id,),
        )
        out = []
        for row in cur.fetchall():
            out.append(GeneRecord(
                run_id=row["run_id"],
                gen_idx=row["gen_idx"],
                pop_idx=row["pop_idx"],
                birth_gen=row["birth_gen"],
                birth_gene=row["birth_gene"],
                gene_vector=np.asarray(json.loads(row["gene_vector"]), dtype=float),
                fitness=tuple(json.loads(row["fitness"])),
                rank=row["rank"],
            ))
        return out

    def load_case_outputs(
        self,
        run_id: str,
        *,
        birth_gen: int,
        birth_gene: int,
        sim_case_idx: int,
    ) -> Optional[CaseResultSet]:
        """Reconstruct a CaseResultSet from the archive.

        Returns ``None`` if no outputs are archived for the given
        key — matches the contract of the disk-based
        :func:`~workflow_common.postprocess.load_case_results`.

        The returned CaseResultSet's ``ctx`` has
        ``generation=birth_gen, gene=birth_gene, obj=sim_case_idx``
        — the same coordinates the sim was run under.
        ``source_path`` values on the TabularResults are synthesized
        as ``archive://<output_name>`` because there's no single
        on-disk path to report.
        """
        self._require_open()
        cur = self._conn.execute(
            "SELECT output_name, data_blob FROM case_outputs "
            "WHERE run_id=? AND birth_gen=? AND birth_gene=? AND sim_case_idx=?",
            (run_id, birth_gen, birth_gene, sim_case_idx),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        tables: Dict[str, TabularResult] = {}
        for row in rows:
            name = row["output_name"]
            df = pickle.loads(row["data_blob"])
            tables[name] = TabularResult(
                name=name,
                df=df,
                source_path=Path(f"archive://{name}"),
            )
        ctx = CaseContext(
            generation=birth_gen, gene=birth_gene, obj=sim_case_idx,
        )
        return CaseResultSet(ctx=ctx, tables=tables)

    # --- Experimental reference data ---------------------------------

    def record_experiment(
        self,
        run_id: str,
        *,
        sim_case_idx: int,
        df: "pd.DataFrame",
        label: Optional[str] = None,
        minmax_strain: Optional[Tuple[Optional[float], Optional[float]]] = None,
        extractor_config: Optional[Dict[str, object]] = None,
    ) -> None:
        """Store experimental reference data for one SimCase.

        Called by the driver at run start so post-run plotting tools
        can recover the experimental curves without needing the user
        to re-supply CSV paths. The DataFrame is pickled into a BLOB
        — same storage strategy as ``record_case_outputs`` so any
        DataFrame shape (column-name conventions, index types,
        non-numeric metadata columns) round-trips faithfully.

        Re-recording the same ``(run_id, sim_case_idx)`` overwrites
        the previous blob. This matters when a long calibration is
        resumed — the driver will call this on every run start, and
        the experimental data hasn't changed, so an upsert is the
        correct semantic.

        Args:
            run_id: The run this experimental data belongs to.
            sim_case_idx: Which SimCase this data is the reference
                for. Matches the ``sim_case_idx`` used in
                ``case_outputs`` for the same row coordinates.
            df: The experimental DataFrame. Most calibrations store
                strain + stress columns; the framework doesn't
                enforce a schema because different sim setups may
                care about different signals.
            label: Optional human-readable label, typically the
                ``SimCase.label`` value. Useful for plot titles.
            minmax_strain: Optional ``(lo, hi)`` strain window the
                optimizer was constrained to. Stored as JSON
                ``[lo, hi]`` (with JSON nulls for either side).
                Plotting tools use this to shade the optimized
                region against the full experimental curve so users
                can sanity-check what the optimizer was actually
                fitting. Either side may be ``None`` for "unbounded"
                — same convention as the extractor's ``window`` field
                except that the extractor cropped against ``|strain|``,
                while the value stored here is the user-supplied
                ``case_data["minmax_strain"]`` verbatim.
            extractor_config: Optional dict from
                :meth:`StressStrainExtractor.to_dict` capturing the
                extractor configuration used at run time. Stored as
                JSON. Plotting tools use this to reconstruct
                ``(strain, stress)`` curves with the EXACT extractor
                settings the optimizer scored against — including
                ``strain_source``, ``strain_rate``, column-name
                overrides, etc. — instead of guessing at defaults.
                Without this, a default-built extractor at plot time
                may silently fail or produce different curves than
                the optimizer saw.
        """
        self._require_writable()
        blob = pickle.dumps(df)
        # JSON-encode the window, preserving Nones as JSON nulls so
        # round-trip yields exactly what was supplied. Storing as
        # text rather than two REAL columns keeps the schema simple
        # and makes "the field was added in a later revision"
        # migrations require only a single ALTER TABLE.
        if minmax_strain is None:
            mm_json: Optional[str] = None
        else:
            lo, hi = minmax_strain
            mm_json = json.dumps([lo, hi])
        ext_json: Optional[str] = (
            None if extractor_config is None
            else json.dumps(extractor_config)
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO experiments(run_id, sim_case_idx, "
            "label, data_blob, minmax_strain, extractor_config) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, sim_case_idx, label, blob, mm_json, ext_json),
        )
        self._conn.commit()

    def load_experiment(
        self, run_id: str, sim_case_idx: int,
    ) -> Optional[Tuple[Optional[str], "pd.DataFrame"]]:
        """Retrieve the experimental DataFrame for one SimCase.

        Returns ``(label, df)`` if found, or ``None`` if no
        experimental data was archived for this ``(run_id,
        sim_case_idx)``. The ``label`` is whatever was passed at
        record time (typically ``SimCase.label`` — may be ``None``).

        Returns ``None`` (not an exception) if the ``experiments``
        table doesn't exist at all. This handles the
        backward-compat case: archives written before this table
        was added remain readable; callers (e.g. plotter
        ``_resolve_experimental_for_case``) gracefully fall back
        to user-supplied CSVs. Read-only opens never run the
        schema-creation script — that's by design — so the
        "missing table" path is real and routine, not a corruption
        signal.

        Note: the strain window (``minmax_strain``) is fetched via
        the separate :meth:`load_experiment_window` method. Keeping
        the load API split lets callers that only need the window
        (e.g. a plotter shading the optimized region) skip the
        DataFrame deserialize.
        """
        self._require_open()
        if not self._has_table("experiments"):
            return None
        cur = self._conn.execute(
            "SELECT label, data_blob FROM experiments "
            "WHERE run_id=? AND sim_case_idx=?",
            (run_id, sim_case_idx),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return row["label"], pickle.loads(row["data_blob"])

    def load_experiment_window(
        self, run_id: str, sim_case_idx: int,
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        """Retrieve the stored optimization strain window for one SimCase.

        Returns the ``(lo, hi)`` tuple as supplied at record time
        (either side may be ``None``), or ``None`` if no record
        exists for this ``(run_id, sim_case_idx)`` OR if the
        ``minmax_strain`` column is absent (older archives that
        pre-date the column addition).

        Distinguishing "no record" from "record exists with NULL
        window": both return ``None``. A caller that needs to tell
        them apart can call :meth:`load_experiment` to check
        existence; for the plotting use case the distinction
        doesn't matter — either way there's no shading to draw.
        """
        self._require_open()
        if not self._has_column("experiments", "minmax_strain"):
            return None
        cur = self._conn.execute(
            "SELECT minmax_strain FROM experiments "
            "WHERE run_id=? AND sim_case_idx=?",
            (run_id, sim_case_idx),
        )
        row = cur.fetchone()
        if row is None:
            return None
        raw = row["minmax_strain"]
        if raw is None:
            return None
        try:
            lo, hi = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # A malformed entry shouldn't block the plotter from
            # rendering the rest of the figure — log and return None.
            logger.warning(
                "archive: malformed minmax_strain JSON for "
                "run=%s sim_case=%d: %r", run_id, sim_case_idx, raw,
            )
            return None
        return (lo, hi)

    def load_extractor_config(
        self, run_id: str, sim_case_idx: int,
    ) -> Optional[Dict[str, object]]:
        """Retrieve the stored StressStrainExtractor config dict.

        Returns the JSON-decoded dict (suitable for passing to
        :meth:`StressStrainExtractor.from_dict`), or ``None`` if no
        record exists, no config was supplied at record time, or
        the column is absent (older archives that pre-date this
        feature). The plotter falls back to a default extractor in
        any of those cases.

        Why this exists: a default-built extractor at plot time
        often differs from what the optimizer used (e.g. a user who
        ran with ``strain_source='time_rate'`` but the plot's
        default uses ``'biot'`` — no exception, just silently
        wrong curves). Storing the config sidesteps the guesswork.
        """
        self._require_open()
        if not self._has_column("experiments", "extractor_config"):
            return None
        cur = self._conn.execute(
            "SELECT extractor_config FROM experiments "
            "WHERE run_id=? AND sim_case_idx=?",
            (run_id, sim_case_idx),
        )
        row = cur.fetchone()
        if row is None:
            return None
        raw = row["extractor_config"]
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(
                "archive: malformed extractor_config JSON for "
                "run=%s sim_case=%d: %r", run_id, sim_case_idx, raw,
            )
            return None

    def list_experiments(
        self, run_id: str,
    ) -> List[Tuple[int, Optional[str]]]:
        """Return ``[(sim_case_idx, label), ...]`` for every archived experiment.

        Doesn't return the DataFrames themselves — that's
        :meth:`load_experiment` 's job. Useful for tools that want
        to know which SimCases have stored references before
        deciding whether to fetch the (potentially large) data.
        Ordered by ``sim_case_idx``.

        Returns ``[]`` if the ``experiments`` table doesn't exist
        — same backward-compat reasoning as :meth:`load_experiment`.
        """
        self._require_open()
        if not self._has_table("experiments"):
            return []
        cur = self._conn.execute(
            "SELECT sim_case_idx, label FROM experiments "
            "WHERE run_id=? ORDER BY sim_case_idx ASC",
            (run_id,),
        )
        return [(row["sim_case_idx"], row["label"]) for row in cur.fetchall()]

    def _has_table(self, name: str) -> bool:
        """True if ``name`` is a real table in the open database.

        Used to keep newly-added tables (e.g. ``experiments``)
        readable on archives created by older versions of this
        code that had a smaller schema. Read-only opens skip the
        ``executescript(_SCHEMA_SQL)`` call (you can't write to a
        read-only DB), so missing-table handling has to be
        defensive on the read path.

        ``sqlite_master`` is queried directly rather than
        ``sqlite_schema`` so this works on SQLite versions
        predating 3.33; everywhere else they're aliases.
        """
        cur = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
        return cur.fetchone() is not None

    def _has_column(self, table: str, column: str) -> bool:
        """True if ``table`` exists AND has a column named ``column``.

        Used by the read paths that touch columns added in a later
        revision than the table itself (e.g. ``experiments.minmax_strain``,
        added after ``experiments`` first shipped). On read-only
        opens of an old archive the column may be absent even though
        the table exists; we fall back to "no value" gracefully
        rather than raise OperationalError.

        Returns False for both "no such table" and "table exists,
        no such column" — callers usually want the same fallback
        behavior in either case.
        """
        if not self._has_table(table):
            return False
        # PRAGMA table_info returns one row per column with the
        # column name in the second position. Lower-cost than a
        # SELECT * LIMIT 0 introspection for this purpose.
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cur.fetchall())

    def _migrate_add_missing_columns(self) -> None:
        """Add columns that were introduced after a table first shipped.

        Each migration is idempotent — guarded by ``_has_column`` so
        re-running this on a fresh DB (where the schema-script
        already created the column) is a no-op. Each entry is
        ``(table, column, definition)``; the definition is the SQL
        type clause, NOT including the column name (since that's
        the ``column`` arg).

        Why migrations live here instead of as standalone scripts:
        Robert is the only user, and "open the archive in writable
        mode and the columns appear" is the simplest workflow.
        Standalone migration scripts would be necessary if there
        were data transformations involved, but every column added
        so far has defaulted to NULL on existing rows, which is
        exactly the right value for "this column didn't exist when
        we wrote that row."
        """
        migrations = [
            # Added in session 9: minmax_strain captures the strain
            # window the optimizer was constrained to during the run.
            ("experiments", "minmax_strain", "TEXT"),
            # Added in session 10: JSON-encoded StressStrainExtractor
            # config so post-run plotters reconstruct curves exactly
            # the way the optimizer scored them, instead of guessing
            # at extractor defaults.
            ("experiments", "extractor_config", "TEXT"),
        ]
        for table, column, definition in migrations:
            if not self._has_table(table):
                # Table itself is from a later revision than the
                # archive we're opening. The schema script handled
                # creating it; if it's still missing here something
                # weird happened. Skip rather than ALTER a missing
                # table.
                continue
            if self._has_column(table, column):
                continue
            # PRAGMA-derived type strings are sanitized in the source
            # tuple; no user input flows here. Plain f-string is fine
            # and ALTER TABLE doesn't accept parameter binding for
            # column names anyway.
            self._conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    # --- Internal helpers ---------------------------------------------

    def _require_open(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "archive is not open; call open() or use as context manager"
            )

    def _require_writable(self) -> None:
        self._require_open()
        if self._readonly:
            raise RuntimeError(
                f"archive at {self.path} was opened read-only"
            )

    @staticmethod
    def _row_to_run_summary(row: sqlite3.Row) -> RunSummary:
        labels_raw = row["objective_labels"]
        return RunSummary(
            run_id=row["run_id"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            seed=row["seed"],
            param_names=json.loads(row["param_names"]),
            objective_labels=(
                json.loads(labels_raw) if labels_raw is not None else []
            ),
            n_generations=row["n_gen"],
        )


# --- Helpers -----------------------------------------------------------


def _utc_now_iso() -> str:
    """Current UTC time in ISO-8601, second precision.

    Stored as TEXT in SQLite; this format sorts lexicographically
    so ``ORDER BY started_at`` works as expected.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_default(o: Any) -> Any:
    """Fallback JSON encoder for numpy scalars and Path objects.

    Paths and numpy scalars are common inside RunConfig dicts that
    users pass to :meth:`ArchiveDB.start_run`. The stdlib JSON
    encoder rejects them by default; this coerces the obvious ones.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"cannot JSON-encode {type(o).__name__}")
