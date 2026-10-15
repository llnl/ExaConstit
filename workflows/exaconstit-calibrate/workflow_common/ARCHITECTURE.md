# workflow_common Architecture

This document explains the package's layering, the reasoning behind
its key design choices, and the module-selection decision table.
For a user-facing walkthrough of how to migrate an existing driver,
see `MIGRATION.md`. For per-module details, see the docstring at
the top of each module.

## Why this package exists

Most computational-mechanics optimization workflows end up
re-inventing the same five things:

1. **Input templating** — rendering per-case files from master
   templates with substituted parameter values.
2. **Path layout** — deciding where each case's working directory
   lives and where its outputs will land.
3. **Job launching** — running simulations locally, via Flux,
   or via SLURM, and collecting results.
4. **Crash safety** — surviving an HPC allocation being killed
   partway through a multi-day run without redoing completed
   work.
5. **Objective evaluation** — reading simulation outputs and
   computing numerical error values the optimizer can use.

The pre-refactor code bundled all of these into one monolith
(`ExaProb` plus related scripts). This package separates them into
independent, self-contained modules that cooperate through small
well-defined interfaces.

The framework is intentionally **code-agnostic**. Nothing in
`workflow_common` knows the name of a specific simulation binary,
the format of a specific input file, or the conventions of a
specific output file. All of that is supplied by the caller,
either as a template, a path pattern, or a user-written callback.
The same infrastructure drives ExaConstit today and a completely
different mechanics code tomorrow without modification.

## Layering

Reading bottom-up: each layer depends only on the ones below it.

```
┌─────────────────────────────────────────────────────────┐
│          Optimizer (NSGA-III / user driver)             │
│          Calls problem.evaluate_population()            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     Problem                             │
│   (problem.py) — orchestrator, wires everything together│
└─┬──────────┬──────────┬──────────┬──────────┬──────────┬┘
  │          │          │          │          │          │
┌─▼──┐    ┌──▼───┐   ┌──▼────┐  ┌──▼────┐  ┌──▼────┐  ┌──▼───────┐
│Case│    │Prop  │   │Back   │  │Result │  │Smooth │  │Objectives│
│Temp│    │Writer│   │end    │  │Reader │  │er     │  │          │
└─┬──┘    └──┬───┘   └──┬────┘  └──┬────┘  └──┬────┘  └──┬───────┘
  │          │          │          │          │          │
┌─▼──────────▼──────────▼──────────▼──────────▼──────────▼──────┐
│      PathResolver + Manifest + Sentinel + Templates            │
│                   (low-level plumbing)                         │
└────────────────────────────────────────────────────────────────┘
```

Everything above the bottom row communicates via **Protocols**
(minimal typed interfaces), not concrete classes. A user who wants
to replace `LocalBackend` with a custom Slurm backend writes one
class implementing `JobBackend` and passes it to `Problem`; nothing
else in the stack needs to know.

## SimCase vs. ObjectiveSpec — the sim/objective distinction

One of the most important non-obvious design points: **the number
of simulations per gene is NOT equal to the number of objectives**.

Consider a single quasi-static tension simulation:

* The stress-strain RMSE is one objective.
* The tangent-modulus / slope RMSE is a second objective.
* The peak-stress-at-failure error is a third objective.

All three are computed from the *same* simulation output. Running
three separate simulations would waste cluster time.

`SimCase` is a unit of simulation work (one `case_data` mapping
of per-experiment context, one label). `ObjectiveSpec` is one
evaluator-plus-sim-case-reference. Multiple `ObjectiveSpec`s can
reference the same `SimCase` by its integer index; when they do,
one simulation runs and all those evaluators score its output.

```
  sim_cases      = [SimCase(sr=1e-3), SimCase(sr=1e1)]
                   └────── 0 ──────┘  └───── 1 ─────┘

  objective_specs = [
      ObjectiveSpec(stress_eval, sim_case=0),  ┐
      ObjectiveSpec(slope_eval,  sim_case=0),  │ shared: one sim
      ObjectiveSpec(peak_eval,   sim_case=0),  ┘
      ObjectiveSpec(stress_eval, sim_case=1),  ┐ its own sim
  ]

  per gene: 2 sims run, 4 errors returned.
```

## Crash-safety: sentinel-first-then-manifest

The single most important crash-safety rule in the package:

> After a case completes, write the sentinel FIRST, then append to
> the manifest.

**Sentinel** is a small `.done` file inside the case's working
directory. It is the *authoritative* signal that a case's output
files are valid and the case is complete. Written atomically
(tempfile + rename) so it is never seen half-written.

**Manifest** is a JSONL file (one JSON object per line, append-only)
that logs every state transition across the entire run. Useful for
tooling (what succeeded last night, how many failed, etc.) but not
the primary source of truth.

Why sentinel first? A crash between the two writes can leave the
state inconsistent. If we write the manifest first and the sentinel
never gets written, the manifest claims the case is done but there's
no sentinel to prove it — next restart will see the manifest claim
and not re-run a case whose outputs may actually be corrupt. With
sentinel-first, a crash between the writes leaves the sentinel
present on disk (case is treated as done, correctly) and the
manifest missing one line (at worst, our audit log lost an entry).
Restart logic reads sentinels, not the manifest. Manifest is for
humans and tooling.

## Decision table — "which module do I reach for?"

| I need to… | Use |
|---|---|
| Run simulations in parallel on a desktop | `LocalBackend` |
| Run simulations as Flux jobs on a cluster | `FluxBackend` |
| Render a per-case input file from a master template | `render_template_file` |
| Render several per-case inputs at once | `CaseTemplater` |
| Write a per-gene material-property file | `TemplatePropertyWriter` or `DelimitedPropertyWriter` |
| Figure out where a case's working directory or output files live | `TemplatePathResolver` |
| Track which cases have finished so I can restart a killed run without redoing work | `Manifest` + `write_sentinel` / `read_sentinel` |
| Write a state file that readers never see half-written | `atomic_write_text` |
| Run code inside a different cwd and have cwd restored automatically | `cd` (context manager) |
| Read simulation output files (text tables) into pandas DataFrames | `TextTableReader` + `CaseLayout` |
| Load experimental reference data from CSV | `load_experimental_csv` |
| Align a simulation curve in time to match experimental sample times | `interpolate_to` / `common_time_range` |
| Smooth a monotonic stress-strain curve without overshoot | `PchipSmoother` |
| Smooth a non-monotonic curve (necking, snap-back, softening) | `ArcLengthSmoother` |
| Match pre-refactor smoothing behavior bit-for-bit on monotonic data | `LegacyLinearSmoother` |
| Pick a smoother automatically based on whether x is monotonic | `auto_smoother` |
| Pull strain and stress out of a simulation's CaseResultSet | `StressStrainExtractor` |
| Compute RMSE (or MAE / max-abs) between sim and exp stress-strain | `StressStrainObjective` |
| Decide what objective value to return when a simulation fails | `InfinityFailureHandler`, `ConstantPenaltyFailureHandler`, or `PartialProgressFailureHandler` |
| Orchestrate everything for one optimization problem | `Problem` |
| Archive per-case sim outputs and per-gen stats into a single SQLite file | `ArchiveDB` (pass on `Problem`, opt-in) |
| Delete old case directories during a run while keeping the data | `RunConfig(cleanup_keep_generations=2)` + `ArchiveDB` |
| Load a pickle checkpoint written by the driver | `load_checkpoint` → `CheckpointData` |
| Pick the best solution(s) from a final generation | `best_solution_eudist` / `best_solution_asf` / `BestSol` |
| Re-read a gene's simulation output from disk after a run | `extract_gene_results` + `load_case_results` |
| Re-read a gene's simulation output from the SQLite archive | `extract_gene_results` + `load_case_results_from_archive` |
| Overlay a sim stress-strain curve against experimental data | `plot_stress_strain_overlay` |
| Plot a 2-objective Pareto front with best-solution highlights | `plot_pareto_front` |
| Log consistently across all framework modules | `configure_logging` + `get_logger(__name__)` |

## Module map

| Module | Purpose |
|---|---|
| `_fs.py` | Filesystem helpers: `cd` context manager, atomic text writes. |
| `logging_utils.py` | Stdlib-logging-based replacement for `ExaConstit_Logger` plus a compat shim. |
| `templates.py` | `%%key%%` placeholder substitution for rendering per-case text files. |
| `paths.py` | `PathResolver` abstraction + `TemplatePathResolver` driven by Python `str.format` patterns. |
| `manifest.py` | JSONL manifest for persistent, crash-safe tracking of which cases have submitted / completed / failed. |
| `sentinel.py` | Per-case `.done` marker files written atomically once outputs are validated. |
| `platform_detect.py` | Small helpers for machine-specific quirks (Spectrum MPI hosts, etc.). |
| `results.py` | `CaseLayout`, `ResultReader` Protocol, `TextTableReader` for parsing sim output into pandas DataFrames. |
| `smoothing.py` | PCHIP / arc-length / linear smoothers for resampling `(x, y)` curves. |
| `case_setup.py` | `CaseTemplater` for rendering inputs; `PropertyWriter` Protocol for encoding gene values. |
| `objectives.py` | `StressStrainExtractor`, `ObjectiveEvaluator` Protocol, `FailureHandler` Protocol. |
| `problem.py` | `Problem` orchestrator + `SimCase` and `ObjectiveSpec` records. |
| `postprocess.py` | Checkpoint loader, best-solution pickers, on-demand sim-output reader, and matplotlib helpers for stress-strain / Pareto plots. |
| `archive.py` | SQLite-backed archive: per-case DataFrames, per-generation stats, and run metadata. Enables rolling-cleanup of case directories during long runs without losing data. |
| `backends/` | `JobBackend` Protocol, `SimJobSpec` / `JobResult` dataclasses, `LocalBackend`, `FluxBackend`. |

## Frequently asked questions

### Why JSONL for the manifest instead of SQLite?

JSONL is append-only plain text. On a shared network filesystem
with multiple concurrent writers, single-line appends are atomic
(POSIX guarantees this up to `PIPE_BUF`, typically 4kB, and our
manifest entries fit). SQLite requires a coordinated lock that
is flaky over NFS, Lustre, and GPFS. The format is also trivially
greppable and diffable, which matters for debugging.

### Why `%%key%%` placeholders instead of Python `str.format`?

Python `str.format` uses `{key}` delimiters, which collide with
the literal braces in most input-file formats (TOML arrays, JSON,
shell scripts, CMake fragments). `%%key%%` is unambiguous and
requires no escaping of the host file's native syntax.

### Why are backends and readers Protocols, not base classes?

Two reasons. First, Protocols (PEP 544) let users implement their
own backends and readers without inheriting from framework types —
useful for mocking in tests and for wrapping third-party tools.
Second, structural typing avoids diamond-inheritance problems and
keeps the framework dependency graph acyclic.

### Why does `CaseContext` still have a field called `obj` when it
### now means "sim_case index"?

Backward compatibility with existing path patterns. The old
semantic was "one objective = one sim = one directory". When we
split `ObjectiveSpec` and `SimCase` apart, we kept the `obj` field
(now reinterpreted as "sim_case index") so that resolver patterns
like `"gen_{generation}/gene_{gene}_obj_{obj}"` continue to work.
The docs recommend `_sc_{obj}` in new patterns to make the
semantics clear.

### How does the archive relate to the old
### ``ind.stress`` pickle hack?

The pre-refactor driver stashed the simulated stress history on
each DEAP Individual so the pickle checkpoint carried everything
needed for post-processing. That let you delete case directories
and still have the data. But it made the checkpoint file huge
(megabytes per generation times ~100 generations = GB-scale
pickles), and it hard-coupled post-processing to DEAP.

The archive replaces that pattern cleanly. `ArchiveDB` captures
exactly the data the evaluator saw (the `CaseResultSet`) in a
separate SQLite file; the checkpoint stays small and DEAP-only.
With `cleanup_keep_generations=2` on the driver, case directories
are rolling-deleted after archival, keeping the filesystem inode
count bounded. Post-processing uses
`load_case_results_from_archive` instead of `load_case_results` —
same return type, different source.

### Why SQLite (vs. HDF5 / Parquet / ...) ?

SQLite ships with Python stdlib, requires no daemon, and tolerates
concurrent readers while the driver is writing (via WAL journaling
mode). A single archive file is trivial to rsync off a compute
node, archive to cold storage, or email to a collaborator. HDF5
would be a reasonable alternative — similar single-file story, and
potentially more compact for numeric data — but adds a dependency
(h5py or tables) that nothing else in the framework needs, and its
concurrent-access story is weaker. Parquet would need pyarrow for
similar reasons. SQLite is the minimum-dependency choice.

### Why pickle DataFrames into BLOB columns instead of normalizing
### into SQL tables?

The data flowing through the archive is a dict of
`pandas.DataFrame` per case, with arbitrary columns depending on
which simulation outputs were registered. Normalizing into SQL
tables would require per-run schema discovery and per-column
typing; pickle round-trips every DataFrame bit-identically without
touching any of that. The tradeoff: you can't `SELECT stress FROM
...` directly; you'd pull the DataFrame back and query in pandas.
For this workload that's the right tradeoff — nobody actually
wants to run SQL against stress-strain curves.

### How does archive-aware resume work?

A pickle checkpoint at end-of-gen-K contains an `archive_run_id`
field. The recommended driver pattern on resume: build the
`Problem` with `archive_run_id=None` (don't call `start_run`).
The library reads the UUID out of the pickle, assigns it to
`problem.archive_run_id`, then calls
`archive.discard_from_generation(run_id, K+1)`, which cascades
through foreign keys to drop any archive rows for generations
> K. Then it re-runs gen K+1 and writes fresh rows. Upshot: even
if the crash left half-written archive entries from a partial
generation, the resume cleans them before new writes.

If the caller accidentally passes a different `archive_run_id`
than the pickled one (typically by calling `start_run` a second
time on resume, creating a fresh UUID), the library refuses with
a `ValueError` naming both UUIDs and showing the fix. The
Problem constructor accepts `archive=..., archive_run_id=None`
precisely so this pattern can be expressed cleanly;
`archive_run_id=None` with `archive=None` is still rejected
since there's nowhere to write. Covered by
`test_archive_resume_discards_stale_generations`,
`test_archive_resume_adopts_pickled_run_id_when_problem_has_none`,
and `test_archive_resume_rejects_conflicting_run_id_with_helpful_error`.

### What housekeeping does `ArchiveDB` support?

Two explicit operations beyond the usual read/write interface:

- `delete_run(run_id)` — remove a single run and every dependent
  row (generations, genes, case_outputs) via schema-level
  `ON DELETE CASCADE`. Raises `KeyError` on unknown IDs rather
  than silent no-op.
- `prune_empty_runs(*, min_age_minutes=60.0, dry_run=False)` —
  bulk-delete runs that have zero generations AND zero case
  outputs. Default age gate protects in-progress runs that
  haven't written their first generation yet; pass
  `min_age_minutes=0` to consider all empty runs. Completed
  runs (those with `completed_at` set) are eligible regardless
  of age.

Both operations require a writable connection and cascade through
the same FK rules. The `inspect_archive` CLI exposes
`prune_empty_runs` via `--clean-empty-runs [--dry-run]
[--age-minutes N]` so users don't need to write Python to tidy
up aborted debug sessions.

### How do I use the archive without the NSGA-III driver?

`ArchiveDB` is fully standalone — you can attach it to a `Problem`
without any driver:

```python
archive = ArchiveDB("opt.db")
archive.open()
run_id = archive.start_run(seed=0, param_names=["x", "y"])
problem = Problem(..., archive=archive, archive_run_id=run_id)

# Your custom GA / optimizer loop:
for gen in range(n_gen):
    results = problem.evaluate_population(genes, generation=gen)
    # ... your own selection logic ...

archive.end_run(run_id)
archive.close()
```

Case outputs are archived automatically by the Problem.
`record_generation` is driver-owned — if you want per-generation
stats archived, call it yourself with whatever stats your
optimizer produces. The `GeneRecord` dataclass in
`workflow_common.archive` is the expected input.

### How do I parallelize across cases?

Pass a backend with a concurrency limit. `LocalBackend(max_workers=8)`
runs up to 8 cases at once on one node via a `ThreadPoolExecutor`
(each worker launches a subprocess; simulations themselves are
multiprocess, not multithreaded). `FluxBackend` submits each case
as its own Flux job and lets the resource manager decide. Either
way, `Problem.evaluate_population` hands the full batch to the
backend in one call via `stream_batch`, so concurrency is a knob
on the backend, not on the Problem.

### How does restart work?

`Problem` checks for a sentinel in each case's working directory
before launching. If present (and `skip_completed=True`), the
case is marked done and its outputs are re-read and re-scored
without re-running the simulation. A restartable optimization
is thus: use a stable manifest path across runs, let the filesystem
persist state, and restart by running the same driver again.
See `test_skips_completed_on_rerun` and
`test_shared_sim_skip_serves_many_objectives` for working examples.
