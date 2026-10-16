# Audit Findings & Action Plan — `exaconstit-calibrate`

Date: 2026-04-23
Scope: systematic review of path handling, subprocess launches, default
values, and validation coverage across `workflow_common`, `workflows`,
and the backends.

## Bug-class meta-pattern

Three families of bugs surfaced in recent debugging, all variants of the
same meta-pattern: **a signature that accepts values it cannot honor, or
a handoff where two sides each think they own the same step.**

1. **Silent-default-disables-feature.** A default value makes an
   adjacent feature quietly inert. Example: `LocalBackend` defaulted
   `mpi_launcher=None`, so `SimCase.num_tasks=4` silently produced
   single-rank runs. The user never sees a warning.

2. **Redundant handoff / double-compose.** Two layers each do the same
   composition (path join, validation, substitution). Idempotent on
   absolute inputs, so tests on tmp-dir absolute paths never see it;
   surfaces only in production on relative user paths. Example:
   `problem.py::_handle_backend_result` pre-joined `working_dir / p`
   before calling `validate_outputs` which ALSO joined against
   `working_dir`, producing `<workdir>/<workdir>/...` on relative
   workspaces.

3. **API-drift / invented kwarg.** Example code against a docstring
   mental model of the API instead of the real signature. Example:
   `TemplateTarget(substitute=False)` was invented before that field
   existed; `configure_logging(restart=...)` used the old ExaConstit
   logger's kwarg name.

All three share one root cause: **tests exercise happy-path absolute
tmp-dir inputs only.** Production users have relative `WORKSPACE`
paths, unusual kwarg combinations, and shapes the tests never cover.

## Findings

### Finding 1 — Silent escape via `TemplateTarget.dest` (FIXED)

**Severity:** Medium — data can land outside the case dir.

**Mechanism:** `CaseTemplater.render` joined `layout.working_dir /
target.dest` without validating `dest`. On POSIX, `Path("/abs") /
"/absolute"` returns `/absolute`, silently dropping the working-dir
prefix. An unvalidated `dest="../escape/foo.toml"` escapes the case
dir. An unvalidated `dest="{gene}.toml"` produces a literal `{gene}.toml`
filename.

**Fix:** `_validate_case_relative_dest(dest, field_name)` helper in
`case_setup.py`, called from `TemplateTarget.__post_init__`. Rejects
absolute paths, `..` parent refs, `{...}` placeholder syntax, and
non-string / empty values.

**Status:** Fixed.

### Finding 2 — Same silent escape via `TemplatePropertyWriter.dest` and `DelimitedPropertyWriter.dest` (FIXED)

**Severity:** Medium — same mechanism as #1, different classes.

**Mechanism:** `TemplatePropertyWriter` and `DelimitedPropertyWriter`
both have `dest: str` fields that feed directly into `layout.working_dir
/ self.dest` inside `write()`. Same validation gap as #1.

**Fix:** Same `_validate_case_relative_dest` helper, wired in via
`__post_init__` on both classes. `CallablePropertyWriter.dest_hint` is
cosmetic (logging only) and intentionally left unvalidated — the user's
callable does the real writing.

**Status:** Fixed.

### Finding 3 — `skip_completed` reuses stale results after config change (NOT FIXED)

**Severity:** **High** — silent wrong-answer on restart.

**Mechanism:** `Problem.skip_completed=True` (the default) uses
`is_case_complete(layout.working_dir)` to decide whether to re-run a
case. If a sentinel is present, the cached result is loaded. The
sentinel file does NOT fingerprint:

- The gene vector that produced the result
- The `param_names` ordering
- The `ProblemConfig.binary` or `binary_args`
- `SimCase.case_data` (boundary conditions, strain rate, ...)

So a user who:

1. Runs generation 0 to completion
2. Realizes their `param_names` order was wrong, edits the driver
3. Resumes with `--resume-from checkpoint_gen_0.pkl`

will get their new gene-vector interpretation evaluated against the
OLD sentinel-cached results. Objective values are silently wrong for
every case that didn't need re-running. No error, no warning.

**Design questions for the fix:**

1. **What counts as "the config changed"?** At minimum: the gene
   vector itself (hash the numpy array); `param_names` (hash the
   tuple); `ProblemConfig.binary` path + `binary_args` tuple; the
   full `SimCase.case_data` dict. Omit `num_tasks`, `duration_s`,
   and other resource fields — they don't affect the *physics* of the
   simulation, only its execution.

2. **Storage:** add `config_fingerprint: Optional[str]` to
   `Sentinel`. Missing (old sentinel) → treat as untrusted, force
   re-run with a one-line log message. Mismatched → force re-run with
   a LOUD warning that includes the diff. Matched → skip as today.

3. **Fingerprint canonicalization:** `hashlib.sha256(json.dumps({...},
   sort_keys=True, default=_canonicalize_value).encode()).hexdigest()`.
   The `_canonicalize_value` helper handles numpy arrays (convert to
   lists), Path objects (str), and tuples (list, so JSON stable).

4. **Backward compatibility:** existing on-disk sentinels don't have
   the fingerprint. Treat missing-fingerprint as "force re-run" — the
   "safe" direction. Log once per restart: "N sentinels lack a config
   fingerprint; re-running those cases for safety. This is expected
   after a framework upgrade."

5. **Opt-out:** some users will legitimately want to skip the
   fingerprint check (they know their edits were cosmetic). Add
   `ProblemConfig.fingerprint_policy: Literal["check", "force",
   "skip"] = "check"`. `"force"` ignores the fingerprint (current
   behavior). `"skip"` — cases without a fingerprint still skip
   rather than re-run. Default `"check"` is the safe behavior.

**Test plan:** at least five new tests:

- Matching fingerprint → skip as expected
- Missing fingerprint (old sentinel) → re-run with log
- Mismatched fingerprint → re-run with warning
- `fingerprint_policy="force"` → skip regardless
- `fingerprint_policy="skip"` → skip when missing, warn-and-re-run
  when mismatched

**Estimated effort:** 200 lines of code, 5 test cases, plus an update
to `MIGRATION.md` explaining the new sentinel field and its
backward-compatibility path.

**Status:** Not fixed. Highest-priority follow-up item.

### Finding 4 — `clear_outputs` could delete files outside the case dir (FIXED)

**Severity:** Medium — data-destruction hazard if a user misconfigures
`output_file_patterns` with an absolute path.

**Mechanism:** `CaseLayout.clear_outputs()` iterates known outputs and
`unlink()`s each. If a user's `TemplatePathResolver` has an
`output_file_patterns` entry whose rendered path is absolute (or
escapes via `..`), `clear_outputs_on_rerun=True` would delete that
file on every restart.

**Fix:** Clamp-to-workdir safety gate in `results.py::CaseLayout.clear_outputs`.
Resolve both the target path and the working dir (with symlinks) and
refuse to delete anything whose resolved path isn't `relative_to(wd)`.
Log the refusal at WARNING level with the offending path so the
misconfiguration is visible.

**Status:** Fixed.

### Finding 5 — `pickle.load` on `resume_from` path (DOCSTRING WARNING)

**Severity:** Low — requires an attacker with write access to the
checkpoint workspace.

**Mechanism:** `_load_checkpoint` calls `pickle.load` on a user-supplied
path. A malicious pickle executes arbitrary Python during unpickling.
On shared HPC scratch filesystems this is a real risk vector.

**Fix:** Docstring warning on `_load_checkpoint` explaining the risk
and suggesting JSON+npz as an alternative exchange format. Did not
switch the serialization format — that's a bigger change and the risk
profile is low for the typical single-user workspace.

**Status:** Warned. Possible follow-up: optional JSON+npz checkpoint
format for cross-user / cross-host exchange.

### Finding 6 — FluxBackend passed raw (possibly relative) paths to Flux (FIXED)

**Severity:** Low — version-dependent Flux behavior, would manifest
as "job runs in unexpected cwd" or "stdout lands nowhere obvious" on
relative-workspace runs.

**Mechanism:** `FluxBackend._build_jobspec` set `js.cwd = str(spec.working_dir)`
and `js.stdout = spec.stdout` using the RAW user strings. Flux's
interpretation of a relative `cwd` depends on the broker version;
`LocalBackend` uses `resolved_stdout()` (absolute) consistently.

**Fix:** Both `js.cwd` and `js.stdout/stderr` now use resolved
absolute paths at the boundary. Matches LocalBackend's behavior.

**Status:** Fixed.

### Finding 7 — Zero test coverage of relative-workspace roots (ACTION ITEM)

**Severity:** Meta — this is the reason findings 3 and the prior
double-join bug reached production.

**Mechanism:** Every existing integration test uses `tmp_path` (an
absolute pytest-fixture path) as the workspace. Users naturally write
`WORKSPACE = Path("./calibration_run")` in their drivers. The double-join
bug only surfaced when a join chain produced a relative path that was
then re-joined.

**Action item:** parameterize the main end-to-end integration test
over `workspace_style` in `{"absolute", "relative"}`. The relative
variant uses `monkeypatch.chdir(tmp_path)` then `Path("calibration_run")`
as the workspace. Expected to shake out any remaining double-join or
relative-vs-absolute confusion.

**Estimated effort:** 30-50 lines of test code, applied to
`test_integration.py` and `test_nsga3_driver.py`. No framework
changes.

**Status:** Not done. Medium-priority follow-up.

## Bugs confirmed NOT present after audit

Documented for future maintainers so the same ground isn't re-covered:

- No duplicate template-substitution passes. `{working_dir}` format
  key is injected only during `output_file()` rendering; the
  `working_dir_pattern` render raises cleanly on self-reference.
- No SQL concatenation in `archive.py`. All queries use parameterized
  bindings.
- `LocalBackend._run_one` is the only subprocess-launch site. Already
  hardened with the `mpi_launcher` check and the running-jobs lock.
- Backends don't duplicate `validate_outputs` / `write_sentinel`.
  `Problem` is the single authority for both.
- `StressStrainExtractor.strain_rate=None` fails loudly when combined
  with `strain_source="time_rate"`. Safe default.
- `FluxBackend.spectrum_mpi=None` triggers `is_spectrum_machine()`
  auto-detection. Safe default.
- Path joins in `paths.py:355`, `sentinel.py:325`,
  `backends/base.py:257/267` all gate on `.is_absolute()` correctly.
- `CaseTemplater` with `targets=[]` is a safe no-op (already tested).
- `_fs.atomic_write_text` uses tempfile+rename, crash-safe.

## Action-item summary

| # | Item | Priority | Effort | Status |
|---|------|----------|--------|--------|
| 3 | Sentinel config fingerprint for `skip_completed` safety | **High** | ~200 lines code + 5 tests + MIGRATION.md | **Open** |
| 7 | Parameterize integration tests over absolute/relative workspace | Medium | ~50 lines | **Done** |
| 5 | JSON+npz optional checkpoint format for cross-user exchange | Low | ~100 lines + 2 tests | Open |
| 8 | `output_file_patterns` bare-relative auto-prepend symmetry | Medium | ~10 lines + 2 tests | **Done** |
| 9 | Restore `logbook1_stats.log` / `logbook2_solutions.log` writes | Medium | ~100 lines + 3 tests | **Done** |
| — | `inspect_archive` CLI for SQL-free archive viewing | Medium | ~400 lines + 10 tests | **Done** |
| — | Cross-gen `--pareto-only` + `--top` | Medium | ~150 lines + 5 tests | **Done** |
| — | Scale work: vectorized ranking + `load_all_genes` batch | Medium | ~80 lines + bench | **Done** |
| — | `--gens-best` convergence view + `--limit` cap | Medium | ~180 lines + 6 tests | **Done** |
| — | `l2_norm` column + `--pareto-only` implies `--genes` | Low | ~60 lines + 6 tests | **Done** |
| — | `--resume-latest` / integer `--resume-from N` | Medium | ~80 lines in example | **Done** |
| — | Relax `Problem.__init__` for resume orphan case | Medium | ~15 lines + 2 tests | **Done** |
| — | Diagnostic resume logging (`resume path:`, before/after discard) | Low | ~25 lines | **Done** |
| — | `delete_run` + `prune_empty_runs` API + CLI | Medium | ~150 lines + 12 tests | **Done** |
| — | `regenerate_logbook_files` CLI for lost `.log` recovery | Low | ~120 lines + 6 tests | **Done** |
| — | Binary-BLOB gene-vector storage for >100k records | Low | Schema migration | Future |
| — | Run full regression after fixes; add 6-8 tests | Medium | ~100 lines of test code | **Done** |

## Things for future consideration

- **Linter or runtime check for API-drift in examples.** The
  `test_example_imports.py` guard catches module-level import breakage
  but not kwarg drift inside function bodies. Consider a
  `main(dry_run=True)` path that constructs every framework object
  without actually running simulations, and have the guard call it.
- **`ExaConstit_Problems.py` "partially successful" simulation
  behavior.** Old code zero-padded the result tail when
  `error_strain > 0.01`. `ZeroPadPartialHandler` was discussed but
  never implemented. Flagged in MIGRATION troubleshooting.
- **Migration from `required_outputs` toward reader-owned validation.**
  The two pre-flight checks overlap. Consider deprecating
  `ProblemConfig.required_outputs` once every user has moved to
  `TextTableSpec(required=True)`.
- **`FluxBackend.poll_stats` coverage.** Two error-handling branches
  marked `# pragma: no cover`. Not a correctness issue; CI coverage
  gap only.

### Finding 8 — `output_file_patterns` bare-relative asymmetry (FIXED)

**Severity:** Medium — silent-wrong-path bug that manifested as
"reader found no files" at runtime.

**Mechanism:** `TemplatePathResolver.output_file()` rendered patterns
via `str.format_map` and returned `Path(rendered)` directly. Patterns
that included `{working_dir}/` got the correct per-case path;
patterns written as "obviously relative to the case dir" without
that prefix (e.g. `"results/avg_stress.txt"`) got rendered as bare
relative paths that `path.exists()` then resolved against the
driver's cwd — missing the case dir entirely. The asymmetry vs
`working_dir_pattern` (which auto-prepends `root`) was not
documented clearly enough, and a user writing what seemed like the
obvious config got silently wrong behavior.

**Fix:** `output_file()` now auto-prepends `working_dir` when the
rendered pattern is (a) not absolute AND (b) doesn't contain the
`{working_dir}` marker. Patterns that explicitly use the marker
are unchanged; absolute patterns are honored exactly as before.
Symmetry with `working_dir_pattern`'s root-prepend behavior
restored.

**Test coverage:** two new tests in `test_paths.py`
(`test_output_file_auto_prepends_working_dir_for_bare_relative_pattern`
and `test_output_file_bare_relative_matches_working_dir_regardless_of_root_type`).

**Status:** Fixed.

## Session 2 deltas

Between the initial audit and this second pass, the following
additional work was completed:

- **Column-name convention aligned to ExaConstit.** All framework
  defaults, docstring examples, fixtures, and the fake binary now
  use the ``# Time  Volume  Sxx/Syy/Szz/Sxy/Sxz/Syz`` header
  format from
  ``ExaConstit/src/postprocessing/postprocessing_file_manager.hpp``
  ``GetVolumeAverageHeader``. `StressStrainExtractor` defaults are
  `stress_column="Szz"`, `def_grad_column="F33"`, `time_column="Time"`;
  `common_time_range` and `interpolate_to` default `time_column`
  is `"Time"`. Non-ExaConstit users override explicitly.

- **Pandas engine auto-switch for commented files.** The C engine
  was mishandling ExaConstit's "indented `#` header + indented
  data rows" shape, raising a misleading
  `EmptyDataError: No columns to parse from file`. `TextTableReader`
  now switches to the python engine whenever `spec.comment` is
  set, and wraps `EmptyDataError` in a `ValueError` that includes
  the first 512 bytes of the file so users can see what pandas
  actually saw.

- **`strain_source` renamed.** `"f11_minus_one"` → `"axial_minus_one"`,
  `"log_f11"` → `"log_axial"`. The old labels stay accepted via a
  normalization shim in `extract()` that emits a `DeprecationWarning`
  naming the new label. The internal local variable `F11` was also
  renamed to `axial_stretch` so the code reads correctly when
  `def_grad_column="F33"` (ExaConstit z-axis default).

- **Finding 7 closed.** `test_integration.py` now has a
  `@pytest.mark.parametrize("workspace_style", ["absolute", "relative"])`
  variant on the happy-path pipeline. The relative branch uses
  `monkeypatch.chdir(tmp_path)` + `Path("wf_rel")` as the
  workspace root, exactly matching how users in production
  wrote `WORKSPACE = Path("./calibration_run")`.

- **Finding 3 still not addressed.** The sentinel config
  fingerprint for safe `skip_completed` restart is the remaining
  silent-wrong-answer hazard. Detailed design is in this document;
  needs a dedicated session.

Test count progression across the two audit passes:
- Start of audit: 287
- After audit fixes (Findings 1, 2, 4, 6): 290
- After Finding 8 (bare-relative auto-prepend): 290
- After column-name sweep + ExaConstit-first defaults: 291
- After reader C-engine fix: 291
- After `strain_source` rename: 292
- After Finding 7 parameterization: 294

## Session 3 deltas

### Finding 9 — Logbook text files dropped during refactor (FIXED)

**Severity:** Medium — observability regression vs pre-refactor
driver. Not a correctness bug, but "the run is producing zero
user-visible output about its own progress" is a serious quality
regression that hid the health of multi-day runs.

**Mechanism:** The pre-refactor `ExaConstit_NSGA3.py` wrote two
tab-delimited text files every generation:
- `logbook1_stats.log` — avg/std/min/max fitness per generation,
  plus ND / GD / HV for multi-objective runs
- `logbook2_solutions.log` — per-individual gene vector + fitness

The refactor preserved the in-memory DEAP `Logbook` objects and
even pickled them into the checkpoint, but the file writes were
dropped. Users had no way to watch the optimization's progress
mid-run short of inspecting the pickle or the SQLite archive.

**Fix:** Added `_LogbookWriter` helper that writes both files using
DEAP's `logbook.stream` delta mechanic — each call emits only
records added since the previous access. Calls are inserted after
every `logbook.record(...)` site in `run_nsga3`. Resume-from-
checkpoint truncates both files and replays the loaded logbooks
from scratch (using DEAP's `buffindex` cursor field, verified by
reading `deap.tools.Logbook.stream`'s source) so no gaps or
duplicates result.

Two new `RunConfig` fields:
- `log_dir: Optional[Path] = None` — default behavior picks
  checkpoint-adjacent or cwd
- `write_logbook_files: bool = True` — opt-out for tests /
  silent workflows; in-memory logbooks still populated regardless

Three new tests in `test_nsga3_driver.py` cover the happy path,
the opt-out, and the resume rewrite.

**Status:** Fixed.

### New deliverable — `inspect_archive` CLI

Not a bug — a usability gap flagged alongside Finding 9. The
SQLite archive at `archive.db` is the authoritative per-run record
but users couldn't inspect it without a SQLite client. Added
`workflows/optimization/inspect_archive.py` — a standalone CLI
with three views (`--runs`, `--gens`, `--genes`), three formats
(`table` / `csv` / `json`), and `--pareto-only` / `--run` / `--gen`
filters. Expands gene vectors into named columns using the run's
stored `param_names` + `objective_labels`.

Ten tests in `test_inspect_archive.py` cover every view × every
format × error paths.

Invocation examples:
```
python -m workflows.optimization.inspect_archive ./wf --runs
python -m workflows.optimization.inspect_archive ./wf --gens
python -m workflows.optimization.inspect_archive ./wf --genes \
    --gen 10 --pareto-only --format csv > front.csv
```

### Test count through session 3

- End of session 2: 294
- +3 logbook writer tests: 297
- +10 inspect CLI tests: 307

## Session 4 deltas — Scale work

User flagged that real calibrations run at 40k–50k total simulations
(100 gens × 60–80 pop × ~8 objectives per gene, or 500 gens × 100 pop).
Two categories of work addressed.

### Performance on large archives

Baseline profile of `inspect_archive --genes --pareto-only` on a
50k-record archive was 2.1 seconds. Two changes dropped this to
~1.2 s without adding any new dependencies:

1. **`ArchiveDB.load_all_genes(run_id)`** — one SQL query instead of
   one per generation. Eliminates ~500 round-trips on a 500-gen
   archive. Saved ~150 ms of pure query overhead.
2. **Vectorized top-N ranking** — the three separate Python loops
   (`_top_n_by_l2`, `_top_n_by_objective`, `_finite_fitness`) were
   replaced with one `(N, M)` numpy fitness matrix and `np.argsort`
   with non-finite values masked to `+inf`. Saved ~700 ms on 50k
   rows; dropped the 450,000 `math.isfinite` calls in the profile.

Remaining cost is dominated by JSON decode in
`sqlite3 → GeneRecord` hydration (~0.67s of the remaining 1.2s).
Further reduction would require either a faster JSON library
(orjson — adds a dependency) or a schema migration to store
`gene_vector` / `fitness` as a binary BLOB. Neither fits under the
"tuning" bucket, so the JSON cost was left alone. Filed for a
future dedicated turn if scaling above ~100k records becomes
common.

Benchmark table (8 objectives, 6 params, Python 3.12):

| Scale                | --runs | --gens | --gens-best | --pareto-only |
|----------------------|--------|--------|-------------|---------------|
| 8000 records         | 4 ms   | 6 ms   | 156 ms      | 228 ms        |
| 50000 records        | 3 ms   | 5 ms   | 1043 ms     | 1316 ms       |

### New views

- **`--gens-best`** (convergence view). Per-generation row with
  running-best fitness on each objective, current L2 champion's
  birth generation, and cumulative `n_seen`. Designed to answer
  "is the GA still improving?" at a glance — if
  `champion_birth_gen` holds the same value for many rows, the
  run has plateaued.
- **`--limit N`** cap on the raw `--genes` dump (default 200).
  Prevents accidentally flooding the terminal when a generation
  has a huge population. `--limit 0` disables. Ignored for
  `--pareto-only` (already bounded). Stderr notice when
  truncation happens so the row count isn't mysteriously round.

Six new tests bring total to 327.

## Session 5 deltas — Resume UX, observability, housekeeping

Four distinct threads of work in this session. All driven by
Robert hitting real usability issues during his first multi-day
calibration run. Each thread ended up teaching us something
worth keeping documented.

### UX thread: the `--resume-latest` / `--resume-from N` flow

Robert typed `python nsga3_calibration.py --resume-from 16` to pick
up from generation 16 and got a terse `FileNotFoundError: '16'`.
The example's argparse had `type=Path`, which dutifully wrapped the
string "16" as a Path object that didn't exist on disk.

Shipped a three-behavior `--resume-from` accepting any of:
- a plain integer like `15` → looks up
  `checkpoint_dir/checkpoint_gen_15.pkl`
- a full path to a pickle → used verbatim
- nothing (absent) → fresh run

Plus a separate `--resume-latest` flag that scans for the
highest-numbered `checkpoint_gen_*.pkl` in the configured
directory — the "just keep going from wherever we left off"
ergonomic default that shouldn't require a number.

A cosmetic `Path.relative_to()` call in the display line blew up
when the checkpoint path was relative and cwd was absolute. Wrapped
in try/except with a fallback to the unresolved path. Minor but
would have bounced users with perfectly valid invocations.

### Usability thread: archive_run_id mismatch on resume

Robert's first successful `--resume-from <path>` immediately died
with `ValueError: checkpoint archive_run_id='3155cb92-...' differs
from Problem.archive_run_id='679fde72-...'`. Root cause: the
example unconditionally called `archive.start_run(...)` on every
invocation, generating a fresh UUID on resume that conflicted with
the one baked into the pickle.

Two fixes, one in the library and one in the example:

1. **Library `Problem.__init__` relaxed** to accept
   `archive=..., archive_run_id=None` — the orphan case that
   lets resume work cleanly. The deferred check moved to
   `_archive_case`, which raises `RuntimeError` at first write
   if `archive_run_id` is still None by then. Same invariant,
   enforced at the real failure point.
2. **Example now gates** `start_run` on `resume_pickle_path is None`.
   On resume it leaves `archive_run_id=None`; the library reads
   the pickled UUID and assigns it to `problem.archive_run_id`
   before any writes happen. The existing `discard_from_generation`
   call in the resume path then cleans stale post-crash rows.

The mismatch ValueError itself got upgraded to name both UUIDs
and quote the correct driver pattern verbatim:

```
archive_run_id mismatch on resume:
  checkpoint's run_id:  '3155cb92-...'
  Problem's run_id:     '679fde72-...'

Fix: in your driver, only call start_run() when NOT resuming:
    if args.resume_from is None:
        problem.archive_run_id = archive.start_run(...)
```

### Observability thread: diagnostic resume logging

Robert reported "resume starts from gen 0, logs wiped, DB wiped"
and we spent time guessing where the bug was before realizing
the symptoms all described the library taking the fresh path
instead of the resume path. The example driver was passing
`resume_from=args.resume_from` to `RunConfig` when it should have
been passing `resume_pickle_path` (which is what the resolver
block populated). `--resume-latest` only sets `args.resume_latest`,
not `args.resume_from`, so the library saw `None` and took fresh.

Fixed the line. Ship diagnostic logging so the **next** time a
resume misbehaves it's one-glance diagnosable:

```
resume path: loading checkpoint from ...
resume: last completed gen=15, pop_library has 16 entries,
  logbook1 has 16 records, logbook2 has 320 records,
  pickled archive_run_id=3155cb92-...
resume: next generation to run is 16
archive resume: run=3155cb92-... has gens [0..15] before discard;
  keeping gens <= 15, dropping >= 16
archive resume: after discard, run=3155cb92-... has gens [0..15]
```

The two-line before/after discard lines specifically address the
"my archive got wiped" class of report — confirm at a glance that
the framework is only touching the current run's post-crash rows.

### Housekeeping thread: archive cleanup tooling

Multiple debug sessions left Robert with 4+ empty aborted run
rows cluttering his archive. He deleted them manually because no
API existed. That's wrong.

Two new `ArchiveDB` methods:

- **`delete_run(run_id) -> int`** — single-run deletion. Schema
  CASCADE handles dependent rows in `generations` / `genes` /
  `case_outputs`. Raises `KeyError` on unknown IDs so typos
  surface rather than silently no-op. Returns the parent-row
  delete count (1 on success).
- **`prune_empty_runs(*, min_age_minutes=60.0, dry_run=False)`** —
  bulk cleanup. "Empty" means no `generations` AND no
  `case_outputs` rows; a partial crash mid-gen-0 that wrote some
  case outputs doesn't qualify because those are real simulation
  artifacts. Two safety gates protect in-progress runs: completed
  runs (those with `end_run` called) are always eligible; still-
  running runs must be older than `min_age_minutes` (default 60)
  before being touched. Pass 0 to override.

CLI surface on `inspect_archive`:

- `--clean-empty-runs` — wraps `prune_empty_runs`
- `--dry-run` — preview without deleting; prints candidates on
  stderr
- `--age-minutes N` — override the age gate

Zero-delete path prints a helper line pointing at
`--age-minutes 0` so users don't have to `--help` to find the
escape hatch.

### Logbook recovery tool

Separate deliverable: `workflows/optimization/regenerate_logbook_files`
CLI. When the `.log` files on disk have been lost (truncated by
a misconfigured run, accidentally deleted), the pickle still
carries the full DEAP `Logbook` history. This tool reads the
pickle and reuses the driver's own `_LogbookWriter.rewrite_from`
to regenerate the `.log` files byte-identically to a live run.
Defaults output dir to the pickle's parent directory (where the
live driver would have put them); `--output-dir` overrides.

### `l2_norm` column on the cross-gen view

Smaller cosmetic change flagged by Robert mid-session. The cross-gen
`--pareto-only` view ranks genes by L2 norm but didn't display the
value. Added `l2_norm` column at the end of every row — L2-block
rows read monotonically, per-objective rows let users spot narrow
specialists vs balanced-and-specialist winners at a glance.

Also: `--pareto-only` alone (without `--genes`) silently fell
through to the default `--gens` view. Fixed — it now implies
`--genes`. Combining `--pareto-only` with `--runs` / `--gens` /
`--gens-best` / `--clean-empty-runs` is now an explicit error
rather than a silent flag-drop.

### Test count through session 5

- End of session 4: 327
- Cross-gen view + L2 column + pareto-only implies genes: 333
- Resume UX fixes + mismatch error + orphan archive case: 336
- `delete_run`: 339
- `regenerate_logbook_files`: 345
- `prune_empty_runs` + CLI: 357

## Session 6 deltas — SimCase API consolidation

One focused thread: collapse `SimCase.template_values` and
`SimCase.context_extra` into a single `SimCase.case_data` field.

### Why the split was a usability rough edge

Robert asked how to provide per-case constants to a property
writer (e.g. temperature-dependent elastic constants). The answer
was conceptually simple — "put them in `template_values` and read
them off `sim_case.template_values` inside the writer" — but only
because `template_values` happened to be visible to the writer
through SimCase passthrough. Meanwhile, `context_extra` was the
field for "things the path resolver needs," visible to the writer
in a different way (`layout.ctx.extra`). The semantic was:

- `template_values` — read by the templater; ALSO visible to the
  writer via `sim_case.template_values`
- `context_extra` — read by the path resolver; ALSO visible to
  the writer via `layout.ctx.extra`

Two fields, both functionally per-case constants, distinguished
only by which downstream consumer reads them. This forced the
user to remember the decision tree:

1. Will it appear as `%%key%%` in a template? → `template_values`
2. Will it appear as `{key}` in a path pattern? → `context_extra`
3. Both? → `template_values` works, but path resolver doesn't
   see it…

The decision tree itself is the smell. Same data, two boxes,
arbitrary partition.

### What changed

`SimCase.case_data` replaces both fields. The single mapping
flows to all three consumers:

- the **templater** for `%%key%%` substitution
- the **path resolver** for `{key}` substitution
- the **property writer** via `sim_case.case_data`

One dict, three readers. No decision tree.

The path resolver is still strict about missing keys (it raises
`KeyError` with a helpful "available: [...]" message), so typos
still surface — the merge doesn't sacrifice that safety net.

### Files touched

- `workflow_common/problem.py` — field rename, docstring rewrite
  (Fields section, Example block, Problem class docstring example
  using `case_data=`). Two `extra=dict(sim_case.context_extra)`
  call sites changed to `extra=dict(sim_case.case_data)`. The
  templater-values build site simplifies because `ctx.extra` is
  now populated from the same source — kept the existing
  `setdefault` loop for defensive clarity but it's a no-op in
  practice now.
- `workflow_common/case_setup.py` — `CallablePropertyWriter`
  docstring example updated to read from `sim_case.case_data`.
- `workflow_common/MIGRATION.md` — Step 3 ("DEP_UNOPT") rewritten
  to introduce `case_data` as the single home with a brief
  historical note about the previous two-field design.
- `workflow_common/ARCHITECTURE.md` — SimCase prose updated.
- `examples/nsga3_calibration.py` — large per-case-constants
  comment block rewritten (was a 3-row table, now a unified
  description). Two prose comments fixed. Two SimCase definitions
  updated. write_properties docstring updated.
- `workflows/optimization/nsga3_driver.py` — two docstring
  examples updated.
- `tests/test_problem.py` — appended new behavioral test
  `test_case_data_visible_to_templater_path_resolver_and_writer`
  that pins the merged-field design end-to-end. The test runs a
  real Problem with a working_dir_pattern that includes
  `{rve_name}` and a templater target that references
  `%%temperature_k%%`, then asserts: (1) the case dir lands at
  the expected path (proves resolver consumed case_data), (2)
  the rendered file content carries the substituted value
  (proves templater consumed case_data), (3) the writer's
  capture-dict shows both fields read off `sim_case.case_data`
  (proves writer consumed case_data). If anyone ever re-splits
  the field, this test fails because all three flow paths
  exercise from one dict.
- `tests/test_case_setup.py` —
  `test_callable_property_writer_receives_sim_case` updated to
  use `case_data` and assert on it.
- `AUDIT_PLAN.md` — Finding 3's design references updated to
  `case_data`.

No backward-compat alias was added. Robert is the only user;
hard rename keeps the API surface clean for incoming users
who'll never have seen the old names.

### Test count through session 6

- End of session 5: 357
- + new behavioral test pinning case_data flow: 358

## Session 6 deltas — addendum: case_data design clarifications

After the rename landed, two follow-on improvements based on
Robert's feedback that the example was steering users wrong:

### Per-case constants live IN case_data, not in the writer

The illustrative example in `examples/nsga3_calibration.py` had
shown a Python `ELASTIC_TABLE = {temp: {c11, c12, c44}}` lookup
table inside `write_properties`, with the writer indexing by
`sim_case.case_data["temperature_k"]`. That's the wrong split:
it splits per-case data between two places (the SimCase
definition and the writer's source code), forcing users who
add a new case to edit two files.

The corrected guidance: **anything that varies per case goes
directly into that SimCase's `case_data` dict**. The writer
just reads the keys. No Python lookup tables, no logic that
maps cases to values.

```python
sim_cases = [
    SimCase(case_data={
        "temperature_k": 298.0,
        "c11": 168.4, "c12": 121.4, "c44": 75.4,
        # ... loading + path-pattern fields ...
    }, label="cold"),
    SimCase(case_data={
        "temperature_k": 600.0,
        "c11": 156.0, "c12": 117.0, "c44": 72.0,
    }, label="hot"),
]
```

The example's illustrative comment block, the SimCase docstring
in `problem.py`, and the matching MIGRATION.md walkthrough all
got rewritten to show this pattern. Every reference to the
old "lookup table inside the writer" pattern was removed.

### Templater is template-driven, not data-driven

Robert flagged that case_data shouldn't fail when it carries
keys the master template doesn't reference. Verified the
existing behavior — `render_template`'s `strict=True` mode
already only complains about template `%%key%%` placeholders
that have NO matching value in the supplied dict; extra keys
in the dict are silently ignored. So Ask 2 was already met
at the mechanism level.

What WASN'T great: the error message from
`UnresolvedPlaceholderError` says "available keys: [...]" but
doesn't tell the user that the missing value should be added
to `case_data` specifically. From the user's point of view
they see "%%c11%% has no value" and have to figure out where
to add c11. The right place is the SimCase's `case_data` dict
— the framework knows this; the template machinery doesn't.

Fix: `Problem._dispatch_one_case` now wraps `templater.render`
in a try/except that catches `UnresolvedPlaceholderError`,
preserves the original message (key name + available-keys
list), and appends a one-line hint pointing the user at
`SimCase(label=..., case_data=...)` for the specific case.

This puts the "what to fix" hint exactly where the
case-specific context lives, without coupling the generic
`render_template` machinery to the Problem layer's
abstractions.

### Test count through session 6 addendum

- Start of addendum: 358 (after the rename)
- + extra-keys-in-case_data-don't-fail test: 359
- + missing-template-key-helpful-error test: 360

## Session 7 deltas — Plotting example, public selection API

Two threads in this session, both driven by Robert's request for
"a way to plot the optimized cases vs the experimental data":

### Promoting inspect_archive's selection logic to a public API

The `inspect_archive` CLI had several internal helpers
(`_pareto_history_rows`, `_dedup_on_gene_vector`, `_pick_run`,
`_collect_all_genes`, `_resolve_archive_path`) that contained the
exact logic any post-run analysis tool would want: locate the
archive, pick a run, top-N rank by L2 / per-objective. Robert's
plotting ask would have meant re-implementing that, which is the
worst kind of duplication — same math in two places that drift
apart over time.

Fix: promote five helpers to public API on
`workflows.optimization.inspect_archive`:

- `RankedGene` — new public dataclass carrying a `GeneRecord` plus
  rank/category/score metadata. The `GeneRecord` keeps the archive
  lookup keys (birth_gen/birth_gene); the metadata lets a consumer
  know why this gene was selected and color/label accordingly.
- `select_top_genes(genes, *, objective_labels, top_n,
  categories=("l2","per_objective"), dedup=True)` — the headline
  selector. Returns `Dict[str, List[RankedGene]]` keyed by category
  name. The CLI's `_pareto_history_rows` was rewritten to call this
  and just format the row dicts on top.
- `pick_run(archive, run_id=None, *, latest_if_none=True)` — picks
  a run; raises `ValueError` rather than the CLI's `SystemExit(1)`,
  so library callers can catch it cleanly.
- `dedup_on_gene_vector(genes)` — drops duplicate gene-vectors
  (caused by NSGA-III elitism carrying winners across generations).
  First-sighting-wins ordering preserved.
- `collect_all_genes(archive, run_id)` — public name for
  `archive.load_all_genes`.
- `resolve_archive_path(path, default_name)` — public alias of
  the existing `_resolve_archive_path` (which already had a clean
  injectable signature for testability).

The CLI now calls into these. Tests pin the new public contracts
independently of the CLI behavior tests.

### `examples/plot_solutions.py` — interactive plotter

Single-file example: ~600 lines including docstrings, importable
from notebooks AND runnable as `python examples/plot_solutions.py
calibration_run --top 10 ...`. Composes the public selection API
with `ArchiveDB.load_case_outputs` and `StressStrainExtractor` to
produce:

1. **Headline figure** — one subplot per SimCase, top-N simulated
   stress-strain curves overlaid on the experimental reference.
   Curves are colored by rank using a red→blue colormap so rank 0
   is visually obvious.
2. **Slider** below the subplots — sets the visible rank threshold
   K. Ranks < K are full opacity; ranks ≥ K fade to 7% opacity.
   Lets users interactively narrow focus from "show me the top 10"
   to "show me the top 3" without re-running.
3. **Click-to-show parameter panel** — clicking any simulated curve
   prints that gene's parameter values + fitness + birth coordinates
   in a monospace text strip below the slider. Uses matplotlib's
   `pick_event` with a 5-pixel tolerance.
4. **Pareto-front side plot** (`--pareto i,j`) — 2-D scatter for the
   chosen objective pair with the L2-closest gene highlighted in
   red. Reuses `workflow_common.postprocess.plot_pareto_front`.

Three ranking modes: `l2` (default — closest to utopian origin),
`objective` (best on a single named objective), `last-gen` (every
individual in the final generation, no ranking). The objective
specifier accepts both indices and label strings.

The plotter reads exclusively from the SQLite archive — gene
records via `load_all_genes`, simulation outputs via
`load_case_outputs`. The on-disk case directories don't need to
still exist; the archive carries everything needed to reconstruct
stress-strain curves. This was specifically by design — the
archive's `case_outputs` table was added in an earlier session
precisely so post-run analysis would survive workspace cleanup.

### Test count through session 7

- End of session 6: 360
- Promotion: select_top_genes / pick_run / dedup / collect / resolve
  contract tests: 372
- Plot solutions integration tests (mode selection, sim_case probe,
  CLI error handling): 382

## Session 8 — Plotter polish + Pareto interactivity

Robert hit several bugs in the post-run plotter and asked for the
Pareto plot to gain the same interactive affordances as the headline
overlay.

### Bug fixes in `examples/plot_solutions.py`

* **`--objective N` no longer silently runs L2 ranking.**
  Previously `--mode l2` was the default and `--objective` had no
  effect unless the user also explicitly passed `--mode objective`.
  This was the root cause of "changing --objective shows the same
  plot." `main()` now auto-promotes `--mode l2` → `"objective"` when
  `--objective` is supplied. `--objective` with `--mode last-gen` is
  reported as an incompatible combination (last-gen mode has no
  ranking step, so the flag would have nothing to apply to).

* **`--no-overlay` flag added.** Previously the headline overlay was
  unconditional; passing `--pareto` produced two figures whether the
  user wanted both or just the Pareto. Now `--no-overlay --pareto i,j`
  produces only the Pareto plot. `--no-overlay` alone reports an
  error rather than silently exiting with no figures.

* **`top_n` honored on the Pareto plot.** The previous implementation
  scattered every finite-fitness gene regardless of `--top`. Now
  `top_n > 0` restricts the scatter to the N lowest-L2 genes (using
  `select_top_genes` for consistency with the inspect-archive
  CLI's `--pareto-only` table). `top_n=0` (default) plots every
  rank-0 gene — the actual Pareto front — with gene-vector dedup so
  elitism-carried duplicates don't pile on.

* **Slider skip when `n_total == 1`.** A single-solution overlay used
  to trigger a matplotlib "identical low/high xlim" warning when
  building the slider with `valmin=valmax=1`. The slider would also
  be functionless. Now the slider is skipped and its axes hidden
  when there's nothing to toggle.

### Pareto interactivity

The Pareto figure is now a 2-column layout: scatter on the left,
response inset on the right.

* **Clickable points.** Each scatter point has `picker=5`. Click
  fires a `pick_event` that updates a parameter panel below the
  plots (gene vector, fitness, birth coordinates, L2 norm) AND
  redraws the response inset.
* **Response inset.** The right axes shows the clicked gene's
  simulated stress-strain curves (one line per SimCase, colored by
  `tab10`) overlaid with the experimental references (dashed,
  same color per SimCase). Initially populated with the L2
  winner's response.
* **L2-winner ring** stays — same red ring on the L2-closest point,
  with the colorbar of L2 norms making "balance vs specialization"
  legible at a glance.
* **Archive-first experimental data** — the inset reuses
  `_resolve_experimental_for_case` so users get the right
  reference curve per SimCase without supplying CSV paths.

### Slope plotting (already in session 7)

The overlay plot has both stress-strain and slope-strain rows.
Slope is computed via `np.gradient(stress, strain)` for both
simulated and experimental curves so the values match what the
slope objective scored against. Toggleable via `show_slopes=False`.

### Test count through session 8

* Start of session: 390
* Pareto top_n / inset / picker / experimental tests: +5 → 395
* CLI auto-promote / no-overlay / l2-vs-objective title tests: +5 → 400
* Archive-first experimental + slope helper + overlay no-CSV: +3 → 403

Total: **403 passing** (33 driver + 370 non-driver).

## Session 8 addendum — Backward compat for archives without `experiments`

Robert hit `sqlite3.OperationalError: no such table: experiments`
running the plotter against an archive created before the
`experiments` table was added (older code, real .db file on his
machine). He had supplied `--experimental` CSV paths as the fallback,
but the load_experiment call crashed before the fallback was reached.

Root cause: `ArchiveDB.open()` only runs the schema-creation script
on writable opens (a read-only SQLite connection can't modify the
schema, by design). So a read-only open of a pre-experiments-table
archive never gets the `IF NOT EXISTS` table-creation statement
executed; the table is genuinely absent, and the first SELECT
against it raises `OperationalError`.

Fix: `load_experiment` and `list_experiments` now check for the
table's existence via a small `_has_table(name)` helper before
querying. Missing → return `None` / `[]` respectively, matching the
contract those methods already had for "no row found." The plotter's
`_resolve_experimental_for_case` therefore reaches its CSV-fallback
branch cleanly when the archive is too old to have stored
experimental data.

Writable opens are unaffected: they run the full schema script on
every open, so missing tables get created on first write. A user
who opens a pre-experiments archive in write mode and calls
`record_experiment` gets the table created on the fly with no
manual migration step required.

### Test count through session 8 addendum

- Start: 403
- Archive-level missing-table tests (load returns None,
  list returns [], writable open creates table): +3 → 406
- Plot-level tests reproducing Robert's command line
  (resolve falls back to CSV, returns None without fallback,
  full main() flow): +3 → 409

Total: **409 passing** (33 driver + 376 non-driver).

## Session 9 — Optimization windowing, generic strain, plotter polish

Robert's feedback covered four threads:

1. Slope plots needed semi-log y because elastic-vs-plastic slopes
   span orders of magnitude and the optimized curves were
   visually invisible compressed against the elastic spike.
2. Optimized curves weren't appearing on slope axes (turned out to
   be the same compression issue — the lines were drawn but
   indistinguishable from the x-axis at linear scale).
3. The old framework's `minmax_strain` field was documented in
   `case_data` but unwired anywhere — optimizer was scoring against
   the full curve, often dominated by elastic regime instead of
   the plastic regime users actually care about.
4. `StressStrainExtractor`'s field names baked def-grad-specific
   assumptions that mismatched users with sims that already write
   strain measures (Lagrange/Euler) directly.

### Extractor refactor (workflow_common/objectives.py)

* `def_grad_output` → `strain_source_output`
* `def_grad_column` → `strain_source_column`
* `axial_minus_one` → `biot` (it IS Biot strain in 1D; the old
  name was inscrutable)
* `log_axial` → `log`
* New `direct` strain source: reads the column verbatim, no
  transformation. Use case: sim already wrote
  `avg_lagrangian_strain.txt` and the user wants `E33` straight.
* New `window: Optional[Tuple[float, float]]` field — crops
  ``(strain, stress)`` to a strain interval. Compares against
  ``|strain|`` so the same value works for tension and compression
  without sign-handling at the call site.
* Legacy aliases (`f11_minus_one`, `log_f11`) and their
  `DeprecationWarning` machinery removed entirely. No deprecation
  period — Robert is the only user.

### Optimization windowing wired through

`StressStrainExtractor.window` is the single point of windowing.
Both `StressStrainObjective` and any custom evaluator that uses an
extractor get cropping for free, since they consume the
extractor's output. The example `nsga3_calibration.py` reads
`sim_case.case_data["minmax_strain"]` and passes it to the
extractor as `window=`. The custom `_StdNormalizedStress/Slope`
evaluators dropped their `desired_strain` parameter — that was
half-windowing (upper bound only) and is now subsumed by the
extractor's `window`.

### Archive: minmax_strain column on experiments

Added `minmax_strain TEXT` column storing JSON `[lo, hi]` so
plotter can shade the optimized region against the full curve.
Both sides may be null for unbounded.

* `record_experiment(..., minmax_strain=)` — new optional kwarg.
* `load_experiment_window(run_id, sim_case_idx)` — new method
  returning the stored `(lo, hi)` tuple. Separate from
  `load_experiment` so callers that only need the window
  (e.g. plotter shading) skip the DataFrame deserialize.
* Backward compat for archives missing the column:
  `load_experiment_window` returns None gracefully via a new
  `_has_column` helper.
* Forward compat for archives missing the column on writable
  open: a new `_migrate_add_missing_columns` runs after the
  schema script and `ALTER TABLE`s the column in. Idempotent
  (guarded by `_has_column`) so reopens of fresh archives
  don't raise "duplicate column."

### Driver wiring

`_record_experiments_from_problem` now reads
`SimCase.case_data["minmax_strain"]`, defensively unpacks it as a
2-tuple of floats/Nones, and passes it to `record_experiment`.
Bad input (e.g. a string) logs a warning and drops the window —
the experiment still gets recorded so plotting keeps working.

### Plotter (examples/plot_solutions.py)

* Slope axes now use `set_yscale("symlog", linthresh=...)` with
  `linthresh = 0.001 * max(|slope|)`. Resolves both the "elastic
  spike compresses everything" issue and noise-induced sign-flip
  brittleness in one shot.
* Window shading: `axvspan` on both stress and slope axes for
  each SimCase whose archive entry has a non-None window.
  Translucent green (alpha=0.08), zorder=0 so it sits behind
  curves. Mirrors to negative-strain region for compression
  loadings (sign detected from experimental or first sim curve).
* `per_case_window` collected alongside `per_case_curves` and
  `per_case_exp` inside the same `with ArchiveDB(...)` block so
  there's only one connection lifecycle to manage.

### Tests added

* Extractor: `direct` strain, window cropping (tension), window
  cropping (compression via abs), exclude-all error, inverted
  bounds error. (5)
* Archive: window persistence, unbounded sides, missing-record
  returns None, missing-column returns None, writable open
  migrates, migration idempotent on fresh archive. (6)
* Driver: case_data minmax_strain plumbed to archive (with
  multiple SimCase variants), malformed minmax_strain handled
  gracefully. (2)
* Plot: optimized curves present on slope axes, slope yscale is
  symlog, window shading drawn when archive has it, no shading
  when window is None. (4)

Existing tests updated where API names changed: `test_objectives`
log/biot renames + drop legacy-alias test, `test_example_imports`
drop `desired_strain` arg, `_make_result_set` helper accepts a
custom output name for direct-strain tests.

### Test count through session 9

- Start: 409
- Extractor (direct strain + window): +5 → 414
- Archive (minmax_strain column + migration): +6 → 420
- Driver (case_data plumbing): +2 → 422
- Plot (slope curves + symlog + shading): +4 → 426

Note: also dropped the legacy-aliases test on the old field
names, so net delta to test_objectives was +5 minus 1 = +4. Net
total **425 passing** (35 driver + 390 non-driver).

## Session 10 — Pareto interactivity fixes, archived extractor configs

Robert reported the new Pareto figure was clean but had three real
issues plus one usability nit:

1. Bug — clicking a Pareto point only plotted "1 of 2 SimCases" in
   the inset response. The same SimCase failed regardless of which
   Pareto pair was selected, so the bug wasn't in pair-selection
   logic.
2. The headline overlay still showed alongside `--pareto`. Robert
   wanted bare `--pareto` to mean "only the Pareto."
3. The L2 winner ring used full-objective L2, not the projected
   pair's L2 — surprising when the user picks 2 of N objectives
   and expects the "balanced winner ON THIS PROJECTION."
4. Coloring by full L2 made it hard to see whether a Pareto front
   was a true tradeoff curve. Different metrics highlight
   different aspects (proximity to Y=0 plane, X=0 plane,
   asymmetry between specialists and compromises).

### Root cause for the SimCase-disappearing bug

The plotter's click handler built a default
`StressStrainExtractor()` for every SimCase. Robert's calibration
used `strain_source="time_rate"` with case-specific `strain_rate`;
the default uses `strain_source="biot"` against `F33`. For ANY
SimCase whose archived `case_outputs` shape didn't fit the default
Biot-strain expectations (or whose curve was indistinguishable
from noise after default extraction), `_extract_curve` returned
None silently — and the SimCase vanished from the inset.

The architectural fix is to archive the extractor config the
optimizer actually used. The plotter consults the archive first,
so a user is guaranteed to see plots that match what the optimizer
scored against — not whatever the plotter's default would produce.

### Archive: extractor_config column on experiments

* New `extractor_config TEXT` column on the `experiments` table.
  JSON-encoded dict from `StressStrainExtractor.to_dict()`.
* `record_experiment(..., extractor_config=)` accepts the dict.
* `load_extractor_config(run_id, sim_case_idx)` returns the dict
  or None (missing column / row / null value all fold to None).
* Migration entry added to `_migrate_add_missing_columns` so old
  archives gain the column on writable open.

### Extractor JSON contract

* `StressStrainExtractor.to_dict()` — serializes all 8 fields,
  converting `window: tuple` to `[lo, hi]` for JSON safety.
* `StressStrainExtractor.from_dict(d)` — tolerant of missing keys
  (defaults fill in) and extra keys (silently ignored), so
  forward-compat works in both directions: an older archive on
  newer code OR a newer archive on older code both load.

### Driver wiring

`_record_experiments_from_problem` now also reads
`evaluator.extractor.to_dict()` if the evaluator exposes
`.extractor`. Custom evaluators without that attribute (or whose
extractor doesn't have `to_dict`) get a warning and the experiment
is recorded with extractor_config=NULL. Plot-time fallback to a
default extractor still works in that case.

### Plotter resolver

New `_resolve_extractor_for_case(archive, run_id, sc_idx, factory=)`:

  archive's stored config → user-supplied factory → default

The window field is **stripped** from any archived extractor
before plot-time use: the plotter shows full curves with the
window shaded as an overlay, so cropping at extraction would hide
exactly the context users want to see.

Both `plot_top_solutions_overlay` and the Pareto plotter use the
resolver. The Pareto plotter also hoists resolution outside the
gene loop (extractors are per-SimCase, not per-gene, so building
them per-gene was wasted work).

### Pareto subset L2 + color modes

The L2 winner ring now lands on the gene with minimum
**subset L2** (sqrt(x² + y²) on the projected pair) — answering
"best balanced compromise on what I'm looking at" rather than
"globally best across all objectives."

New `--color-by` flag with five options:

* `subset_l2` (default) — distance from utopia in the projected
  plane. Iso-curves are circles. Uniform color along the front
  signals a true tradeoff curve.
* `full_l2` — distance across all objectives. Catches points
  that look balanced on the projection but are bad on the
  hidden objectives.
* `x` / `y` — single axis. Iso-curves are vertical/horizontal
  lines. Reveals proximity to Y=0 / X=0 plane.
* `asymmetry` — `|x - y| / (x + y)`. Zero = perfectly balanced
  on the pair. Highlights specialists vs compromises.

### CLI inversion

* `--no-overlay` removed.
* `--overlay` added — explicit opt-in to keep the overlay alongside
  `--pareto`.
* Bare `--pareto i,j` → only Pareto figure produced.
* Bare command (no `--pareto`) → still shows overlay (backward
  compat with the "user just wants to look at results" workflow).

### Tests added

* Archive (5): extractor_config persists, missing record returns
  None, null record returns None, old archive without column
  returns None, writable open auto-migrates.
* Objectives (4): JSON round-trip with tuple→tuple, missing-keys
  tolerance, extra-keys tolerance, window=None round-trip.
* Driver (2): extractor config from evaluator flows through to
  archive, evaluators without `.extractor` skip cleanly.
* Plot (12): subset-L2 winner ring (not full-L2), color_by exact
  values for subset_l2 and full_l2, x/y/asymmetry render OK,
  invalid color_by raises, Pareto inset shows all SimCases on
  initial draw, archived extractor used when present (with
  custom column names that would fail the default), bare
  `--pareto` skips overlay, `--pareto --overlay` shows both,
  no-pareto-no-overlay still shows overlay.

The autouse `_agg_backend` fixture now also runs `plt.close("all")`
on test teardown — without that, the figure-leak warning escalated
once test count exceeded matplotlib's 20-figure threshold.

### Test count through session 10

- Start: 425
- Archive (extractor_config column + migration): +5 → 430
- Objectives (to_dict / from_dict round-trip): +4 → 434
- Driver (extractor config plumbing): +2 → 436
- Plot (subset L2 + color modes + click-handler + extractor
  resolver + CLI inversion): +12, minus 2 dropped `--no-overlay`
  tests → **net +10** → **446**

Total: **446 passing** (37 driver + 409 non-driver).

## Session 11 — Save the simulation curve too

Robert's response to session 10 had two parts: (1) the diagnosis
of "only 1 of 2 SimCases plotted" was incomplete (the same
default-extractor logic worked for the overlay so something else
was going on, and the architectural fix is still right but my
narrative was wrong), and (2) more substantively, the framework
should save the SIMULATION-side (independent, dependent) curve
alongside the experimental data. Storing only the experimental
side and reconstructing the sim side after the fact loses
information and makes re-analysis harder than it needs to be.

Robert called out three motivations:

* Direct retrieval beats reconstruction. With the sim curve
  persisted, post-run analyses pull it back as data, not as
  something rebuilt by re-running the extraction pipeline (which
  could drift if extractor logic ever changes).
* Re-running analyses with different metrics (RMSE → MAE,
  different windowing, different weighting) becomes a query, not
  a re-extraction. Especially valuable when deciding to evaluate
  parameter sets against new criteria after the fact without
  re-running the whole optimization.
* Bayesian / surrogate model training. Mapping
  ``gene_vector → response_curve`` lets surrogate-driven search
  speed up future optimizations; that requires the curves to be
  available and stable.

Critical detail: store the FULL extraction range, not the
windowed range. The optimizer's window crops which region the
metric was computed over, but downstream analyses might want
data outside that region. Strip the window before extraction
when archiving.

### Schema: case_curves table

New table keyed on ``(run_id, birth_gen, birth_gene, sim_case_idx)``
matching the case_outputs key shape. Columns:

* ``independent_blob`` — pickled np.ndarray (typically strain).
* ``dependent_blob`` — pickled np.ndarray (typically stress).
* ``independent_label`` / ``dependent_label`` — TEXT hints; the
  framework defaults to "strain" / "stress" but the columns let
  future extractor types (thermal, creep) carry their own labels.
* FK cascade on ``runs(run_id)``.

Why a separate table rather than a column on case_outputs:
case_outputs holds raw simulator output (avg_stress.txt etc.),
which is an INPUT to extraction. The extracted curve is the
comparison target the optimizer scored against. Different
conceptual layer; different table.

### Archive API

* ``record_case_curve(run_id, *, birth_gen, birth_gene, sim_case_idx, independent, dependent, independent_label=None, dependent_label=None)``.
  Validates 1-D shapes and matching length.
* ``load_case_curve(run_id, *, birth_gen, birth_gene, sim_case_idx)``
  returns ``(ind, dep, ind_label, dep_label)`` or None. Tolerates
  missing table on read-only opens of older archives.
* ``discard_from_generation`` extended to also delete case_curves
  rows past the resume point.
* ``delete_run`` cascades automatically via FK.

### Problem-level wiring

``Problem._archive_case`` now calls a new
``_archive_case_curve(ctx, rs)`` after writing case_outputs.
Logic:

* Cache the extractor per ``sim_case_idx`` on the Problem
  instance (lazy, populated on first hit). The extractor is
  found by walking ``objective_specs`` for the first one with
  ``spec.sim_case == sim_case_idx`` AND a usable
  ``.extractor`` attribute.
* Strip the window from the cached extractor via the
  ``to_dict`` / ``from_dict`` round-trip so the archived curve
  covers the full range. Falls back to using the original
  extractor if to_dict isn't available — better windowed than
  nothing.
* Run the extractor, archive the result. Failures log at debug
  (not warning) since the evaluator's own failure handler is
  the primary path for reporting extraction problems; this is
  a bonus archive write.

Custom evaluators without an ``.extractor`` attribute leave
case_curves empty for that SimCase. The plotter handles that
by falling back to ``case_outputs`` + extraction.

### Plotter

New ``_load_or_extract_curve`` helper:

  archived case_curve → re-extract from case_outputs → None

Both ``plot_top_solutions_overlay`` and the Pareto plotter
route through this helper. Old archives without case_curves
hit path 2 and continue working unchanged.

### Tests added

* Archive (7): record/load round-trip, shape validation, missing
  record returns None, missing table returns None on read-only,
  collision-replace, FK cascade on delete, discard_from_generation.
* Driver (2): full-range curves archived per (gene, sim_case)
  with window stripped, custom evaluators without extractor
  skip cleanly.
* Plot (1): plotter prefers archived curves over re-extraction.

### Test count through session 11

- Start: 446
- Archive (case_curves CRUD + cascade + discard): +7 → 453
- Driver (curves archived from Problem hook): +2 → 455
- Plot (archived-curve preference): +1 → 456

Total: **456 passing** (39 driver + 417 non-driver).

## Session 12 — Sign correction for compressive loading

Robert reported the plotter rendering a compressive case with
positive strain alongside negative stress — strain axis flipped
relative to the stress axis. Root cause: the example calibration
script applied ``abs()`` to ``case_data["strain_rate"]`` before
constructing the extractor, so the extractor's
``strain = strain_rate * time`` produced positive strain even on
runs the user had configured as compression (signed strain rate).
Stress (Szz) was naturally negative. The two axes ended up
mirrored relative to each other.

### Three layers of fix

**1. Example fix (preventive).** Drop ``abs()`` on the strain
rate so it carries sign through to the extractor unchanged.
Evaluators that score on magnitudes (the example's
``_StdNormalized*Evaluator`` family) already apply ``np.abs``
themselves before computing RMSE, so the sign change doesn't
affect optimization scores. The fix only changes what gets
passed to the extractor for archiving and plotting.

**2. Post-processing fix (Robert's specific ask).** This is
where users notice the bug — plots, not run logs. The plotter's
``_load_or_extract_curve`` helper now accepts an
``experimental_reference`` kwarg ``(exp_ind, exp_dep)``; when
supplied, both the loaded simulated arrays are sign-matched
against the experimental columns via
``match_sign_to_reference``. Both ``plot_top_solutions_overlay``
and the Pareto plotter resolve experimental data BEFORE the
gene-curve loop and thread it through, so the sign-correction
sits at the single curve-loading point. Old archives written
before the save-off fix get corrected at plot time without any
upgrade path needed.

**3. Save-off fix (defensive, archive-side).** ``Problem.
_archive_case_curve`` now also runs the same correction before
calling ``record_case_curve``. The same evaluator that supplied
the extractor must also supply ``.experimental`` and the
matching column-name attributes (``experimental_strain_col`` /
``experimental_stress_col``, defaulting to ``"strain"`` /
``"stress"``). Resolved together via a new
``_resolve_case_curve_state(sim_case_idx) ->
Tuple[extractor, exp_df]`` so both come from the same evaluator
choice — keeps them consistent. Result: archived ``case_curves``
rows are correctly-signed for any downstream consumer
(surrogate-model trainers, alternative post-processing scripts),
not just the bundled plotter.

### The helper itself

Originally drafted with a "compute dominant signs of both arrays,
multiply by -1 if they disagree" approach. Robert pointed out
``np.copysign`` is the right primitive — simpler and conveys
intent more clearly. Final form:

```python
def match_sign_to_reference(sim, reference):
    if sim.size == 0 or reference.size == 0:
        return sim
    ref_dominant = float(reference[int(np.argmax(np.abs(reference)))])
    if ref_dominant == 0.0:
        return sim
    return np.copysign(sim, ref_dominant)
```

Picking a single representative sign from the reference (its
max-magnitude value) sidesteps numerical noise around zero —
a Voce stress curve's max-magnitude point is at peak load,
unambiguous in sign. Then ``np.copysign(sim, scalar)`` forces
every element of ``sim`` to that sign while preserving
magnitude. For monotonic loading (the framework's target) this
is correct; for cyclic loading where the curve genuinely
crosses zero this would over-correct, but the framework
doesn't currently target cyclic fits.

The helper is exported from ``workflow_common`` so external
code can apply the same correction.

### Tests added

* Objectives unit (7): compression flip, already-aligned no-op,
  unaligned lengths handled, empty sim, empty ref, all-zero ref,
  noise-near-zero robustness.
* Driver (2): sign-correction at save-off given a compression
  experimental reference, no correction when experimental data
  is absent.
* Plot (2): sign-correction at plot time given wrong-signed
  archived curves, no correction when experimental data is
  absent.

### Test count through session 12

- Start: 456
- Objectives helper: +7 → 463
- Driver save-off: +2 → 465
- Plot post-processing: +2 → 467

Total: **467 passing** (41 driver + 426 non-driver).

## Session 13 — Use the smoothing module, in the right direction

Robert called out two related issues with the example evaluators
``_StdNormalizedStressEvaluator`` / ``_StdNormalizedSlopeEvaluator``
in ``nsga3_calibration.py``:

1. **Wrong direction.** They were interpolating sim onto exp's
   strain grid (``np.interp(target_strain=exp, sim_strain,
   sim_stress)``). The original ExaConstit code interpolated
   experimental data onto the simulation's grid — which is the
   right way: the optimizer is computing error at sim's own
   sample points.

2. **Bypassing the framework.** Used raw ``np.interp``
   instead of the framework's smoothing module
   (``PchipSmoother``, ``ArcLengthSmoother``,
   ``auto_smoother``). The smoothing module exists exactly for
   this — picks PCHIP for monotonic data, ``ArcLengthSmoother``
   for non-monotonic, and the ``sample_at`` method gives
   "evaluate at these target x's" semantics with PCHIP's
   no-overshoot guarantee.

### Evaluator rewrite

Both ``_StdNormalized{Stress,Slope}Evaluator.evaluate`` now:

* Mask SIM (not exp) to the common strain range — sim's sample
  points become the comparison grid.
* ``PchipSmoother(strict_monotonic=False).sample_at(exp_strain,
  exp_stress, sim_strain_in)`` brings exp onto sim's grid via
  PCHIP. ``strict_monotonic=False`` accepts mild noise in
  exp_strain (auto-sort); it doesn't handle genuinely
  non-monotonic data, which the framework's ``ArcLengthSmoother``
  is for. Mechanical-test exp data is cleaned and clipped before
  calibration anyway, per Robert.
* RMSE / std computed on the sim-grid-aligned arrays.

### Sign-correction sites updated

The post-processing plotter and the archive-time
``Problem._archive_case_curve`` previously used a
``match_sign_to_reference`` helper that wrapped ``np.copysign``
with a parametric ``np.interp`` resample. Robert pointed out
that's the wrong tool — the framework already has
``PchipSmoother`` for the same job. Both sites updated to:

```python
smoother = PchipSmoother(strict_monotonic=False)
exp_dep_at_sim = smoother.sample_at(
    np.abs(exp_ind), exp_dep, np.abs(sim_ind),
).y
sim_dep = np.copysign(sim_dep, exp_dep_at_sim)
sim_ind = np.copysign(sim_ind, exp_ind[-1])  # scalar from last point
```

Element-wise ``np.copysign`` on the dependent axis preserves
cyclic structure (each sim point inherits the sign of the
corresponding exp point at the same strain magnitude). The
independent axis uses a scalar sign drawn from exp's last
strain value — unambiguous for monotonic loading; cyclic data
ending back at zero would degenerate, but the framework
targets monotonic loading near-term.

The thin ``match_sign_to_reference`` helper has been removed —
``np.copysign`` doesn't need a wrapper.

### Tests changed

* Removed: 7 ``match_sign_to_reference`` unit tests (helper gone).
* Added: 2 evaluator-direction tests pinning the
  PchipSmoother-based score against a hand-computed
  expected value, ensuring the rewrite uses exp→sim alignment
  and not sim→exp.
* The 2 existing driver tests (sign-corrected curves at
  save-off) and 2 plot tests (sign-corrected curves at plot
  time) continue to exercise the new PCHIP-aligned path —
  they pass unchanged.

### Test count through session 13

- Start: 467
- Removed match_sign helper tests: -7 → 460
- Added evaluator-direction tests: +2 → 462

Total: **462 passing** (41 driver + 421 non-driver).

## Session 14 — Slope plot truncation

Robert's report: with slope data like ``[strain=-1e-6..-0.12, slope=-128, -40, +210, +128, ..., 0.7]`` the slope subplot only shows "the very start" — the rest of the curve appears unplotted. Confirmed with and without symlog. Only ``abs(slope)`` made the rest visible, which Robert explicitly didn't want.

### Diagnosis

Matplotlib was actually plotting all 133 points. Verified with standalone repro on Robert's exact arrays — every point made it onto the canvas. The problem was **visualization**, not data: an elastic spike (~+210) and a brief start-of-test numerical-noise spike (~-128) dominated the auto-scaled y-axis. The plastic-regime bulk (slope ≈ 0.7-3, the actual response of interest) got compressed into a hairline near the chart edge. To Robert's eye, that hairline read as "only the start was plotted." ``abs(slope)`` happened to fix it because the negative spike's contribution to ylim disappeared, doubling the bulk's relative real estate — but at the cost of losing sign information.

Old code (``examples/plot_solutions.py``):

```python
max_abs = float(np.max(np.abs(concat)))
linthresh = max(max_abs * 0.001, 1e-9)
slope_ax.set_yscale("symlog", linthresh=linthresh)
```

Two failure modes baked in: ``max(|slope|)`` lets a single outlier dictate ``linthresh``, and there's no explicit ``set_ylim`` so matplotlib's auto-scale also sizes against the spike.

### Fix

Drive both the y-axis range AND ``linthresh`` from robust percentiles of ``|slope|``, not extremes:

```python
finite = concat[np.isfinite(concat)]
abs_finite = np.abs(finite)
nonzero = abs_finite[abs_finite > 0]
# 90th percentile: captures the bulk Voce response, excludes
# spikes (typically ≤5% of points). 1.5x for headroom.
p90 = float(np.percentile(abs_finite, 90))
bulk_extent = max(p90 * 1.5, 1.0)
slope_ax.set_ylim(-bulk_extent, bulk_extent)
# Linthresh from 5th percentile of nonzero |slope| — keeps the
# bulk in the LOG region, not the linear band.
linthresh = max(float(np.percentile(nonzero, 5)) * 0.5, 1e-9)
slope_ax.set_yscale("symlog", linthresh=linthresh)
```

For Robert's data:
- ``p5(|slope|)`` ≈ 0.77, ``p50`` ≈ 1.43, ``p90`` ≈ 2.54, ``max`` ≈ 210.9
- Old: ``linthresh = 0.21``, ylim auto ≈ ``(-323, +530)`` — bulk visually compressed.
- New: ``linthresh = 0.38``, ylim = ``(-3.81, +3.81)`` — bulk fills the chart; spikes extend off the top and bottom edges, still visible as line segments leaving the visible region.

Crucially the spike data is **not** clipped from the line — matplotlib draws the line in full and just renders the portions outside ``ylim`` past the chart edge. Tested explicitly via ``line.get_ydata().max() > 100`` and ``< -50`` after plotting.

### Tests

Added ``test_plot_overlay_slope_ylim_robust_against_outliers``:
- Builds a sim curve with the failure pattern (elastic + noise spikes plus smooth bulk)
- Asserts symmetric ylim with ``abs_extent < 50`` (bulk-scale, not spike-scale)
- Asserts plotted y-data still includes the spike values (range, not data, was the fix)

### Test count through session 14

- Start: 462
- Added robust-ylim slope test: +1 → 463

Total: **463 passing**.
