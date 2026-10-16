# exaconstit-calibrate

Calibration infrastructure for [ExaConstit](https://github.com/LLNL/ExaConstit):
material-parameter fitting via (U-)NSGA-III on the
[rcarson3/deap](https://github.com/rcarson3/deap) fork, plus a
code-agnostic framework for driving external simulation codes from
Python workflows.

**Current scope:** NSGA-III-based crystal-plasticity parameter
calibration (replaces `ExaConstit_NSGA3.py` + friends).

**Planned scope:** Macroscale yield-surface data generation and Barlat
surface fitting. The framework layer is code-agnostic and already
supports arbitrary drivers; only the yld-specific driver is pending.

**Out of scope:** Full additive-manufacturing challenge-problem
workflows (those live under `workflows/Stage3/` in ExaConstit and
are a separate pipeline).

The package ships two importable names:

- `workflow_common` — the code-agnostic framework (Problem
  orchestrator, ArchiveDB, result readers, backends). No DEAP
  dependency.
- `workflows.optimization` — the NSGA-III driver. Requires the
  DEAP fork. Named `workflows` to preserve import paths from
  migrated pre-refactor drivers.

## Installation

Requires **Python 3.12 or newer**. Most HPC sites are standardizing
on 3.12+; older versions are not supported because the codebase
uses post-3.10 typing idioms (`X | Y`, PEP 604 unions, etc.).

### Core framework only

```bash
pip install exaconstit-calibrate
```

This installs `workflow_common` and `workflows` with `numpy`,
`pandas`, and `scipy` only. No DEAP, no matplotlib, no flux-core.
Suitable for users who want the Problem orchestrator, archive,
and result readers but are plugging in their own optimizer or
using a different plotting stack. `import workflows.optimization`
still works but driver functions raise `ModuleNotFoundError` on
first use (they are lazily imported).

### With the NSGA-III driver (typical user setup)

```bash
pip install "exaconstit-calibrate[nsga3,plot]"
```

Adds the DEAP fork and matplotlib. This is what you want for
material-parameter-calibration workflows and for running the
`run_nsga3` driver.

### Development install

```bash
git clone https://github.com/LLNL/ExaConstit.git
cd ExaConstit/workflows    # or wherever this package lives
pip install -e ".[test]"
python -m pytest tests/ -q
```

`-e` gives you an editable install so your checkout is what gets
imported. `.[test]` pulls in pytest, matplotlib, and the DEAP fork.
Expected output: `254 passed` in 30-45 seconds.

### HPC install (LLNL, ...)

```bash
pip install "exaconstit-calibrate[nsga3,plot,flux]"
```

The `flux` extra pulls `flux-python` for the `FluxBackend`. This
will only succeed on systems where `flux-core` is already
installed (that is, inside a Flux allocation on an HPC system).
On a developer laptop you almost certainly do not want this
extra.

## Quick start

```python
from pathlib import Path
import numpy as np
import pandas as pd
from workflow_common import (
    CaseTemplater, TemplateTarget,
    TemplatePathResolver, TemplatePropertyWriter,
    TextTableReader, TextTableSpec,
    Problem, ProblemConfig, SimCase, ObjectiveSpec,
    StressStrainExtractor, StressStrainObjective,
    LocalBackend, Manifest, ArchiveDB,
    load_experimental_csv,
)
from workflows.optimization import Bounds, RunConfig, run_nsga3

# ... build templater / writer / resolver / reader / evaluator ...

archive = ArchiveDB("opt.db")
archive.open()
run_id = archive.start_run(seed=42, param_names=["yield", "hardening"])

problem = Problem(
    config=ProblemConfig(binary=Path("/path/to/mechanics"), ...),
    param_names=["yield", "hardening"],
    sim_cases=[SimCase()],
    objective_specs=[ObjectiveSpec(evaluator, sim_case=0)],
    templater=templater,
    property_writer=writer,
    resolver=resolver,
    backend=LocalBackend(max_workers=4),
    reader=reader,
    manifest=Manifest("wf/manifest.jsonl"),
    archive=archive,
    archive_run_id=run_id,
)

result = run_nsga3(
    problem,
    bounds=Bounds(lower=np.array([100., 1000.]),
                  upper=np.array([500., 3000.])),
    config=RunConfig(
        n_generations=50, population_size=60,
        unsga3=True, seed=42,
        checkpoint_dir=Path("ck"),
        cleanup_keep_generations=2,       # keep inode count bounded
    ),
)

archive.close()
print("Pareto size:", len(result.pareto_front))
```

See `workflow_common/MIGRATION.md` for a full walk-through of
porting an existing `ExaConstit_NSGA3.py` driver to this package,
and `workflow_common/ARCHITECTURE.md` for the design-rationale
overview, layering diagram, and decision table.

For a runnable side-by-side example that mirrors the old script
top-to-bottom — same parameter bounds, same two-experiment
topology, same GA defaults — with inline comments quoting the
pre-refactor code next to each new-framework equivalent, see
`examples/nsga3_calibration.py`. It's the "show, don't tell"
counterpart to the migration guide.

## Watching a run in progress

During a run, the driver writes two human-readable text files
(tab-delimited, matches the pre-refactor driver's format exactly):

- `logbook1_stats.log` — per-generation avg / std / min / max
  fitness, plus ND / GD / HV for multi-objective runs
- `logbook2_solutions.log` — one row per individual per
  generation with gene vector and fitness

These land in `RunConfig.log_dir`, which defaults to the
`checkpoint_dir` if set, otherwise `cwd`. `less`, `tail -f`, and
`grep` all work. Turn them off with `write_logbook_files=False`.

If you enabled archiving (`ArchiveDB`), a `archive.db` SQLite
file sits alongside. You don't need a SQLite client to read it —
there's a CLI:

```
python -m workflows.optimization.inspect_archive ./run_dir --runs
python -m workflows.optimization.inspect_archive ./run_dir --gens
python -m workflows.optimization.inspect_archive ./run_dir --gens-best
python -m workflows.optimization.inspect_archive ./run_dir \
    --genes --pareto-only --top 5 --format csv > best.csv
python -m workflows.optimization.inspect_archive ./run_dir \
    --clean-empty-runs --dry-run
```

Views:

- `--runs` — every run in the archive
- `--gens` — per-generation summary stats (default)
- `--gens-best` — convergence view. Running best fitness per
  objective, plus the generation the current L2 champion was born
  in. Plateau detection: if `champion_birth_gen` holds steady for
  many rows, the GA is stuck
- `--genes` — individual gene records; capped at 200 rows by
  default (`--limit N` to override, `--limit 0` for no cap). Pair
  with `--pareto-only` for top-N-across-history (top 3 by L2 norm
  plus top 3 per objective, with an explicit `l2_norm` column so
  every row is self-verifying), or with `--gen N --pareto-only`
  for the rank-0 set at one specific generation. Passing
  `--pareto-only` by itself implies `--genes`
- `--clean-empty-runs` — prune runs that have no generations and
  no case outputs (typical residue of aborted debugging sessions).
  Pair with `--dry-run` to preview; `--age-minutes 0` overrides
  the default 60-minute safety gate that protects in-progress
  runs

Formats: `table` (default), `csv`, `json`. Filters: `--run RUN_ID`,
`--gen GEN_IDX`, `--pareto-only`, `--top N`, `--limit N`. Gene
vectors and fitness tuples are expanded into named columns using
the run's stored `param_names` and `objective_labels`, so the
output has meaningful headers (`yield_stress`, `hardening`,
`rmse`) rather than cryptic positional indices.

Scale: on a 50 000-record archive (500 gens × 100 pop × 8
objectives), `--runs` / `--gens` are near-instant; `--gens-best`
and `--pareto-only` run in about a second, dominated by SQLite
JSON decode.

## Resuming a run

Two ways to pick up from the last checkpoint:

```
python nsga3_calibration.py --resume-latest
python nsga3_calibration.py --resume-from 15        # gen number
python nsga3_calibration.py --resume-from path/to/checkpoint_gen_15.pkl
```

`--resume-latest` picks the highest-numbered pickle in the
checkpoint directory — the common case after a crash.
`--resume-from N` looks up `checkpoint_gen_N.pkl` in the
configured `checkpoint_dir`; giving a path uses it verbatim.

On resume the example driver deliberately does NOT call
`archive.start_run()`, leaving `Problem.archive_run_id=None` so
the library adopts the UUID stored in the pickle. Calling
`start_run` on resume generates a fresh UUID that doesn't match
the pickle, which the library rejects with a diagnostic error
pointing at the fix. Stale archive rows from a pre-crash partial
generation are cleaned automatically via
`discard_from_generation`.

## Recovering blown-away logs

If the `.log` files have been lost or truncated but the pickle is
still around:

```
python -m workflows.optimization.regenerate_logbook_files \
    calibration_run/checkpoint_files/checkpoint_gen_15.pkl
```

Writes fresh `logbook1_stats.log` + `logbook2_solutions.log` next
to the pickle (or elsewhere with `--output-dir`). Output is
byte-identical to what a live run would have produced, because
it reuses the driver's own `_LogbookWriter`.

## Plotting solutions vs experimental data

Once a calibration finishes, point the example plotter at the
workspace:

```
python examples/plot_solutions.py calibration_run --top 10
```

The plotter reads everything from the SQLite archive — gene
records, simulation outputs, AND the experimental reference
DataFrames (the driver records these at run start so you don't
have to keep CSVs around or worry about CSV-to-SimCase ordering
post-run). The on-disk case directories are not required.

The headline figure is a grid of subplots, one column per SimCase,
with two rows: stress-strain on top, slope-strain on the bottom
(slope = `np.gradient(stress, strain)`, computed identically for
sim and exp so it matches what the slope objective scored).
Below the subplots:

- a slider that fades curves below a chosen rank — narrow focus
  from N to K without re-running;
- a click panel that prints the gene's parameters and fitness
  when you click a curve.

Selection modes:

- `--objective 0` (or `--objective stress_rmse_0`) ranks by a single
  objective. Pass `--objective` and the plotter picks `--mode
  objective` automatically.
- `--mode last-gen` plots every individual in the final generation
  with no ranking.
- L2 norm is the default — closest-to-utopian-origin across all
  objectives.

Pareto front:

- `--pareto 0,1` produces a 2-D scatter of two objectives. Points
  are colored by L2 norm; the L2 winner is ringed in red. **Every
  point is clickable** — click reveals the gene's parameters AND
  draws its stress-strain response in an inset axes overlaid against
  the experimental references. With `--top N` the scatter is
  restricted to the N lowest-L2 genes; without it, every rank-0
  gene from the run is shown.
- `--no-overlay` skips the headline overlay so you can request only
  the Pareto.

Headless / batch:

```
python examples/plot_solutions.py calibration_run \
    --top 10 --pareto 0,1 \
    --save overlay.png --save-pareto pareto.png --no-show
```

The plotter reuses the selection logic from `inspect_archive`
(the public `select_top_genes` API) so the L2 / per-objective /
dedup behavior matches what `inspect_archive --pareto-only` would
show.

If for any reason the archive doesn't carry experimental data
(older archives written before the experiments table existed),
fall back to passing CSVs explicitly:

```
python examples/plot_solutions.py calibration_run \
    --top 10 --experimental experiments/exp1.txt experiments/exp2.txt
```

## Package layout

```
exaconstit-calibrate/
├── pyproject.toml
├── README.md
├── LICENSE
├── workflow_common/             Framework package
│   ├── __init__.py              Public API
│   ├── ARCHITECTURE.md          Layering, decision table, FAQ
│   ├── MIGRATION.md             Step-by-step port guide
│   ├── _fs.py                   Atomic writes, cd context manager
│   ├── logging_utils.py         Stdlib-logging wrapper
│   ├── templates.py             %%key%% substitution
│   ├── paths.py                 PathResolver, TemplatePathResolver
│   ├── manifest.py              JSONL run manifest
│   ├── sentinel.py              Per-case atomic .done markers
│   ├── platform_detect.py       HPC-specific quirks
│   ├── results.py               CaseLayout, ResultReader, TextTableReader
│   ├── smoothing.py             PCHIP / arc-length / legacy smoothers
│   ├── case_setup.py            CaseTemplater + PropertyWriter
│   ├── objectives.py            Extractor, evaluator, failure handlers
│   ├── problem.py               Problem orchestrator
│   ├── postprocess.py           BestSol, plotting, checkpoint loader
│   ├── archive.py               SQLite archive (ArchiveDB)
│   └── backends/                LocalBackend + FluxBackend
│
├── workflows/
│   └── optimization/            (U-)NSGA-III driver on DEAP fork
│
├── examples/
│   ├── README.md                How to adapt and run the example
│   └── nsga3_calibration.py     Side-by-side port of ExaConstit_NSGA3.py
│
└── tests/                       255 tests
```

## Testing

After an editable install:

```bash
python -m pytest tests/ -q           # full suite (~30-45s)
python -m pytest tests/test_archive.py -v     # one module
python -m pytest tests/ -k "determin"         # keyword filter
```

The test suite uses a fake Python "simulation binary" defined in
`tests/conftest.py`; no ExaConstit build is required.

## A note on the package name

`exaconstit-calibrate` captures the current and planned scope —
crystal-plasticity parameter calibration today, macroscale
yield-surface fitting tomorrow — without overreaching into other
ExaConstit workflow territory (AM challenge-problem orchestration,
etc.). The importable names (`workflow_common`, `workflows`) are
deliberately kept generic and unchanged from the pre-refactor
code so migrated drivers don't need import-path rewrites.

## License

BSD-3-Clause (matches ExaConstit). See `LICENSE`.

## For maintainers

Cutting a new release? See `RELEASING.md` for the build-and-publish
workflow. Most of the time the whole sequence is:

```bash
# edit version in pyproject.toml, then:
python -m pytest tests/ -q
rm -rf dist build *.egg-info
python -m build
git tag -a v<VERSION> -m "Release v<VERSION>"
```

but there are failure modes (wheel sanity check, stale build dirs,
PyPI immutability) worth being aware of before publishing. The
guide covers all of them.
