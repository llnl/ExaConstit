# Migration guide: `ExaConstit_NSGA3.py` → new framework

This guide is for users of the pre-refactor `workflows/optimization/`
scripts (`ExaConstit_NSGA3.py`, `ExaConstit_Problems.py`,
`ExaConstit_Logger.py`, `normal_map.py`) who want to port their
existing drivers to the new `workflow_common` + new
`nsga3_driver.py`.

## What stayed the same

Algorithmically, nothing changes. The new `nsga3_driver.py` still
uses the same DEAP fork (https://github.com/rcarson3/deap) that
the old driver depends on, and uses the same operators with the
same default parameters:

- `tools.uniform_reference_points` — Das-Dennis with optional
  second hyperplane
- `tools.selNSGA3(nd="standard")` — environmental selection
- `tools.niching_selection_UNSGA3` — fork-only U-NSGA-III niching
  step (required for single-objective, recommended for 2-obj)
- `tools.cxSimulatedBinaryBounded(eta=30)` — SBX crossover
- `tools.mutPolynomialBounded(eta=20, indpb=1/ndim)` — polynomial
  mutation
- `algorithms.varAnd(pop, toolbox, 1, 1)` — always crossover +
  mutate
- `deap.benchmarks.tools.hypervolume(pop, [1]*n_obj)` — HV
  indicator
- Pickle checkpoint format with exactly the same keys:
  `pop_library`, `iter_tot`, `generation`, `fail_count`,
  `stop_count`, `logbook1`, `logbook2`, `rndstate`
- DEAP `Logbook` stats layout: `gen`, `iter`, `simRuns`, and (for
  multi-obj) `ND`, `GD`, `HV`, `std`, `min`, `avg`, `max`
- Stopping criteria: `fail_limit` on cumulative sim failures,
  `stop_limit` consecutive "ND == NPOP" generations after `Imin`

A checkpoint pickled by the old driver loads cleanly into the new
one, and vice versa. Same seed + same bounds + same ref-point
config = same per-generation population.

## What changed

Everything BELOW the DEAP layer:

| Old concept                               | New concept                                                      |
| ----------------------------------------- | ---------------------------------------------------------------- |
| `ExaProb(...)` monolith                   | `Problem(sim_cases=..., objective_specs=...)` orchestrator       |
| One sim per objective (hardcoded)         | `SimCase` decoupled from `ObjectiveSpec`; M sims, N objectives   |
| `normal_map.map_custom(problem, gen, ind)` | `_deap_evaluate()` inside the driver calls `evaluate_population` |
| `normal_map.map_custom_fail` retry path   | Driver-internal fail-retry loop in `run_nsga3`                   |
| `ExaProb.is_simulation_done(igene)`       | `FailureHandler` + `failure_threshold` on `RunConfig`            |
| `ExaConstit_Logger` globals               | `configure_logging()` + `get_logger(__name__)`                   |
| Hand-rolled `.done` flag files            | `write_sentinel` / `read_sentinel` (atomic)                      |
| `smoothening_ss_data_fcn.smooth_curve`    | `PchipSmoother`, `ArcLengthSmoother`, `LegacyLinearSmoother`     |
| Hand-rolled `pd.read_csv` of sim output   | `TextTableReader` + `CaseLayout`                                 |
| Hand-rolled stress-strain comparison      | `StressStrainExtractor` + `StressStrainObjective`                |
| `loc_mechanics`, `NUM_NODES`, etc. globals | `ProblemConfig` dataclass (defaults) + per-`SimCase` resource overrides — see step 4a. |
| Parallel arrays `ncpus`, `ngpus`, `nnodes`, `timeout` indexed by experiment | `SimCase(num_tasks=..., gpus_per_task=..., num_nodes=..., duration_s=...)` — one field per concept, inherits from `ProblemConfig` when omitted. |
| User-supplied function that writes `property.txt` per case | `CallablePropertyWriter(func=your_function)` — same callable, framework-wrapped — see step 4b. |
| Top-level global `BOUND_LOW`, `NDIM`      | `Bounds` dataclass                                               |
| Top-level global `UNSGA3`, `NGEN`, `mut_eta` | `RunConfig` dataclass                                         |
| `ind.stress` cached in the pickle         | `ArchiveDB` (SQLite) stores sim outputs separately; `cleanup_keep_generations` rolling-deletes case dirs. Opt-in — see step 11. |

## What is gone

Two things from the old driver do not have direct replacements:

1. **`ind.stress`** — the old driver stashed the simulated stress
   history on each DEAP individual for post-processing
   (`ExaConstit_PostProcess.py` reads it). The new framework does
   not carry the `CaseResultSet` through the fitness path (it is
   consumed by the evaluator and discarded). If your post-processor
   needs the sim output, read it back from disk using
   `CaseLayout(ctx, resolver)` + `reader.read(layout)`. The case
   directories are still there as long as the workspace hasn't
   been wiped.
2. **`ExaProb.is_simulation_done(igene)`** — sim failure now flows
   through a `FailureHandler`, which converts it into a fitness
   value (inf by default). The driver detects failed genes by
   comparing returned fitness values against
   `RunConfig.failure_threshold` (default `np.inf`) and triggers
   the same fail-retry logic the old driver had.

## Step-by-step port of `ExaConstit_NSGA3.py`

The walk-through below mirrors the structure of the old driver
section by section.

### 1. "Basic Parameter" section

Old:

```python
UNSGA3 = True
NEXP = 2
NOBJ = NEXP * 2
```

New: these go on `RunConfig` and on the SimCase/ObjectiveSpec lists.
There is no global `NOBJ`; it is derived from
`len(objective_specs)`.

```python
config = RunConfig(unsga3=True, ...)
# NEXP * 2 == len(objective_specs) == 4; see step 5 for the list.
```

### 2. "CP Parameter Constraints" section

Old:

```python
IND_LOW = [150, 100, 50, 1500, 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]
IND_UP  = [200, 150, 100, 2500, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]
DEP_LOW = None
DEP_UP  = None
BOUND_LOW = IND_LOW
BOUND_UP  = IND_UP
# If DEP_LOW/UP are set, they are extended per objective:
# for i in range(NOBJ): BOUND_LOW.extend(DEP_LOW); BOUND_UP.extend(DEP_UP)
NDIM = len(BOUND_LOW)
```

New: construct the extended bounds yourself and pass them to
`Bounds`:

```python
import numpy as np

ind_low = [150, 100, 50, 1500, 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]
ind_up  = [200, 150, 100, 2500, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]
dep_low = None                 # or a list of values
dep_up  = None

low = list(ind_low)
up  = list(ind_up)
if dep_low is not None and dep_up is not None:
    for _ in range(len(objective_specs)):   # NOBJ
        low.extend(dep_low)
        up.extend(dep_up)

bounds = Bounds(lower=np.array(low), upper=np.array(up))
# bounds.n_params replaces NDIM.
```

### 3. "DEP_UNOPT" (non-optimized per-experiment parameters)

Old:

```python
DEP_UNOPT_S = [298.0]
DEP_UNOPT = [DEP_UNOPT_S for _ in range(NEXP)]
```

This fed temperatures/strain-rates into the options template,
PER experiment, without being part of the gene.

New: put them on the `SimCase.case_data` mapping, one
SimCase per experiment:

```python
sim_cases = [
    SimCase(case_data={"temperature_k": 298.0, "strain_rate": 1e-3},
            label="exp1"),
    SimCase(case_data={"temperature_k": 298.0, "strain_rate": 1e-2},
            label="exp2"),
]
```

The template can reference `%%temperature_k%%` and
`%%strain_rate%%`; per-experiment values are substituted by the
templater at render time.

The same `case_data` dict is also visible to:

* the **path resolver** — any key here can appear as `{key}` in
  `working_dir_pattern` or `output_file_patterns`. So if you want
  `gen_5/grain_32/exp1/...` as your working dir, put `rve_name`
  in `case_data` and reference `{rve_name}` in the pattern.
* a **`CallablePropertyWriter`** via `sim_case.case_data`.

If you have constants that vary per case — elastic moduli at
different temperatures, anisotropy ratios per orientation, lattice
parameters per phase — put them directly in each SimCase's
`case_data` dict. There is no need for a Python lookup table
inside the writer; the data is already where it belongs:

```python
sim_cases = [
    SimCase(case_data={
        "temperature_k": 298.0,
        "c11": 168.4, "c12": 121.4, "c44": 75.4,
        # ... loading conditions etc ...
    }, label="cold"),
    SimCase(case_data={
        "temperature_k": 600.0,
        "c11": 156.0, "c12": 117.0, "c44": 72.0,
        # ...
    }, label="hot"),
]
```

The writer reads keys directly:

```python
def write_properties(case_dir, gene, names, sim_case):
    d = sim_case.case_data
    # ... write properties.txt using d['c11'], d['c12'], d['c44'] ...
```

Or — even simpler — if your sim code reads these from a TOML/INI
file, put `%%c11%%`, `%%c12%%`, `%%c44%%` placeholders in the
master template and the templater fills them straight from
`case_data` with zero writer code involved.

`case_data` may carry keys not referenced by any template or path
pattern; those are silently ignored by the templater. The
templater only complains when its template has a `%%key%%` with
no matching entry in `case_data` — exactly the typo-detection
behavior that catches missing keys without nagging about extras.

One mapping serves all three consumers. The previous design had
two separate fields (`template_values` + `context_extra`) that
meant the same thing but were visible to different consumers;
they were merged into a single `case_data` for clarity.

### 4. ExaProb initialization

Old:

```python
from ExaConstit_Problems import ExaProb

problem = ExaProb(
    n_obj=NOBJ,
    n_dep=n_dep,
    ndim=NDIM,
    loc_mechanics="/path/to/mechanics",
    Exper_input_files=["exp1.txt", "exp2.txt"],
    Sim_input_files=["options_exp1.toml", "options_exp2.toml"],
    DEP_UNOPT=DEP_UNOPT,
)
```

New: build up the framework components and assemble a `Problem`.
This is the longest part of the migration but each line replaces a
hidden responsibility inside the old ExaProb.

```python
from pathlib import Path
import pandas as pd
from workflow_common import (
    CallablePropertyWriter, CaseTemplater, TemplateTarget,
    TemplatePathResolver,
    TextTableReader, TextTableSpec,
    Problem, ProblemConfig, SimCase, ObjectiveSpec,
    StressStrainExtractor, StressStrainObjective,
    LocalBackend, FluxBackend, Manifest,
    InfinityFailureHandler,
    load_experimental_csv, configure_logging,
)

configure_logging(level="info", logfile="logbook3_ExaProb.log")

# Per-experiment experimental reference data.
exp_dfs = [
    load_experimental_csv("exp1.txt"),
    load_experimental_csv("exp2.txt"),
]

# The old Sim_input_files was a per-experiment list of master-template
# filenames. In the new framework there is one template per rendered
# file, and the templater is code-agnostic.
templater = CaseTemplater([
    TemplateTarget(
        source=Path("master_options.toml"),
        dest="options.toml",
    ),
])

# The pre-refactor code had a user-supplied Python function that
# wrote a per-case `properties.txt` file from the gene vector.
# CallablePropertyWriter is the direct port. Your write function
# gets (case_dir, gene, names, sim_case) and returns the path it
# wrote. See MIGRATION step 4b for the full pattern and rationale;
# use TemplatePropertyWriter only if your properties actually are
# a `%%key%%` substitution and nothing else.
def _write_properties(case_dir, gene, names, sim_case):
    gd = dict(zip(names, gene))
    path = case_dir / "properties.txt"
    path.write_text(
        f"g0_1 = {gd['g0_1']}\n"
        f"g0_2 = {gd['g0_2']}\n"
        # ... etc per your material model ...
    )
    return path

property_writer = CallablePropertyWriter(func=_write_properties)

resolver = TemplatePathResolver(
    working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
    output_file_patterns={
        "avg_stress":   "{working_dir}/avg_stress.txt",
        "avg_def_grad": "{working_dir}/avg_def_grad.txt",
    },
)
reader = TextTableReader({
    "avg_stress": TextTableSpec(columns=[
        "time", "s11", "s22", "s33", "s12", "s23", "s13",
    ]),
    "avg_def_grad": TextTableSpec(
        columns=["time", "F11", "F12", "F13",
                 "F21", "F22", "F23", "F31", "F32", "F33"],
        required=False,
    ),
})
```

### 4a. Per-SimCase resource overrides

Different experiments in a calibration workflow routinely need
different compute allocations. The pre-refactor code carried
these as parallel arrays:

```python
# OLD (ExaConstit_NSGA3.py, top-level):
ncpus   = [4, 4]
ngpus   = [0, 0]
nnodes  = [1, 1]
timeout = [6 * 60, 6 * 60]     # seconds
```

Each row of those arrays corresponded to a row of `test_dataframe`
(i.e. one experiment). The values flowed through into the
subprocess launch one experiment at a time.

In the new framework these collapse into optional fields on each
`SimCase`. Any field left at `None` falls through to the
`ProblemConfig` default, so you only override what differs:

```python
# NEW: per-SimCase overrides. Fields omitted here inherit from
# ProblemConfig (which can still set a sensible workstation default).
sim_cases = [
    SimCase(
        label="quasi_static",
        case_data={"strain_rate": 1e-3, ...},
        num_nodes=1, num_tasks=4,
        gpus_per_task=0,
        duration_s=6 * 60,
        binary_args=("-opt", "options.toml"),
    ),
    SimCase(
        label="dynamic",
        case_data={"strain_rate": 1e1, ...},
        num_nodes=4,                    # more nodes for explicit dynamics
        num_tasks=16,                   # more MPI ranks too
        gpus_per_task=1,                # this one runs on GPU
        duration_s=2 * 3600,            # and needs more wall time
        binary_args=("-opt", "options.toml"),
    ),
]
```

The resource-override fields on `SimCase` are:

- `num_nodes` — nodes to request
- `num_tasks` — MPI ranks per simulation
- `cores_per_task` — cores per rank (typically 1 for HPC CPUs)
- `gpus_per_task` — GPUs per rank; set 0 to run on CPU
- `duration_s` — per-sim wall-time budget (seconds)
- `binary` — override the simulation binary itself (rare)
- `binary_args` — override the CLI arguments passed to the binary

All of them map 1:1 onto the `SimJobSpec` the framework hands to
the backend. `FluxBackend` honors them natively — each case is
submitted as its own Flux job with its own resource footprint.
`LocalBackend` uses them for bookkeeping (tags, logs) but runs all
cases through the same `ThreadPoolExecutor`. If you need strict
per-case resource isolation on a single node, use Flux even
locally via `flux start`.

### 4b. CallablePropertyWriter — the "user-supplied write function" pattern

Most crystal-plasticity workflows produce a per-case properties
file whose format is specific to the material model (ExaCMech
slip-system blocks, Voce hardening parameters, elastic constants,
etc.). Three framework-provided writers cover the common cases:

| Writer                      | When to use                                             |
| --------------------------- | ------------------------------------------------------- |
| `TemplatePropertyWriter`    | Properties are a straight `%%key%%` substitution into a text template with no conditional logic, no derived values, no per-experiment tweaks. |
| `DelimitedPropertyWriter`   | The binary reads a bare list of numbers (one per line, or CSV). |
| `CallablePropertyWriter`    | Anything else. This is the direct migration target for the pre-refactor "user supplies a Python function that writes property.txt" pattern. |

`CallablePropertyWriter` is a thin wrapper around a user function
with signature `(case_dir, gene, names, sim_case) -> Path`. The
function gets:

- `case_dir` — absolute path to the case's working directory.
  Write files INSIDE here.
- `gene` — the parameter vector as a sequence of floats.
- `names` — the parameter names aligned to `gene` by index.
- `sim_case` — the `SimCase` for this evaluation. Lets the
  function do per-experiment logic (different orientation files,
  temp-dependent derived properties, etc.) without any separate
  plumbing path.

```python
# OLD (ExaConstit_Problems.py, roughly):
#
#     def write_props(fdironl, gene, ...):
#         with open(fdironl + "/property.txt", "w") as f:
#             f.write(f"g0_1 = {gene[0]}\n")
#             # ... etc
#     # called inside ExaProb.preprocess()

# NEW: CallablePropertyWriter is a PropertyWriter that just hands
# execution to your function at the right moment.
def write_props(case_dir, gene, names, sim_case):
    gd = dict(zip(names, gene))
    path = case_dir / "properties.txt"
    # Exactly the same body you had before; just indent and add
    # the `sim_case` arg if you want per-experiment branching.
    with path.open("w") as f:
        f.write(f"# case: {sim_case.label}\n")
        f.write(f"g0_1 = {gd['g0_1']}\n")
        f.write(f"g0_2 = {gd['g0_2']}\n")
        # ... etc ...
    return path

property_writer = CallablePropertyWriter(func=write_props)
```

The function must **return the path it wrote** so the framework can
log it and surface it in the manifest. Exceptions from inside the
function propagate unchanged — catching them is the caller's job,
because a config error (malformed gene bounds, missing input file)
should fail loudly, not masquerade as a sim failure.

One gotcha: if your function is closed over any mutable state
between gene evaluations (counters, caches), the framework calls
it concurrently across the population when LocalBackend's worker
count is > 1. Either make the function stateless (the usual
choice) or protect the shared state with a lock.

### 5. Objectives — the `NOBJ = NEXP * 2` pattern

This is where the new framework really pays off. The old driver's
convention is "per experiment, we produce TWO objective values:
stress-RMSE and slope-RMSE, both from ONE simulation." The old
ExaProb hard-coded `f[iobj * 2] = ...` and `f[iobj * 2 + 1] = ...`
inside its `evaluate` method.

In the new framework this is the motivating case for `SimCase`
sharing: N experiments → N SimCases (one sim per experiment), 2
objectives per SimCase → 2*N ObjectiveSpecs, all sharing their
respective SimCase.

```python
# One SimCase per experiment.
sim_cases = [
    SimCase(case_data={"strain_rate": 1e-3, "temp_k": 298.0},
            label="exp1"),
    SimCase(case_data={"strain_rate": 1e-2, "temp_k": 298.0},
            label="exp2"),
]

# Two evaluators per experiment, both pointing at the same SimCase.
# See the test_problem.py `_slope_evaluator` helper for a minimal
# slope-matching evaluator that reuses the stress extractor.
def make_specs(sim_case_idx, exp_df, strain_rate):
    stress_eval = StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=strain_rate,
        ),
    )
    slope_eval = _slope_evaluator(stress_eval)  # your helper
    return [
        ObjectiveSpec(stress_eval, sim_case=sim_case_idx,
                      label=f"stress_{sim_case_idx}"),
        ObjectiveSpec(slope_eval,  sim_case=sim_case_idx,
                      label=f"slope_{sim_case_idx}"),
    ]

objective_specs = []
for i, (exp_df, sc) in enumerate(zip(exp_dfs, sim_cases)):
    objective_specs.extend(make_specs(
        i, exp_df, sc.case_data["strain_rate"]))

# Per gene: 2 sims run (one per SimCase), 4 objective values returned
# (stress + slope per SimCase). Matches NOBJ = NEXP * 2.
```

### 6. Failure policy

Old: the `while fail_count < fail_limit` retry loop inside the
driver, with `ExaProb.flag` as the success signal.

New: two layers cooperate:

- The framework's `FailureHandler` decides what fitness value to
  return on sim failure. Default `InfinityFailureHandler()` is
  the simplest and works out-of-the-box with
  `RunConfig.failure_threshold=inf`.
- The driver detects those sentinel values and retries with a
  fresh random individual, up to `fail_limit` times cumulatively.
  No driver changes needed.

```python
problem = Problem(
    ...,
    failure_handler=InfinityFailureHandler(),  # or ConstantPenaltyFailureHandler(1e9)
)

result = run_nsga3(problem, bounds, RunConfig(
    ...,
    fail_limit=10,                       # match the old driver's default
    failure_threshold=float("inf"),      # match the default handler
))
```

If you want the driver to treat a finite penalty as a failure
(useful with `ConstantPenaltyFailureHandler`), set
`failure_threshold` to a value below the penalty:

```python
failure_handler=ConstantPenaltyFailureHandler(penalty=1e9),
# in RunConfig:
failure_threshold=1e8,   # any fitness >= 1e8 triggers retry
```

### 7. Assembling Problem + running

```python
problem = Problem(
    config=ProblemConfig(
        binary=Path("/path/to/mechanics"),
        binary_args=("-opt", "options.toml"),
        num_nodes=1,
        num_tasks=4,
        duration_s=3600,
        required_outputs=("avg_stress.txt",),
    ),
    param_names=[
        "athermal_1", "athermal_2", "athermal_3", "athermal_4",
        "rate_1", "rate_2", "rate_3", "rate_4", "rate_5",
    ],
    sim_cases=sim_cases,
    objective_specs=objective_specs,
    templater=templater,
    property_writer=property_writer,
    resolver=resolver,
    backend=LocalBackend(max_workers=4),      # or FluxBackend(...)
    reader=reader,
    failure_handler=InfinityFailureHandler(),
    manifest=Manifest("wf/manifest.jsonl"),
)
```

### 8. Running the GA

Old:

```python
from ExaConstit_NSGA3 import main   # calls DEAP directly
main(seed=42, checkpoint=None, checkpoint_freq=1)
```

New:

```python
from nsga3_driver import Bounds, RunConfig, run_nsga3

result = run_nsga3(
    problem, bounds,
    RunConfig(
        n_generations=100,
        unsga3=True,                        # UNSGA3=True in old script
        ref_dirs_partitions=(10, 0),        # p = [10, 0] in old script
        ref_dirs_scaling=(1.0, 0.0),        # scaling = [1, 0]
        seed=42,
        mate_eta=30.0,                      # mat_eta
        mut_eta=20.0,                       # mut_eta
        fail_limit=10,
        imin_fraction=0.5,                  # Imin = round(NGEN/2)
        stop_limit=5,
        checkpoint_dir=Path("checkpoint_files"),
        checkpoint_freq=1,
        track_hypervolume=True,
    ),
)
```

### 9. Resuming from a checkpoint

Old:

```python
# in ExaConstit_NSGA3.py top-level:
checkpoint = "checkpoint_files/checkpoint_gen_2.pkl"
main(seed=None, checkpoint=checkpoint, ...)
```

New:

```python
result = run_nsga3(
    problem, bounds,
    RunConfig(
        ...,
        resume_from=Path("checkpoint_files/checkpoint_gen_2.pkl"),
    ),
)
```

Old pickles work as input — the new driver uses the same pickle
format (same keys: `pop_library`, `iter_tot`, `generation`,
`fail_count`, `stop_count`, `logbook1`, `logbook2`, `rndstate`).

Two levels of restart are now available and they compose:

1. **GA state** is pickled at the end of each generation. The
   `resume_from` path restores `random` state + population +
   logbooks.
2. **Sim state** is restored by the framework via sentinels on
   disk. If the workspace directory survives, every completed case
   is auto-skipped on rerun regardless of pickle state.

The right mental model: pickle recovers the GA's "where am I";
sentinels avoid re-running the expensive simulations.

### 10. Post-processing (`ExaConstit_PostProcess.py`)

The old post-processor reads a checkpoint, walks
`pop_library[gen][ind].stress` to reconstruct per-gene stress
histories, picks best solutions via `ExaConstit_SolPicker.BestSol`,
and plots them with `ExaPlots.StressStrain`.

All three of those pieces are ported to the new
`workflow_common.postprocess` module:

| Old                                  | New                                              |
| ------------------------------------ | ------------------------------------------------ |
| `pickle.load(ckp_file)` + key access | `load_checkpoint(path)` → `CheckpointData`       |
| `ExaConstit_SolPicker.BestSol`       | `workflow_common.postprocess.BestSol` (same API) |
| `ind.stress` attribute               | `load_case_results(result, sim_case, resolver, reader)` (re-reads from disk) OR `load_case_results_from_archive(archive, result, sim_case)` (re-reads from SQLite — see step 11) |
| `ExaPlots.StressStrain`              | `plot_stress_strain_overlay`                     |
| `ExaPlots.ObjFun2D`                  | `plot_pareto_front`                              |
| `ExaPlots.ObjFun3D`                  | (intentionally not ported — see postprocess docstring) |

A full post-processing script in the new framework:

```python
import numpy as np
from workflow_common import (
    TemplatePathResolver, TextTableReader, TextTableSpec,
)
from workflow_common.postprocess import (
    load_checkpoint,
    extract_gene_results,
    BestSol,
    load_case_results,
    plot_stress_strain_overlay,
    plot_pareto_front,
)

# Rebuild the same resolver + reader the original run used.
resolver = TemplatePathResolver(
    working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
    output_file_patterns={
        "avg_stress":   "{working_dir}/avg_stress.txt",
        "avg_def_grad": "{working_dir}/avg_def_grad.txt",
    },
)
reader = TextTableReader({
    "avg_stress": TextTableSpec(columns=[
        "time", "s11", "s22", "s33", "s12", "s23", "s13",
    ]),
})

# Step 1 - load checkpoint and turn it into typed records.
ckp = load_checkpoint("checkpoint_files/checkpoint_gen_125.pkl")
all_results = extract_gene_results(ckp.pop_library)
final_gen = all_results[-1]

# Step 2 - pick the 3 best solutions from the final generation
# using the pre-refactor API, unchanged.
fits = np.array([r.fitness for r in final_gen])
best_idx = BestSol(fits, weights=[1] * fits.shape[1], nsmallest=3).EUDIST()

# Step 3 - for each best gene, read its sim output back from disk
# and plot it against experimental data. Replaces `ind.stress`.
for rank, idx in enumerate(best_idx):
    gene_result = final_gen[idx]
    for sc_idx in range(n_sim_cases):          # e.g. NEXP experiments
        case = load_case_results(
            gene_result, sim_case_idx=sc_idx,
            resolver=resolver, reader=reader,
        )
        if case is None:
            continue   # workspace cleaned; skip
        df = case.df("avg_stress")
        plot_stress_strain_overlay(
            sim_strain=df["time"].to_numpy() * strain_rate[sc_idx],
            sim_stress=df["s11"].to_numpy(),
            exp_strain=exp_data[sc_idx]["strain"].to_numpy(),
            exp_stress=exp_data[sc_idx]["stress"].to_numpy(),
            title=f"rank {rank}, experiment {sc_idx}",
        )

# Step 4 - Pareto scatter for 2-objective runs.
plot_pareto_front(fits, best_idx=best_idx, show=True)
```

Three things worth noticing:

1. **`BestSol` is a drop-in.** Your existing picker code keeps
   working; only the import path changes
   (`ExaConstit_SolPicker` → `workflow_common.postprocess`).
2. **Re-reading from disk is per-gene per-sim-case.** A single
   `load_case_results` call gets one `CaseResultSet`. Loop over
   the sim_cases if your run has multiple. This is verbose on
   purpose — the old code implicitly did it inside ExaProb for
   every call.
3. **Birth-generation correctness.** The `GeneResult.generation`
   field is the BIRTH generation, not the pop_library index.
   An individual that survived from gen 2 into gen 50 via
   selection is still looked up under `gen_2/gene_X_sc_Y/`. The
   postprocess module handles this automatically; your code
   should use `GeneResult.generation` when building case paths,
   not the outer loop variable.


### 11. Filesystem-scale problem and the archive

This section is about a second production problem that the
pre-refactor code solved in a subtle, DEAP-coupled way: keeping
the filesystem sane over long optimization runs.

#### The problem

A realistic run of 100 generations × 300 genes × 4 experiments
leaves 120,000 case directories behind, each containing the
input files, output files, stdout/stderr logs, and `.done`
sentinel. Lustre, GPFS, and NFS all start to struggle somewhere
between 500k and 1M small files: `ls` takes minutes, `rm -rf` can
take hours, and on some sites you'll hit inode quota before
convergence. You can't just `rm` the old directories during the
run, because post-processing needs the per-gene stress history
that only lives in those output files.

The pre-refactor workaround was to stash the stress history
directly on each DEAP Individual (`ind.stress`), so the pickle
checkpoint carried everything post-processing would need. That
worked but (a) bloated the checkpoint to GB scale on long runs,
and (b) hard-coupled post-processing to DEAP's class hierarchy.

#### The new approach

`workflow_common.archive.ArchiveDB` is an opt-in SQLite archive
that captures, per case, exactly the `CaseResultSet` that went
INTO the evaluator. Pair it with
`RunConfig(cleanup_keep_generations=2)` and the driver
rolling-deletes old case directories once they've been archived.
At any moment, at most two generations of case dirs live on disk
(the current one and the previous one as a crash-safety margin);
everything else is in a single SQLite file.

**Default behavior is unchanged.** Without `archive=...` on the
Problem and without `cleanup_keep_generations=...` on the
RunConfig, the code behaves exactly like before — case dirs
accumulate on disk, no DB file is created. Opt-in all at once or
not at all.

#### Two-line opt-in

```python
from workflow_common import ArchiveDB

archive = ArchiveDB("opt.db")
archive.open()                                   # creates the DB file
run_id = archive.start_run(                      # opens a new run entry
    seed=42,
    param_names=["yield_stress", "hardening"],
    objective_labels=["stress_rmse"],
    config={"n_gens": 100, ...},                 # any JSON-serializable dict
)

problem = Problem(
    # ...all the usual arguments...
    archive=archive,
    archive_run_id=run_id,
)

result = run_nsga3(problem, bounds, RunConfig(
    # ...the usual GA knobs...
    cleanup_keep_generations=2,                  # rolling cleanup
))

archive.close()                                  # flush and release the file
```

That's it. The driver calls `archive.end_run(run_id)` internally
on success, so `completed_at` gets stamped. A crashed run leaves
`completed_at` NULL — useful for tooling that scans a directory
of archives.

#### What gets captured

Four SQLite tables hold:

- **`runs`** — run metadata: `run_id`, `started_at`, `completed_at`,
  `seed`, `param_names`, `objective_labels`, plus any JSON dict
  you passed as `config=`.
- **`generations`** — per-generation: `gen_idx`, `recorded_at`,
  `n_pop`, and the stats dict from DEAP's `Statistics.compile`.
- **`genes`** — per-individual-per-generation: gene vector,
  fitness tuple, rank, plus the BIRTH coordinates
  (`birth_gen`, `birth_gene`) needed to look up its case output.
- **`case_outputs`** — the pickled `pandas.DataFrame` for every
  output registered on the ResultReader, keyed by
  `(run_id, birth_gen, birth_gene, sim_case_idx, output_name)`.

Pickled DataFrames round-trip bit-identical — same dtype, same
index, same values. This matters for reproducibility of
post-hoc analyses.

#### Post-processing swap

Disk-based (case dirs still present):

```python
from workflow_common.postprocess import (
    load_checkpoint, extract_gene_results, best_solution_eudist,
    load_case_results,        # <-- disk version
)

ckp = load_checkpoint("ck/checkpoint_gen_100.pkl")
final_gen = extract_gene_results(ckp.pop_library)[-1]
fits = np.array([r.fitness for r in final_gen])
best = final_gen[best_solution_eudist(fits)[0]]

for sc_idx in range(n_sim_cases):
    case = load_case_results(best, sc_idx, resolver, reader)
    # ... plot case.df("avg_stress") ...
```

Archive-based (case dirs were cleaned up):

```python
from workflow_common import ArchiveDB
from workflow_common.postprocess import (
    load_checkpoint, extract_gene_results, best_solution_eudist,
    load_case_results_from_archive,       # <-- archive version
)

ckp = load_checkpoint("ck/checkpoint_gen_100.pkl")
final_gen = extract_gene_results(ckp.pop_library)[-1]
fits = np.array([r.fitness for r in final_gen])
best = final_gen[best_solution_eudist(fits)[0]]

with ArchiveDB("opt.db", readonly=True) as archive:
    for sc_idx in range(n_sim_cases):
        case = load_case_results_from_archive(archive, best, sc_idx)
        # ... plot case.df("avg_stress") ...
```

Same return type (`CaseResultSet`), same DataFrame semantics.
The only differences at the call site are the function name, the
`ArchiveDB` context manager, and the dropped `resolver`/`reader`
arguments (the archive is self-describing).

#### Resume interaction

The pickle checkpoint carries an `archive_run_id` field. The
recommended pattern: on resume, build the `Problem` with
`archive=archive, archive_run_id=None` (do **not** call
`archive.start_run()` a second time). The library reads the
pickled UUID out of the checkpoint and assigns it to
`problem.archive_run_id` before any archive writes happen,
picking up exactly where the crashed run left off.

In example driver form:

```python
archive = ArchiveDB("opt.db"); archive.open()
if args.resume_from is None:
    # Fresh run — create the archive row.
    run_id = archive.start_run(seed=seed, param_names=...)
else:
    # Resume — leave run_id None; library pulls it from pickle.
    run_id = None
problem = Problem(..., archive=archive, archive_run_id=run_id)
run_nsga3(problem, bounds, config)
```

If you DO pass a different `archive_run_id` at construction time
(usually because you accidentally called `start_run` again on
resume), the library refuses with a diagnostic ValueError naming
both UUIDs and showing the correct driver pattern. This guards
against cross-run contamination — gene records from run A
silently landing in archive row B.

Before any writes, the library calls
`archive.discard_from_generation(run_id, K+1)` which
cascade-deletes stale generations/genes/case_outputs from a
pre-crash partial write. End result: even a crash mid-generation
leaves a clean archive after resume. Verified in
`test_archive_resume_discards_stale_generations` and
`test_archive_resume_adopts_pickled_run_id_when_problem_has_none`.

Old (pre-archive-aware) pickles don't carry `archive_run_id`; the
driver detects that and starts a fresh archive run, logging a
warning that pre-crash archive entries from the original run ID
won't be carried forward.

#### Using the archive without the driver

If you rolled your own GA (or use a different optimizer on top of
`Problem`), `ArchiveDB` works standalone:

```python
archive = ArchiveDB("opt.db")
archive.open()
run_id = archive.start_run(seed=0, param_names=["x", "y"])

problem = Problem(..., archive=archive, archive_run_id=run_id)

for gen in range(my_n_gens):
    results = problem.evaluate_population(genes, generation=gen)
    # Problem archives each case's DataFrames automatically.
    # You own the generation-level record since that's optimizer-specific:
    gene_records = [...]  # build GeneRecord objects per your optimizer
    archive.record_generation(run_id, gen_idx=gen, genes=gene_records,
                              stats={...})

archive.end_run(run_id)
archive.close()
```

Rolling cleanup without the driver: call
`problem.cleanup_generation_dirs(gen_idx=..., pop_size=...)`
yourself after each generation. The driver's logic is simply
"after gen N, clean up gen (N - keep)" with `keep >= 1`.

#### When to turn it on

Production rule of thumb: **always** for runs longer than ~20
generations or with populations larger than ~50. The archive
overhead is a few MB per generation and the per-case write is
one SQLite INSERT (sub-millisecond); the filesystem win is
orders of magnitude greater.

The one scenario where the archive is not useful: very short
debugging runs where you actively want the case dirs on disk for
inspection. In that case, leave `archive=None` and
`cleanup_keep_generations=None` and everything behaves as
before.


## Troubleshooting

- **"All fitness values are `inf` / the penalty value"** — the
  simulation is failing on every gene. Check `stdout.log` /
  `stderr.log` in one of the case directories; the path is in
  the log output. If the files are empty and the rc is 0, your
  `required_outputs` entry may not match what the binary actually
  writes. If the sim crashed, look at the stderr for the real
  error.

- **"Checkpoints are not being written"** — confirm
  `checkpoint_dir` is set on `RunConfig`. The default is `None`
  which disables checkpointing entirely.

- **"Resumed run produces different results from a fresh run"** —
  the `random` state inside the pickle must match what the seed
  would have produced at that generation. If your driver manually
  calls `random.seed()` somewhere after the driver starts, that
  will desync things. Leave seeding to `run_nsga3`.

- **"UNSGA3 niching throws an ImportError / AttributeError"** — you
  are using stock DEAP, not the fork. Install from
  `https://github.com/rcarson3/deap` as documented in
  `workflows/README.md`.

- **"GA converges but parameters don't match expectation"** — the
  objective values are probably correct but scaled differently
  than in the old driver. The old ExaProb normalized stress RMSE
  by `np.std(s_exp)` (see `ExaConstit_Problems.py`). The default
  `StressStrainObjective` does NOT — it returns raw RMSE. Either
  normalize your exp data before passing it in, or supply a
  custom `metric` callable to `StressStrainObjective` that divides
  by `np.std(exp)` the way the old code did:

  ```python
  def std_normalized_rmse(sim, exp):
      return float(np.sqrt(np.mean((sim - exp) ** 2)) / np.std(exp))

  StressStrainObjective(
      experimental=exp_df,
      extractor=StressStrainExtractor(),
      metric=std_normalized_rmse,
  )
  ```

  The migration is not automatic precisely because this is a
  judgment call — some users want the normalization, some
  don't, and the framework should not silently apply one
  convention.
