# Examples

## `nsga3_calibration.py`

Drop-in replacement for the pre-refactor
`ExaConstit_NSGA3.py` + `ExaConstit_Problems.py` + `normal_map.py`
+ `ExaConstit_Logger.py` set. Runs the real ExaConstit `mechanics`
binary against the same two-experiment topology, the same parameter
bounds, and the same GA defaults as the original.

Every section in the file has a commented-out block quoting the
relevant slice of the old script, followed by its new equivalent.
Useful for mapping old code mentally onto new code in one sitting.

### Running it

For Slurm users, start with `run_nsga3_slurm.sh`,
`nsga3_slurm_helpers.sh`, and `SLURM_RUN_GUIDE.md`. The Slurm script
is the small file users edit. The helper contains the longer shell
functions and should stay in the same directory unless
`HELPER_SCRIPT` points to its full path.

1. Install the package with the NSGA-III extra:

   ```bash
   pip install "exaconstit-calibrate[nsga3,plot]"
   ```

2. Make sure the example lives inside an ExaConstit checkout. The
   script auto-discovers:
   - the enclosing ExaConstit repo root
   - `test/data` inputs such as `voce_quats.ori`, `grains.txt`,
     and `state_cp_voce.txt`
   - a built `mechanics` binary in common locations such as
     `build_cpu/bin/mechanics`

   Override discovery only if your layout differs:
   - `EXACONSTIT_ROOT=/path/to/ExaConstit`
   - `EXACONSTIT_MECHANICS=/path/to/mechanics`

3. Pick a backend:
   - `--backend flux` is the default and is the intended HPC path.
     Flux reads the per-case `SimCase` resource fields directly
     (`num_nodes`, `num_tasks`, `cores_per_task`, `gpus_per_task`,
     `duration_s`).
   - `--backend local` is mainly for workstation/debug use. The
     shipped example cases request `num_tasks=4`, so local mode is
     not a drop-in replacement unless you either:
     - reduce the cases to `num_tasks=1` for serial debugging, or
     - adapt the backend construction in the example to provide an
       MPI launcher such as `mpirun`/`srun` for multi-rank local runs

4. Run:

   ```bash
   python nsga3_calibration.py --backend flux
   ```

   For a Flux allocation started from Slurm, a typical launch looks like:

   ```bash
   srun -n 1 --pty --mpi=none --mpibind=off flux start \
     python nsga3_calibration.py --backend flux
   ```

   For a serial local debug run, first lower the example's
   `SimCase.num_tasks` values to `1`, then run:

   ```bash
   python nsga3_calibration.py --backend local
   ```

   Optional flags:
   - `--no-archive` — disable the SQLite archive + rolling cleanup
     (most literal reproduction of the old script's behavior)
   - `--resume-from PATH/checkpoint_gen_N.pkl` — resume a previous run
   - `--resume-latest` — resume from the highest checkpoint in the run dir
   - `--seed N` — override the RNG seed

### Paths And Templates

- `examples/template_options.toml` is repo-portable: the example
  fills in `%%properties_file%%`, `%%state_vars_file%%`,
  `%%grain_file%%`, and the other case-specific placeholders at render
  time.
- The example script's local package root is
  `ExaConstit/workflows/exaconstit-calibrate`, while the ExaConstit
  repo root is the higher-level `ExaConstit` directory. The code uses
  separate names for those on purpose: `PACKAGE_ROOT` versus
  `EXACONSTIT_ROOT`.

### Resource Configuration

- Backend choice is separate from resource description. The resource
  shape lives on each `SimCase`.
- Typical CPU-only Flux case:
  `num_nodes=1, num_tasks=4, cores_per_task=1, gpus_per_task=0`
- Typical GPU Flux case with one rank per GPU:
  `num_nodes=1, num_tasks=4, cores_per_task=1, gpus_per_task=1`
- Hybrid MPI+threads case:
  `num_nodes=1, num_tasks=4, cores_per_task=7, gpus_per_task=1`
  which requests 28 CPU cores plus 4 GPUs for that one simulation.

### Things worth knowing before you port your own driver

- **Normalization.** The built-in `StressStrainObjective` returns
  plain RMSE. The old script divided by `np.std(exp)`. The example
  includes two small custom evaluator classes
  (`_StdNormalizedStressEvaluator`, `_StdNormalizedSlopeEvaluator`)
  that reproduce the old normalization exactly. Copy those into
  your own driver or replace with whatever metric you prefer.

- **Population size.** The old script derived `NPOP` from the
  reference-point count. The new `RunConfig` does the same when
  `population_size=None` (default). You get `NPOP = 288` for
  `NOBJ=4, p=10` either way.

- **Case-directory naming.** The example uses
  `gen_{generation}/gene_{gene}_obj_{obj}` to exactly match the
  pre-refactor layout, so a mix of old + new runs in the same
  workspace doesn't collide.

- **SimCase vs ObjectiveSpec.** One SimCase per experiment (two
  in this example). Two ObjectiveSpecs per SimCase (stress and
  slope). That's the `NOBJ = NEXP * 2` pattern, preserved exactly
  but with the sim-vs-evaluator split that the framework makes
  explicit.

For the conceptual walkthrough that this file is the
"show, don't tell" counterpart to, read
`workflow_common/MIGRATION.md` — especially the step-by-step
section 4-8 for the Problem construction.
