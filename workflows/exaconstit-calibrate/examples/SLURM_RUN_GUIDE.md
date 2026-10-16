# Slurm Run Guide For `nsga3_calibration.py`

This guide is for running the NSGA-III calibration example on a Slurm
cluster that starts a Flux instance inside the Slurm job.

The short version:

```bash
cd ExaConstit/workflows/exaconstit-calibrate/examples
sbatch run_nsga3_slurm.sh
```

Before you submit, edit the top of `run_nsga3_slurm.sh`.

Keep these two files together in the same directory:

- `run_nsga3_slurm.sh`
- `nsga3_slurm_helpers.sh`

Most users should edit only `run_nsga3_slurm.sh`. The helper file holds
the longer shell functions so the job script stays short and harder to
break by accident.

If you move `nsga3_slurm_helpers.sh` somewhere else, set
`HELPER_SCRIPT` in `run_nsga3_slurm.sh` to the full path:

```sh
HELPER_SCRIPT="/usr/workspace/myname/scripts/nsga3_slurm_helpers.sh"
```

## What To Edit First

Open `run_nsga3_slurm.sh` and edit the `#SBATCH` block:

```sh
#SBATCH -A wbronze
#SBATCH -N 2
#SBATCH -n 224
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH -J exaconstit_nsga3
#SBATCH -o exaconstit_nsga3.%j.out
```

Meaning:

- `-A` is your bank, project, or allocation account.
- `-N` is the number of nodes.
- `-n` is the total Slurm task count. A common choice is nodes times
  CPU cores per node.
- `-t` is the wall-clock time limit.
- `-p` is the queue or partition.
- `-J` is the job name shown by Slurm.
- `-o` is the output log file. `%j` expands to the Slurm job id.

Concrete examples:

```sh
# Short debug job on 2 nodes:
#SBATCH -A wbronze
#SBATCH -N 2
#SBATCH -n 224
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH -J exaconstit_debug
#SBATCH -o exaconstit_debug.%j.out
```

```sh
# Longer production-style job on 4 nodes:
#SBATCH -A your_bank_name
#SBATCH -N 4
#SBATCH -n 448
#SBATCH -t 08:00:00
#SBATCH -p pbatch
#SBATCH -J exaconstit_calibration
#SBATCH -o exaconstit_calibration.%j.out
```

Then check the `User settings` block:

```sh
PYTHON="/usr/tce/packages/python/python-3.12.2/bin/python"
FLUX_PYTHONPATH="/usr/lib64/flux/python3.12"
ACTION="all"
INSTALL_PACKAGE="0"
TOP_N="10"
```

Most users only need to change `PYTHON`, `ACTION`, and maybe
`INSTALL_PACKAGE`.

Important shell syntax rule: do not add spaces around `=`.

Correct:

```sh
ACTION="all"
```

Incorrect:

```sh
ACTION = "all"
```

More examples:

```sh
# Use python3 from your loaded environment:
PYTHON="python3"
```

```sh
# Use a different Flux Python module directory:
FLUX_PYTHONPATH="/usr/lib64/flux/python3.11"
```

```sh
# Show 25 best solutions instead of 10:
TOP_N="25"
```

## Mechanics Binary

The example tries to find the ExaConstit checkout and the `mechanics`
binary automatically. It looks for common paths like:

```text
ExaConstit/build_cpu/bin/mechanics
ExaConstit/build/bin/mechanics
ExaConstit/build_hip/bin/mechanics
ExaConstit/build_cuda/bin/mechanics
```

If your build is somewhere else, set this in `run_nsga3_slurm.sh`:

```sh
EXACONSTIT_MECHANICS="/full/path/to/mechanics"
```

If the script cannot find the ExaConstit checkout, set:

```sh
EXACONSTIT_ROOT="/full/path/to/ExaConstit"
```

## Running A New Calibration

Set:

```sh
ACTION="run"
```

Then submit:

```bash
sbatch run_nsga3_slurm.sh
```

This runs:

```bash
srun -n 1 --mpi=none --mpibind=off flux start \
  python nsga3_calibration.py --backend flux
```

The driver writes its run directory under:

```text
examples/calibration_run
```

Important files:

- `calibration_run/calibration.db` is the SQLite archive.
- `calibration_run/checkpoint_files/checkpoint_gen_N.pkl` are restart files.
- `calibration_run/checkpoint_files/logbook1_stats.log` has generation stats.
- `calibration_run/checkpoint_files/logbook2_solutions.log` has individuals.

## Running And Post-Processing In One Job

Set:

```sh
ACTION="all"
```

This runs the calibration first. If the calibration finishes, the script
then prints archive summaries and writes plots under:

```text
examples/postprocess
```

This is the recommended first mode because it gives you useful outputs
without needing to remember separate commands.

## Resuming A Run

The calibration driver writes a checkpoint every generation.

To resume from the newest checkpoint:

```sh
ACTION="resume-latest"
```

To resume from a generation number:

```sh
ACTION="resume-from"
CHECKPOINT_GEN="15"
CHECKPOINT_PATH=""
```

To resume from an explicit checkpoint path:

```sh
ACTION="resume-from"
CHECKPOINT_GEN=""
CHECKPOINT_PATH="calibration_run/checkpoint_files/checkpoint_gen_15.pkl"
```

Only set one of `CHECKPOINT_GEN` or `CHECKPOINT_PATH`.

## Inspecting Results Without Running More Simulations

Set:

```sh
ACTION="inspect"
```

This runs archive inspection commands only:

```bash
python -m workflows.optimization.inspect_archive calibration_run --runs
python -m workflows.optimization.inspect_archive calibration_run --gens-best
python -m workflows.optimization.inspect_archive calibration_run \
  --genes --pareto-only --top 10
```

What these mean:

- `--runs` lists runs stored in the archive.
- `--gens-best` shows convergence by generation.
- `--genes --pareto-only --top 10` shows the best solutions across the
  whole run, not just the last generation.

The script also writes:

```text
postprocess/top_solutions.csv
```

## Making Plots Without Running More Simulations

Set:

```sh
ACTION="plots"
```

This writes several figures under:

```text
examples/postprocess
```

Useful plotting commands shown by the script:

```bash
python plot_solutions.py calibration_run \
  --top 10 \
  --save postprocess/top_10_balanced_l2_overlay.png \
  --no-show
```

This plots the top 10 most balanced solutions across all objectives.

```bash
python plot_solutions.py calibration_run \
  --top 10 --objective stress_1 \
  --save postprocess/top_10_stress_1_overlay.png \
  --no-show
```

This ranks by one objective only. Available objective labels in this
example are:

- `stress_1`
- `slope_1`
- `stress_2`
- `slope_2`

You can also use integer objective indices:

- `0` means `stress_1`
- `1` means `slope_1`
- `2` means `stress_2`
- `3` means `slope_2`

## Pareto Plots

Pareto plots show tradeoffs between two objectives.

Stress-vs-slope for experiment 1:

```bash
python plot_solutions.py calibration_run \
  --top 10 --pareto 0,1 \
  --save-pareto postprocess/pareto_stress_1_vs_slope_1.png \
  --no-show
```

Stress objective from experiment 1 vs stress objective from experiment 2:

```bash
python plot_solutions.py calibration_run \
  --top 10 --pareto 0,2 \
  --save-pareto postprocess/pareto_stress_1_vs_stress_2.png \
  --no-show
```

To make both a Pareto plot and the stress-strain overlay in one command,
add `--overlay`:

```bash
python plot_solutions.py calibration_run \
  --top 10 --pareto 0,1 --overlay \
  --save postprocess/top_10_overlay.png \
  --save-pareto postprocess/pareto_0_1.png \
  --no-show
```

## Common Problems

### The job says it cannot import `flux`

The Flux Python module is not visible to your Python interpreter. Check:

```sh
FLUX_PYTHONPATH="/usr/lib64/flux/python3.12"
```

That path must match your site and Python version.

### The job says it cannot find `mechanics`

Set:

```sh
EXACONSTIT_MECHANICS="/full/path/to/mechanics"
```

### I want a quick test, not a full production run

Use the `pdebug` partition, a short time limit, and `ACTION="run"` or
`ACTION="all"`. For a very small local test, edit `nsga3_calibration.py`
so each `SimCase` has `num_tasks=1`, then run with `--backend local`.

### The run stopped before all generations

That can be normal. The NSGA-III driver has early stopping criteria. Look
at:

```bash
python -m workflows.optimization.inspect_archive calibration_run --gens-best
```

and:

```text
calibration_run/checkpoint_files/logbook1_stats.log
```

to see whether it converged or stopped due to a failure.
