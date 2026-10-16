#!/bin/sh
#
# Slurm job script for the ExaConstit NSGA-III calibration example.
#
# Most users should only edit:
#   1. The #SBATCH lines.
#   2. The SIMPLE SETTINGS block.
#
# Submit with:
#
#   sbatch run_nsga3_slurm.sh
#
# Keep this file in the same directory as nsga3_slurm_helpers.sh unless
# you set HELPER_SCRIPT below to the full path of that helper file.

# ----------------------------------------------------------------------
# 1. SLURM SETTINGS
# ----------------------------------------------------------------------
#
# These lines ask Slurm for compute resources.
#
# Examples:
#
#   One short debug job on two nodes:
#     #SBATCH -A your_bank_name
#     #SBATCH -N 2
#     #SBATCH -n 224
#     #SBATCH -t 01:00:00
#     #SBATCH -p pdebug
#
#   A longer job on four nodes:
#     #SBATCH -A your_bank_name
#     #SBATCH -N 4
#     #SBATCH -n 448
#     #SBATCH -t 08:00:00
#     #SBATCH -p pbatch
#
# Edit the active lines below. Do not put quotes around these values.
#
#SBATCH -A your_bank_name
#SBATCH -N 2
#SBATCH -n 224
#SBATCH -t 01:00:00
#SBATCH -p pdebug
#SBATCH -J exaconstit_nsga3
#SBATCH -o exaconstit_nsga3.%j.out

# ----------------------------------------------------------------------
# 2. SIMPLE SETTINGS
# ----------------------------------------------------------------------
#
# Change the text to the right of each equals sign.
# Do not add spaces around the equals sign.
#
# Correct:
#   ACTION="all"
#
# Incorrect:
#   ACTION = "all"

# Python to use for this workflow.
#
# Example for this LLNL setup:
#   PYTHON="/usr/tce/packages/python/python-3.12.2/bin/python"
#
# Example if your environment provides python on PATH:
#   PYTHON="python3"
PYTHON="/usr/tce/packages/python/python-3.12.2/bin/python"

# Where Flux's Python module is installed.
#
# Example for this LLNL setup:
#   FLUX_PYTHONPATH="/usr/lib64/flux/python3.12"
#
# Example for a different Python version:
#   FLUX_PYTHONPATH="/usr/lib64/flux/python3.11"
FLUX_PYTHONPATH="/usr/lib64/flux/python3.12"

# What should this job do?
#
# Choose exactly one:
#   ACTION="run"            starts a new calibration run
#   ACTION="resume-latest"  resumes from the newest checkpoint
#   ACTION="resume-from"    resumes from CHECKPOINT_GEN or CHECKPOINT_PATH
#   ACTION="inspect"        prints tables from an existing run
#   ACTION="plots"          makes plots from an existing run
#   ACTION="postprocess"    does inspect and plots, but no new simulations
#   ACTION="all"            runs calibration, then inspect, then plots
ACTION="run"

# Resume settings. These only matter when ACTION="resume-from".
#
# Resume from generation 15:
#   CHECKPOINT_GEN="15"
#   CHECKPOINT_PATH=""
#
# Resume from an exact checkpoint file:
#   CHECKPOINT_GEN=""
#   CHECKPOINT_PATH="calibration_run/checkpoint_files/checkpoint_gen_15.pkl"
CHECKPOINT_GEN=""
CHECKPOINT_PATH=""

# Set this to "1" the first time you run, or after Python dependencies
# changed. Set it to "0" for normal runs so you do not spend allocation
# time reinstalling the package.
#
# Example first-time setup:
#   INSTALL_PACKAGE="1"
#
# Example normal production run:
#   INSTALL_PACKAGE="0"
INSTALL_PACKAGE="0"

# Where nsga3_calibration.py writes case directories, checkpoints,
# logs, and the SQLite archive.
#
# Most users can leave this as-is.
WORKSPACE="calibration_run"

# Where post-processing tables and figures should be written.
#
# Most users can leave this as-is.
POST_DIR="postprocess"

# Number of best solutions to print or plot.
#
# Example:
#   TOP_N="5"
#   TOP_N="10"
#   TOP_N="25"
TOP_N="10"

# Optional path overrides.
#
# Most users should leave these blank because nsga3_calibration.py can
# find the ExaConstit checkout and common mechanics build directories.
#
# Example if your ExaConstit checkout is somewhere unusual:
#   EXACONSTIT_ROOT="/usr/workspace/myname/ExaConstit"
#
# Example if mechanics is in a custom build directory:
#   EXACONSTIT_MECHANICS="/usr/workspace/myname/ExaConstit/build_cpu/bin/mechanics"
# For most people, you just need to provide the ExaConstit binary location
EXACONSTIT_ROOT=""
EXACONSTIT_MECHANICS=""

# Helper script with the actual shell functions.
#
# If nsga3_slurm_helpers.sh is in this same directory, leave this blank.
#
# If you keep the helper somewhere else, use the full path:
#   HELPER_SCRIPT="/usr/workspace/myname/scripts/nsga3_slurm_helpers.sh"
HELPER_SCRIPT=""

# ----------------------------------------------------------------------
# 3. DO NOT EDIT BELOW THIS LINE FOR NORMAL USE
# ----------------------------------------------------------------------

set -eu

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)

if [ -z "${HELPER_SCRIPT}" ]; then
    HELPER_SCRIPT="${SCRIPT_DIR}/nsga3_slurm_helpers.sh"
fi

if [ ! -f "${HELPER_SCRIPT}" ]; then
    echo "ERROR: helper script not found:"
    echo "  ${HELPER_SCRIPT}"
    echo
    echo "Keep nsga3_slurm_helpers.sh next to run_nsga3_slurm.sh, or set"
    echo "HELPER_SCRIPT to the helper's full path."
    exit 2
fi

# Load helper functions, then run the workflow selected by ACTION.
. "${HELPER_SCRIPT}"
exa_nsga3_main
