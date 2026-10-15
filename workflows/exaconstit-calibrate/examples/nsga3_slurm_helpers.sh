#!/bin/sh
#
# Helper functions for run_nsga3_slurm.sh.
#
# Most users should not edit this file. It exists so the Slurm job
# script can stay short and easy to edit.
#
# Required location:
#   Keep this file in the same directory as run_nsga3_slurm.sh.
#
# Alternative location:
#   If you store this helper somewhere else, set HELPER_SCRIPT in
#   run_nsga3_slurm.sh to this file's full path.

# exa_nsga3_slurm_tasks_text
# --------------------------
# Returns a human-readable Slurm task count. Some interactive/exclusive
# allocations are created by node count and do not export SLURM_NTASKS;
# that is not an error for this workflow because the launch command
# below explicitly starts Flux with ``srun -n 1``.
exa_nsga3_slurm_tasks_text()
{
    if [ -n "${SLURM_NTASKS:-}" ]; then
        printf "%s" "${SLURM_NTASKS}"
    else
        printf "not set (common in interactive/exclusive allocations)"
    fi
}

# exa_nsga3_slurm_cpu_text
# ------------------------
# Reports whichever Slurm CPU-count variable is available. These are
# useful sanity checks in interactive jobs where SLURM_NTASKS is unset.
exa_nsga3_slurm_cpu_text()
{
    if [ -n "${SLURM_CPUS_ON_NODE:-}" ]; then
        printf "%s (SLURM_CPUS_ON_NODE)" "${SLURM_CPUS_ON_NODE}"
    elif [ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]; then
        printf "%s (SLURM_JOB_CPUS_PER_NODE)" "${SLURM_JOB_CPUS_PER_NODE}"
    else
        printf "unknown"
    fi
}

# exa_nsga3_setup
# ----------------
# Sets paths, exports PYTHONPATH so Python can import this package and
# Flux, moves into the package directory, and optionally runs pip install.
exa_nsga3_setup()
{
    PACKAGE_ROOT=$(CDPATH= cd "${SCRIPT_DIR}/.." && pwd)

    export PYTHONPATH="${FLUX_PYTHONPATH}:${PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    export MPLBACKEND=Agg

    if [ -n "${EXACONSTIT_ROOT}" ]; then
        export EXACONSTIT_ROOT
    fi

    if [ -n "${EXACONSTIT_MECHANICS}" ]; then
        export EXACONSTIT_MECHANICS
    fi

    echo "=== ExaConstit NSGA-III Slurm job ==="
    echo "date:              $(date)"
    echo "host:              $(hostname)"
    echo "job id:            ${SLURM_JOB_ID:-not-running-under-slurm}"
    echo "nodes:             ${SLURM_JOB_NUM_NODES:-unknown}"
    echo "slurm tasks:       $(exa_nsga3_slurm_tasks_text)"
    echo "slurm cpus/node:   $(exa_nsga3_slurm_cpu_text)"
    echo "flux launcher:     srun -n 1 starts one Flux broker inside this allocation"
    echo "python:            ${PYTHON}"
    echo "package root:      ${PACKAGE_ROOT}"
    echo "examples dir:      ${SCRIPT_DIR}"
    echo "workspace:         ${SCRIPT_DIR}/${WORKSPACE}"
    echo "postprocess dir:   ${SCRIPT_DIR}/${POST_DIR}"
    echo "action:            ${ACTION}"
    echo

    cd "${PACKAGE_ROOT}"

    if [ "${INSTALL_PACKAGE}" = "1" ]; then
        echo "=== Installing editable package ==="
        "${PYTHON}" -m pip install -e ".[test,nsga3,plot]"
        echo
    fi

    cd "${SCRIPT_DIR}"
}

# exa_nsga3_calibration_args
# --------------------------
# Converts ACTION and checkpoint settings into command-line arguments
# understood by nsga3_calibration.py.
exa_nsga3_calibration_args()
{
    CALIBRATION_ARGS="--backend flux"

    case "${ACTION}" in
        run|all)
            ;;
        resume-latest)
            CALIBRATION_ARGS="${CALIBRATION_ARGS} --resume-latest"
            ;;
        resume-from)
            if [ -n "${CHECKPOINT_PATH}" ]; then
                CALIBRATION_ARGS="${CALIBRATION_ARGS} --resume-from ${CHECKPOINT_PATH}"
            elif [ -n "${CHECKPOINT_GEN}" ]; then
                CALIBRATION_ARGS="${CALIBRATION_ARGS} --resume-from ${CHECKPOINT_GEN}"
            else
                echo "ERROR: ACTION=\"resume-from\" requires CHECKPOINT_GEN or CHECKPOINT_PATH."
                exit 2
            fi
            ;;
        inspect|plots|postprocess)
            ;;
        *)
            echo "ERROR: unknown ACTION=\"${ACTION}\""
            echo "Allowed actions:"
            echo "  run"
            echo "  resume-latest"
            echo "  resume-from"
            echo "  inspect"
            echo "  plots"
            echo "  postprocess"
            echo "  all"
            exit 2
            ;;
    esac
}

# exa_nsga3_run_calibration
# -------------------------
# Starts one Flux instance inside the Slurm job and runs the Python
# NSGA-III calibration driver inside that Flux instance.
exa_nsga3_run_calibration()
{
    echo "=== Starting calibration ==="
    echo "command:"
    echo "  srun -n 1 --mpi=none --mpibind=off flux start ${PYTHON} nsga3_calibration.py ${CALIBRATION_ARGS}"
    echo

    srun -n 1 --mpi=none --mpibind=off \
        flux start "${PYTHON}" nsga3_calibration.py ${CALIBRATION_ARGS}

    echo
}

# exa_nsga3_inspect_results
# -------------------------
# Prints easy-to-read archive summaries and writes a CSV table of the
# best solutions across the whole run.
exa_nsga3_inspect_results()
{
    echo "=== Inspecting archive tables ==="
    mkdir -p "${POST_DIR}"

    echo
    echo "--- Runs in the archive ---"
    "${PYTHON}" -m workflows.optimization.inspect_archive "${WORKSPACE}" --runs

    echo
    echo "--- Per-generation convergence summary ---"
    "${PYTHON}" -m workflows.optimization.inspect_archive "${WORKSPACE}" --gens-best

    echo
    echo "--- Best solutions across the whole run ---"
    "${PYTHON}" -m workflows.optimization.inspect_archive \
        "${WORKSPACE}" --genes --pareto-only --top "${TOP_N}"

    echo
    echo "--- Writing CSV table of best solutions ---"
    "${PYTHON}" -m workflows.optimization.inspect_archive \
        "${WORKSPACE}" --genes --pareto-only --top "${TOP_N}" --format csv \
        > "${POST_DIR}/top_solutions.csv"
    echo "wrote: ${SCRIPT_DIR}/${POST_DIR}/top_solutions.csv"
    echo
}

# exa_nsga3_make_plots
# --------------------
# Writes PNG plots for balanced best solutions, per-objective best
# solutions, and a few common Pareto tradeoff views.
exa_nsga3_make_plots()
{
    echo "=== Making solution plots ==="
    mkdir -p "${POST_DIR}"

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" \
        --save "${POST_DIR}/top_${TOP_N}_balanced_l2_overlay.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --objective stress_1 \
        --save "${POST_DIR}/top_${TOP_N}_stress_1_overlay.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --objective slope_1 \
        --save "${POST_DIR}/top_${TOP_N}_slope_1_overlay.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --objective stress_2 \
        --save "${POST_DIR}/top_${TOP_N}_stress_2_overlay.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --objective slope_2 \
        --save "${POST_DIR}/top_${TOP_N}_slope_2_overlay.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --pareto 0,1 \
        --save-pareto "${POST_DIR}/pareto_stress_1_vs_slope_1.png" \
        --no-show

    "${PYTHON}" plot_solutions.py "${WORKSPACE}" \
        --top "${TOP_N}" --pareto 0,2 \
        --save-pareto "${POST_DIR}/pareto_stress_1_vs_stress_2.png" \
        --no-show

    echo "wrote plots under: ${SCRIPT_DIR}/${POST_DIR}"
    echo
}

# exa_nsga3_main
# --------------
# Main dispatcher called by run_nsga3_slurm.sh after it loads this
# helper. It runs the action selected in the user-edited Slurm script.
exa_nsga3_main()
{
    exa_nsga3_setup
    exa_nsga3_calibration_args

    case "${ACTION}" in
        run|resume-latest|resume-from)
            exa_nsga3_run_calibration
            ;;
        all)
            exa_nsga3_run_calibration
            exa_nsga3_inspect_results
            exa_nsga3_make_plots
            ;;
        inspect)
            exa_nsga3_inspect_results
            ;;
        plots)
            exa_nsga3_make_plots
            ;;
        postprocess)
            exa_nsga3_inspect_results
            exa_nsga3_make_plots
            ;;
    esac

    echo "=== Done ==="
    echo "date: $(date)"
}
