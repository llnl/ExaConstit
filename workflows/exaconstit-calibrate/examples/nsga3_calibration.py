"""
ExaConstit NSGA-III calibration — side-by-side with the pre-refactor script.

This file is a drop-in replacement for the pre-refactor
``ExaConstit_NSGA3.py`` + ``ExaConstit_Problems.py`` + ``normal_map.py`` +
``ExaConstit_Logger.py`` set. It runs the real ``mechanics`` binary
against the same two-experiment topology, the same parameter bounds,
the same per-case resource allocation, and the same GA knobs as the
original.

How to read this file
---------------------
The whole thing is a single ``main()``. Each top-level section starts
with a commented block showing the relevant slice of the OLD script,
followed by the NEW equivalent. The OLD blocks are quoted verbatim
from the pre-refactor sources so you can grep for them and verify
the line-for-line mapping.

Numerical defaults match the old script exactly. Running both against
the same seed, same experimental data, and same binary should produce
trajectories that agree to within float roundoff.

How to run this file
--------------------
1. Ensure the example lives inside an ExaConstit checkout. By default
   it auto-discovers:

   * the repo root by walking upward from this file,
   * ``test/data`` inputs like ``voce_quats.ori`` and ``grains.txt``,
   * a built ``mechanics`` binary in common locations such as
     ``build_cpu/bin/mechanics``.

   Override discovery with ``EXACONSTIT_ROOT=/path/to/ExaConstit`` or
   ``EXACONSTIT_MECHANICS=/path/to/mechanics`` if your layout differs.

2. Install the package with the NSGA-III extra:

       pip install "exaconstit-calibrate[nsga3,plot]"

3. Run:

       python nsga3_calibration.py

   Or, to run without filesystem cleanup and without the archive
   (closer to the old script's default behavior):

       python nsga3_calibration.py --no-archive

For the conceptual overview of each framework component this script
uses, see ``workflow_common/MIGRATION.md``. This file is the
"show me working code" counterpart to that prose walkthrough.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = EXAMPLE_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    # Make the example runnable directly from a source checkout
    # without requiring an editable install first.
    sys.path.insert(0, str(PACKAGE_ROOT))


def _find_exaconstit_root(start: Path) -> Path:
    """Find the surrounding ExaConstit checkout.

    Preference order:
    1. ``EXACONSTIT_ROOT`` environment variable.
    2. Walk upward from this example until we find ``test/data`` and
       ``workflows`` together, which identifies the repo root in a
       normal checkout.
    3. Fall back to the expected ``.../ExaConstit`` ancestor relative
       to the example file.
    """
    env_root = os.environ.get("EXACONSTIT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    for candidate in (start, *start.parents):
        if (candidate / "test" / "data").is_dir() and (
            candidate / "workflows"
        ).is_dir():
            return candidate.resolve()

    return start.parents[2].resolve()


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _mechanics_candidates(exaconstit_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_mechanics = os.environ.get("EXACONSTIT_MECHANICS")
    if env_mechanics:
        candidates.append(Path(env_mechanics).expanduser())

    for build_dir in (
        "build_cpu",
        "build",
        "build_hip",
        "build_cuda",
        "build_gpu",
        "build_debug",
        "build_release",
    ):
        candidates.append(exaconstit_root / build_dir / "bin" / "mechanics")
    return candidates


@dataclass(frozen=True)
class ParameterSpec:
    """One optimized material parameter.

    Edit the PARAMS table in main(), not three separate lists. Bounds,
    names, log output, checkpoint metadata, and write_properties() all
    derive from this single table.
    """

    name: str
    lower: float
    upper: float
    units: str = ""
    description: str = ""


from workflow_common import (
    ArchiveDB,
    CallablePropertyWriter,
    CaseTemplater,
    HAS_FLUX,
    InfinityFailureHandler,
    LocalBackend,
    Manifest,
    ObjectiveSpec,
    PchipSmoother,
    Problem,
    ProblemConfig,
    SimCase,
    StressStrainExtractor,
    TemplatePathResolver,
    TemplateTarget,
    TextTableReader,
    TextTableSpec,
    configure_logging,
)


# ================================================================
# CONFIGURATION — repo-relative defaults for the shipped example.
# Everything below this block should work in a normal ExaConstit checkout.
# ================================================================
#
# USER EDIT MAP
# -------------
# If you are new to this workflow, start here. Most users only need
# to edit the items in this block and the matching sections called
# out below.
#
# 1. New experiment data:
#    * Put your stress-strain text/CSV files under examples/experiments
#      or point EXPERIMENT_FILES at their full paths.
#    * Add one matching SimCase in the ``sim_cases = [...]`` section
#      for each experiment. One SimCase means one simulation setup:
#      strain rate, final strain, time stepping, mesh, orientation,
#      and CPU/GPU resource request.
#    * In the OBJECTIVES section, set ``stress_column`` to the stress
#      component matching your loading direction: Sxx for x, Syy for y,
#      Szz for z.
#
# 2. New orientations, grains, or state variables:
#    * Point ORIENTATION_FILES, GRAIN_FILE, and STATE_VARS_FILE at the
#      files you want copied/referenced by each case.
#    * If different experiments use different orientations, keep one
#      ORIENTATION_FILES entry per experiment and set each SimCase's
#      ``ori_file`` to the filename that should appear inside the
#      case directory.
#
# 3. New material model or new ExaConstit options:
#    * Edit examples/template_options.toml. That is the options.toml
#      template rendered into every case directory.
#    * Edit ``write_properties()`` if your material model expects a
#      different properties-file order or extra material constants.
#      The property order must match the active ExaCMech material
#      shortcut/model selected in template_options.toml.
#
# 4. New fitted parameters:
#    * Edit the PARAMS table in main(). Bounds and parameter names are
#      derived from that one table.
#    * Then update ``write_properties()`` so each fitted parameter is
#      written into the correct material-model slot. The order still
#      matters when ExaConstit reads properties.txt.
#
# 5. Backend and hardware:
#    * Pick ``--backend flux`` for HPC Flux runs or ``--backend local``
#      for small/debug runs.
#    * CPU/GPU shape is controlled on each SimCase with num_nodes,
#      num_tasks, cores_per_task, gpus_per_task, and duration_s.
#      Changing only the Slurm header does not change what each
#      individual simulation requests from Flux.

EXACONSTIT_ROOT = _find_exaconstit_root(EXAMPLE_DIR)
TEST_DATA_DIR = EXACONSTIT_ROOT / "test" / "data"
MECHANICS_BINARY = (
    _first_existing(_mechanics_candidates(EXACONSTIT_ROOT))
    or (EXACONSTIT_ROOT / "build_cpu" / "bin" / "mechanics")
)
STATE_VARS_FILE = TEST_DATA_DIR / "state_cp_voce.txt"
GRAIN_FILE = TEST_DATA_DIR / "grains.txt"

# Template options TOML with ``%%key%%`` placeholders. The keys
# substituted in per-case come from SimCase.template_values
# (see the ``sim_cases = [...]`` section). A minimal viable master
# file references:
#
#   temperature_k = %%temperature_k%%
#   orientation_file = "%%ori_file%%"
#   t_final = %%t_final%%
#   dt_min = %%dt_min%%
#   dt_max = %%dt_max%%
#   dt_scale = %%dt_scale%%
#   essential_vals = %%essential_vals%%
#   mesh_lengths = %%mesh_lengths%%
#   mesh_cuts = %%mesh_cuts%%
#   p_refinement = %%p_refinement%%
#   properties_file = "%%properties_file%%"
#   state_vars_file = "%%state_vars_file%%"
#   grain_file = "%%grain_file%%"
#
# Note: ``strain_rate`` remains in SimCase.case_data because the
# objective extractor uses it to convert time to strain. The mechanics
# input currently receives direct velocity values via
# ``essential_vals``; if you switch to velocity-gradient BCs, map
# ``%%strain_rate%%`` into that block in template_options.toml.
MASTER_TOML = EXAMPLE_DIR / "template_options.toml"

# One stress-strain CSV per experiment. Two-column format:
# strain, stress. Whitespace or comma-separated both work.
EXPERIMENT_FILES = [
    EXAMPLE_DIR / "experiments" / "expt_strain_rate_1m3.txt",
    EXAMPLE_DIR / "experiments" / "expt_strain_rate_1m1.txt",
]

# Orientation files (one per experiment) copied into each case dir.
# Matches the old ``ori_file=["voce_quats.ori", "voce_quats.ori"]``.
ORIENTATION_FILES = [
    TEST_DATA_DIR / "voce_quats.ori",
    TEST_DATA_DIR / "voce_quats.ori",
]

WORKSPACE = EXAMPLE_DIR / "calibration_run"

# ================================================================


def main(argv: Sequence[str] | None = None) -> None:
    from workflows.optimization import (
        Bounds,
        RunConfig,
        build_reference_points,
        derive_population_size,
        run_nsga3,
    )

    parser = argparse.ArgumentParser(
        description="NSGA-III calibration of ExaConstit parameters",
    )
    parser.add_argument(
        "--backend",
        choices=("flux", "local"),
        default="flux",
        help=(
            "Execution backend. 'flux' submits each case into the "
            "current Flux instance. 'local' runs cases as ordinary "
            "subprocesses on the current host; for this example, "
            "local mode is mainly for serial debugging unless you also "
            "adapt the resource settings and/or provide an MPI launcher."
        ),
    )
    parser.add_argument(
        "--resume-from", default=None,
        help=(
            "Resume from an existing checkpoint. Accepts either: "
            "(a) a path to a checkpoint pickle, e.g. "
            "'calibration_run/checkpoint_files/checkpoint_gen_15.pkl'; "
            "or (b) a plain integer like '15' meaning "
            "'checkpoint_gen_15.pkl inside the run's checkpoint dir'. "
            "Mutually exclusive with --resume-latest."
        ),
    )
    parser.add_argument(
        "--resume-latest", action="store_true",
        help=(
            "Resume from the highest-numbered checkpoint pickle in "
            "the run's checkpoint directory. Convenient after a "
            "crash when you just want to pick back up wherever "
            "the last good generation landed."
        ),
    )
    parser.add_argument(
        "--no-archive", action="store_true",
        help="Disable the SQLite archive + rolling cleanup. "
             "Closest match to the pre-refactor script's disk behavior.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args(argv)

    required_files = [
        ("mechanics binary", MECHANICS_BINARY),
        ("master options template", MASTER_TOML),
        ("state vars file", STATE_VARS_FILE),
        ("grain file", GRAIN_FILE),
    ]
    required_files.extend(
        (f"experiment file #{i + 1}", path)
        for i, path in enumerate(EXPERIMENT_FILES)
    )
    required_files.extend(
        (f"orientation file #{i + 1}", path)
        for i, path in enumerate(ORIENTATION_FILES)
    )
    missing = [(label, path) for label, path in required_files if not path.is_file()]
    if missing:
        details = "\n".join(
            f"  - {label}: {path}" for label, path in missing
        )
        parser.error(
            "required example inputs were not found:\n"
            f"{details}\n"
            "Set EXACONSTIT_ROOT to your checkout root or "
            "EXACONSTIT_MECHANICS to an explicit mechanics binary path "
            "if your build lives elsewhere."
        )

    # ============================================================
    # RESOLVE --resume-from / --resume-latest
    # ============================================================
    #
    # We let --resume-from accept either a checkpoint path or a
    # generation number, because the user's mental model is "pick
    # up from gen N" — they shouldn't have to remember the exact
    # checkpoint filename convention. --resume-latest is a separate
    # flag for the very common "just keep going from wherever we
    # left off" case.
    #
    # The end product is ``resume_pickle_path``: either a Path
    # pointing at the pickle to load, or None for "fresh run".
    # This is what the library's RunConfig.resume_from wants.
 
    if args.resume_from is not None and args.resume_latest:
        parser.error(
            "--resume-from and --resume-latest are mutually exclusive"
        )
 
    CHECKPOINT_DIR = WORKSPACE / "checkpoint_files"
 
    resume_pickle_path: Path | None = None
    if args.resume_latest:
        # Find the highest-numbered pickle in the checkpoint dir.
        # Pattern is fixed by the library as "checkpoint_gen_{N}.pkl".
        if not CHECKPOINT_DIR.is_dir():
            parser.error(
                f"--resume-latest: checkpoint directory not found at "
                f"{CHECKPOINT_DIR}. Is this a fresh workspace?"
            )
        candidates = list(CHECKPOINT_DIR.glob("checkpoint_gen_*.pkl"))
        if not candidates:
            parser.error(
                f"--resume-latest: no checkpoint files under "
                f"{CHECKPOINT_DIR}. Nothing to resume from."
            )
        # Extract the integer part and pick the max. Robust against
        # non-matching filenames — ``int(...)`` on a bad stem raises
        # and we skip that candidate, preserving the behavior for
        # any user's stray logs or backups in the same dir.
        def _gen_idx_of(p: Path) -> int:
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1
        resume_pickle_path = max(candidates, key=_gen_idx_of)
        # Display-only line. Try to shorten to a cwd-relative path
        # since that's nicer to read, but fall back to the full
        # string if the checkpoint dir and cwd have no ancestor
        # relationship (happens when the user runs from elsewhere,
        # or when CHECKPOINT_DIR is itself a relative path that
        # argparse hasn't resolved).
        try:
            display_path = resume_pickle_path.resolve().relative_to(
                Path.cwd().resolve()
            )
        except ValueError:
            display_path = resume_pickle_path
        print(
            f"--resume-latest: picked {display_path} "
            f"(gen {_gen_idx_of(resume_pickle_path)})"
        )
    elif args.resume_from is not None:
        # Try integer first; if it parses, it's a generation number.
        # Otherwise treat as a path.
        raw = str(args.resume_from)
        try:
            gen_idx = int(raw)
        except ValueError:
            # Not an integer → treat as path.
            resume_pickle_path = Path(raw)
        else:
            # Integer → look up the canonical filename. We do this
            # eagerly so the user gets a clear "no such checkpoint"
            # error right here, not deep inside run_nsga3's pickle
            # load.
            resume_pickle_path = (
                CHECKPOINT_DIR / f"checkpoint_gen_{gen_idx}.pkl"
            )
        if not resume_pickle_path.is_file():
            parser.error(
                f"--resume-from: no checkpoint at {resume_pickle_path}"
            )

    # ============================================================
    # LOGGING
    # ============================================================
    #
    # OLD (ExaConstit_NSGA3.py):
    #
    #     initialize_ExaProb_log(
    #         glob_loglvl="info",
    #         filename="logbook3_ExaProb.log",
    #         restart=restart,
    #     )
    #
    # NEW: configure_logging wraps stdlib logging. Inside framework
    # modules, loggers come from get_logger(__name__).

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    configure_logging(
        level="info",
        logfile=WORKSPACE / "logbook3_ExaProb.log",
        append=(resume_pickle_path is not None),
    )

    # ============================================================
    # PARAMETER BOUNDS — Bounds replaces BOUND_LOW / BOUND_UP / NDIM
    # ============================================================
    #
    # OLD (ExaConstit_NSGA3.py):
    #
    #     IND_LOW = [150, 100, 50, 1500, 1e-5, 1e-3, 1e-4, 1e-5, 1e-6]
    #     IND_UP  = [200, 150, 100, 2500, 1e-3, 1e-1, 1e-2, 1e-3, 1e-4]
    #     BOUND_LOW = IND_LOW; BOUND_UP = IND_UP
    #     NDIM = len(BOUND_LOW)
    #
    # NEW: edit one PARAMS table. Bounds, names, archive metadata,
    # log output, and write_properties() all derive from this table.
    #
    # IMPORTANT USER CONTRACT:
    # PARAMS defines the optimized-parameter vector. The order here is
    # the gene order:
    #
    #   gene[0] -> PARAMS[0].name
    #   gene[1] -> PARAMS[1].name
    #   ...
    #
    # To add a fitted parameter, add one ParameterSpec below and then
    # use that name in write_properties(). Example: to fit the Voce
    # saturation strength, add
    #
    #   ParameterSpec("crss_sat", 80.0e-3, 180.0e-3, "GPa",
    #                 "CRSS saturation strength")
    #
    # and replace the fixed crss_sat value in write_properties() with
    # gene_dict["crss_sat"].

    PARAMS = [
        ParameterSpec(
            name="mprime",
            lower=0.01e0,
            upper=0.05e0,
            description="rate sensitivity exponent for power-law slip",
        ),
        ParameterSpec(
            name="h0",
            lower=200.0e-3,
            upper=500.0e-3,
            units="GPa",
            description="Voce hardening coefficient",
        ),
        ParameterSpec(
            name="crss0",
            lower=10.0e-3,
            upper=20.0e-3,
            units="GPa",
            description="initial critical resolved shear stress",
        ),
    ]

    param_names = [spec.name for spec in PARAMS]
    bounds = Bounds(
        lower=np.array([spec.lower for spec in PARAMS]),
        upper=np.array([spec.upper for spec in PARAMS]),
    )
    assert len(param_names) == bounds.n_params

    # ============================================================
    # SIMCASES — one per experiment, with FULL per-experiment dict
    # ============================================================
    #
    # OLD (ExaConstit_NSGA3.py, all the per-experiment parallel arrays):
    #
    #     ncpus = [4, 4]
    #     ngpus = [0, 0]
    #     nnodes = [1, 1]
    #     temperature_k = [DEP_UNOPT[0][0], DEP_UNOPT[1][0]]
    #     ori_file = ["voce_quats.ori", "voce_quats.ori"]
    #     strain_rate = [-1.0e-3, -1.0e-1]
    #     desired_strain = [-0.1301, -0.1001]
    #     timeout = [6 * 60, 6 * 60]
    #     t_final = [120.0, 1.0]
    #     dt_min = [0.001, 0.00001]
    #     dt_max = [1.0, 0.01]
    #     dt_scale = [0.125, 0.125]
    #     minmax_strain = [None, None]
    #     essential_vals = [[0.0, 0.0, 0.0, 0.0, 0.0, -0.001],
    #                       [0.0, 0.0, 0.0, 0.0, 0.0, -0.001]]
    #     mesh_lengths = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    #     mesh_cuts = [[1, 1, 1], [1, 1, 1]]
    #     p_refinement = [1, 1]
    #
    #     test_dataframe = {
    #         "experiments": exper_input_files,
    #         "temp_k": temperature_k, "strain_rate": strain_rate,
    #         "desired_strain": desired_strain, "ori_file": ori_file,
    #         "t_final": t_final, "dt_min": dt_min, "dt_max": dt_max,
    #         "dt_scale": dt_scale, "essential_vals": essential_vals,
    #         "mesh_lengths": mesh_lengths, "mesh_cuts": mesh_cuts,
    #         "p_refinement": p_refinement, "timeout": timeout,
    #         "minmax_strain": minmax_strain,
    #     }
    #     test_dataframe = pd.DataFrame(data=test_dataframe)
    #
    # NEW: one SimCase per row of the old DataFrame. Per-experiment
    # overlay (the old `test_dataframe` columns minus the experiment
    # file itself) goes in SimCase.case_data and gets substituted
    # into template_options.toml at render time. Per-experiment resource
    # requests (ncpus/ngpus/nnodes/timeout) go in the SimCase's
    # resource-override fields and flow through to the backend's
    # SimJobSpec.
    #
    # PLAIN-LANGUAGE VERSION:
    # One SimCase is one experiment you are trying to match. If you
    # measured the same material at two strain rates, use two SimCases.
    # If you measured tension and compression, use two SimCases. If
    # you measured three temperatures, use three SimCases.
    #
    # To add an experiment, copy one SimCase block and change:
    #   * label: a short name that will appear in logs
    #   * EXPERIMENT_FILES above: add the measured curve file
    #   * strain_rate: signed strain rate for the experiment
    #   * desired_strain and minmax_strain: comparison strain range
    #   * t_final/dt_min/dt_max/dt_scale: simulation time stepping
    #   * essential_vals: imposed boundary values in options.toml
    #   * ori_file / ORIENTATION_FILES if orientation changes
    #   * mesh_lengths, mesh_cuts, p_refinement if the RVE changes
    #   * num_tasks, cores_per_task, gpus_per_task if resource needs
    #     change for that case

    # ============================================================
    # PER-CASE CONSTANTS — what to put on each SimCase
    # ============================================================
    #
    # SimCase.case_data is the single mapping for everything that's
    # specific to one experiment but NOT part of the gene vector
    # being optimized. Whatever you put here flows to all three
    # framework components that need per-case context:
    #
    #   * the TEMPLATER substitutes %%key%% references in
    #     template_options.toml (and any other rendered file)
    #   * the PATH RESOLVER substitutes {key} references in
    #     working_dir_pattern and output_file_patterns
    #   * the PROPERTY WRITER reads sim_case.case_data inside the
    #     CallablePropertyWriter callable
    #
    # One dict, three readers. No need to remember which field a
    # particular variable belongs in.
    #
    # Common cases:
    #   * "different elastic constants per temperature" — put the
    #     constants directly in case_data as "c11", "c12", and
    #     "c44". The first shipped SimCase below does this explicitly;
    #     the second omits them and therefore uses write_properties()
    #     defaults.
    #   * "case-specific orientation file" — put the filename in
    #     case_data["ori_file"] so it lands in options.toml AND
    #     the property writer can read it.
    #   * "case-specific RVE name in the path" — put it in
    #     case_data["rve_name"] and use {rve_name} in
    #     working_dir_pattern.
    #
    # Per-case RESOURCE overrides (num_nodes / num_tasks /
    # gpus_per_task / duration_s / binary / binary_args) are
    # separate top-level fields on SimCase — see step 4a below.
    # Resources are consumed by the backend, not by templates or
    # paths, which is why they have dedicated fields rather than
    # living inside case_data.
    #
    # NOTE: TemplatePropertyWriter ALSO has an `extra_values=` field,
    # but those are constants applied to EVERY case (not per-case);
    # for per-case property-writer constants, read them out of
    # sim_case.case_data inside your CallablePropertyWriter callable.
    #
    # ILLUSTRATIVE EXAMPLE — not used in this run, but the pattern
    # to imitate when constants need to vary per simulation case
    # (elastic constants, lattice parameters, anisotropy ratios,
    # whatever else is gene-independent but case-dependent):
    #
    # PUT EVERY VARYING CONSTANT DIRECTLY IN case_data. No Python
    # lookup tables, no logic in the writer to map cases to values.
    # The data lives next to the SimCase that uses it; modifying
    # it later is one place to edit:
    #
    #     sim_cases = [
    #         SimCase(
    #             label="exp_298K",
    #             case_data={
    #                 # Loading + path-pattern fields:
    #                 "strain_rate":   -1.0e-3,
    #                 "temperature_k":  298.0,
    #                 "ori_file":      "voce_298k.ori",
    #                 "rve_name":      "grain_32",
    #                 # Per-case material constants — elastic moduli
    #                 # at this case's temperature, lattice params,
    #                 # whatever else varies. Just dictionary entries:
    #                 "c11": 168.4,
    #                 "c12": 121.4,
    #                 "c44":  75.4,
    #             },
    #         ),
    #         SimCase(
    #             label="exp_600K",
    #             case_data={
    #                 "strain_rate":   -1.0e-3,
    #                 "temperature_k":  600.0,
    #                 "ori_file":      "voce_600k.ori",
    #                 "rve_name":      "grain_32",
    #                 # Different elastic constants for this temperature.
    #                 # Same KEYS, different VALUES — that's all:
    #                 "c11": 156.0,
    #                 "c12": 117.0,
    #                 "c44":  72.0,
    #             },
    #         ),
    #     ]
    #
    # CONSUMING THIS DATA — three options, pick whichever fits your
    # sim code's input format. All three see the same case_data;
    # the framework wires it to all three readers automatically:
    #
    # OPTION 1 — let the templater put the constants in options.toml.
    # If your template_options.toml has %%c11%%, %%c12%%, %%c44%%
    # placeholders, the templater fills them from case_data. Zero
    # writer code; nothing to change in the writer at all.
    #
    # OPTION 2 — let the writer pull from case_data and write a
    # separate properties.txt. Useful if your sim code reads
    # material constants from a dedicated properties file rather
    # than options.toml. The writer reads case_data by key:
    #
    #     def write_properties(case_dir, gene, names, sim_case):
    #         path = case_dir / "properties.txt"
    #         data = sim_case.case_data
    #         lines = [
    #             f"c11 = {data['c11']}",
    #             f"c12 = {data['c12']}",
    #             f"c44 = {data['c44']}",
    #             # ... gene-derived values follow ...
    #         ]
    #         path.write_text("\n".join(lines) + "\n")
    #         return path
    #
    # OPTION 3 — split: some constants in options.toml via
    # templater (Option 1), some in properties.txt via writer
    # (Option 2). Same case_data dict, different consumers. The
    # framework doesn't impose a particular layout.
    #
    # ABOUT EXTRA KEYS: case_data may contain keys not referenced
    # by any template or path pattern. Those are silently ignored
    # by the templater (validated at template-load time, not at
    # data-load time) — the templater only complains when its
    # template has a %%key%% with no matching case_data entry,
    # which is exactly the typo-detection users want.

    # RESOURCE SHAPE PER CASE
    # -----------------------
    # These four fields tell the backend how much hardware ONE
    # simulation case should consume:
    #
    #   num_nodes       - nodes requested for this case
    #   num_tasks       - MPI ranks for this case
    #   cores_per_task  - CPU cores per MPI rank
    #   gpus_per_task   - GPUs per MPI rank
    #
    # Examples:
    #
    #   CPU-only, 4 MPI ranks on 1 node:
    #       num_nodes=1, num_tasks=4, cores_per_task=1, gpus_per_task=0
    #
    #   GPU run, 1 rank per GPU on a 4-GPU node:
    #       num_nodes=1, num_tasks=4, cores_per_task=1, gpus_per_task=1
    #
    #   Hybrid MPI+threads, 4 ranks each with 7 CPU cores and 1 GPU:
    #       num_nodes=1, num_tasks=4, cores_per_task=7, gpus_per_task=1
    #
    # FluxBackend consumes these fields directly when building the
    # jobspec. LocalBackend only launches local subprocesses, so
    # ``num_tasks > 1`` requires wiring in an MPI launcher there.
    # The shipped example keeps the old 4-rank CPU-only shape.
    sim_cases = [
        SimCase(
            label="exp1_quasi_static",
            case_data={
                # Loading condition / boundary conditions.
                "strain_rate":    -1.0e-3,
                "desired_strain": -0.1301,
                "temperature_k":   298.0,
                "ori_file":       "voce_quats.ori",    # filename IN case dir
                # Example of per-case material constants. These are
                # the same cubic elastic constants used as defaults
                # in write_properties(), so adding them here verifies
                # the case_data path without changing the shipped
                # numerical behavior. To calibrate or compare
                # experiments at different temperatures, put the
                # temperature-specific elastic constants on each
                # SimCase.
                "c11": 168.4e0,
                "c12": 121.4e0,
                "c44": 75.2e0,
                # Time stepping.
                "t_final": 120.0,
                "dt_min":    0.001,
                "dt_max":    1.0,
                "dt_scale":  0.125,
                # Essential (Dirichlet) BC values for each dof.
                # For this template, the six values are ordered like
                # the six velocity/strain-rate components expected by
                # the ExaConstit options file. The last entry controls
                # the z-direction loading used by this example, so it
                # matches strain_rate for uniaxial z compression.
                # Negative means compression; positive means tension.
                "essential_vals": [0.0, 0.0, 0.0, 0.0, 0.0, -0.001],
                # Mesh controls.
                "mesh_lengths": [1.0, 1.0, 1.0],
                "mesh_cuts":    [5, 5, 5],
                "p_refinement": 1,
                # Reference to the per-case properties file written
                # by CallablePropertyWriter below.
                "properties_file": "properties.txt",
                # Repo-local support files resolved from this example's
                # location so users can run from any working directory.
                "state_vars_file": str(STATE_VARS_FILE),
                "grain_file": str(GRAIN_FILE),
                # Optimization window. (lo, hi) absolute-value strain
                # bounds; the extractor crops both stress and slope
                # comparisons to this interval. The elastic regime
                # in metals is typically below ~0.002 strain — set
                # lo above that to skip elastic, leaving the optimizer
                # focused on the plastic portion the gene parameters
                # actually control. Use None on either side for "no
                # bound." None for the whole field disables windowing.
                "minmax_strain": (-0.01, -0.13),
            },
            # Per-case resource request. This example keeps the old
            # CPU-only 4-rank shape:
            #   1 node
            #   4 MPI ranks
            #   1 CPU core per rank
            #   0 GPUs per rank
            #
            # To move to GPUs under Flux, typically change only
            # ``gpus_per_task`` (and, if needed, ``cores_per_task``).
            num_nodes=1,
            num_tasks=4,           # ncpus[0]
            cores_per_task=1,
            gpus_per_task=0,       # ngpus[0]
            duration_s=6 * 60,     # timeout[0] (seconds)
            # Per-SimCase args for the binary's CLI.
            binary_args=("-opt", "options.toml"),
        ),
        SimCase(
            label="exp2_high_rate",
            case_data={
                "strain_rate":    -1.0e-1,
                "desired_strain": -0.1001,
                "temperature_k":   298.0,
                "ori_file":       "voce_quats.ori",
                "t_final": 1.0,
                "dt_min":    0.00001,
                "dt_max":    0.01,
                "dt_scale":  0.125,
                # Same convention as exp1: the last entry imposes
                # z-direction compression at the faster rate.
                "essential_vals": [0.0, 0.0, 0.0, 0.0, 0.0, -0.1],
                "mesh_lengths": [1.0, 1.0, 1.0],
                "mesh_cuts":    [5, 5, 5],
                "p_refinement": 1,
                "properties_file": "properties.txt",
                "state_vars_file": str(STATE_VARS_FILE),
                "grain_file": str(GRAIN_FILE),
                "minmax_strain": (-0.01, -0.1),
            },
            num_nodes=1,
            num_tasks=4,
            cores_per_task=1,
            gpus_per_task=0,
            duration_s=6 * 60,
            binary_args=("-opt", "options.toml"),
        ),
    ]

    # ============================================================
    # OBJECTIVES — stress + slope per experiment (2 * NEXP)
    # ============================================================
    #
    # OLD (ExaConstit_Problems.py::evaluate):
    #
    #     f[iobj * 2]     = RMSE(sim_stress, exp_stress) / np.std(exp_stress)
    #     f[iobj * 2 + 1] = RMSE(sim_slope,  exp_slope)  / np.std(exp_slope)
    #
    # NEW: one ObjectiveSpec per objective value, each pointing at
    # its SimCase by index. Custom evaluator classes defined at the
    # bottom of this file reproduce the OLD std-normalized RMSE
    # exactly. The framework's built-in StressStrainObjective returns
    # plain RMSE without normalization — that's a judgment call the
    # framework doesn't make silently.
    #
    # Choosing the stress/strain comparison:
    #   * stress_column must match the loading direction in the
    #     simulation output: Sxx for x loading, Syy for y loading,
    #     Szz for z loading.
    #   * strain_rate should keep the experimental sign convention.
    #     This example uses negative strain for compression.
    #   * minmax_strain in each SimCase crops the comparison window.
    #     Use it to ignore elastic transients, machine seating, noisy
    #     tails, or strain ranges where the experiment is unreliable.
    #   * Some evaluators score using absolute strain/stress magnitudes
    #     so tension/compression signs do not dominate RMSE. Plots and
    #     physical interpretation still need the correct sign.

    objective_specs = []
    for i, (exp_file, sim_case) in enumerate(zip(EXPERIMENT_FILES, sim_cases)):
        exp_df = _load_experimental_csv(exp_file)
        # Pass strain_rate signed so the extractor's
        # strain = strain_rate * time produces signed strain
        # matching the loading direction (negative for compression,
        # positive for tension). Stripping the sign here would
        # flip the strain axis relative to the stress axis on
        # compression cases, leaving plotted curves looking like
        # tension responses with negative stress — wrong sign on
        # the strain side, mismatched against experimental data.
        # Evaluators that score on magnitudes still work because
        # they apply np.abs themselves before computing RMSE.
        strain_rate = float(sim_case.case_data["strain_rate"])

        # Optional optimization window. case_data["minmax_strain"] is
        # a (lo, hi) pair or None: lo=None or hi=None means "no
        # bound on that side." If both bounds are missing the
        # extractor sees window=None and returns the full curve.
        # Anything real, including upper bounds derived from
        # "desired_strain" if that's all the user wants, can go
        # here.
        minmax = sim_case.case_data.get("minmax_strain")
        if minmax is not None and (minmax[0] is not None or minmax[1] is not None):
            # Substitute 0 for missing lo and a huge number for
            # missing hi — both are absolute-value bounds, so 0 means
            # "include from origin" and 1e9 means "no upper limit".
            lo = 0.0 if minmax[0] is None else float(abs(minmax[0]))
            hi = 1e9 if minmax[1] is None else float(abs(minmax[1]))
            window = (lo, hi)
        else:
            # Fall back to "desired_strain" as the upper bound when
            # minmax_strain isn't set — this preserves the old
            # convention where users only declared the maximum strain
            # they cared about.
            window = (0.0, abs(sim_case.case_data["desired_strain"]))

        # Build one extractor per SimCase. The window crops the
        # extracted (strain, stress) arrays before any objective
        # sees them, so both stress and slope evaluators below
        # automatically respect the user's chosen interval.
        extractor = StressStrainExtractor(
            stress_output="avg_stress",
            stress_column="Szz",       # z-axis load — override for x/y loading
            strain_source="time_rate",
            strain_rate=strain_rate,
            time_column="Time",
            window=window,
        )

        stress_eval = _StdNormalizedStressEvaluator(
            experimental=exp_df, extractor=extractor,
        )
        slope_eval = _StdNormalizedSlopeEvaluator(
            experimental=exp_df, extractor=extractor,
        )
        objective_specs.extend([
            ObjectiveSpec(stress_eval, sim_case=i, label=f"stress_{i + 1}"),
            ObjectiveSpec(slope_eval,  sim_case=i, label=f"slope_{i + 1}"),
        ])

    n_obj = len(objective_specs)

    # ============================================================
    # PROPERTY WRITER — CallablePropertyWriter matching the old pattern
    # ============================================================
    #
    # OLD: a user-supplied Python function wrote a per-case properties
    # file. The ExaProb class invoked that callable during each
    # evaluation, right before running mechanics.
    #
    # NEW: CallablePropertyWriter is the direct analogue. Define your
    # write function with signature
    # ``(case_dir, gene, param_names, sim_case) -> Path`` and hand it
    # to the writer. Everything else about the framework is unchanged.
    #
    # The writer's callable receives the SimCase for this evaluation,
    # so per-experiment logic (different orientation files, temp-
    # dependent derived values, etc.) works naturally.
    #
    # This example writer is intentionally minimal. It is enough to
    # exercise the workflow and reproduce the old quick-start behavior,
    # but it is not a complete material-model authoring interface.
    # For production calibration work, port or write a material
    # generator that names each physical parameter explicitly, similar
    # to the older Matgen-style scripts. That keeps units, model
    # assumptions, and parameter ordering visible to the mechanics
    # user instead of hiding them in a list of numbers.

    def write_properties(case_dir: Path, gene: Sequence[float],
                         names: Sequence[str], sim_case: SimCase) -> Path:
        """Write material-model properties for one case.

        Produces a `properties.txt` file whose format matches what
        the old `ExaConstit_Problems.py` wrote. Adapt the body to
        whatever format your mechanics binary actually reads. The
        framework does not care about the file's internal format,
        only that the file exists and is referenced correctly in
        options.toml. ExaConstit does care: the numeric order here
        must match the active material model selected in
        template_options.toml.

        ``sim_case`` lets this callable behave differently per
        experiment. Read per-case constants out of
        ``sim_case.case_data`` (set up the SimCase) — typical
        examples include ``temperature_k`` for a temp-dependent
        elastic-constant lookup, ``ori_file`` for case-specific
        orientation files, or any other field you put in the
        SimCase's case_data dict. See the SimCase definitions
        above for the conventions; the comment block before
        ``sim_cases = [...]`` explains how case_data flows to the
        templater, path resolver, and property writer.
        """
        path = case_dir / "properties.txt"
        # Zip gene values with their names so the output is both
        # machine-readable AND human-auditable for debugging.
        gene_dict = dict(zip(names, gene))

        # ------------------------------------------------------------------
        # Voce material properties for this example
        # ------------------------------------------------------------------
        #
        # This section mirrors the older Matgen-style scripts: assign
        # each physical quantity to a named Python variable first, then
        # write the final numeric list in the exact order ExaConstit
        # expects for the selected material model.
        #
        # Per-simulation constants:
        # SimCase.case_data is available here as ``case_data``. To make
        # a constant vary by experiment, add the key to each SimCase and
        # read it with ``case_data.get("name", default)`` below.
        #
        # Example for temperature-dependent elastic constants:
        #
        #   In each SimCase.case_data, add:
        #       "c11": 168.4, "c12": 121.4, "c44": 75.2
        #
        #   Then the c11/c12/c44 assignments below automatically use
        #   the case-specific values. If the keys are absent, the
        #   defaults shown here are used.
        #
        # Example for a temperature-driven reference energy:
        #
        #   simulation_temperature_k = case_data["temperature_k"]
        #   reference_temperature_k = simulation_temperature_k
        #
        # This example keeps the historical 300 K reference energy so
        # it reproduces the old quick-start behavior.
        case_data = sim_case.case_data

        # Basic thermo-physical constants.
        density = 8.920e-6
        heat_capacity_cv = 0.003435984
        tolerance = 1.0e-10

        # Cubic elastic constants. Put c11/c12/c44 in SimCase.case_data
        # to vary these by experiment, temperature, orientation set, or
        # any other case-specific condition.
        c11 = float(case_data.get("c11", 168.4e0))
        c12 = float(case_data.get("c12", 121.4e0))
        c44 = float(case_data.get("c44", 75.2e0))

        # Voce hardening and power-law slip parameters.
        # Values pulled from gene_dict are fitted by NSGA-III. The
        # remaining values are fixed model constants for this example.
        mu = (c11 - c12) / 2.0
        nu = c44
        voigt_shear = 0.2 * (2.0 * mu + 3.0 * nu)
        reuss_shear = (mu * nu) / (nu + 3.0 * (mu - nu) * 0.2)
        # Shear modulus calculation if not available in literature
        shear_modulus = (voigt_shear + reuss_shear) / 2.0
        mprime = float(gene_dict["mprime"])
        gdot0 = 1.0e0
        h0 = float(gene_dict["h0"])
        crss0 = float(gene_dict["crss0"])
        crss_sat = 122.4e-3
        crss_sat_scaling_exponent = 0.0
        crss_sat_scaling_coefficient = 5.0e9
        hardening_initial = crss0

        # Equation-of-state style constants used by the current model
        # input. reference_internal_energy is tied to a reference
        # temperature, not necessarily the SimCase's loading temperature.
        gruneisen_parameter = 0.0
        reference_temperature_k = 300.0
        reference_internal_energy = -heat_capacity_cv * reference_temperature_k

        # IMPORTANT: this order is the file format. Keep it aligned
        # with the active ExaCMech material shortcut/model in
        # template_options.toml. The names above are for humans; the
        # file written below remains one numeric value per line.
        properties = [
            density,
            heat_capacity_cv,
            tolerance,
            c11,
            c12,
            c44,
            shear_modulus,
            mprime,
            gdot0,
            h0,
            crss0,
            crss_sat,
            crss_sat_scaling_exponent,
            crss_sat_scaling_coefficient,
            hardening_initial,
            gruneisen_parameter,
            reference_internal_energy,
        ]
        path.write_text(
            "\n".join(f"{value:.12g}" for value in properties) + "\n"
        )
        return path

    property_writer = CallablePropertyWriter(
        func=write_properties, dest_hint="properties.txt",
    )

    # ============================================================
    # TEMPLATER — renders options.toml per case + copies ori file
    # ============================================================
    #
    # The CaseTemplater renders text files with %%key%% substitution.
    # TWO targets here:
    #
    #   1. template_options.toml -> options.toml, with ALL the
    #      SimCase.case_data substituted.
    #   2. voce_quats.ori -> voce_quats.ori (plain copy, no
    #      substitution needed since .ori files are static).
    #
    # The orientation file's source path is per-SimCase, so we build
    # the target list dynamically per case. The ``TemplateTarget``
    # per_case_source mapping lets a single TemplateTarget resolve
    # differently per SimCase's index. See the framework's
    # CaseTemplater docstring for alternative patterns.

    # For maximum clarity in a short example, we use a single
    # templater with one global target (the options.toml render)
    # and copy the ori files in as part of each SimCase's
    # case_data by passing the source path — the framework
    # resolves it naturally because %%ori_file%% is a string in
    # template_options.toml.
    #
    # In production you typically have orientation files that
    # differ per SimCase; the simplest way is to place them in a
    # known location OUTSIDE the case dir and reference by absolute
    # path in template_options.toml, so no copy is needed. The
    # example uses that pattern: each SimCase sets
    # ``ori_file = "voce_quats.ori"`` as a filename IN the case dir,
    # and a second TemplateTarget copies it in.
    templater_targets = [
        TemplateTarget(source=MASTER_TOML, dest="options.toml"),
    ]
    # Only include the ori-file copy if the user actually pointed
    # at a real file — matches how the old code handled optional
    # orientation inputs.
    for i, ori_path in enumerate(ORIENTATION_FILES):
        if ori_path.is_file():
            templater_targets.append(
                TemplateTarget(
                    source=ori_path,
                    dest="voce_quats.ori",
                    substitute=False,  # binary-safe copy, no %%key%% pass
                )
            )
            break   # single global copy is fine if both experiments share it
    templater = CaseTemplater(templater_targets)

    # ============================================================
    # PATH RESOLVER — where each case's working dir and outputs live
    # ============================================================
    #
    # The PathResolver maps a (generation, gene, obj) CaseContext onto
    # a directory and a set of output filenames. Two concerns:
    #
    # 1. ``working_dir_pattern`` — template for the case's directory.
    #    Placeholders available: {generation}, {gene}, {obj}, plus
    #    anything in SimCase.case_data. The old code used
    #    "gen_{gen}/gene_{gene}_obj_{obj}", which we preserve so a
    #    mixed old+new workspace doesn't collide.
    #
    # 2. ``output_file_patterns`` — logical names -> template paths.
    #    These are the files you want to read BACK after the sim.
    #    Each key becomes a valid argument to reader.read() later,
    #    and each {working_dir} inside the pattern is expanded via
    #    the working_dir_pattern above.
    #
    # Think of output_file_patterns as "which output files does the
    # reader care about, and where does the binary write them?" The
    # filenames can differ from the binary's hard-coded output names
    # via a simple rename step; here we match the old names directly.

    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress":   "{working_dir}/results/options/avg_stress_global.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad_global.txt",
        },
        root=WORKSPACE,
    )

    # ============================================================
    # RESULT READER — which output columns matter, how to parse them
    # ============================================================
    #
    # TextTableReader is built for whitespace-separated numeric output
    # like ExaConstit's avg_stress.txt. One TextTableSpec per logical
    # output name (the keys match PathResolver.output_file_patterns).
    # Each spec declares:
    #
    #   columns  - name each column in disk order; the resulting
    #              DataFrame uses these as .columns. This is where you
    #              say "column Szz" vs the old hardcoded [:, 2]
    #              indexing in ExaConstit_Problems.py.
    #   required - if True (default) a missing or empty file fails
    #              the case; if False the reader silently skips it
    #              and the evaluator handles the missing DataFrame.
    #
    # The extractor above picks from these DataFrames by column name.
    # Adjust column order to match your binary's output format exactly,
    # or use `comment="#"` / `delimiter=","` / etc. for non-default
    # formats (see TextTableSpec docstring for the full option set).

    reader = TextTableReader({
        # avg_stress_global.txt: Time + Volume + 6 Cauchy-stress
        # components. ExaConstit writes its volume-averaged output
        # files with a ``# Time  Volume  ...`` header prefix followed
        # by calc-type-specific columns. For stress the columns are
        # Sxx/Syy/Szz/Sxy/Sxz/Syz (capital S, x/y/z axis indexing —
        # Load direction here is z so stress_column="Szz" above.
        # See ExaConstit/src/postprocessing/postprocessing_file_manager.hpp
        # GetVolumeAverageHeader for the authoritative column lists.
        "avg_stress": TextTableSpec(columns=[
            "Time", "Volume",
            "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz",
        ]),
        # avg_def_grad_global.txt: Time + Volume + 9 F-components.
        # Written by mechanics but optional for calibration; we keep
        # it around for post-processing plots.
        "avg_def_grad": TextTableSpec(
            columns=[
                "Time", "Volume",
                "F11", "F12", "F13",
                "F21", "F22", "F23",
                "F31", "F32", "F33",
            ],
            required=False,
        ),    
    })

    # ============================================================
    # ARCHIVE + CLEANUP (opt-in, new feature)
    # ============================================================
 
    archive = None
    archive_run_id = None
    if not args.no_archive:
        archive = ArchiveDB(WORKSPACE / "calibration.db")
        archive.open()
        if resume_pickle_path is None:
            # Fresh run — create a new archive row and get back its
            # UUID. That UUID is threaded into the Problem below so
            # gene records write to the right row.
            archive_run_id = archive.start_run(
                seed=args.seed,
                param_names=param_names,
                objective_labels=[spec.label for spec in objective_specs],
                config={
                    "mechanics_binary": str(MECHANICS_BINARY),
                    "master_toml": str(MASTER_TOML),
                    "experiment_files": [str(p) for p in EXPERIMENT_FILES],
                },
            )
        else:
            # Resume — do NOT call start_run(). A fresh UUID here
            # would conflict with the one baked into the pickle,
            # and the library would refuse the mismatch with a
            # ValueError. Leave archive_run_id = None; the library
            # reads the pickled UUID out of the checkpoint and
            # assigns it to Problem.archive_run_id on our behalf.
            # The existing archive row keeps receiving gene records
            # under its original UUID, uninterrupted.
            pass

    # ============================================================
    # BACKEND
    # ============================================================
    #
    # OLD: normal_map.map_custom ran cases sequentially via subprocess.
    #
    # NEW: pluggable backend. LocalBackend for workstation use;
    # FluxBackend for HPC. The per-SimCase resource fields above
    # (num_nodes, num_tasks, cores_per_task, gpus_per_task, duration_s)
    # flow through to whichever backend you pick.
    #
    # FluxBackend:
    #   * turns each SimCase into a Flux jobspec
    #   * requests the CPU/GPU shape encoded on that SimCase
    #   * handles MPI launch internally via Flux's job shell
    #
    # LocalBackend:
    #   * starts local subprocesses on the current host
    #   * is useful for serial debugging or small workstation runs
    #   * does NOT magically create MPI ranks by itself; multi-rank
    #     cases need an explicit MPI launcher in the backend config
    #
    # NOTE on mpi_launcher: ExaConstit's ``mechanics`` binary uses
    # MPI internally, so any SimCase with ``num_tasks > 1`` must run
    # under an MPI launcher. LocalBackend will refuse at submit time
    # if ``num_tasks > 1`` and ``mpi_launcher`` is not set — prevents
    # the failure mode where a four-rank request silently ran
    # single-rank. Set this to "mpirun", "srun", "jsrun", or
    # whichever launcher is on your PATH. If your launcher uses a
    # non-standard ntasks flag (jsrun wants ``--nrs``, lrun wants
    # ``-T``) pass ``mpi_launcher_ntasks_flag`` too.

    if args.backend == "flux":
        if not HAS_FLUX:
            parser.error(
                "--backend=flux requested, but this Python interpreter "
                "cannot import Flux's bindings. Use the same interpreter "
                "for both install and run, and ensure the Flux module is "
                "visible (for example via "
                "PYTHONPATH=/usr/lib64/flux/python3.12 on this system)."
            )
        from workflow_common.backends.flux_backend import FluxBackend

        backend = FluxBackend()
    else:
        # Debug-oriented default. The SimCases in this example request
        # 4 MPI ranks, so local mode assumes ``mpirun`` is available
        # on PATH and uses it to launch those ranks. For a simpler
        # serial debug run, change each SimCase to ``num_tasks=1`` and
        # set mpi_launcher=None here. If your system uses ``srun`` or
        # another launcher instead of ``mpirun``, replace the string
        # below with that command.
        backend = LocalBackend(max_workers=4, mpi_launcher="mpirun")

    # On a Flux-managed HPC allocation:
    #
    #     from workflow_common import FluxBackend
    #     backend = FluxBackend()
    #
    # (The per-case resource fields on each SimCase drive Flux's
    # job submission; no global resource config is needed. For
    # example, a SimCase with ``num_tasks=4, cores_per_task=7,
    # gpus_per_task=1`` requests 4 MPI ranks, 28 CPU cores total,
    # and 4 GPUs for that one simulation. Flux handles MPI rank
    # launching internally; no mpi_launcher argument is involved
    # for FluxBackend.)

    # The driver automatically attaches a ProgressReporter per
    # generation, rendering a line like:
    #
    #   gen 3/100 | 47/288 (16.3%) | running 4/4 | cores 16/16 | elapsed 2m14s | ETA 11m42s
    #
    # Interactive runs get in-place updates; captured-stdout runs
    # (HPC job logs) get one line per update. Turn off with
    # ``RunConfig(show_progress=False)`` if you want silent mode.

    # ============================================================
    # ASSEMBLE THE PROBLEM
    # ============================================================
    #
    # ProblemConfig now holds only the FALLBACK resource settings.
    # If any SimCase leaves a resource field as None, the value
    # here is used. In this example all fields are per-case, so
    # ProblemConfig's resource numbers don't actually govern
    # anything — they're shown for completeness.

    problem = Problem(
        config=ProblemConfig(
            binary=MECHANICS_BINARY,
            binary_args=("-opt", "options.toml"),
            num_nodes=1, num_tasks=4,
            cores_per_task=1, gpus_per_task=0,
            duration_s=3600,
            stdout="stdout.log", stderr="stderr.log",
            required_outputs=("results/options/avg_stress_global.txt",),
        ),
        param_names=param_names,
        sim_cases=sim_cases,
        objective_specs=objective_specs,
        templater=templater,
        property_writer=property_writer,
        resolver=resolver,
        backend=backend,
        reader=reader,
        failure_handler=InfinityFailureHandler(),
        manifest=Manifest(WORKSPACE / "manifest.jsonl"),
        archive=archive,
        archive_run_id=archive_run_id,
    )

    # ============================================================
    # RUN CONFIG — every knob is 1:1 with the old globals
    # ============================================================
    #
    # OLD -> NEW mapping:
    #
    #     NGEN = 100                -> n_generations=100
    #     UNSGA3 = True             -> unsga3=True
    #     p = [10, 0]               -> ref_dirs_partitions=(10, 0)
    #     scaling = [1, 0]          -> ref_dirs_scaling=(1.0, 0.0)
    #     seed = <arg>              -> seed=args.seed
    #     mat_eta = 30.0            -> mate_eta=30.0
    #     mut_eta = 20.0            -> mut_eta=20.0
    #     indpb = 1.0 / NDIM        -> mut_indpb=None  (default = 1/n_params)
    #     fail_limit = 10           -> fail_limit=10
    #     Imin = round(NGEN / 2)    -> imin_fraction=0.5
    #     stop_limit = 5            -> stop_limit=5
    #     checkpoint_freq = 1       -> checkpoint_freq=1
    #
    # cleanup_keep_generations=2 is NEW and requires the archive.

    ref_dirs_partitions = (3, 0)
    ref_dirs_scaling = (1.0, 0.0)

    # Compute NPOP the same way the driver will, using the framework's
    # public helpers. This replaces a hand-rolled P/H/NPOP calculation
    # that's easy to get wrong for the two-hyperplane case (p_inner > 0).
    _ref_points, h_count = build_reference_points(
        n_obj, ref_dirs_partitions, ref_dirs_scaling,
    )
    npop = derive_population_size(n_obj, h_count)

    # Pull n_generations out up front so the summary print and the
    # RunConfig below share one source of truth. Same story for any
    # other knob you want to surface at the top of the log.
    n_generations = 30

    print(f"\nNumber of experiments        = {len(sim_cases)}")
    print(f"Number of objectives         = {n_obj}")
    print(f"Number of parameters         = {bounds.n_params}")
    print(f"Number of generations        = {n_generations}")
    print(f"Number of reference points H = {h_count}")
    print(f"Population size NPOP         = {npop}")
    print(f"Total simulation runs        = "
          f"{npop * len(sim_cases) * n_generations} "
          f"(pop x n_sim_cases x gens)\n")

    config = RunConfig(
        n_generations=n_generations,
        unsga3=True,
        ref_dirs_partitions=ref_dirs_partitions,
        ref_dirs_scaling=ref_dirs_scaling,
        seed=args.seed,
        mate_eta=30.0,
        mut_eta=20.0,
        mut_indpb=None,                      # = 1/n_params, old default
        fail_limit=10,
        imin_fraction=0.5,                   # Imin = NGEN / 2
        stop_limit=5,
        checkpoint_dir=WORKSPACE / "checkpoint_files",
        checkpoint_freq=1,
        resume_from=resume_pickle_path,
        track_hypervolume=True,
        cleanup_keep_generations=2 if not args.no_archive else None,
        # log_dir defaults to checkpoint_dir if set (here, the
        # checkpoint_files dir above); override if you want logs
        # somewhere else. Two files are written per generation:
        #   * logbook1_stats.log — aggregate fitness stats (avg /
        #     std / min / max) plus ND / GD / HV for multi-objective
        #   * logbook2_solutions.log — every individual's gene
        #     vector + fitness
        # Both are DEAP-style tab-delimited text — greppable,
        # less-able, and parseable by the pre-refactor driver's
        # downstream tooling. Set write_logbook_files=False to
        # skip them.
    )

    # ============================================================
    # GO
    # ============================================================
    #
    # TIP: for a CLI peek at the SQLite archive during or after
    # the run (if --archive is on), try:
    #
    #   python -m workflows.optimization.inspect_archive \
    #       ./calibration_run --gens
    #
    # Add --genes --gen N --pareto-only to see just the Pareto
    # front at a specific generation. --format csv pipes into
    # Excel / LibreOffice Calc. --format json is friendly to
    # notebook post-processing.

    try:
        result = run_nsga3(problem, bounds, config)
    finally:
        if archive is not None:
            archive.close()

    # ============================================================
    # QUICK POST-RUN SUMMARY
    # ============================================================

    print(f"\nRun complete. Generations run: {result.generations_run}")
    print(f"Final pop size: {len(result.final_pop)}")
    print(f"Pareto front size: {len(result.pareto_front)}")
    if result.stopped_early:
        print("Stopped early via ND == NPOP criterion.")

    from workflow_common.postprocess import best_solution_eudist
    fits = np.array([ind.fitness.values for ind in result.final_pop])
    best_idx = best_solution_eudist(fits, nsmallest=1)[0]
    best_gene = result.final_pop[best_idx]
    print(f"\nBest gene fitness: {best_gene.fitness.values}")
    print("Best gene parameters:")
    for name, val in zip(param_names, list(best_gene)):
        print(f"  {name:10s} = {val:.6g}")


# ============================================================================
# Custom evaluators that reproduce the OLD std-normalized RMSE
# ============================================================================
#
# The pre-refactor ExaProb computed:
#
#     f[iobj * 2]     = RMSE(sim_stress, exp_stress) / np.std(exp_stress)
#     f[iobj * 2 + 1] = RMSE(sim_slope,  exp_slope)  / np.std(exp_slope)
#
# where slope = np.diff(stress) / np.diff(strain). The framework's
# built-in StressStrainObjective does plain RMSE without normalization;
# these two classes reproduce the old normalization exactly. If you
# prefer a different normalization (the old ExaConstit_Problems.py has
# commented alternatives with std / IQR / min-max / mean denominators),
# change the `denom = ...` line below and leave everything else alone.
#
# An evaluator is any object with an ``evaluate(results, ctx) -> float``
# method. No base class to inherit from; duck typing suffices.


class _StdNormalizedStressEvaluator:
    """RMSE(sim_stress, exp_stress) / std(exp_stress).

    Brings the experimental curve onto the simulation's strain
    grid via the framework's PCHIP smoother (the right tool: it
    can't overshoot, so a smoothed exp curve will not introduce
    stress values that never appeared in the source data). This
    is the direction the original ExaConstit calibration code
    used and it's the one that matters for scoring: the
    optimizer is computing sim error at sim's own sample points,
    not at noisy raw exp points.

    The extractor is responsible for cropping the simulated
    curve to the user's optimization window (via its ``window``
    field). This evaluator just compares whatever the extractor
    returns against the experimental reference, restricted to
    the common strain range.
    """

    def __init__(
        self,
        experimental: pd.DataFrame,
        extractor: StressStrainExtractor,
    ):
        self.experimental = experimental
        self.extractor = extractor

    def evaluate(self, results: Any, ctx: Any) -> float:
        sim_strain, sim_stress = self.extractor.extract(results)
        sim_strain = np.abs(sim_strain)
        sim_stress = np.abs(sim_stress)
        exp_strain = np.abs(self.experimental.iloc[:, 0].to_numpy())
        exp_stress = np.abs(self.experimental.iloc[:, 1].to_numpy())

        # Mask SIM (not exp) to the common strain range — we want
        # sim's sample points as the comparison grid since the
        # optimizer is implicitly fitting at sim's resolution.
        lo = max(float(sim_strain.min()), float(exp_strain.min()))
        hi = min(float(sim_strain.max()), float(exp_strain.max()))
        sim_mask = (sim_strain >= lo) & (sim_strain <= hi)
        if sim_mask.sum() < 2:
            return float("inf")
        sim_strain_in = sim_strain[sim_mask]
        sim_stress_in = sim_stress[sim_mask]

        # Bring exp onto sim's grid using PCHIP. ``strict_monotonic
        # =False`` accepts mild noise in exp_strain (a tiny backward
        # step from sampling jitter) by sorting; it does NOT handle
        # genuinely non-monotonic data. Mechanical-test exp data is
        # cleaned and clipped before reaching this code, per the
        # standard calibration workflow.
        smoother = PchipSmoother(strict_monotonic=False)
        exp_at_sim = smoother.sample_at(
            exp_strain, exp_stress, sim_strain_in,
        ).y

        denom = float(np.std(exp_at_sim))
        if denom <= 0:
            return float("inf")
        residual = sim_stress_in - exp_at_sim
        return float(np.sqrt(np.mean(residual ** 2)) / denom)


class _StdNormalizedSlopeEvaluator:
    """RMSE(sim_slope, exp_slope) / std(exp_slope).

    Same architecture as the stress evaluator: brings exp onto
    sim's strain grid via PCHIP smoothing, then compares
    finite-difference slopes on that common grid.
    """

    def __init__(
        self,
        experimental: pd.DataFrame,
        extractor: StressStrainExtractor,
    ):
        self.experimental = experimental
        self.extractor = extractor

    def evaluate(self, results: Any, ctx: Any) -> float:
        sim_strain, sim_stress = self.extractor.extract(results)
        sim_strain = np.abs(sim_strain)
        sim_stress = np.abs(sim_stress)
        exp_strain = np.abs(self.experimental.iloc[:, 0].to_numpy())
        exp_stress = np.abs(self.experimental.iloc[:, 1].to_numpy())

        lo = max(float(sim_strain.min()), float(exp_strain.min()))
        hi = min(float(sim_strain.max()), float(exp_strain.max()))
        sim_mask = (sim_strain >= lo) & (sim_strain <= hi)
        if sim_mask.sum() < 3:
            return float("inf")
        sim_strain_in = sim_strain[sim_mask]
        sim_stress_in = sim_stress[sim_mask]

        smoother = PchipSmoother(strict_monotonic=False)
        exp_at_sim = smoother.sample_at(
            exp_strain, exp_stress, sim_strain_in,
        ).y

        # Slopes via finite difference on the common grid.
        # ``np.diff`` shrinks length by one; using sim's strain
        # spacing is correct because exp has now been sampled at
        # exactly those points.
        diff_strain = np.diff(sim_strain_in)
        sim_slope = np.diff(sim_stress_in) / diff_strain
        exp_slope = np.diff(exp_at_sim) / diff_strain

        denom = float(np.std(exp_slope))
        if denom <= 0:
            return float("inf")
        residual = sim_slope - exp_slope
        return float(np.sqrt(np.mean(residual ** 2)) / denom)

def _load_experimental_csv(path: Path) -> pd.DataFrame:
    """Load a two-column (strain, stress) file. Any whitespace separator."""
    arr = np.loadtxt(path, dtype=float, ndmin=2)
    return pd.DataFrame({"strain": arr[:, 0], "stress": arr[:, 1]})


if __name__ == "__main__":
    main()
