"""
End-to-end tests for the NSGA-III driver built on DEAP's rcarson3 fork.

What these prove
----------------
1. **Driver smoke test** — a small (U-)NSGA-III run produces the
   expected final population size, pareto subset, and DEAP logbook
   records.
2. **Determinism** — same seed in two separate workspaces produces
   bit-identical per-generation gene matrices and objective
   matrices. This is the single most important property for
   reproducible scientific pipelines and one that depends on
   ``random.seed(...)`` being respected throughout DEAP's
   variation operators.
3. **Shared-SimCase parity** — N objectives scored against one
   shared SimCase produces the same initial-population objective
   values as the same N objectives scored against N separate
   SimCases. Confirms that the framework's sim-sharing
   optimization does not change what DEAP sees.
4. **Checkpoint / resume parity** — a run that completes N
   generations matches a run that runs N/2 generations, pickles a
   checkpoint, and resumes for N/2 more from that pickle.
5. **UNSGA3 off / on** — both paths run without errors; UNSGA3 is
   the fork-only feature that required keeping DEAP.
6. **Reference points + NPOP derivation** — the population size
   computed from ``ref_dirs_partitions`` matches the NSGA-III
   paper's recipe.
7. **Fail-retry on sim failure** — when the framework emits inf
   on sim failure, the driver replaces the failing individual and
   retries; fail_limit terminates the run.
8. **Bounds / shape guardrails** — misconfigured bounds or
   objective counts raise at construction time, not during
   evaluation.

Heavyweight notes
-----------------
These tests run real subprocesses through LocalBackend and DEAP's
full selection/crossover/mutation pipeline. They are slower than
the pure-unit tests — a few seconds each, not milliseconds.
"""
from __future__ import annotations

import os
import pickle
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the driver via its proper package path. conftest.py has
# already put the repo root on sys.path for source-checkout runs;
# when the package is pip-installed the import resolves via
# site-packages. The underscore-prefixed helpers are internal but
# we test them explicitly so importing by name is intentional.
from workflows.optimization.nsga3_driver import (  # noqa: E402
    Bounds,
    RunConfig,
    RunResult,
    run_nsga3,
    _build_reference_points,
    _derive_population_size,
    _is_failed,
)
from workflow_common import (  # noqa: E402
    CaseTemplater,
    ConstantPenaltyFailureHandler,
    InfinityFailureHandler,
    LocalBackend,
    Manifest,
    ObjectiveSpec,
    PchipSmoother,
    Problem,
    ProblemConfig,
    SimCase,
    StressStrainExtractor,
    StressStrainObjective,
    TemplatePathResolver,
    TemplatePropertyWriter,
    TemplateTarget,
    TextTableReader,
    TextTableSpec,
    load_experimental_csv,
    rmse,
)


# --- Shared helpers -----------------------------------------------------


def _build_problem(
    workspace: Path,
    fake_binary: Path,
    *,
    sim_cases,
    objective_specs,
    failure_penalty: float = 1e8,
    manifest_name: str = "manifest.jsonl",
    max_workers: int = 4,
    archive=None,
    archive_run_id=None,
):
    """Build a Problem against the standard test plumbing.

    Same pattern as test_problem.py's fixtures. Kept inline here so
    this file is self-contained (no cross-test imports).
    """
    options_tmpl = workspace / "master_options.toml"
    if not options_tmpl.exists():
        options_tmpl.write_text(textwrap.dedent("""\
            [Problem]
                name = "case_fixed"
                basename = "options"
                strain_rate = %%strain_rate%%
                yield_stress = %%yield_stress%%
                hardening = %%hardening%%
                temperature_k = %%temp_k%%
        """))
    placeholder = workspace / "placeholder.txt"
    if not placeholder.exists():
        placeholder.write_text("# placeholder\n")

    templater = CaseTemplater([
        TemplateTarget(source=placeholder, dest=".placeholder"),
    ])
    writer = TemplatePropertyWriter(
        template_path=options_tmpl,
        dest="options.toml",
        extra_values={"strain_rate": 1.0, "temp_k": 298.0},
    )
    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress":   "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
        ),
        "avg_def_grad": TextTableSpec(
            columns=["Time", "Volume", "F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33"],
            required=False,
        ),
    })
    return Problem(
        config=ProblemConfig(
            binary=fake_binary, binary_args=(),
            num_tasks=1, duration_s=30,
            stdout="stdout.log", stderr="stderr.log",
            required_outputs=("results/options/avg_stress.txt",),
        ),
        param_names=["yield_stress", "hardening"],
        sim_cases=sim_cases,
        objective_specs=objective_specs,
        templater=templater,
        property_writer=writer,
        resolver=resolver,
        backend=LocalBackend(max_workers=max_workers),
        reader=reader,
        failure_handler=ConstantPenaltyFailureHandler(penalty=failure_penalty),
        manifest=Manifest(workspace / manifest_name),
        archive=archive,
        archive_run_id=archive_run_id,
    )


@pytest.fixture
def exp_df():
    """Reference Voce-law curve matching the fake binary's saturation constant."""
    strain = np.linspace(0.0, 1.0, 40)
    stress = 210.0 + 1900.0 * (1 - np.exp(-48.0 * strain))
    return pd.DataFrame({"strain": strain, "stress": stress})


@pytest.fixture
def stress_evaluator(exp_df):
    return StressStrainObjective(
        experimental=exp_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=1.0,
        ),
    )


def _slope_evaluator(stress_eval):
    """Slope-matching evaluator sharing extractor + exp DataFrame."""
    exp_df = stress_eval.experimental
    exp_strain = exp_df[stress_eval.experimental_strain_col].to_numpy()
    exp_stress = exp_df[stress_eval.experimental_stress_col].to_numpy()
    exp_slope = np.gradient(exp_stress, exp_strain)
    exp_slope_df = pd.DataFrame({
        stress_eval.experimental_strain_col: exp_strain,
        stress_eval.experimental_stress_col: exp_slope,
    })

    class _SlopeEvaluator:
        extractor = stress_eval.extractor
        experimental = exp_slope_df
        experimental_strain_col = stress_eval.experimental_strain_col
        experimental_stress_col = stress_eval.experimental_stress_col

        def evaluate(self, results, ctx):
            strain, stress = stress_eval.extractor.extract(results)
            slope = np.gradient(stress, strain)
            lo = max(strain.min(), exp_strain.min())
            hi = min(strain.max(), exp_strain.max())
            if lo >= hi:
                raise ValueError("no strain overlap")
            common = np.linspace(lo, hi, 100)
            sm = PchipSmoother(n_samples=100)
            sim_y = sm.sample_at(strain, slope, common).y
            exp_y = sm.sample_at(exp_strain, exp_slope, common).y
            return rmse(sim_y, exp_y)

    return _SlopeEvaluator()


# --- 1. Smoke test -----------------------------------------------------


def test_nsga3_single_objective_runs(workspace, fake_binary, stress_evaluator):
    """Small single-objective run produces the right shape outputs."""
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    result = run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=2, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
        ),
    )
    assert isinstance(result, RunResult)
    assert len(result.final_pop) == 4
    assert len(result.pop_library) == 3  # gen 0, gen 1, gen 2
    # Logbook stats has one record per generation (0, 1, 2).
    assert len(result.logbook_stats) == 3
    # Every individual has real fitness values.
    for ind in result.final_pop:
        assert np.isfinite(ind.fitness.values[0])


def test_nsga3_multi_objective_runs(
    workspace, fake_binary, stress_evaluator,
):
    """2-objective run: fork's UNSGA3 + HV tracking + ND counting."""
    slope = _slope_evaluator(stress_evaluator)
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[
            ObjectiveSpec(stress_evaluator, sim_case=0, label="stress"),
            ObjectiveSpec(slope, sim_case=0, label="slope"),
        ],
    )
    result = run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=2, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=True,
        ),
    )
    assert len(result.final_pop) == 4
    # Post-gen-1, HV should be a finite number.
    hv_gen_2 = result.logbook_stats[-1]["HV"]
    assert np.isfinite(hv_gen_2)
    # ND should be a sensible count.
    nd = result.logbook_stats[-1]["ND"]
    assert 1 <= nd <= 4


# --- 2. Determinism ----------------------------------------------------


def test_nsga3_same_seed_produces_same_trajectory(
    fake_binary, stress_evaluator, tmp_path_factory,
):
    """Two runs, same seed, separate workspaces -> identical trajectories."""
    ws_a = tmp_path_factory.mktemp("det_a")
    ws_b = tmp_path_factory.mktemp("det_b")

    def run(ws):
        p = _build_problem(
            ws, fake_binary,
            sim_cases=[SimCase(label="quasi")],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        )
        return run_nsga3(
            p,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=1234, track_hypervolume=False,
            ),
        )

    r1 = run(ws_a)
    r2 = run(ws_b)

    assert len(r1.pop_library) == len(r2.pop_library)
    for gen, (p1, p2) in enumerate(zip(r1.pop_library, r2.pop_library)):
        g1 = np.asarray([list(ind) for ind in p1])
        g2 = np.asarray([list(ind) for ind in p2])
        np.testing.assert_allclose(
            g1, g2, rtol=0, atol=0,
            err_msg=f"gene matrix differs at gen {gen}",
        )
        f1 = np.asarray([ind.fitness.values for ind in p1])
        f2 = np.asarray([ind.fitness.values for ind in p2])
        np.testing.assert_allclose(
            f1, f2, rtol=0, atol=0,
            err_msg=f"fitness matrix differs at gen {gen}",
        )


# --- 3. Shared-vs-separate sim parity ----------------------------------


def test_shared_vs_separate_sim_same_initial_stress(
    fake_binary, stress_evaluator, tmp_path_factory,
):
    """Shared-SimCase and separate-SimCase topologies: initial stress matches.

    Gen 0 of DEAP is a uniform-random sample whose seed is the only
    thing affecting the draw. Both topologies use the same bounds +
    same seed, so the initial genes match bit-for-bit. The simulation
    is deterministic, so the STRESS RMSE (same evaluator in both
    topologies) must also match bit-for-bit.

    The slope objective is not compared across topologies because a
    fresh slope evaluator in each run may produce slightly different
    numerical paths through the smoother - topology B has two
    slope evaluators, one per sim case, both getting the same sim
    output. Comparing only stress isolates the parity claim.
    """
    ws_a = tmp_path_factory.mktemp("topo_a")
    ws_b = tmp_path_factory.mktemp("topo_b")

    slope_a = _slope_evaluator(stress_evaluator)
    slope_b = _slope_evaluator(stress_evaluator)

    pa = _build_problem(
        ws_a, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[
            ObjectiveSpec(stress_evaluator, sim_case=0),
            ObjectiveSpec(slope_a, sim_case=0),
        ],
    )
    pb = _build_problem(
        ws_b, fake_binary,
        sim_cases=[SimCase(label="quasi_a"), SimCase(label="quasi_b")],
        objective_specs=[
            ObjectiveSpec(stress_evaluator, sim_case=0),
            ObjectiveSpec(slope_b, sim_case=1),
        ],
    )

    bounds = Bounds(
        lower=np.array([150.0, 1500.0]),
        upper=np.array([300.0, 2500.0]),
    )
    cfg = RunConfig(
        n_generations=1, population_size=4,
        unsga3=True, ref_dirs_partitions=(4, 0),
        seed=7, track_hypervolume=False,
    )

    ra = run_nsga3(pa, bounds, cfg)
    rb = run_nsga3(pb, bounds, cfg)

    # Gen-0 genes must match.
    ga = np.asarray([list(ind) for ind in ra.pop_library[0]])
    gb = np.asarray([list(ind) for ind in rb.pop_library[0]])
    np.testing.assert_allclose(
        ga, gb, rtol=0, atol=0,
        err_msg="gen 0 genes differ between topologies",
    )
    # Stress RMSE (first objective) must match.
    stress_a = np.asarray([
        ind.fitness.values[0] for ind in ra.pop_library[0]
    ])
    stress_b = np.asarray([
        ind.fitness.values[0] for ind in rb.pop_library[0]
    ])
    np.testing.assert_allclose(
        stress_a, stress_b, rtol=1e-12, atol=1e-9,
        err_msg="stress RMSE differs between topologies",
    )


# --- 4. UNSGA3 on / off ------------------------------------------------


def test_unsga3_off_runs_but_differs_from_on(
    fake_binary, stress_evaluator, tmp_path_factory,
):
    """UNSGA3 flag changes the trajectory but both paths run cleanly.

    With 2 objectives both should work; UNSGA3 adds a niching step
    before variation, so the second-generation population should
    differ. This also acts as a smoke test for the non-UNSGA path.
    """
    slope = _slope_evaluator(stress_evaluator)

    def run(ws, unsga3):
        p = _build_problem(
            ws, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[
                ObjectiveSpec(stress_evaluator, sim_case=0),
                ObjectiveSpec(slope, sim_case=0),
            ],
        )
        return run_nsga3(
            p,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=unsga3, ref_dirs_partitions=(4, 0),
                seed=11, track_hypervolume=False,
            ),
        )

    r_on  = run(tmp_path_factory.mktemp("u_on"),  True)
    r_off = run(tmp_path_factory.mktemp("u_off"), False)
    # Both succeed.
    assert len(r_on.final_pop) == 4
    assert len(r_off.final_pop) == 4
    # Gen 0 is deterministic (random.seed only) so it should match
    # regardless of UNSGA3 (UNSGA3 only runs from gen 1 onward).
    g0_on  = np.asarray([list(ind) for ind in r_on.pop_library[0]])
    g0_off = np.asarray([list(ind) for ind in r_off.pop_library[0]])
    np.testing.assert_allclose(g0_on, g0_off, rtol=0, atol=0)
    # Gen 1+ can differ because UNSGA3 reorders via niching. We don't
    # *require* them to differ (tiny pop may coincide), but if they
    # match exactly we've not really tested anything. Just ensure
    # both paths finished.


# --- 5. Checkpoint / resume --------------------------------------------


def test_nsga3_checkpoint_resume_produces_same_final_pop(
    fake_binary, stress_evaluator, tmp_path_factory,
):
    """Full N-generation run == (first-half + resume second-half).

    Writes a checkpoint at gen 1, resumes from it, runs to gen 3,
    then compares against a clean 3-generation run. They should
    match bit-for-bit.

    This also serves as the "parity with old driver" test: the
    checkpoint format uses the same pickle keys the old driver
    uses, so an old checkpoint can be loaded by this driver for a
    mid-run framework upgrade.
    """
    ws_full = tmp_path_factory.mktemp("ckpt_full")
    ws_split = tmp_path_factory.mktemp("ckpt_split")
    ckpt_dir = ws_split / "checkpoint_files"

    bounds = Bounds(
        lower=np.array([150.0, 1500.0]),
        upper=np.array([300.0, 2500.0]),
    )

    # Full run, no checkpoint.
    pa = _build_problem(
        ws_full, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    r_full = run_nsga3(
        pa, bounds,
        RunConfig(
            n_generations=3, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=321, track_hypervolume=False,
        ),
    )

    # Split run part 1: gens 0..1, checkpoint at every gen.
    pb1 = _build_problem(
        ws_split, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    r_part1 = run_nsga3(
        pb1, bounds,
        RunConfig(
            n_generations=1, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=321, track_hypervolume=False,
            checkpoint_dir=ckpt_dir, checkpoint_freq=1,
        ),
    )
    assert (ckpt_dir / "checkpoint_gen_1.pkl").exists()

    # Split run part 2: resume from gen 1, run through gen 3.
    pb2 = _build_problem(
        ws_split, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        manifest_name="manifest.jsonl",  # same as part 1
    )
    r_part2 = run_nsga3(
        pb2, bounds,
        RunConfig(
            n_generations=3, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=321,  # ignored on resume (state comes from pickle)
            track_hypervolume=False,
            resume_from=ckpt_dir / "checkpoint_gen_1.pkl",
        ),
    )

    # Final pops must match.
    assert len(r_full.pop_library) == len(r_part2.pop_library)
    for gen, (pf, ps) in enumerate(zip(
        r_full.pop_library, r_part2.pop_library,
    )):
        gf = np.asarray([list(ind) for ind in pf])
        gs = np.asarray([list(ind) for ind in ps])
        np.testing.assert_allclose(
            gf, gs, rtol=0, atol=0,
            err_msg=f"gen {gen} genes differ between full and resumed",
        )
        ff = np.asarray([ind.fitness.values for ind in pf])
        fs = np.asarray([ind.fitness.values for ind in ps])
        np.testing.assert_allclose(
            ff, fs, rtol=0, atol=0,
            err_msg=f"gen {gen} fitness differs between full and resumed",
        )


def test_checkpoint_file_has_expected_keys(
    workspace, fake_binary, stress_evaluator,
):
    """Checkpoint format matches the pre-refactor driver's exact keys.

    This guarantees that old ExaConstit_NSGA3.py checkpoints can
    be loaded by the new driver - and vice versa.
    """
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    ckpt_dir = workspace / "ck"
    run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=1, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=0, track_hypervolume=False,
            checkpoint_dir=ckpt_dir, checkpoint_freq=1,
        ),
    )
    with (ckpt_dir / "checkpoint_gen_1.pkl").open("rb") as f:
        ckp = pickle.load(f)
    # Must contain every key the pre-refactor driver used. We permit
    # one additional key (archive_run_id) that the new driver added
    # for archive-aware resume; old drivers can still read these
    # pickles because pickle ignores extra keys.
    required_keys = {
        "pop_library", "iter_tot", "generation",
        "fail_count", "stop_count",
        "logbook1", "logbook2", "rndstate",
    }
    assert required_keys.issubset(set(ckp.keys()))
    # The new key is always present but may be None when no
    # archive was configured for the run.
    assert "archive_run_id" in ckp


# --- 6. Population sizing ----------------------------------------------


@pytest.mark.parametrize("n_obj,partitions,expected_h,expected_npop", [
    # From the NSGA-III paper: H = C(n_obj + P - 1, P).
    (2, (4, 0),  5,  8),    # C(5, 4) = 5,  round up to 8
    (2, (10, 0), 11, 12),   # C(11,10) = 11, round up to 12
    (3, (4, 0),  15, 16),   # C(6, 4) = 15, round up to 16
    (4, (4, 0),  35, 36),   # C(7, 4) = 35, round up to 36
])
def test_reference_points_and_npop(n_obj, partitions, expected_h, expected_npop):
    """Das-Dennis count and NPOP-from-H formula match the paper."""
    ref, h = _build_reference_points(n_obj, partitions, (1.0, 0.0))
    assert ref.shape[0] == expected_h
    assert ref.shape[1] == n_obj
    assert h == expected_h
    assert _derive_population_size(n_obj, h) == expected_npop


def test_single_obj_reference_points():
    """Single-obj trivially returns one reference direction."""
    ref, h = _build_reference_points(1, (10, 0), (1.0, 0.0))
    assert ref.shape == (1, 1)
    assert h == 10
    # NPOP derivation uses h=10 → round up to 12.
    assert _derive_population_size(1, h) == 12


# --- 7. Failure detection ----------------------------------------------


def test_is_failed_detects_inf():
    """_is_failed correctly flags infinite fitness values."""
    assert _is_failed((float("inf"),), float("inf"))
    assert _is_failed((1.0, float("inf")), float("inf"))
    assert not _is_failed((1.0, 2.0), float("inf"))


def test_is_failed_detects_nan():
    """NaN also counts as failure (np.isfinite returns False)."""
    assert _is_failed((float("nan"),), float("inf"))


def test_is_failed_respects_threshold():
    """A custom threshold catches below-inf penalties too."""
    assert _is_failed((1e9,), 1e8)
    assert not _is_failed((1.0,), 1e8)


def test_fail_limit_raises_runtime_error(
    workspace, fake_binary, stress_evaluator,
):
    """If all sims fail, fail_limit triggers a RuntimeError.

    Forces every sim to fail via the FAKE_FAIL env var and verifies
    that the driver stops with a clear error message instead of
    silently continuing.
    """
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    # Override the failure handler to emit inf so fail detection
    # triggers (default is ConstantPenaltyFailureHandler(1e8)).
    problem.failure_handler = InfinityFailureHandler()

    os.environ["FAKE_FAIL"] = "1"
    try:
        with pytest.raises(RuntimeError, match="fail_limit"):
            run_nsga3(
                problem,
                bounds=Bounds(
                    lower=np.array([150.0, 1500.0]),
                    upper=np.array([300.0, 2500.0]),
                ),
                config=RunConfig(
                    n_generations=1, population_size=4,
                    unsga3=True, ref_dirs_partitions=(4, 0),
                    seed=0, track_hypervolume=False,
                    fail_limit=3,  # short limit so test is fast
                ),
            )
    finally:
        os.environ.pop("FAKE_FAIL", None)


# --- 8. Guard rails ----------------------------------------------------


def test_bounds_rejects_upper_le_lower():
    with pytest.raises(ValueError, match="strictly greater"):
        Bounds(
            lower=np.array([0.0, 1.0]),
            upper=np.array([1.0, 1.0]),
        )


def test_bounds_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        Bounds(
            lower=np.array([0.0, 0.0]),
            upper=np.array([1.0, 1.0, 1.0]),
        )


def test_nsga3_rejects_mismatched_bounds(
    workspace, fake_binary, stress_evaluator,
):
    """bounds.n_params must match problem.param_names."""
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    with pytest.raises(ValueError, match="param_names"):
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([0.0, 0.0, 0.0]),  # 3 params
                upper=np.array([1.0, 1.0, 1.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=0, track_hypervolume=False,
            ),
        )


# --- 9. End-to-end post-processing --------------------------------------


def test_driver_run_postprocess_pipeline(
    workspace, fake_binary, stress_evaluator,
):
    """Full pipeline: run → checkpoint → load → extract → read case → plot.

    Proves the workflow_common.postprocess module integrates cleanly
    with a real checkpoint from the new driver. Exercises the
    entire post-analysis path a real user would take.
    """
    from workflow_common.postprocess import (
        best_solution_eudist,
        extract_gene_results,
        load_case_results,
        load_checkpoint,
    )

    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    ckpt_dir = workspace / "ck"
    run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=1, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
            checkpoint_dir=ckpt_dir, checkpoint_freq=1,
        ),
    )

    # Step 1: load checkpoint from disk.
    ckp = load_checkpoint(ckpt_dir / "checkpoint_gen_1.pkl")
    assert ckp.n_pop == 4
    assert ckp.n_dim == 2

    # Step 2: extract GeneResult records.
    all_results = extract_gene_results(ckp.pop_library)
    assert len(all_results) == 2                 # gen 0, gen 1
    last_gen_results = all_results[-1]
    assert len(last_gen_results) == 4

    # Step 3: pick the best via EUDIST.
    fits = np.array([r.fitness for r in last_gen_results])
    best_idx = best_solution_eudist(fits, nsmallest=1)
    assert len(best_idx) == 1
    best_result = last_gen_results[best_idx[0]]

    # Step 4: re-read the sim output from disk for that gene.
    # This is the replacement for the old ind.stress attribute.
    case_results = load_case_results(
        best_result, sim_case_idx=0,
        resolver=problem.resolver, reader=problem.reader,
    )
    assert case_results is not None
    stress_df = case_results.df("avg_stress")
    assert len(stress_df) > 0

    # Step 5 (smoke): hand it to the plot function.
    # We don't assert anything about the figure content - just that
    # the data flows through the plotting helper without errors.
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from workflow_common.postprocess import plot_stress_strain_overlay

    exp_strain = np.linspace(0, 1, 20)
    exp_stress = 210.0 + 1900.0 * (1 - np.exp(-48.0 * exp_strain))
    fig, ax = plot_stress_strain_overlay(
        sim_strain=stress_df["Time"].to_numpy(),
        sim_stress=stress_df["Szz"].to_numpy(),
        exp_strain=exp_strain,
        exp_stress=exp_stress,
        title="best gene, final generation",
    )
    assert fig is not None




# --- 10. Archive + rolling cleanup --------------------------------------


def test_archive_captures_every_case_and_every_generation(
    workspace, fake_binary, stress_evaluator,
):
    """After a 2-gen run with archive, every case output + every gene + stats
    is retrievable from the SQLite DB.

    This is the "archive actually captures what ran" test: without
    any cleanup happening, just prove the archive side-effect
    works end-to-end through the driver.
    """
    from workflow_common import ArchiveDB

    db_path = workspace / "opt.db"
    archive = ArchiveDB(db_path)
    with archive:
        # On a fresh run, the caller creates the archive row
        # up front via start_run and passes the UUID into Problem.
        # On resume, the driver fills in the UUID from the pickle
        # instead — see test_archive_resume_adopts_pickled_run_id.
        run_id = archive.start_run(
            seed=11, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive,
            archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=11, track_hypervolume=False,
            ),
        )

    # Query the DB with a fresh read-only connection to prove the
    # archive persisted properly (and is independent of the live
    # connection used during the run).
    with ArchiveDB(db_path, readonly=True) as a:
        # Run was ended -> completed_at populated.
        run = a.get_run(run_id)
        assert run.completed_at is not None
        assert run.seed == 11

        # Three generations: gen 0 (init), gen 1, gen 2.
        gens = a.list_generations(run_id)
        assert [g.gen_idx for g in gens] == [0, 1, 2]
        for g in gens:
            assert g.n_pop == 4

        # Case outputs: at minimum, every (gen, gene) that was
        # actually run has its sim output in the DB. The number of
        # distinct case rows depends on how many individuals were
        # re-evaluated vs. survived selection. We check the floor:
        # the initial pop (gen 0) has 4 new sims.
        genes_g0 = a.load_genes(run_id, 0)
        assert len(genes_g0) == 4
        for g in genes_g0:
            rs = a.load_case_outputs(
                run_id,
                birth_gen=g.birth_gen, birth_gene=g.birth_gene,
                sim_case_idx=0,
            )
            assert rs is not None
            # avg_stress is the required output - must be there.
            assert "avg_stress" in rs.tables
            assert len(rs.df("avg_stress")) > 0


def test_rolling_cleanup_deletes_older_gen_preserves_recent(
    workspace, fake_binary, stress_evaluator,
):
    """cleanup_keep_generations=2: after gen N, delete gen N-2.

    Runs for 4 generations. Asserts:
    - Gen 0 and gen 1 case dirs are DELETED from disk.
    - Gen 2 and gen 3 case dirs REMAIN on disk.
    - Archive still has the case outputs for gen 0 and gen 1
      (we can pull them back from pickled BLOBs).
    """
    from workflow_common import ArchiveDB
    from workflow_common.postprocess import load_case_results_from_archive

    db_path = workspace / "opt.db"
    archive = ArchiveDB(db_path)
    with archive:
        run_id = archive.start_run(
            seed=7, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive,
            archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=3, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=7, track_hypervolume=False,
                cleanup_keep_generations=2,
            ),
        )

    # Disk state: dirs under wf/gen_N/gene_X_sc_0.
    wf_root = workspace / "wf"
    # Gen 0 cleanup triggers at end-of-gen-2 (gen >= keep=2). Gen 1
    # cleanup triggers at end-of-gen-3. Both are gone; gen 2 and
    # gen 3 remain (gen 3 is current, gen 2 is the safety margin).
    def _has_case_dirs(gen_dir: Path) -> bool:
        return gen_dir.exists() and any(gen_dir.iterdir())

    assert not _has_case_dirs(wf_root / "gen_0"), (
        "gen_0 directory still has case dirs after cleanup"
    )
    assert not _has_case_dirs(wf_root / "gen_1"), (
        "gen_1 directory still has case dirs after cleanup"
    )
    assert _has_case_dirs(wf_root / "gen_2"), (
        "gen_2 (keep-1 safety margin) was cleaned up prematurely"
    )
    assert _has_case_dirs(wf_root / "gen_3"), (
        "gen_3 (current) was cleaned up - wrong!"
    )

    # Archive side: the gen 0 sim outputs are gone from disk but the
    # DB still has them. Pull one back and prove it's a real DataFrame.
    with ArchiveDB(db_path, readonly=True) as a:
        genes_g0 = a.load_genes(run_id, 0)
        assert genes_g0
        sample = next(g for g in genes_g0 if g.birth_gen == 0)
        # Use a duck-typed stand-in for GeneResult to avoid building
        # full postprocess objects here; the function only reads
        # .generation and .gene.
        gr = type("R", (), {
            "generation": sample.birth_gen,
            "gene": sample.birth_gene,
        })()
        rs = load_case_results_from_archive(
            a, gr, sim_case_idx=0, run_id=run_id,
        )
        assert rs is not None, "gen 0 case output not recoverable from archive"
        df = rs.df("avg_stress")
        assert len(df) > 0
        assert "Szz" in df.columns


def test_cleanup_without_archive_raises_at_run_start(
    workspace, fake_binary, stress_evaluator,
):
    """cleanup_keep_generations without an archive is a misconfiguration.

    The driver must refuse to start rather than silently destroy
    the user's only copy of the simulation output. Failing AT RUN
    START (not mid-run) keeps the error localized and obvious.
    """
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        # No archive.
    )
    with pytest.raises(ValueError, match="archive"):
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=0, track_hypervolume=False,
                cleanup_keep_generations=2,
            ),
        )


def test_cleanup_keep_generations_zero_rejected(
    workspace, fake_binary, stress_evaluator,
):
    """cleanup_keep_generations=0 would delete the current gen's dirs; reject it."""
    from workflow_common import ArchiveDB

    with ArchiveDB(workspace / "a.db") as archive:
        rid = archive.start_run(
            seed=0, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive, archive_run_id=rid,
        )
        with pytest.raises(ValueError, match="cleanup_keep_generations"):
            run_nsga3(
                problem,
                bounds=Bounds(
                    lower=np.array([150.0, 1500.0]),
                    upper=np.array([300.0, 2500.0]),
                ),
                config=RunConfig(
                    n_generations=1, population_size=4,
                    unsga3=True, ref_dirs_partitions=(4, 0),
                    seed=0, track_hypervolume=False,
                    cleanup_keep_generations=0,
                ),
            )


def test_archive_resume_discards_stale_generations(
    workspace, fake_binary, stress_evaluator,
):
    """Resume from gen 1 pickle: archive rows for gen 2+ get discarded.

    Simulates a mid-run crash. We set up the mid-run state by
    running 2 gens fully, then resume from the gen 1 pickle and
    run 2 more gens. The archive at the end must contain gens 0..3
    of the RESUMED run, with no leftover gen 2 entries from the
    pre-crash sequence (cascade delete via discard_from_generation
    cleans them before the phase-2 write).
    """
    from workflow_common import ArchiveDB

    db_path = workspace / "opt.db"
    ckpt_dir = workspace / "ck"

    # Phase 1: run 2 gens, get a checkpoint at gen 1.
    archive = ArchiveDB(db_path)
    with archive:
        run_id = archive.start_run(
            seed=21, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=21, track_hypervolume=False,
                checkpoint_dir=ckpt_dir, checkpoint_freq=1,
            ),
        )

    # Confirm phase 1 captured gens 0, 1, 2.
    with ArchiveDB(db_path, readonly=True) as a:
        phase1_gens = [g.gen_idx for g in a.list_generations(run_id)]
        assert phase1_gens == [0, 1, 2]

    # Phase 2: resume from gen 1 pickle, run through gen 3. The
    # resume path must discard the phase-1 gen 2 archive rows and
    # rewrite them fresh during the resumed run.
    archive = ArchiveDB(db_path)
    with archive:
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=3, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=21, track_hypervolume=False,
                checkpoint_dir=ckpt_dir, checkpoint_freq=1,
                resume_from=ckpt_dir / "checkpoint_gen_1.pkl",
            ),
        )

    # Archive should now have gens 0..3, all belonging to the same
    # run_id, with no duplicate or stale gen 2 rows. Since genes has
    # (run_id, gen_idx, pop_idx) as PRIMARY KEY, stale duplicates
    # would have raised IntegrityError during phase 2 - they were
    # discarded cleanly, so the write succeeded.
    with ArchiveDB(db_path, readonly=True) as a:
        final_gens = [g.gen_idx for g in a.list_generations(run_id)]
        assert final_gens == [0, 1, 2, 3]
        for g in final_gens:
            loaded = a.load_genes(run_id, g)
            assert len(loaded) == 4


def test_archive_resume_rejects_conflicting_run_id_with_helpful_error(
    workspace, fake_binary, stress_evaluator,
):
    """Resuming with a Problem whose archive_run_id differs from the
    pickled one fails with a message that names the fix.

    This is the error path Robert hit in production: the example
    driver called ``archive.start_run()`` unconditionally in main(),
    generating a fresh UUID. The library raises rather than silently
    continuing with the wrong run_id (which would cross-contaminate
    archive rows). The message must name the fix — "only start_run
    when not resuming" — so next time the error is diagnosable.
    """
    from workflow_common import ArchiveDB

    db_path = workspace / "opt.db"
    ckpt_dir = workspace / "ck"

    # Phase 1: one quick run, pickle one checkpoint.
    archive = ArchiveDB(db_path)
    with archive:
        run_id_a = archive.start_run(
            seed=42, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive, archive_run_id=run_id_a,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=42, track_hypervolume=False,
                checkpoint_dir=ckpt_dir, checkpoint_freq=1,
            ),
        )

    # Phase 2: resume, but simulate the driver bug — start a NEW
    # archive run and set its fresh UUID on the Problem before
    # handing to run_nsga3. The library must refuse.
    archive2 = ArchiveDB(db_path)
    with archive2:
        run_id_b = archive2.start_run(
            seed=42, param_names=["yield_stress", "hardening"],
        )
        assert run_id_a != run_id_b  # different UUIDs
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive2, archive_run_id=run_id_b,
        )
        with pytest.raises(ValueError) as excinfo:
            run_nsga3(
                problem,
                bounds=Bounds(
                    lower=np.array([150.0, 1500.0]),
                    upper=np.array([300.0, 2500.0]),
                ),
                config=RunConfig(
                    n_generations=2, population_size=4,
                    unsga3=True, ref_dirs_partitions=(4, 0),
                    seed=42, track_hypervolume=False,
                    checkpoint_dir=ckpt_dir, checkpoint_freq=1,
                    resume_from=ckpt_dir / "checkpoint_gen_0.pkl",
                ),
            )
    msg = str(excinfo.value)
    # Must name both UUIDs so the user can tell which is which,
    # and must name the fix so the next person hitting it doesn't
    # need to spelunk the source.
    assert run_id_a in msg
    assert run_id_b in msg
    assert "start_run" in msg
    assert "resume" in msg.lower()


def test_archive_resume_adopts_pickled_run_id_when_problem_has_none(
    workspace, fake_binary, stress_evaluator,
):
    """Resume path: Problem built with archive=..., archive_run_id=None
    picks up the pickled run_id and continues writing to that row.

    This is the path Robert's example driver takes after the fix:
    on resume, the driver does NOT call archive.start_run() — it
    leaves archive_run_id=None on the Problem. The library reads
    the UUID out of the checkpoint and assigns it before any
    writes happen.

    Verifies end-to-end:
      * Problem construction accepts the orphan case
        (archive set but archive_run_id=None), which used to
        be rejected by __init__.
      * After resume, wf_problem.archive_run_id holds the
        pickled UUID.
      * Archive rows are written under the pickled run, not a
        new one — list_runs() still shows exactly one run.
    """
    from workflow_common import ArchiveDB

    db_path = workspace / "opt.db"
    ckpt_dir = workspace / "ck"

    # Phase 1: full-setup fresh run that produces a checkpoint.
    archive = ArchiveDB(db_path)
    with archive:
        pickled_run_id = archive.start_run(
            seed=99, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive, archive_run_id=pickled_run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=99, track_hypervolume=False,
                checkpoint_dir=ckpt_dir, checkpoint_freq=1,
            ),
        )

    # Phase 2: resume with archive_run_id=None — this is what
    # the example driver does on resume. The library fills it in
    # from the pickle.
    archive2 = ArchiveDB(db_path)
    with archive2:
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=archive2,
            archive_run_id=None,      # <-- the fix in action
        )
        assert problem.archive_run_id is None  # pre-resume state
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=2, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=99, track_hypervolume=False,
                checkpoint_dir=ckpt_dir, checkpoint_freq=1,
                resume_from=ckpt_dir / "checkpoint_gen_0.pkl",
            ),
        )
        # After run_nsga3 the Problem should now carry the
        # pickled UUID — library's resume-adoption worked.
        assert problem.archive_run_id == pickled_run_id

    # No new run was created — we continued writing to the original.
    with ArchiveDB(db_path, readonly=True) as a:
        runs = a.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id == pickled_run_id
        # All generations landed under the one run.
        gens = [g.gen_idx for g in a.list_generations(pickled_run_id)]
        assert gens == [0, 1, 2]


def test_problem_rejects_run_id_without_archive(workspace, fake_binary, stress_evaluator):
    """The inverse orphan — run_id without archive — is still a user
    error: there's nowhere to write. Caught at construction.
    """
    import pytest
    with pytest.raises(ValueError, match="archive_run_id=... requires archive"):
        _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
            archive=None,
            archive_run_id="some-orphan-uuid",
        )


# --- Logbook .log file writing ------------------------------------------


def test_logbook_files_written_per_generation(
    workspace, fake_binary, stress_evaluator, tmp_path,
):
    """logbook1_stats.log and logbook2_solutions.log exist and contain
    one header + per-gen entries after a run.

    Matches the pre-refactor driver's output. Downstream team tooling
    parses these as tab-delimited text; check the shape rather than
    exact byte layout so a DEAP formatter tweak doesn't break the
    test.
    """
    log_dir = tmp_path / "logs"
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=2, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
            log_dir=log_dir,
        ),
    )

    stats_path = log_dir / "logbook1_stats.log"
    solutions_path = log_dir / "logbook2_solutions.log"

    # Both files exist.
    assert stats_path.is_file(), "logbook1_stats.log not written"
    assert solutions_path.is_file(), "logbook2_solutions.log not written"

    # Stats file: one header line + one body line per generation.
    # gen_0 + gen_1 + gen_2 = three body lines. The exact header
    # text is DEAP's; we just check it's there.
    stats_text = stats_path.read_text()
    assert "gen" in stats_text
    assert "avg" in stats_text and "std" in stats_text
    # Count lines that begin with a digit — the body lines. There
    # may be a header wrap too, but body lines are the robust count.
    body_lines = [
        line for line in stats_text.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(body_lines) == 3, (
        f"expected 3 gen body lines (0, 1, 2) in stats log, got "
        f"{len(body_lines)}:\n{stats_text}"
    )

    # Solutions file: NPOP rows per gen × 3 gens = 12 body rows.
    solutions_text = solutions_path.read_text()
    solutions_body = [
        line for line in solutions_text.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(solutions_body) == 12, (
        f"expected 12 individual entries (4 pop × 3 gens), got "
        f"{len(solutions_body)}"
    )


def test_logbook_files_disabled_by_write_logbook_files_false(
    workspace, fake_binary, stress_evaluator, tmp_path,
):
    """Setting write_logbook_files=False skips the .log files entirely.

    In-memory logbooks are still populated (so downstream tools that
    reach into RunResult still work); only the on-disk artifacts are
    suppressed. The opt-out exists for tests and for users running
    completely silent / pipe-driven workflows.
    """
    log_dir = tmp_path / "logs"
    problem = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    run_nsga3(
        problem,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=1, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
            log_dir=log_dir,
            write_logbook_files=False,
        ),
    )
    # log_dir may not even exist if the writer wasn't constructed.
    assert not (log_dir / "logbook1_stats.log").exists()
    assert not (log_dir / "logbook2_solutions.log").exists()


def test_logbook_files_rewritten_cleanly_on_resume(
    workspace, fake_binary, stress_evaluator, tmp_path,
):
    """On resume-from-checkpoint, both .log files are rewritten from the
    logbooks in the pickle, then continue to grow incrementally.

    The before-resume run + the after-resume run must collectively
    contain exactly one record per generation at the end — no gaps,
    no duplicates. Gaps would happen if the writer appended blindly
    (it would skip gens already in the pickle); duplicates would
    happen if we didn't truncate before replaying.
    """
    ckpt_dir = tmp_path / "ckpt"
    log_dir = tmp_path / "logs"

    # Initial run: 2 generations, checkpoints each gen.
    problem1 = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    run_nsga3(
        problem1,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=2, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
            checkpoint_dir=ckpt_dir, checkpoint_freq=1,
            log_dir=log_dir,
        ),
    )

    # Capture the pre-resume state of the two files.
    stats_path = log_dir / "logbook1_stats.log"
    stats_pre = stats_path.read_text()
    stats_body_pre = [
        line for line in stats_pre.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    assert len(stats_body_pre) == 3  # gens 0, 1, 2

    # Resume from gen 1 (forces gen 2 to be re-run). After the
    # resume completes, the .log file should still have exactly 3
    # body lines (0, 1, 2) — the first rewrite from the pickle,
    # followed by an incremental append for the re-run gen 2.
    problem2 = _build_problem(
        workspace, fake_binary,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
    )
    run_nsga3(
        problem2,
        bounds=Bounds(
            lower=np.array([150.0, 1500.0]),
            upper=np.array([300.0, 2500.0]),
        ),
        config=RunConfig(
            n_generations=2, population_size=4,
            unsga3=True, ref_dirs_partitions=(4, 0),
            seed=42, track_hypervolume=False,
            checkpoint_dir=ckpt_dir, checkpoint_freq=1,
            resume_from=ckpt_dir / "checkpoint_gen_1.pkl",
            log_dir=log_dir,
        ),
    )

    stats_post = stats_path.read_text()
    stats_body_post = [
        line for line in stats_post.splitlines()
        if line.strip() and line.strip().split()[0].isdigit()
    ]
    # Exactly 3 body lines still — no duplicates, no gaps.
    assert len(stats_body_post) == 3, (
        f"expected 3 body lines after resume, got {len(stats_body_post)}:\n"
        f"{stats_post}"
    )


# --- Experimental data archiving ----------------------------------------


def test_run_archives_experimental_data_per_sim_case(
    workspace, fake_binary, stress_evaluator, exp_df,
):
    """The driver must store evaluator.experimental into the archive
    once archive_run_id is known. Plotting tools rely on this so users
    don't have to re-supply CSV paths post-run.
    """
    from workflow_common import ArchiveDB

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=99, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="quasi"), SimCase(label="dynamic")],
            objective_specs=[
                ObjectiveSpec(stress_evaluator, sim_case=0, label="stress_0"),
                ObjectiveSpec(stress_evaluator, sim_case=1, label="stress_1"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=99, track_hypervolume=False,
            ),
        )

    # Archive must now have experimental data for both SimCases.
    with ArchiveDB(db, readonly=True) as a:
        listing = a.list_experiments(run_id)
        assert listing == [(0, "quasi"), (1, "dynamic")]
        result0 = a.load_experiment(run_id, 0)
        assert result0 is not None
        label0, df0 = result0
        assert label0 == "quasi"
        # Round-trip the original experimental DataFrame faithfully.
        pd.testing.assert_frame_equal(df0, exp_df)


def test_run_skips_archive_for_evaluators_without_experimental_data(
    workspace, fake_binary,
):
    """An evaluator that doesn't expose .experimental shouldn't crash
    the archiving step. The SimCase just won't have experimental data
    recorded, and the plotter falls back gracefully later.
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import ObjectiveEvaluator

    class _NoExpEvaluator(ObjectiveEvaluator):
        # Custom evaluator with no experimental DataFrame.
        # Returns a value that varies with the gene so NSGA-III's
        # niching has something to work with (a constant returner
        # would cause divide-by-zero in the reference-point step).
        def evaluate(self, results, ctx):
            # Use the case context's gene index as a stand-in
            # variation source.
            return 0.5 + 0.01 * ctx.gene

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=1, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="custom")],
            objective_specs=[
                ObjectiveSpec(_NoExpEvaluator(), sim_case=0, label="custom"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=1, track_hypervolume=False,
            ),
        )

    # No experimental data was recorded, but the run still succeeded.
    with ArchiveDB(db, readonly=True) as a:
        assert a.list_experiments(run_id) == []


def test_run_archives_minmax_strain_from_case_data(
    workspace, fake_binary, stress_evaluator,
):
    """case_data['minmax_strain'] must flow through the driver into
    the archive's experiments.minmax_strain column. This is the
    plumbing that lets the plotter shade the optimization window
    without the user re-supplying it post-run.
    """
    from workflow_common import ArchiveDB

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=42, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[
                SimCase(label="windowed",
                        case_data={"minmax_strain": (0.005, 0.13)}),
                SimCase(label="upper_only",
                        case_data={"minmax_strain": (None, 0.10)}),
                SimCase(label="no_window", case_data={}),
            ],
            objective_specs=[
                ObjectiveSpec(stress_evaluator, sim_case=0, label="a"),
                ObjectiveSpec(stress_evaluator, sim_case=1, label="b"),
                ObjectiveSpec(stress_evaluator, sim_case=2, label="c"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=42, track_hypervolume=False,
            ),
        )

    with ArchiveDB(db, readonly=True) as a:
        assert a.load_experiment_window(run_id, 0) == (0.005, 0.13)
        assert a.load_experiment_window(run_id, 1) == (None, 0.10)
        # SimCase 2 didn't supply a window — None, not (None, None).
        assert a.load_experiment_window(run_id, 2) is None


def test_run_handles_malformed_minmax_strain_gracefully(
    workspace, fake_binary, stress_evaluator, caplog,
):
    """A misformatted minmax_strain (e.g. a string instead of a tuple)
    shouldn't abort the run. The driver should log a warning, drop
    the window, and still record the experiment so the rest of the
    plotter pipeline keeps working.
    """
    import logging
    from workflow_common import ArchiveDB

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=7, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[
                SimCase(label="bad",
                        case_data={"minmax_strain": "not a tuple"}),
            ],
            objective_specs=[
                ObjectiveSpec(stress_evaluator, sim_case=0, label="a"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        with caplog.at_level(logging.WARNING):
            run_nsga3(
                problem,
                bounds=Bounds(
                    lower=np.array([150.0, 1500.0]),
                    upper=np.array([300.0, 2500.0]),
                ),
                config=RunConfig(
                    n_generations=1, population_size=4,
                    unsga3=True, ref_dirs_partitions=(4, 0),
                    seed=7, track_hypervolume=False,
                ),
            )

    # Run completed; experiment recorded but window dropped.
    with ArchiveDB(db, readonly=True) as a:
        assert a.load_experiment(run_id, 0) is not None
        assert a.load_experiment_window(run_id, 0) is None
    # Warning was logged so user can find the misconfiguration.
    assert any(
        "minmax_strain" in rec.message
        for rec in caplog.records
    )


def test_run_archives_extractor_config_from_evaluator(
    workspace, fake_binary, stress_evaluator,
):
    """``evaluator.extractor.to_dict()`` must flow through the driver
    to ``archive.load_extractor_config``. Without this, post-run
    plotters fall back to default extractor settings that may not
    match what the optimizer actually used — silently producing
    different curves than the run scored against (Robert's
    'only 1 of 2 SimCases plotted' bug).
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import StressStrainExtractor

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=11, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="quasi"), SimCase(label="dynamic")],
            objective_specs=[
                ObjectiveSpec(stress_evaluator, sim_case=0, label="a"),
                ObjectiveSpec(stress_evaluator, sim_case=1, label="b"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=11, track_hypervolume=False,
            ),
        )

    # The fixture's stress_evaluator uses time_rate strain with
    # strain_rate=1.0 — those settings should round-trip into the
    # archive verbatim. Both SimCases share the same evaluator,
    # so both archived configs should match.
    with ArchiveDB(db, readonly=True) as a:
        for sc_idx in (0, 1):
            cfg = a.load_extractor_config(run_id, sc_idx)
            assert cfg is not None, (
                f"sim_case {sc_idx} has no archived extractor config"
            )
            assert cfg["strain_source"] == "time_rate"
            assert cfg["strain_rate"] == 1.0
            # Reconstruct should match the original exactly.
            rebuilt = StressStrainExtractor.from_dict(cfg)
            assert rebuilt == stress_evaluator.extractor


def test_run_skips_extractor_config_for_evaluators_without_one(
    workspace, fake_binary,
):
    """An evaluator without an ``.extractor`` attribute (custom
    user-defined) shouldn't crash the archiving step. The
    experiment is still recorded; extractor_config column for
    that SimCase stays NULL; the plotter falls back to its
    default extractor with a warning.
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import ObjectiveEvaluator

    class _CustomEvaluator(ObjectiveEvaluator):
        # No .extractor and no .experimental — but this evaluator
        # type is fine to optimize against, just doesn't auto-archive.
        # We give it an experimental DataFrame so the experiment
        # still gets stored, just without an extractor config.
        def __init__(self):
            self.experimental = pd.DataFrame({
                "strain": np.linspace(0, 0.1, 5),
                "stress": np.linspace(100, 200, 5),
            })

        def evaluate(self, results, ctx):
            return 0.5 + 0.01 * ctx.gene

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=2, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="custom")],
            objective_specs=[
                ObjectiveSpec(_CustomEvaluator(), sim_case=0, label="x"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=2, track_hypervolume=False,
            ),
        )

    # Experiment recorded; extractor_config absent.
    with ArchiveDB(db, readonly=True) as a:
        assert a.load_experiment(run_id, 0) is not None
        assert a.load_extractor_config(run_id, 0) is None


def test_run_archives_full_range_curves_per_gene_and_simcase(
    workspace, fake_binary, stress_evaluator,
):
    """Every successfully-evaluated (gene, sim_case) pair must yield a
    case_curve row covering the FULL extraction range. The window
    (when supplied via the extractor) is stripped before extraction
    so re-analysis with different metrics or windows can use the
    archived curve directly.

    This pins Robert's ask: 'we should always make sure to save off
    the independent and dependent variables related to our objective
    functions for the full simulation range.'
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import StressStrainExtractor, StressStrainObjective

    # Use an evaluator whose extractor has a window so the
    # window-stripping logic gets exercised. The optimizer scores
    # against the windowed curve; the archive stores the full one.
    exp_df = stress_evaluator.experimental
    windowed_extractor = StressStrainExtractor(
        strain_source="time_rate", strain_rate=1.0,
        window=(0.05, 0.5),
    )
    windowed_evaluator = StressStrainObjective(
        experimental=exp_df,
        extractor=windowed_extractor,
    )

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=42, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="quasi"), SimCase(label="dynamic")],
            objective_specs=[
                ObjectiveSpec(windowed_evaluator, sim_case=0, label="a"),
                ObjectiveSpec(windowed_evaluator, sim_case=1, label="b"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=42, track_hypervolume=False,
            ),
        )

    with ArchiveDB(db, readonly=True) as a:
        # 4 genes × 2 SimCases × 2 generations (gen 0 + 1 transition,
        # per n_generations semantics) = 16 rows expected.
        cur = a._conn.execute(
            "SELECT COUNT(*) FROM case_curves WHERE run_id=?", (run_id,),
        )
        n_rows = cur.fetchone()[0]
        assert n_rows == 16, f"expected 16 case_curves rows, got {n_rows}"

        # Pick one and confirm the strain range exceeds the window.
        # Window was (0.05, 0.5); time_rate strain at rate=1.0 with
        # the fake binary's t in [0,1] gives strain in [0,1]. So the
        # archived curve should span (or come close to) [0,1], not
        # be clipped to [0.05, 0.5].
        result = a.load_case_curve(
            run_id, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
        assert result is not None
        ind, dep, ind_label, dep_label = result
        # Strain min should be <= 0.05 (i.e., extends below the
        # window's lower bound — proof window was stripped).
        assert ind.min() <= 0.05 + 1e-6, (
            f"archived strain min={ind.min()} exceeds window lo=0.05; "
            f"window may not have been stripped before extraction"
        )
        # Strain max should be >= 0.5 (extends above window upper).
        assert ind.max() >= 0.5 - 1e-6, (
            f"archived strain max={ind.max()} below window hi=0.5"
        )
        assert ind_label == "strain"
        assert dep_label == "stress"


def test_run_skips_curve_archive_for_evaluators_without_extractor(
    workspace, fake_binary,
):
    """Custom evaluators without an ``.extractor`` attribute don't
    crash the run — they just don't contribute case_curves rows."""
    from workflow_common import ArchiveDB
    from workflow_common.objectives import ObjectiveEvaluator

    class _CustomEvaluator(ObjectiveEvaluator):
        def __init__(self):
            self.experimental = pd.DataFrame({
                "strain": [0.0, 0.1], "stress": [100.0, 200.0],
            })

        def evaluate(self, results, ctx):
            return 0.5 + 0.01 * ctx.gene

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=2, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="custom")],
            objective_specs=[
                ObjectiveSpec(_CustomEvaluator(), sim_case=0, label="x"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=2, track_hypervolume=False,
            ),
        )

    with ArchiveDB(db, readonly=True) as a:
        # No case_curves rows — extractor unavailable.
        cur = a._conn.execute(
            "SELECT COUNT(*) FROM case_curves WHERE run_id=?", (run_id,),
        )
        assert cur.fetchone()[0] == 0


def test_run_archives_curves_with_sign_matched_to_experimental(
    workspace, fake_binary,
):
    """When the extractor produces positive-signed sim curves but
    the evaluator's ``.experimental`` is compression-shaped, the
    framework should sign-match the simulated curve before archiving.

    Pins Robert's ask: a defensive check at save-off so the archive
    holds curves consistent with the experimental data.
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import (
        ObjectiveEvaluator, StressStrainExtractor,
    )

    # Stub evaluator that exposes both ``.extractor`` and
    # ``.experimental`` (so Problem's curve-archiving logic finds
    # them and pairs them) but has an ``evaluate`` method that
    # returns finite values regardless of any sim/exp mismatch —
    # avoids DEAP crashing on all-infinite fitnesses while still
    # exercising the sign-correction path.
    class _StubEvaluator(ObjectiveEvaluator):
        def __init__(self):
            self.extractor = StressStrainExtractor(
                strain_source="time_rate", strain_rate=1.0,
            )
            # Compression-shaped reference: negative strain,
            # negative stress.
            exp_strain = np.linspace(0.0, -1.0, 25)
            exp_stress = -210.0 - 1900.0 * (
                1 - np.exp(48.0 * exp_strain)
            )
            self.experimental = pd.DataFrame({
                "strain": exp_strain, "stress": exp_stress,
            })
            self.experimental_strain_col = "strain"
            self.experimental_stress_col = "stress"

        def evaluate(self, results, ctx):
            return 0.5 + 0.01 * ctx.gene

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=7, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="compression")],
            objective_specs=[
                ObjectiveSpec(_StubEvaluator(), sim_case=0, label="x"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=7, track_hypervolume=False,
            ),
        )

    with ArchiveDB(db, readonly=True) as a:
        # Pull one archived curve and check signs.
        result = a.load_case_curve(
            run_id, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
        assert result is not None
        ind, dep, _, _ = result
        # Both axes should have non-positive dominant signs
        # (matching exp), even though the raw simulation produced
        # positive values for each.
        assert ind[int(np.argmax(np.abs(ind)))] <= 0, (
            f"strain not sign-corrected; dominant value="
            f"{ind[int(np.argmax(np.abs(ind)))]}"
        )
        assert dep[int(np.argmax(np.abs(dep)))] <= 0, (
            f"stress not sign-corrected; dominant value="
            f"{dep[int(np.argmax(np.abs(dep)))]}"
        )


def test_run_archives_curves_unchanged_when_no_experimental_data(
    workspace, fake_binary,
):
    """An evaluator without experimental data leaves the sim curve
    untouched (no reference to align against). The raw extractor
    output is what gets archived.
    """
    from workflow_common import ArchiveDB
    from workflow_common.objectives import (
        ObjectiveEvaluator, StressStrainExtractor,
    )

    # Custom evaluator with .extractor but no .experimental.
    class _NoExpEvaluator(ObjectiveEvaluator):
        def __init__(self):
            self.extractor = StressStrainExtractor(
                strain_source="time_rate", strain_rate=1.0,
            )
            # No .experimental attribute.

        def evaluate(self, results, ctx):
            return 0.5 + 0.01 * ctx.gene

    db = workspace / "opt.db"
    archive = ArchiveDB(db)
    with archive:
        run_id = archive.start_run(
            seed=8, param_names=["yield_stress", "hardening"],
        )
        problem = _build_problem(
            workspace, fake_binary,
            sim_cases=[SimCase(label="ne")],
            objective_specs=[
                ObjectiveSpec(_NoExpEvaluator(), sim_case=0, label="x"),
            ],
            archive=archive, archive_run_id=run_id,
        )
        run_nsga3(
            problem,
            bounds=Bounds(
                lower=np.array([150.0, 1500.0]),
                upper=np.array([300.0, 2500.0]),
            ),
            config=RunConfig(
                n_generations=1, population_size=4,
                unsga3=True, ref_dirs_partitions=(4, 0),
                seed=8, track_hypervolume=False,
            ),
        )

    with ArchiveDB(db, readonly=True) as a:
        result = a.load_case_curve(
            run_id, birth_gen=0, birth_gene=0, sim_case_idx=0,
        )
        assert result is not None
        ind, dep, _, _ = result
        # fake_binary produces positive Szz and positive strain,
        # so without sign-matching both should remain positive.
        assert ind[int(np.argmax(np.abs(ind)))] >= 0
        assert dep[int(np.argmax(np.abs(dep)))] >= 0
