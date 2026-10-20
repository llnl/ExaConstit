"""
Integration tests for :class:`workflow_common.problem.Problem`.

What these prove
----------------
The Problem orchestrator is the first thing that lets an optimizer
write a single method call and get back objective values. These
tests exercise that method across the scenarios a real optimization
will encounter:

* Single objective, happy path: one gene, one sim, one number out.
* Multi-objective, single sim case: N evaluators against ONE sim's
  output (the stress-and-slope use case).
* Multi-sim-case, multi-objective: separate sims per loading case,
  one or more evaluators scoring each.
* Population submission via ``evaluate_population``.
* Failure handling, restart, partial-progress handler.

Every test runs the real subprocess pipeline - templater renders
files, fake binary is invoked, reader parses the output, evaluator
computes an error. No mocks at the Problem boundary.
"""
from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflow_common import (
    CaseContext,
    CaseTemplater,
    ConstantPenaltyFailureHandler,
    InfinityFailureHandler,
    LocalBackend,
    Manifest,
    ObjectiveSpec,
    PartialProgressFailureHandler,
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


# --- Shared fixtures and helpers ----------------------------------------


@pytest.fixture
def experimental_df(workspace: Path) -> pd.DataFrame:
    """Reference Voce-law curve whose saturation constant matches the fake binary."""
    strain = np.linspace(0.0, 1.0, 40)
    stress = 210.0 + 1900.0 * (1 - np.exp(-48.0 * strain))
    df = pd.DataFrame({"strain": strain, "stress": stress})
    path = workspace / "exp_reference.csv"
    df.to_csv(path, index=False)
    return load_experimental_csv(path, delimiter=",")


@pytest.fixture
def problem_setup(
    workspace: Path, fake_binary: Path, experimental_df: pd.DataFrame,
):
    """Build the components (resolver, reader, writer, evaluator) for a Problem.

    Returns a dict the individual tests use to assemble a Problem
    with the exact SimCase/ObjectiveSpec combination they need.
    Keeping the fixture at the level of "framework parts" rather
    than "complete Problem" lets each test pick its own sharing
    topology.
    """
    options_tmpl = workspace / "master_options.toml"
    options_tmpl.write_text(textwrap.dedent("""\
        [Problem]
            name = "case_fixed"
            basename = "options"
            strain_rate = %%strain_rate%%
            yield_stress = %%yield_stress%%
            hardening = %%hardening%%
            temperature_k = %%temp_k%%
    """))

    placeholder_src = workspace / "placeholder.txt"
    placeholder_src.write_text("# placeholder\n")
    templater = CaseTemplater([
        TemplateTarget(source=placeholder_src, dest=".placeholder"),
    ])

    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
        "avg_def_grad": TextTableSpec(
            columns=[
                "Time", "Volume", "F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33",
            ],
            required=False,
        ),
    })

    stress_evaluator = StressStrainObjective(
        experimental=experimental_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=1.0,
        ),
    )

    def make_writer(strain_rate: float, temp_k: float = 298.0):
        return TemplatePropertyWriter(
            template_path=options_tmpl,
            dest="options.toml",
            extra_values={"strain_rate": strain_rate, "temp_k": temp_k},
        )

    config = ProblemConfig(
        binary=fake_binary,
        binary_args=(),
        num_tasks=1,
        duration_s=30,
        stdout="stdout.log",
        stderr="stderr.log",
        required_outputs=("results/options/avg_stress.txt",),
    )

    return {
        "workspace": workspace,
        "config": config,
        "templater": templater,
        "resolver": resolver,
        "reader": reader,
        "stress_evaluator": stress_evaluator,
        "experimental_df": experimental_df,
        "make_writer": make_writer,
    }


def _make_problem(s, *, sim_cases, objective_specs, writer,
                  failure_handler=None, manifest=None, max_workers=1):
    if manifest is None:
        manifest = Manifest(s["workspace"] / "manifest.jsonl")
        manifest.load()
    return Problem(
        config=s["config"],
        param_names=["yield_stress", "hardening"],
        sim_cases=sim_cases,
        objective_specs=objective_specs,
        templater=s["templater"],
        property_writer=writer,
        resolver=s["resolver"],
        backend=LocalBackend(max_workers=max_workers),
        reader=s["reader"],
        failure_handler=failure_handler,
        manifest=manifest,
    )


def _slope_evaluator(stress_eval: StressStrainObjective):
    """Slope-matching evaluator that reuses stress_eval's extractor.

    Demonstrates the "two objectives, one simulation" topology:
    both evaluators consume the same CaseResultSet, so both
    ObjectiveSpecs should point at the same SimCase.
    """
    exp_df = stress_eval.experimental
    exp_strain = exp_df[stress_eval.experimental_strain_col].to_numpy()
    exp_stress = exp_df[stress_eval.experimental_stress_col].to_numpy()
    exp_slope = np.gradient(exp_stress, exp_strain)

    # Exp DataFrame with slope in the 'stress' column so the
    # PartialProgressFailureHandler can introspect via the
    # standard experimental/experimental_strain_col/experimental_stress_col
    # attributes.
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


# --- Single SimCase, single objective -----------------------------------


def test_single_case_happy_path(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0, label="stress"),
        ],
        writer=s["make_writer"](strain_rate=1.0),
    )
    errs = problem.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    assert len(errs) == 1
    assert 0 <= errs[0] < 50.0


def test_different_genes_produce_different_errors(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
        max_workers=2,
    )
    good = problem.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    bad = problem.evaluate_gene(
        np.array([500.0, 100.0]), generation=0, gene_idx=1,
    )
    assert good[0] < bad[0]


# --- Shared SimCase (the key new capability) ---------------------------


def test_two_objectives_one_sim_case(problem_setup):
    """Stress + slope from the same simulation: N=2 objectives, M=1 sim.

    Verifies that the Problem runs ONE sim (not two) by checking that
    only one sim directory exists after the call.
    """
    s = problem_setup
    slope = _slope_evaluator(s["stress_evaluator"])
    problem = _make_problem(
        s,
        sim_cases=[SimCase(label="quasi")],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0, label="stress"),
            ObjectiveSpec(slope, sim_case=0, label="slope"),
        ],
        writer=s["make_writer"](strain_rate=1.0),
    )
    errs = problem.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    assert len(errs) == 2
    assert all(np.isfinite(errs))
    # Exactly ONE sim directory was used.
    gene0 = s["workspace"] / "wf" / "gen_0"
    sc_dirs = [d for d in gene0.iterdir() if d.is_dir()]
    assert len(sc_dirs) == 1, (
        f"expected 1 sim directory (shared sim_case), got {len(sc_dirs)}: "
        f"{sc_dirs}"
    )


# --- Multi sim case ------------------------------------------------------


def test_two_sim_cases_two_objectives(problem_setup):
    """Different loading conditions, one evaluator per SimCase."""
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase(label="quasi"), SimCase(label="dynamic")],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0, label="stress_qs"),
            ObjectiveSpec(s["stress_evaluator"], sim_case=1, label="stress_dyn"),
        ],
        writer=s["make_writer"](strain_rate=1.0),
        max_workers=2,
    )
    errs = problem.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    assert len(errs) == 2
    gene_dir = s["workspace"] / "wf" / "gen_0"
    assert (gene_dir / "gene_0_sc_0").is_dir()
    assert (gene_dir / "gene_0_sc_1").is_dir()


def test_three_objectives_two_sim_cases(problem_setup):
    """Stress+slope from quasi-static, stress from dynamic: N=3, M=2."""
    s = problem_setup
    slope = _slope_evaluator(s["stress_evaluator"])
    problem = _make_problem(
        s,
        sim_cases=[SimCase(label="quasi"), SimCase(label="dynamic")],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0, label="stress_qs"),
            ObjectiveSpec(slope,                 sim_case=0, label="slope_qs"),
            ObjectiveSpec(s["stress_evaluator"], sim_case=1, label="stress_dyn"),
        ],
        writer=s["make_writer"](strain_rate=1.0),
        max_workers=2,
    )
    errs = problem.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    assert len(errs) == 3
    assert problem.n_objectives == 3
    assert problem.n_sim_cases == 2
    # Exactly 2 sim dirs, not 3.
    gene_dir = s["workspace"] / "wf" / "gen_0"
    sc_dirs = [d for d in gene_dir.iterdir() if d.is_dir()]
    assert len(sc_dirs) == 2


# --- Population ----------------------------------------------------------


def test_evaluate_population_returns_aligned_results(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
        max_workers=3,
    )
    genes = [
        np.array([210.0, 1900.0]),
        np.array([500.0, 100.0]),
        np.array([100.0, 3000.0]),
    ]
    errs = problem.evaluate_population(genes, generation=0)
    assert len(errs) == 3
    assert errs[0][0] < errs[1][0]
    assert errs[0][0] < errs[2][0]


def test_population_with_shared_sim_case(problem_setup):
    """3 genes x 1 SimCase x 2 objectives -> 3 sims run, 6 errors returned."""
    s = problem_setup
    slope = _slope_evaluator(s["stress_evaluator"])
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0),
            ObjectiveSpec(slope, sim_case=0),
        ],
        writer=s["make_writer"](strain_rate=1.0),
        max_workers=3,
    )
    genes = [
        np.array([210.0, 1900.0]),
        np.array([250.0, 1800.0]),
        np.array([300.0, 1700.0]),
    ]
    errs = problem.evaluate_population(genes, generation=0)
    assert len(errs) == 3
    assert all(len(e) == 2 for e in errs)
    gen0 = s["workspace"] / "wf" / "gen_0"
    for gi in range(3):
        gd = gen0 / f"gene_{gi}_sc_0"
        assert gd.is_dir()


# --- Failure handling ---------------------------------------------------


def test_failed_sim_case_propagates_to_all_its_objectives(problem_setup):
    """If a shared sim fails, EVERY objective tied to it gets the handler value."""
    s = problem_setup
    slope = _slope_evaluator(s["stress_evaluator"])
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[
            ObjectiveSpec(s["stress_evaluator"], sim_case=0),
            ObjectiveSpec(slope, sim_case=0),
        ],
        writer=s["make_writer"](strain_rate=1.0),
        failure_handler=ConstantPenaltyFailureHandler(penalty=999.0),
    )
    os.environ["FAKE_FAIL"] = "1"
    try:
        errs = problem.evaluate_gene(
            np.array([210.0, 1900.0]), generation=0, gene_idx=0,
        )
    finally:
        os.environ.pop("FAKE_FAIL", None)
    assert errs == [999.0, 999.0]


def test_failed_case_returns_infinity(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
        failure_handler=InfinityFailureHandler(),
    )
    os.environ["FAKE_FAIL"] = "1"
    try:
        errs = problem.evaluate_gene(
            np.array([210.0, 1900.0]), generation=0, gene_idx=0,
        )
    finally:
        os.environ.pop("FAKE_FAIL", None)
    assert errs == [float("inf")]


def test_mixed_pass_fail_in_population(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
        failure_handler=ConstantPenaltyFailureHandler(penalty=1e9),
        max_workers=3,
    )
    genes = [
        np.array([210.0, 1900.0]),
        np.array([250.0, 1800.0]),
        np.array([300.0, 1700.0]),
    ]
    errs = problem.evaluate_population(genes, generation=0)
    for per_gene in errs:
        assert per_gene[0] < 1e9


# --- Restart -------------------------------------------------------------


def test_skips_completed_on_rerun(problem_setup):
    s = problem_setup

    def build():
        return _make_problem(
            s,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
            writer=s["make_writer"](strain_rate=1.0),
            manifest=Manifest(s["workspace"] / "manifest.jsonl"),
        )

    p1 = build()
    errs1 = p1.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )

    stress_file = s["resolver"].output_file(
        "avg_stress", CaseContext(0, 0, 0),
    )
    first_mtime = stress_file.stat().st_mtime_ns
    time.sleep(0.05)

    p2 = build()
    errs2 = p2.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )

    assert errs1 == errs2
    assert stress_file.stat().st_mtime_ns == first_mtime


def test_shared_sim_skip_serves_many_objectives(problem_setup):
    """Restart + shared sim: one skip rescores both objectives."""
    s = problem_setup
    slope = _slope_evaluator(s["stress_evaluator"])

    def build():
        return _make_problem(
            s,
            sim_cases=[SimCase()],
            objective_specs=[
                ObjectiveSpec(s["stress_evaluator"], sim_case=0),
                ObjectiveSpec(slope, sim_case=0),
            ],
            writer=s["make_writer"](strain_rate=1.0),
            manifest=Manifest(s["workspace"] / "manifest.jsonl"),
        )

    p1 = build()
    errs1 = p1.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    stress_file = s["resolver"].output_file(
        "avg_stress", CaseContext(0, 0, 0),
    )
    mtime = stress_file.stat().st_mtime_ns
    time.sleep(0.05)

    p2 = build()
    errs2 = p2.evaluate_gene(
        np.array([210.0, 1900.0]), generation=0, gene_idx=0,
    )
    assert errs1 == errs2
    # Sim was NOT re-run.
    assert stress_file.stat().st_mtime_ns == mtime


# --- Partial-progress handler -------------------------------------------


def test_partial_progress_no_output(problem_setup):
    s = problem_setup
    handler = PartialProgressFailureHandler(
        inner_evaluator=s["stress_evaluator"],
        base_penalty=1e6,
        progress_weight=1.0,
        strain_target=1.0,
    )
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
        failure_handler=handler,
    )
    os.environ["FAKE_FAIL"] = "1"
    try:
        errs = problem.evaluate_gene(
            np.array([210.0, 1900.0]), generation=0, gene_idx=0,
        )
    finally:
        os.environ.pop("FAKE_FAIL", None)
    assert errs == [1e6]


# --- Validation ---------------------------------------------------------


def test_gene_length_mismatch_raises(problem_setup):
    s = problem_setup
    problem = _make_problem(
        s,
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        writer=s["make_writer"](strain_rate=1.0),
    )
    with pytest.raises(ValueError, match="entries"):
        problem.evaluate_gene(np.array([210.0]), generation=0, gene_idx=0)


def test_rejects_empty_sim_cases(problem_setup):
    s = problem_setup
    with pytest.raises(ValueError, match="SimCase"):
        _make_problem(
            s,
            sim_cases=[],
            objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
            writer=s["make_writer"](strain_rate=1.0),
        )


def test_rejects_empty_objective_specs(problem_setup):
    s = problem_setup
    with pytest.raises(ValueError, match="ObjectiveSpec"):
        _make_problem(
            s,
            sim_cases=[SimCase()],
            objective_specs=[],
            writer=s["make_writer"](strain_rate=1.0),
        )


def test_rejects_out_of_range_sim_case_index(problem_setup):
    """An ObjectiveSpec referencing a non-existent SimCase fails at construction."""
    s = problem_setup
    with pytest.raises(ValueError, match="sim_case=5"):
        _make_problem(
            s,
            sim_cases=[SimCase()],
            objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=5)],
            writer=s["make_writer"](strain_rate=1.0),
        )


# --- ProblemConfig.required_outputs validation -------------------------


def test_required_outputs_rejects_absolute_path():
    """Absolute paths would produce `workdir / /abs/path` which silently
    loses the working-dir prefix entirely on POSIX; fail loudly at
    construction instead.
    """
    from pathlib import Path
    with pytest.raises(ValueError, match="absolute"):
        ProblemConfig(
            binary=Path("/bin/true"),
            required_outputs=("/tmp/avg_stress.txt",),
        )


def test_required_outputs_rejects_placeholder_syntax():
    """Users sometimes confuse this pre-flight check with the reader's
    output_file_patterns and write '{working_dir}/foo.txt' here. That
    produces doubled paths at runtime; reject at construction.
    """
    from pathlib import Path
    with pytest.raises(ValueError, match="Placeholder syntax"):
        ProblemConfig(
            binary=Path("/bin/true"),
            required_outputs=("{working_dir}/avg_stress.txt",),
        )


def test_required_outputs_rejects_parent_directory_refs():
    """`..` segments escape the case working dir, defeating the
    per-case isolation the framework assumes.
    """
    from pathlib import Path
    with pytest.raises(ValueError, match="parent-"):
        ProblemConfig(
            binary=Path("/bin/true"),
            required_outputs=("../outside.txt",),
        )


def test_required_outputs_rejects_non_string():
    """Path objects, integers, etc. — users occasionally pass these
    thinking the framework will Path-coerce. Be explicit that str is
    required; coercion here hides the real intent.
    """
    from pathlib import Path
    with pytest.raises(TypeError, match="expected str"):
        ProblemConfig(
            binary=Path("/bin/true"),
            required_outputs=(Path("avg_stress.txt"),),  # type: ignore[arg-type]
        )


def test_required_outputs_accepts_bare_filename():
    """Sanity — plain filenames are the common case and must work."""
    from pathlib import Path
    cfg = ProblemConfig(
        binary=Path("/bin/true"),
        required_outputs=("avg_stress.txt",),
    )
    assert cfg.required_outputs == ("avg_stress.txt",)


def test_required_outputs_accepts_multi_segment_relative_path():
    """Binaries that write into `results/<basename>/...` are common;
    multi-segment relative paths must be accepted.
    """
    from pathlib import Path
    cfg = ProblemConfig(
        binary=Path("/bin/true"),
        required_outputs=("results/options/avg_stress.txt",),
    )
    assert cfg.required_outputs == ("results/options/avg_stress.txt",)


def test_required_outputs_validation_does_not_double_join_with_relative_workdir(
    problem_setup, tmp_path, monkeypatch,
):
    """Regression: when the resolver's root is a RELATIVE path
    (e.g. ``Path("./calibration_run")``, which users do all the
    time), the old code pre-joined ``working_dir / required_output``
    and then validate_outputs joined AGAIN because the first join
    produced a relative path too. The result was paths like
    ``<workdir>/<workdir>/...`` in the missing_or_empty log message
    and the actual filesystem check looking at the wrong spot.
    Fix: single join inside validate_outputs.
    """
    from pathlib import Path

    from workflow_common import (
        TemplatePathResolver,
        Problem, ProblemConfig, SimCase, ObjectiveSpec,
        LocalBackend, Manifest,
    )

    s = problem_setup

    # Run from tmp_path so the relative "calibration_run" is scoped
    # to the test's workspace, not the repo root.
    monkeypatch.chdir(tmp_path)
    rel_workspace = Path("calibration_run")
    rel_workspace.mkdir()

    # Reuse the fixture's templater and writer unmodified — they
    # already match the fake binary's expectations. Only the
    # resolver needs to use a RELATIVE root to reproduce the bug.
    resolver = TemplatePathResolver(
        working_dir_pattern="gen_{generation}/gene_{gene}_obj_{obj}",
        output_file_patterns={
            "avg_stress":   "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=rel_workspace,   # RELATIVE — this is what triggers the old bug
    )

    problem = Problem(
        config=ProblemConfig(
            binary=s["config"].binary,
            binary_args=(),
            num_tasks=1,
            duration_s=30,
            stdout="stdout.log",
            stderr="stderr.log",
            required_outputs=("results/options/avg_stress.txt",),
        ),
        param_names=["yield_stress", "hardening"],
        sim_cases=[SimCase()],
        objective_specs=[ObjectiveSpec(s["stress_evaluator"], sim_case=0)],
        templater=s["templater"],
        property_writer=s["make_writer"](strain_rate=1.0),
        resolver=resolver,
        backend=LocalBackend(max_workers=1),
        reader=s["reader"],
        manifest=Manifest(rel_workspace / "manifest.jsonl"),
    )

    errors = problem.evaluate_population([[200.0, 2000.0]], generation=0)
    assert len(errors) == 1
    # A real error value, not the infinity-failure-handler sentinel.
    # If the double-join bug regressed, validate_outputs would look
    # at `calibration_run/gen_0/gene_0_obj_0/calibration_run/...`,
    # find nothing, and the case would be marked FAILED -> inf.
    assert errors[0][0] != float("inf"), (
        "required-files validation incorrectly failed with a "
        "relative workspace path — suggests the double-join bug "
        "has regressed."
    )

    # Case dir exists at the NORMAL single-joined path.
    assert (rel_workspace / "gen_0" / "gene_0_obj_0").is_dir()
    # NOT at the double-joined path.
    assert not (
        rel_workspace / "gen_0" / "gene_0_obj_0" / "calibration_run"
    ).exists()



# --- case_data flows to all three consumers ---------------------------------


def test_case_data_visible_to_templater_path_resolver_and_writer(
    workspace, fake_binary, experimental_df,
):
    """One SimCase.case_data dict should reach all three places that
    need per-case context: the templater (via the values mapping),
    the path resolver (via ``ctx.extra``), and the property writer
    (via ``sim_case.case_data``).

    This test pins the merged-field design — if someone ever splits
    the field again (or breaks one of the three flow paths), this
    test fails because ALL THREE consumers must successfully see
    the same SimCase.case_data values from one source.

    Uses a real fake_binary subprocess like the other tests in this
    module (no monkeypatching), and exercises a path pattern with
    a {rve_name} placeholder so the path resolver's consumption is
    verified by the case dir's actual location on disk.
    """
    from workflow_common import CallablePropertyWriter

    # Capture what the writer saw, to assert later.
    seen = {}

    def write_props(case_dir, gene, names, sim_case):
        # The writer pulls case_data fields directly off sim_case.
        # This is the documented pattern: per-case constants for
        # the writer live in sim_case.case_data alongside
        # template/path values.
        seen["temperature_k"] = sim_case.case_data["temperature_k"]
        seen["rve_name"] = sim_case.case_data["rve_name"]

        # Still produce the options.toml that the fake_binary needs
        # to read. We do this in the writer (rather than via the
        # templater) so this test exercises the writer's responsibility
        # to write everything the binary will read. Mirroring
        # TemplatePropertyWriter's behavior keeps the fake_binary
        # happy: it parses strain_rate/yield_stress/hardening from
        # the options file via regex.
        gene_dict = dict(zip(names, gene))
        case_dir.mkdir(parents=True, exist_ok=True)
        opts = case_dir / "options.toml"
        opts.write_text(textwrap.dedent(f"""\
            [Problem]
                name = "case_fixed"
                basename = "options"
                strain_rate = 1.0
                yield_stress = {gene_dict['yield_stress']}
                hardening = {gene_dict['hardening']}
                temperature_k = {sim_case.case_data['temperature_k']}
        """))
        return opts

    # Path pattern WITH {rve_name} — the resolver only resolves this
    # if case_data["rve_name"] reaches it via ctx.extra.
    resolver = TemplatePathResolver(
        working_dir_pattern="{rve_name}/gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )

    # Templater renders a small file we can inspect to confirm the
    # %%temperature_k%% substitution went through. The writer above
    # produces the actual options.toml; this is a separate render
    # target purely for verifying templater consumption.
    template_src = workspace / "rendered_template.txt"
    template_src.write_text("temperature_k = %%temperature_k%%\nrve = %%rve_name%%\n")
    templater = CaseTemplater([
        TemplateTarget(source=template_src, dest="rendered.txt"),
    ])

    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
        "avg_def_grad": TextTableSpec(
            columns=[
                "Time", "Volume", "F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33",
            ],
            required=False,
        ),
    })

    stress_evaluator = StressStrainObjective(
        experimental=experimental_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=1.0,
        ),
    )

    sc = SimCase(
        case_data={
            "temperature_k": 298.0,    # used by templater + writer
            "rve_name":      "grain_32",  # used by path resolver + writer
        },
        label="exp1",
    )

    config = ProblemConfig(
        binary=fake_binary,
        binary_args=(),
        num_tasks=1,
        duration_s=30,
        stdout="stdout.log",
        stderr="stderr.log",
        required_outputs=("results/options/avg_stress.txt",),
    )

    manifest = Manifest(workspace / "manifest.jsonl")
    manifest.load()

    problem = Problem(
        config=config,
        param_names=["yield_stress", "hardening"],
        sim_cases=[sc],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        templater=templater,
        property_writer=CallablePropertyWriter(func=write_props),
        resolver=resolver,
        backend=LocalBackend(max_workers=1),
        reader=reader,
        manifest=manifest,
    )

    problem.evaluate_gene(gene=[210.0, 1900.0], generation=0, gene_idx=0)

    # ----- Three independent assertions, one per consumer ------------------

    # 1. Path resolver consumed {rve_name} — the case dir lives under
    #    "grain_32/gen_0/gene_0_sc_0" inside the workspace. Existence
    #    of this directory IS the proof that case_data flowed to
    #    the resolver.
    case_dir = workspace / "grain_32" / "gen_0" / "gene_0_sc_0"
    assert case_dir.is_dir(), (
        f"path resolver did not consume case_data['rve_name']; "
        f"expected dir {case_dir} not found"
    )

    # 2. Templater consumed %%temperature_k%% — the rendered file
    #    inside the case dir carries the substituted value.
    rendered = (case_dir / "rendered.txt").read_text()
    assert "temperature_k = 298.0" in rendered
    assert "rve = grain_32" in rendered

    # 3. Property writer read sim_case.case_data — both fields
    #    came through.
    assert seen["temperature_k"] == 298.0
    assert seen["rve_name"] == "grain_32"

# --- case_data validation policy: template-driven, not data-driven ---------


def test_case_data_can_carry_extra_keys_not_in_template(
    workspace, fake_binary, experimental_df,
):
    """Keys in case_data that aren't referenced by any template
    should NOT cause a render failure. The templater is
    template-driven: it complains only when its template demands
    a key that case_data doesn't supply, never the reverse.

    Why: users routinely put per-case material constants in
    case_data even when their master template doesn't reference
    them (because the writer reads them instead). Failing on
    "unused" keys would force users to maintain two parallel
    inventories — what's in the template vs what's in case_data —
    just to keep the templater quiet. That's the wrong direction.
    """
    from workflow_common import CallablePropertyWriter

    options_tmpl = workspace / "master_options.toml"
    # Template ONLY references strain_rate, yield_stress, hardening,
    # temperature_k. Other case_data entries (c11, c12, c44,
    # rve_name) are extras the templater will see and ignore.
    options_tmpl.write_text(textwrap.dedent("""\
        [Problem]
            name = "case_fixed"
            basename = "options"
            strain_rate = %%strain_rate%%
            yield_stress = %%yield_stress%%
            hardening = %%hardening%%
            temperature_k = %%temperature_k%%
    """))

    placeholder_src = workspace / "placeholder.txt"
    placeholder_src.write_text("# placeholder\n")
    templater = CaseTemplater([
        TemplateTarget(source=placeholder_src, dest=".placeholder"),
    ])

    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={
            "avg_stress": "{working_dir}/results/options/avg_stress.txt",
            "avg_def_grad": "{working_dir}/results/options/avg_def_grad.txt",
        },
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=True,
        ),
        "avg_def_grad": TextTableSpec(
            columns=[
                "Time", "Volume", "F11", "F12", "F13", "F21", "F22", "F23", "F31", "F32", "F33",
            ],
            required=False,
        ),
    })
    stress_evaluator = StressStrainObjective(
        experimental=experimental_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=1.0,
        ),
    )

    # Writer that doesn't even READ the extras — proves they pass
    # through harmlessly. Uses TemplatePropertyWriter so the
    # rendered options.toml comes from the master template.
    writer = TemplatePropertyWriter(
        template_path=options_tmpl,
        dest="options.toml",
        extra_values={"strain_rate": 1.0, "temperature_k": 298.0},
    )

    sc = SimCase(
        case_data={
            # Extras the template never references — these MUST
            # NOT cause a render failure:
            "c11": 168.4,
            "c12": 121.4,
            "c44":  75.4,
            "rve_name": "grain_32",
            "lattice_type": "fcc",
            "anisotropy_ratio": 1.5,
        },
        label="exp1",
    )

    config = ProblemConfig(
        binary=fake_binary,
        binary_args=(),
        num_tasks=1,
        duration_s=30,
        stdout="stdout.log",
        stderr="stderr.log",
        required_outputs=("results/options/avg_stress.txt",),
    )

    problem = Problem(
        config=config,
        param_names=["yield_stress", "hardening"],
        sim_cases=[sc],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        templater=templater,
        property_writer=writer,
        resolver=resolver,
        backend=LocalBackend(max_workers=1),
        reader=reader,
        manifest=Manifest(workspace / "manifest.jsonl"),
    )

    # Should NOT raise. If unused keys caused an error we'd see it here.
    errs = problem.evaluate_gene(
        gene=[210.0, 1900.0], generation=0, gene_idx=0,
    )
    assert errs is not None
    assert len(errs) == 1


def test_case_data_missing_template_key_raises_helpful_error(
    workspace, fake_binary, experimental_df,
):
    """When a master template references %%c11%% but case_data
    has no 'c11' key, the framework must raise an error that
    points the user at SimCase.case_data — not just a generic
    "key not found" from the templater.

    The hint matters because users will edit case_data to fix
    the issue; saying just "missing key c11" leaves them guessing
    where to add it.
    """
    from workflow_common.templates import UnresolvedPlaceholderError

    # Master template demands c11 but case_data won't have it.
    options_tmpl = workspace / "master_options.toml"
    options_tmpl.write_text("c11 = %%c11%%\n")

    templater = CaseTemplater([
        TemplateTarget(source=options_tmpl, dest="options.toml"),
    ])

    placeholder_src = workspace / "placeholder.txt"
    placeholder_src.write_text("# placeholder\n")
    # Use a no-op writer so the failure is purely templater-driven.
    from workflow_common import CallablePropertyWriter

    def write_props(case_dir, gene, names, sim_case):
        case_dir.mkdir(parents=True, exist_ok=True)
        p = case_dir / "noop.txt"
        p.write_text("ok\n")
        return p

    writer = CallablePropertyWriter(func=write_props)

    resolver = TemplatePathResolver(
        working_dir_pattern="wf/gen_{generation}/gene_{gene}_sc_{obj}",
        output_file_patterns={"avg_stress": "{working_dir}/x.txt"},
        root=workspace,
    )
    reader = TextTableReader({
        "avg_stress": TextTableSpec(
            columns=["Time", "Volume", "Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz"],
            required=False,
        ),
    })
    stress_evaluator = StressStrainObjective(
        experimental=experimental_df,
        extractor=StressStrainExtractor(
            strain_source="time_rate", strain_rate=1.0,
        ),
    )

    sc = SimCase(
        # No c11 here — template wants it but case_data doesn't have it.
        case_data={"strain_rate": 1.0, "temperature_k": 298.0},
        label="exp_missing_c11",
    )

    problem = Problem(
        config=ProblemConfig(
            binary=fake_binary, num_tasks=1, duration_s=30,
            required_outputs=(),
        ),
        param_names=["yield_stress", "hardening"],
        sim_cases=[sc],
        objective_specs=[ObjectiveSpec(stress_evaluator, sim_case=0)],
        templater=templater,
        property_writer=writer,
        resolver=resolver,
        backend=LocalBackend(max_workers=1),
        reader=reader,
        manifest=Manifest(workspace / "manifest.jsonl"),
    )

    with pytest.raises(UnresolvedPlaceholderError) as exc_info:
        problem.evaluate_gene(
            gene=[210.0, 1900.0], generation=0, gene_idx=0,
        )

    msg = str(exc_info.value)
    # Original message preserved (key name, available list).
    assert "c11" in msg
    # New hint pointing the user at the right field to edit.
    assert "case_data" in msg
    # Names the offending SimCase so users know which entry to edit.
    assert "exp_missing_c11" in msg
