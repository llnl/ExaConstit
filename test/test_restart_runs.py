#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExaConstit Restart Testing Framework
Runs real ExaConstit jobs and checks that restarted runs match uninterrupted ones exactly

Each test case runs a reference job once, uninterrupted, writing checkpoints along the way.
It then checks two kinds of restart against that reference:

* Chained runs: the job is stopped and restarted from its own checkpoints, one segment at a
  time (each segment extends t_final).
* Restarts from the reference's checkpoints: a continuation reads a past checkpoint of the
  reference with `[Restart] read_directory` and runs to the end, writing its own checkpoints
  elsewhere. Any number of past checkpoints can be checked this way without rerunning the
  reference.

Every checkpoint a restarted job writes is compared with the reference checkpoint of the same
cycle (scripts/restart/compare_restart_checkpoints.cpp), and the volume-average outputs are
compared row by row. Both comparisons are bitwise unless --rtol is given.

A bitwise match alone cannot tell a real restart from a run that ignored the checkpoint and
recomputed from step 0 (the runs are deterministic), so two more checks guard against that:

* Each restarted job's log must report restarting from the expected checkpoint, and the job
  must run exactly the remaining steps, starting at the step after the checkpoint.
* Negative control (needs h5py; skipped with a warning otherwise): the state variables in a
  copy of a reference checkpoint are perturbed, a continuation restarts from it, and its final
  checkpoint and outputs must DIFFER from the reference. This proves the restored state is what
  the continuation uses, and that the comparisons detect a difference.

Cases:
  cyclic       Fully reversed strain cycles along z (0 -> +amp -> -amp -> 0) at a fixed dt.
               Chains stop every cycle and every 2 cycles. Cycle ends fall inside a boundary
               condition segment, so the boundary-condition replay on restart is exercised.
  monotonic    Long tension along z at a fixed dt. A chain stops halfway; the reference is also
               restarted from its halfway checkpoint; the negative control runs here.
  empty_ranks  Short tension run on a 2-element mesh, so with more than 2 ranks every rank but
               two owns no elements. Covers checkpointing and restarting with empty ranks.

Defaults: 16^3 = 4096 hex elements, 8 MPI ranks, 4 cycles of +/-1% strain (160 steps), 10%
monotonic strain (100 steps), and a 20-step empty-rank run. Run with --help for the options
(mesh size, rank count, checkpoint interval and retention, which reference checkpoints to
restart from).
"""

import argparse
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# The top face is pulled in z at this velocity and every run uses this fixed time step, giving
# 1e-3 strain per step on the unit cube.
TOP_VELOCITY = 0.01
DT = 0.1


class Failure(Exception):
    """Raised when a run fails outright, which stops the whole test"""


@dataclass
class MeshSpec:
    """Auto-generated mesh used by a test case"""
    nxyz: Tuple[int, int, int]
    grain_file: str

    @property
    def num_elements(self) -> int:
        return self.nxyz[0] * self.nxyz[1] * self.nxyz[2]


# ============================================================================
# INPUT FILES
# ============================================================================

def find_data_dir(explicit: Optional[str]) -> Path:
    """
    Find the directory holding the Voce material input files

    Args:
        explicit: Directory given on the command line, or None to search next to this script

    Returns:
        Directory containing props_cp_voce.txt
    """
    candidates = [Path(explicit)] if explicit else [SCRIPT_DIR, SCRIPT_DIR / "data"]
    for candidate in candidates:
        if (candidate / "props_cp_voce.txt").exists():
            return candidate
    raise Failure(f"cannot find props_cp_voce.txt in {[str(c) for c in candidates]}")


def prepare_inputs(workdir: Path, data_dir: Path, meshes: List[MeshSpec], num_grains: int,
                   seed: int) -> None:
    """
    Copy the material inputs into the work directory and write one grain file per mesh

    Args:
        workdir: Work directory (created if needed)
        data_dir: Directory with the Voce material input files
        meshes: Meshes whose grain files to write
        num_grains: Number of grains (orientations) to draw from
        seed: Seed for the element-to-grain assignment, so every run gets the same grains
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("props_cp_voce.txt", "state_cp_voce.txt", "voce_quats.ori"):
        shutil.copy(data_dir / name, workdir / name)
    for mesh in meshes:
        rng = random.Random(seed)
        with open(workdir / mesh.grain_file, "w") as f:
            for _ in range(mesh.num_elements):
                f.write(f"{rng.randint(1, num_grains)}\n")


def bc_block(update_steps: List[int], top_velocities: List[float]) -> str:
    """
    Build the [BCs] table for a run whose top-face velocity changes at given steps

    The bottom face is fixed in z, the x = 0 and y = 0 faces are symmetry planes, and the top
    face moves in z with the given velocity. The legacy changing-BC format is used because it
    expresses a per-step velocity schedule directly.

    Args:
        update_steps: Steps at which a new velocity takes effect (must start with 1)
        top_velocities: Top-face z velocity for each update step

    Returns:
        TOML text of the [BCs] table
    """
    ids = ", ".join(["[1, 2, 3, 4]"] * len(update_steps))
    comps = ", ".join(["[3, 1, 2, 3]"] * len(update_steps))
    vals = ",\n                      ".join(
        "[" + ", ".join(["0.0"] * 11 + [repr(v)]) + "]" for v in top_velocities)
    return f"""[BCs]
    changing_ess_bcs = true
    update_steps = {list(update_steps)}
    essential_ids = [{ids}]
    essential_comps = [{comps}]
    essential_vals = [{vals}]
"""


def restart_block(restart: Dict[str, object]) -> str:
    """
    Build the [Restart] table

    Args:
        restart: [Restart] keys and values (bool, int, float or string)

    Returns:
        TOML text of the [Restart] table
    """
    lines = ["[Restart]"]
    for key, value in restart.items():
        if isinstance(value, bool):
            lines.append(f"    {key} = {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"    {key} = {value}")
        else:
            lines.append(f'    {key} = "{value}"')
    return "\n".join(lines) + "\n"


def write_toml(path: Path, basename: str, mesh: MeshSpec, num_grains: int, bcs: str,
               t_final: float, restart: Dict[str, object]) -> None:
    """
    Write the options file for one run

    Every run uses the same material (Voce FCC crystal plasticity through ExaCMech), solver
    settings and volume-average outputs; only the mesh, boundary conditions, final time and
    restart settings differ between runs.

    Args:
        path: Options file to write
        basename: Output basename; runs sharing it append to the same output files
        mesh: Mesh of the test case
        num_grains: Number of grains in the orientation file
        bcs: [BCs] table from bc_block()
        t_final: Final simulation time of this run
        restart: [Restart] keys and values
    """
    path.write_text(f"""Version = "0.10.0"
basename = "{basename}"
grain_file = "{mesh.grain_file}"

[[Materials]]
    material_name = "voce_fcc"
    mech_type = "exacmech"
    temperature = 298.0
    [Materials.Properties]
        floc = "props_cp_voce.txt"
        num_props = 17
    [Materials.State_Vars]
        floc = "state_cp_voce.txt"
        num_vars = 24
    [Materials.Grain]
        ori_state_var_loc = 9
        ori_stride = 4
        ori_type = "quat"
        num_grains = {num_grains}
        orientation_file = "voce_quats.ori"
    [Materials.Model]
        crystal_plasticity = true
        [Materials.Model.ExaCMech]
            shortcut = "evptn_FCC_A"

{bcs}
[Time]
    [Time.Fixed]
        dt = {DT}
        t_final = {t_final!r}

[Visualizations]
    visit = false
    conduit = false
    paraview = false

[PostProcessing]
    [PostProcessing.volume_averages]
        enabled = true
        stress = true
        def_grad = true
        euler_strain = true
        plastic_work = true
        output_frequency = 1
        output_directory = "./results"

[Solvers]
    assembly = "FULL"
    rtmodel = "CPU"
    [Solvers.NR]
        iter = 25
        rel_tol = 5e-5
        abs_tol = 5e-10
    [Solvers.Krylov]
        iter = 1000
        rel_tol = 1e-7
        abs_tol = 1e-27
        solver = "CG"

[Mesh]
    type = "auto"
    ref_ser = 0
    ref_par = 0
    p_refinement = 1
    [Mesh.Auto]
        mxyz = [1.0, 1.0, 1.0]
        nxyz = [{mesh.nxyz[0]}, {mesh.nxyz[1]}, {mesh.nxyz[2]}]

{restart_block(restart)}""")


def time_of(steps: int) -> float:
    """
    Simulation time after a number of fixed steps

    Args:
        steps: Number of completed steps

    Returns:
        steps * DT, rounded so it prints and compares cleanly
    """
    return round(steps * DT, 10)


# ============================================================================
# RUNNING AND COMPARING
# ============================================================================

class Harness:
    """Runs ExaConstit jobs and records the outcome of every check"""

    def __init__(self, args: argparse.Namespace, workdir: Path):
        """
        Args:
            args: Parsed command-line options
            workdir: Work directory holding the inputs, checkpoints, outputs and logs
        """
        self.args = args
        self.workdir = workdir
        self.logdir = workdir / "logs"
        self.logdir.mkdir(exist_ok=True)
        self.failures: List[str] = []
        self.checks = 0

    def check(self, ok: bool, message: str) -> None:
        """
        Record and print the outcome of one check

        Args:
            ok: Whether the check passed
            message: Description of what was checked and what was found
        """
        self.checks += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {message}", flush=True)
        if not ok:
            self.failures.append(message)

    def run(self, name: str, toml_path: Path) -> None:
        """
        Run ExaConstit on an options file, logging its output

        With --np 1 the binary is run directly (no MPI launcher) unless --always-mpiexec is
        given, which allows single-rank runs where the launcher cannot be used.

        Args:
            name: Run name, used for the log file
            toml_path: Options file in the work directory

        Raises:
            Failure: If the run exits with a non-zero status
        """
        if self.args.np == 1 and not self.args.always_mpiexec:
            cmd = [str(self.args.exe), "-opt", toml_path.name]
        else:
            cmd = self.args.mpiexec.split() + ["-np", str(self.args.np), str(self.args.exe),
                                               "-opt", toml_path.name]
        log = self.logdir / f"{name}.log"
        start = time.time()
        with open(log, "w") as f:
            result = subprocess.run(cmd, cwd=self.workdir, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        print(f"  ran {name} in {elapsed:.1f} s (log: {log.relative_to(self.workdir.parent)})",
              flush=True)
        if result.returncode != 0:
            tail = log.read_text().splitlines()[-30:]
            raise Failure(f"run {name} failed with exit code {result.returncode}:\n    " +
                          "\n    ".join(tail))

    def verify_restart_log(self, name: str, read_dir: str, cycle: int, end_step: int) -> None:
        """
        Check from a run's log that it really restarted, from the expected checkpoint

        The run must report restarting from read_dir/cycle_<cycle> and restoring the state, and
        it must run exactly steps cycle+1 .. end_step. A run that ignored the checkpoint and
        started from step 0 would otherwise go unnoticed, since it produces the same results.

        Args:
            name: Run name (its log is checked)
            read_dir: Checkpoint directory the run should have read, relative to the work
                directory
            cycle: Cycle of the checkpoint it should have restarted from
            end_step: Last step it should have run
        """
        text = (self.logdir / f"{name}.log").read_text()
        expected = f"Restarting from checkpoint {read_dir}/cycle_{cycle:010d} (cycle {cycle},"
        restarted = expected in text and "Restored checkpoint state" in text
        steps = [int(line.split(":")[1]) for line in text.splitlines()
                 if line.startswith("Simulation cycle:")]
        ran_remaining = steps == list(range(cycle + 1, end_step + 1))
        detail = (f"steps {steps[0]}..{steps[-1]}" if steps else "no steps") + \
            f" (expected {cycle + 1}..{end_step})"
        self.check(restarted and ran_remaining,
                   f"{name}: restarted from {read_dir}/cycle_{cycle:010d} and ran only the "
                   f"remaining steps: {'yes' if restarted else 'NO restart message'}, {detail}")

    def checkpoint(self, ckpt_root: str, cycle: int) -> Path:
        """
        Path of a checkpoint directory

        Args:
            ckpt_root: Checkpoint root directory, relative to the work directory
            cycle: Completed cycles of the checkpoint

        Returns:
            Path of the cycle_<cycle> directory
        """
        return self.workdir / ckpt_root / f"cycle_{cycle:010d}"

    def compare_checkpoints(self, label: str, ckpt_a: Path, ckpt_b: Path,
                            expect_differ: bool = False) -> None:
        """
        Compare two checkpoints with compare_restart_checkpoints

        Args:
            label: Description used in the check message
            ckpt_a: First checkpoint directory (normally the reference)
            ckpt_b: Second checkpoint directory
            expect_differ: If True, the check passes only if the checkpoints differ
        """
        for ckpt in (ckpt_a, ckpt_b):
            if not (ckpt / "commit.json").exists():
                self.check(False, f"{label}: committed checkpoint "
                                  f"{ckpt.relative_to(self.workdir)} exists")
                return
        cmd = [str(self.args.compare_tool), str(ckpt_a), str(ckpt_b)]
        if self.args.rtol > 0:
            cmd += ["--rtol", repr(self.args.rtol)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        summary = (result.stdout.strip().splitlines() or ["(no output)"])[-1]
        if expect_differ:
            self.check(result.returncode == 1, f"{label}: must differ: {summary}")
            return
        self.check(result.returncode == 0, f"{label}: {summary}")
        if result.returncode != 0:
            for line in result.stdout.strip().splitlines()[:15]:
                print(f"      {line}")

    def compare_outputs(self, label: str, ref_basename: str, other_basename: str,
                        after_time: Optional[float] = None, expect_differ: bool = False) -> None:
        """
        Compare every volume-average output file of two runs

        The time columns must match exactly, and the values bitwise (or within --rtol).

        Args:
            label: Description used in the check message
            ref_basename: Basename of the reference run
            other_basename: Basename of the run to compare
            after_time: If given, reference rows with time <= after_time are skipped (for runs
                that restarted at that time and so only wrote the later rows)
            expect_differ: If True, the check passes only if the outputs differ
        """
        ref_dir = self.workdir / "results" / ref_basename
        other_dir = self.workdir / "results" / other_basename
        ref_files = sorted(p.name for p in ref_dir.glob("avg_*.txt"))
        if not ref_files:
            self.check(False, f"{label}: reference outputs exist in {ref_dir}")
            return
        worst = 0.0
        problems = []
        for name in ref_files:
            other = other_dir / name
            if not other.exists():
                problems.append(f"{name} missing")
                continue
            a = np.atleast_2d(np.loadtxt(ref_dir / name, comments="#"))
            b = np.atleast_2d(np.loadtxt(other, comments="#"))
            if after_time is not None:
                a = a[a[:, 0] > after_time + 1e-9]
            if a.shape != b.shape:
                problems.append(f"{name}: {b.shape[0]} rows vs {a.shape[0]} expected")
                continue
            if not np.array_equal(a[:, 0], b[:, 0]):
                problems.append(f"{name}: time columns differ")
                continue
            scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
            rel = np.max(np.abs(a - b) / scale)
            worst = max(worst, rel)
            if (self.args.rtol == 0 and not np.array_equal(a, b)) or rel > self.args.rtol:
                problems.append(f"{name}: max rel diff {rel:.3e}")
        detail = "; ".join(problems) if problems else \
            f"{len(ref_files)} files, max rel diff {worst:.3e}"
        if expect_differ:
            self.check(bool(problems), f"{label}: outputs must differ from reference ({detail})")
            return
        self.check(not problems, f"{label}: outputs vs reference ({detail})")


# ============================================================================
# TEST CASE BUILDING BLOCKS
# ============================================================================

def parse_cycles(spec: Optional[str], available: List[int], default: List[int]) -> List[int]:
    """
    Interpret a --*-restart-from option

    Args:
        spec: "all", "none", a comma-separated list of cycles, or None for the default
        available: Reference checkpoint cycles that can be restarted from
        default: Cycles to use when spec is None

    Returns:
        Cycles to restart from
    """
    if spec == "all":
        return list(available)
    if spec == "none":
        return []
    if spec is None:
        return default
    return [int(s) for s in spec.split(",") if s]


def run_reference(h: Harness, name: str, mesh: MeshSpec, bcs: str, total_steps: int,
                  interval: int) -> List[int]:
    """
    Run a case's uninterrupted reference job, writing checkpoints along the way

    Args:
        h: Test harness
        name: Reference run name (also its basename and checkpoint directory suffix)
        mesh: Mesh of the case
        bcs: [BCs] table
        total_steps: Number of steps to run
        interval: Checkpoint interval in steps

    Returns:
        Cycles of the complete checkpoints the reference kept
    """
    write_toml(h.workdir / f"{name}.toml", name, mesh, h.args.num_grains, bcs,
               time_of(total_steps),
               {"write": True, "interval": interval, "write_final": True,
                "keep_last": h.args.keep_last, "directory": f"ckpt_{name}"})
    h.run(name, h.workdir / f"{name}.toml")
    ref_cycles = sorted(int(p.name.split("_")[1])
                        for p in (h.workdir / f"ckpt_{name}").glob("cycle_*")
                        if (p / "commit.json").exists())
    print(f"  reference checkpoints kept: {ref_cycles}")
    return ref_cycles


def run_chain(h: Harness, ref: str, ref_cycles: List[int], name: str, mesh: MeshSpec, bcs: str,
              stops: List[int]) -> None:
    """
    Run a job in segments, each restarting from the previous one, and compare with the reference

    Segment 1 starts fresh; every later segment restarts from the newest checkpoint in the
    chain's own checkpoint directory and extends t_final to its stop. Every segment writes a
    final checkpoint. All segments share one basename, so their volume-average outputs append
    into the same files, which are then compared with the reference's in full.

    Args:
        h: Test harness
        ref: Reference run name
        ref_cycles: Cycles of the reference's checkpoints
        name: Chain name (its basename and checkpoint directory suffix)
        mesh: Mesh of the case
        bcs: [BCs] table
        stops: Step at which each segment stops, increasing; the last is the end of the run
    """
    for i, stop in enumerate(stops):
        segment = f"{name}_seg{i + 1}"
        restart = {"write": True, "interval": 0, "write_final": True, "keep_last": 0,
                   "directory": f"ckpt_{name}"}
        if i > 0:
            restart["read"] = True
        write_toml(h.workdir / f"{segment}.toml", name, mesh, h.args.num_grains, bcs,
                   time_of(stop), restart)
        h.run(segment, h.workdir / f"{segment}.toml")
        if i > 0:
            h.verify_restart_log(segment, f"ckpt_{name}", stops[i - 1], stop)
    for stop in stops:
        if stop in ref_cycles:
            h.compare_checkpoints(f"{name} cycle {stop}", h.checkpoint(f"ckpt_{ref}", stop),
                                  h.checkpoint(f"ckpt_{name}", stop))
    h.compare_outputs(name, ref, name)


def run_from_reference(h: Harness, ref: str, ref_cycles: List[int], from_cycles: List[int],
                       mesh: MeshSpec, bcs: str, total_steps: int) -> None:
    """
    Restart from past reference checkpoints and compare each continuation with the reference

    Each continuation reads the reference's checkpoint through [Restart] read_directory and
    writes its own checkpoints to a separate directory, so the reference's checkpoints are never
    touched. Its final checkpoint is compared with the reference's final one, and its outputs
    with the reference rows after the restart time.

    Args:
        h: Test harness
        ref: Reference run name
        ref_cycles: Cycles of the reference's checkpoints
        from_cycles: Cycles to restart from
        mesh: Mesh of the case
        bcs: [BCs] table
        total_steps: Last step of the run
    """
    for cycle in from_cycles:
        if cycle not in ref_cycles:
            h.check(False, f"{ref}: requested restart cycle {cycle} is among the kept "
                           f"checkpoints {ref_cycles}")
            continue
        name = f"{ref}_from{cycle}"
        write_toml(h.workdir / f"{name}.toml", name, mesh, h.args.num_grains, bcs,
                   time_of(total_steps),
                   {"write": True, "interval": 0, "write_final": True, "keep_last": 0,
                    "read": True, "read_cycle": cycle, "read_directory": f"ckpt_{ref}",
                    "directory": f"ckpt_{name}"})
        h.run(name, h.workdir / f"{name}.toml")
        h.verify_restart_log(name, f"ckpt_{ref}", cycle, total_steps)
        h.compare_checkpoints(f"{name} final cycle {total_steps}",
                              h.checkpoint(f"ckpt_{ref}", total_steps),
                              h.checkpoint(f"ckpt_{name}", total_steps))
        h.compare_outputs(name, ref, name, after_time=time_of(cycle))


def run_negative_control(h: Harness, ref: str, ref_cycles: List[int], mesh: MeshSpec, bcs: str,
                         total_steps: int) -> None:
    """
    Restart from a perturbed copy of a reference checkpoint; the result must differ

    The latest reference checkpoint before the end is copied, and every beginning-of-step state
    variable on rank 0 is scaled by (1 + 1e-12). The end-of-step partners are stored as copies
    of those values, so they are perturbed consistently. A continuation then restarts from the
    copy, and its final checkpoint and outputs must differ from the reference. This shows that
    the restored state is what the continuation actually uses, and that the comparisons pick up
    even a tiny difference. Skipped with a warning if h5py is not available.

    Args:
        h: Test harness
        ref: Reference run name
        ref_cycles: Cycles of the reference's checkpoints
        mesh: Mesh of the case
        bcs: [BCs] table
        total_steps: Last step of the run
    """
    try:
        import h5py
    except ImportError:
        print("  WARNING: h5py is not available; skipping the negative control")
        return
    candidates = [c for c in ref_cycles if c < total_steps]
    if not candidates:
        return
    cycle = candidates[-1]  # the latest one keeps this extra run short
    name = f"{ref}_perturbed{cycle}"
    source = h.workdir / f"ckpt_{name}_src"
    shutil.copytree(h.checkpoint(f"ckpt_{ref}", cycle), source / f"cycle_{cycle:010d}")
    with h5py.File(source / f"cycle_{cycle:010d}" / "rank_000000.hdf5", "r+") as f:
        paths = [f"sidre/external/qfuncs/{g}/values" for g in f["sidre/external/qfuncs"]
                 if g.startswith("state_var_beg")]
        if not paths:
            h.check(False, f"{name}: found state_var_beg arrays to perturb")
            return
        for path in paths:
            f[path][...] = f[path][...] * (1.0 + 1e-12)
    print(f"  negative control: perturbed {', '.join(p.split('/')[3] for p in paths)} by 1e-12 "
          f"(relative) in a copy of {ref} cycle {cycle}")
    write_toml(h.workdir / f"{name}.toml", name, mesh, h.args.num_grains, bcs,
               time_of(total_steps),
               {"write": True, "interval": 0, "write_final": True, "keep_last": 0,
                "read": True, "read_cycle": cycle, "read_directory": source.name,
                "directory": f"ckpt_{name}"})
    h.run(name, h.workdir / f"{name}.toml")
    h.verify_restart_log(name, source.name, cycle, total_steps)
    h.compare_checkpoints(f"{name} (negative control) final cycle {total_steps}",
                          h.checkpoint(f"ckpt_{ref}", total_steps),
                          h.checkpoint(f"ckpt_{name}", total_steps), expect_differ=True)
    h.compare_outputs(f"{name} (negative control)", ref, name, after_time=time_of(cycle),
                      expect_differ=True)


# ============================================================================
# TEST CASES
# ============================================================================

def run_cyclic(h: Harness, mesh: MeshSpec) -> None:
    """
    Cyclic case: fully reversed strain cycles, chained every 1 and 2 cycles

    Each cycle goes 0 -> +amp -> -amp -> 0 in z, with the top-face velocity reversing at the
    peaks, so a cycle takes 4 * steps_per_quarter steps. The boundary-condition update steps
    fall at the peaks, while the checkpoints fall at cycle ends, in the middle of a segment, so
    restarting exercises the boundary-condition replay.

    Args:
        h: Test harness
        mesh: Mesh of the case
    """
    args = h.args
    quarter = args.steps_per_quarter
    steps_per_cycle = 4 * quarter
    total_steps = args.cycles * steps_per_cycle
    # +v for the first quarter, then alternating -v / +v for half a cycle each.
    update_steps = [1]
    velocities = [TOP_VELOCITY]
    step = 1 + quarter
    sign = -1.0
    while step <= total_steps:
        update_steps.append(step)
        velocities.append(sign * TOP_VELOCITY)
        step += 2 * quarter
        sign = -sign
    bcs = bc_block(update_steps, velocities)
    amplitude = quarter * TOP_VELOCITY * DT
    interval = args.cyclic_checkpoint_interval or steps_per_cycle
    print(f"\n== cyclic: {args.cycles} cycles of +/-{amplitude * 100:g}% strain, {total_steps} "
          f"steps, BC updates at {update_steps}, reference checkpoints every {interval} steps")

    ref = "cyclic_ref"
    ref_cycles = run_reference(h, ref, mesh, bcs, total_steps, interval)

    for stride in args.cyclic_chain_strides:
        stops = list(range(stride * steps_per_cycle, total_steps + 1, stride * steps_per_cycle))
        if stops[-1] != total_steps:
            stops.append(total_steps)
        run_chain(h, ref, ref_cycles, f"cyclic_chain{stride}", mesh, bcs, stops)

    restartable = [c for c in ref_cycles if c < total_steps]
    from_cycles = parse_cycles(args.cyclic_restart_from, restartable, restartable)
    run_from_reference(h, ref, ref_cycles, from_cycles, mesh, bcs, total_steps)


def run_monotonic(h: Harness, mesh: MeshSpec) -> None:
    """
    Monotonic case: long tension in z, chained halfway, plus the negative control

    Args:
        h: Test harness
        mesh: Mesh of the case
    """
    args = h.args
    total_steps = args.monotonic_steps
    half = total_steps // 2
    bcs = bc_block([1], [TOP_VELOCITY])
    interval = args.monotonic_checkpoint_interval or max(1, total_steps // 10)
    print(f"\n== monotonic: {total_steps * TOP_VELOCITY * DT * 100:g}% tensile strain along z, "
          f"{total_steps} steps, reference checkpoints every {interval} steps")

    ref = "mono_ref"
    ref_cycles = run_reference(h, ref, mesh, bcs, total_steps, interval)
    run_chain(h, ref, ref_cycles, "mono_chain_half", mesh, bcs, [half, total_steps])

    restartable = [c for c in ref_cycles if c < total_steps]
    default_from = [half] if half in ref_cycles else \
        [min(restartable, key=lambda c: abs(c - half))] if restartable else []
    from_cycles = parse_cycles(args.monotonic_restart_from, restartable, default_from)
    run_from_reference(h, ref, ref_cycles, from_cycles, mesh, bcs, total_steps)
    if args.negative_control:
        run_negative_control(h, ref, ref_cycles, mesh, bcs, total_steps)


def run_empty_ranks(h: Harness, mesh: MeshSpec) -> None:
    """
    Empty-rank case: a short tension run on a 2-element mesh

    With fewer elements than ranks, MFEM puts element i on rank i, so ranks 0 and 1 own one
    element each (and share a face) while every other rank owns nothing. Those ranks must still
    write and read their (empty) checkpoint files and take part in every collective step. The
    reference is chained halfway and restarted from its halfway checkpoint, like the monotonic
    case.

    Args:
        h: Test harness
        mesh: Mesh of the case (2 x 1 x 1 elements)
    """
    args = h.args
    total_steps = args.empty_ranks_steps
    half = total_steps // 2
    bcs = bc_block([1], [TOP_VELOCITY])
    interval = max(1, total_steps // 4)
    empty = max(0, args.np - mesh.num_elements)
    print(f"\n== empty_ranks: {mesh.num_elements} elements on {args.np} rank(s) ({empty} without "
          f"elements), {total_steps} steps, reference checkpoints every {interval} steps")

    ref = "empty_ref"
    ref_cycles = run_reference(h, ref, mesh, bcs, total_steps, interval)
    run_chain(h, ref, ref_cycles, "empty_chain_half", mesh, bcs, [half, total_steps])
    from_cycles = [half] if half in ref_cycles else []
    run_from_reference(h, ref, ref_cycles, from_cycles, mesh, bcs, total_steps)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """
    Parse the options, run the requested cases and report the results

    Returns:
        0 if every check passed, 1 if a check or run failed, 2 for a setup error
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--np", type=int, default=8, help="MPI ranks (default 8)")
    p.add_argument("--mpiexec", default="mpirun", help="MPI launcher (default mpirun)")
    p.add_argument("--always-mpiexec", action="store_true",
                   help="use the launcher even with --np 1 (default: run the binary directly)")
    p.add_argument("--exe", default=str(SCRIPT_DIR / ".." / "bin" / "mechanics"))
    p.add_argument("--compare-tool",
                   default=str(SCRIPT_DIR / ".." / "bin" / "compare_restart_checkpoints"))
    p.add_argument("--data-dir", default=None, help="directory with props_cp_voce.txt etc.")
    p.add_argument("--workdir", default=str(Path.cwd() / "restart_runs"))
    p.add_argument("--cases", default="cyclic,monotonic,empty_ranks",
                   help="comma-separated cases to run (default cyclic,monotonic,empty_ranks)")
    p.add_argument("--nxyz", type=int, default=16,
                   help="cyclic/monotonic: elements per cube edge (default 16)")
    p.add_argument("--num-grains", type=int, default=500)
    p.add_argument("--seed", type=int, default=12345, help="seed for the element-to-grain map")
    p.add_argument("--rtol", type=float, default=0.0, help="0 = bitwise (default)")
    p.add_argument("--keep-last", type=int, default=0,
                   help="[Restart] keep_last for the reference runs (0 = keep every checkpoint)")
    p.add_argument("--cycles", type=int, default=4, help="cyclic: number of cycles (default 4)")
    p.add_argument("--steps-per-quarter", type=int, default=10,
                   help="cyclic: steps from 0 to peak strain; 1e-3 strain per step "
                        "(default 10 = 1%%)")
    p.add_argument("--cyclic-chain-strides", default="1,2",
                   help="cyclic: chained runs stopping every N cycles (default 1,2)")
    p.add_argument("--cyclic-checkpoint-interval", type=int, default=0,
                   help="cyclic: reference checkpoint interval in steps (default: one cycle)")
    p.add_argument("--cyclic-restart-from", default=None,
                   help="cyclic: reference cycles to restart from: list, 'all' or 'none' "
                        "(default all kept, except the last)")
    p.add_argument("--monotonic-steps", type=int, default=100,
                   help="monotonic: steps; 1e-3 strain per step (default 100 = 10%%)")
    p.add_argument("--monotonic-checkpoint-interval", type=int, default=0,
                   help="monotonic: reference checkpoint interval in steps (default steps/10)")
    p.add_argument("--monotonic-restart-from", default=None,
                   help="monotonic: reference cycles to restart from: list, 'all' or 'none' "
                        "(default: the halfway checkpoint)")
    p.add_argument("--empty-ranks-steps", type=int, default=20,
                   help="empty_ranks: steps (default 20)")
    p.add_argument("--no-negative-control", dest="negative_control", action="store_false",
                   help="skip the perturbed-checkpoint negative control")
    args = p.parse_args()
    args.cyclic_chain_strides = [int(s) for s in args.cyclic_chain_strides.split(",") if s]
    args.exe = Path(args.exe).resolve()
    args.compare_tool = Path(args.compare_tool).resolve()

    for tool in (args.exe, args.compare_tool):
        if not tool.exists():
            print(f"ERROR: {tool} not found")
            return 2

    cases: Dict[str, Tuple[Callable[[Harness, MeshSpec], None], MeshSpec]] = {
        "cyclic": (run_cyclic, MeshSpec((args.nxyz,) * 3, "grains.txt")),
        "monotonic": (run_monotonic, MeshSpec((args.nxyz,) * 3, "grains.txt")),
        "empty_ranks": (run_empty_ranks, MeshSpec((2, 1, 1), "grains_empty_ranks.txt")),
    }
    requested = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in requested if c not in cases]
    if unknown:
        print(f"ERROR: unknown case(s) {unknown}; choose from {list(cases)}")
        return 2

    workdir = Path(args.workdir).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    try:
        meshes = {cases[c][1].grain_file: cases[c][1] for c in requested}
        prepare_inputs(workdir, find_data_dir(args.data_dir), list(meshes.values()),
                       args.num_grains, args.seed)
        h = Harness(args, workdir)
        print(f"Restart run tests: {args.np} rank(s), workdir {workdir}")
        for case in requested:
            run_case, mesh = cases[case]
            run_case(h, mesh)
    except Failure as e:
        print(f"\nERROR: {e}")
        return 1

    print(f"\n{h.checks - len(h.failures)}/{h.checks} checks passed")
    for failure in h.failures:
        print(f"  FAILED: {failure}")
    return 0 if not h.failures else 1


if __name__ == "__main__":
    sys.exit(main())
