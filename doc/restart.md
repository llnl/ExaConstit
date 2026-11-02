# Checkpoint and Restart

ExaConstit can write binary checkpoints during a run and restart a later run from one of them.
A restarted run continues exactly where the checkpoint left off: with the same number of MPI
ranks it produces bitwise identical results to a run that never stopped.

A restart needs only the checkpoint and the options file. The mesh, grain, region-mapping,
orientation and state-variable files of the original run are not read again and may have been
deleted.

## Contents
1. [Requirements](#requirements)
2. [Using checkpoints](#using-checkpoints)
3. [What may change between runs](#what-may-change-between-runs)
4. [How it works](#how-it-works)
5. [Checkpoint layout](#checkpoint-layout)
6. [Testing](#testing)
7. [Limitations](#limitations)

## Requirements

Checkpoints are written with Axom's Sidre library, one HDF5 file per MPI rank. ExaConstit must
be configured with `-DAXOM_DIR=<axom install>/lib/cmake`, where Axom was built with Sidre and
its Conduit and HDF5 dependencies. The install scripts in `scripts/install/` build all three
(Axom v0.14.0, Conduit v0.9.7, HDF5 1.14.6). Without Axom, a `[Restart]` table that asks to
write or read checkpoints is rejected when the options are validated.

## Using checkpoints

Everything is controlled by the `[Restart]` table of the options file (documented with every
key in `src/options.toml`):

```toml
[Restart]
    write          = true       # write checkpoints during this run (default false)
    interval       = 50         # every 50 completed steps; 0 = only the final one (default 0)
    write_final    = true       # also write one when the run finishes (default true)
    keep_last      = 2          # keep the newest 2 checkpoints; 0 = keep all (default 2)
    directory      = "restart"  # where checkpoints are written (default "restart")
    read           = false      # restart this run from a checkpoint (default false)
    read_cycle     = 1200       # which checkpoint to read; newest complete one if omitted
    read_directory = "other"    # read from here instead of `directory` (optional)
```

Each checkpoint is a directory `<directory>/cycle_<completed steps>/`. To continue a run, set
`read = true`, keep `directory` pointing at the same place, and increase `t_final` (or extend
the custom time steps). The restarted run picks the newest complete checkpoint unless
`read_cycle` names one, and keeps writing checkpoints to `directory` if `write` is set.

`read_directory` restarts from another run's checkpoints while writing this run's own
checkpoints to `directory`. The other run's checkpoints are never renamed or removed, which is
useful for trying several continuations of one run, and is how the restart tests reuse a
single reference run.

A checkpoint written at the final time is complete but has nothing left to run; restarting
from it prints a warning unless `t_final` has been increased.

**Output files.** Volume-average output files are opened in append mode, so a continuation
that keeps the same `basename` appends to the original run's files and they read as one
uninterrupted run. Use a different `basename` to keep the continuation's output separate.

## What may change between runs

**Must stay the same** (checked; a mismatch aborts with a message saying what changed):
- the number of MPI ranks, since each rank holds its own piece of the mesh and state;
- the mesh order;
- the material list: the number of materials, their order, names, model types and state
  variable counts (including orientations).

**May change:**
- `t_final` and the time-step settings;
- solver, preconditioner and assembly settings;
- the device (`rtmodel`);
- visualization and post-processing settings. A quadrature function enabled only in the
  continuation starts from its initial value, and one dropped by the continuation is skipped;
  both are reported as warnings;
- material properties, with a warning; the continuation uses its own values.

**Boundary conditions** for the steps already completed must stay the same; later entries of
the schedule may change. **Ignored on restart:** `ref_ser`, `ref_par` and the mesh file and
auto-mesh settings, since the mesh comes from the checkpoint.

## How it works

The implementation is documented in detail in the code. This section is an overview.

| File | Contents |
|---|---|
| `src/sim_state/restart_checkpoint.hpp/.cpp` | `exaconstit::RestartCheckpoint`: writing, selecting, validating and restoring checkpoints |
| `src/sim_state/exa_sidre_parmesh.hpp/.cpp` | Writing a rank's piece of an `mfem::ParMesh` and rebuilding the identical `ParMesh` from it |
| `src/sim_state/restart_sidre_utils.hpp` | Small helpers shared by the two |
| `src/options/option_restart.cpp` | The `[Restart]` table |
| `src/sim_state/simulation_state.cpp` | The restart branches of the `SimulationState` constructor and `TimeManagement::ResumeFromCheckpoint()` |
| `src/mechanics_driver.cpp` | Restoring the state and boundary conditions before the time loop, and writing checkpoints in it |

The background notes at the top of `restart_checkpoint.cpp` and `exa_sidre_parmesh.cpp` are
the place to start before changing anything.

**Build first, overlay last.** A restarted run constructs `SimulationState`, `SystemDriver`
and `PostProcessingDriver` exactly as a fresh run does, then overwrites the evolving state
(coordinates, velocity, state variables, stresses and time) with the checkpointed values.
Anything a constructor initializes is simply replaced, and any quadrature function a component
registers is restored, without the restart code needing to know about it.

The one exception is that the `SimulationState` constructor would read input files that may
no longer exist. On restart runs it takes those few things from the checkpoint instead
("phase 1"): the mesh, the reference coordinates, and the grain and region IDs. It also skips
the state-variable and orientation initialization. Everything else is restored from the driver
once every object exists ("phase 2").

**The mesh** is rebuilt from what each rank wrote, with the same local numbering, element
vertex order, communication groups, shared entities and shared-entity orientation as the
original, so the finite element spaces number their DOFs identically. The standard
`axom::sidre::MFEMSidreDataCollection` is not used: it records shared edges and faces in each
rank's local orientation, which matches shared DOFs incorrectly as soon as a shared face has
more than one interior DOF (H1 order 3 and up on hexes). It also references the mesh's memory
instead of copying it, and does not hand ownership of what it rebuilds to the caller.

**Checks before anything is overwritten.** The restarted run verifies the checkpoint's format
version and rank count, the rebuilt mesh topology, the coordinate space sizes, that every DOF
is numbered as in the original (an element-local hash and a shared-DOF round trip of the
reference coordinates), the material identities, and every quadrature space and function's
shape. Problems found on any rank abort every rank together, with each failing rank reporting
its own details.

**Writing is safe to interrupt.** Each rank writes its own file, and rank 0 writes `commit.json`
only once every rank has succeeded. A checkpoint without it is never read, so a crash during
a write cannot leave a checkpoint that looks complete. A failed write is reported as a warning
and the run continues.

**Ranks without elements** are fully supported: they write and read empty checkpoint files and
take part in every collective step.

**Adding new state.** Quadrature point data registered with
`SimulationState::AddQuadratureFunction()` is checkpointed automatically, including data a new
material model registers. Nodal data is not: only the coordinates and velocity are saved, and
a new grid function or solution vector has to be added to the checkpoint explicitly. Only the
volume mesh is saved; submeshes and other topologies must be rebuilt from it. The developer's
guide section "Keeping New State Restartable" lists the rules.

## Checkpoint layout

```
<directory>/cycle_<completed steps, 10 digits>/
    rank_<rank, 6 digits>.hdf5    one per rank
    commit.json                   written last by rank 0; marks the checkpoint complete
```

Each rank file holds:

| Group | Contents |
|---|---|
| `restart_info/` | format version, rank count, rank, completed steps, driver step, completed time, next, previous and original time steps, time-step type, mesh order, ExaConstit version |
| `mesh/` | the rank's piece of the mesh (Conduit Blueprint layout) and each element's grain ID |
| `fields/` | reference and current coordinates, velocity, and the coordinate space sizes and DOF-layout hash |
| `qspaces/` | validation data for every quadrature space |
| `qfuncs/` | every quadrature function except the tangent stiffness (recomputed before use); an end-of-step function identical to its beginning-of-step partner is stored as `copy_of` instead of repeating its values |
| `materials/` | each region's material name, model type, state variable count and properties |

Two diagnostic tools are built alongside the tests: `inspect_restart_rank_file` lists how each
quadrature function is stored in a rank file, and `compare_restart_checkpoints` compares two
checkpoints view by view (bitwise by default).

## Testing

The restart tests are built when ExaConstit is configured with Axom and tests are enabled.

**Default suite** (fast; runs with `ctest`, including on CI):
- `test_restart_mesh_roundtrip_np{1..4}`: writes and rebuilds hex and tet meshes, with and
  without parallel refinement, and a case where half of the ranks own no elements. Checks the
  rebuilt mesh's topology, communication groups, shared-entity order and orientation, and that
  H1 spaces of order 1 to 4 have identical DOF layouts.
- `test_restart_e2e_np{2,3}`: builds a real `SimulationState` from input files, checkpoints a
  synthetic evolved state, deletes the input files, restarts, and compares everything bitwise.

No test uses more MPI ranks than the machine has cores: rank counts above
`EXACONSTIT_TEST_MAX_MPI_TASKS` (default: the physical core count detected at configure time,
since Open MPI counts physical cores as slots) are skipped.

**Long tests** (not on CI; developers changing the restart code must run them locally):
configure with `-DEXACONSTIT_ENABLE_LONG_TESTS=ON`, then run

```bash
ctest -L long --output-on-failure
```

This runs `test/test_restart_runs.py` on real simulations (Voce FCC crystal plasticity on a
4096-element mesh, 8 ranks by default via `EXACONSTIT_RESTART_RUN_TEST_NP`, capped at
`EXACONSTIT_TEST_MAX_MPI_TASKS`), which takes 10 to 15 minutes:
- **cyclic**: four ±1% strain cycles, chained every cycle and every two cycles, and restarted
  from each past checkpoint of the uninterrupted reference;
- **monotonic**: 10% tension, chained halfway and restarted from the reference's halfway
  checkpoint;
- **empty_ranks**: a 2-element mesh, so all but two ranks own no elements.

Every checkpoint a restarted run writes is compared bitwise with the reference checkpoint of
the same step, and its volume-average output row by row. Each restarted run's log is checked to
confirm it really restarted from the expected checkpoint and ran only the remaining steps, and
a negative control restarts from a slightly perturbed checkpoint and requires the result to
differ. `python3 test_restart_runs.py --help` lists the options, including a smaller mesh, fewer
steps and fewer ranks for quick local checks, and the device (`--rtmodel`) and assembly
(`--assembly`) of the runs.

GPU builds also register `test_restart_runs_gpu` (labels `restart;long;gpu`), which runs the same
cases with `rtmodel = "GPU"` and `EXACONSTIT_RESTART_RUN_TEST_GPU_ASSEMBLY` (default `EA`, the
fastest GPU assembly for these runs). It checks that the restored host data reaches the device and
that GPU runs restart bitwise identically. Run only the GPU tests with `ctest -L gpu`.

## Limitations

- **Same rank count.** Restarting on a different number of ranks would need a
  partition-independent layout and redistribution of the state; it is not supported.
- **One file per rank per checkpoint.** Very large runs produce many files. Axom's `IOManager`
  can group them later without changing the commit protocol.
- **Conforming meshes only**, with one element geometry and one boundary geometry
  (segment, triangle, quadrilateral, tetrahedron or hexahedron). Nonconforming and NURBS meshes
  are rejected.
- **GPU testing so far is CUDA only.** The restored data is written on the host and marked
  valid there, so it is copied to the device on first use. `test_restart_runs_gpu` passes
  bitwise with PA and EA on an NVIDIA GPU (CUDA); it has not been run on an AMD GPU (HIP) yet.
