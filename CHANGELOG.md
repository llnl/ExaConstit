# Changelog

All notable changes to ExaConstit are documented in this file, one line per change.
Entries for releases before v0.10.0 were reconstructed from the release tag notes.

## [Unreleased]

### Added
- `field_location` option in `[Visualizations]`: `"zone"` (default, one element-averaged value per element) or `"quadrature"` (at each H1 node location of every element, the value of the nearest quadrature point, so bounded quantities such as unit quaternions and non-negative plastic strain stay bounded). Both share the same projection kernels through `VizPointMap`.
- Named state variables in the visualization output: `visualize_state_vars` in a `[[Materials]]` table (a list of names or `"all"`) and `region_<N>` in `[PostProcessing.projections] enabled_projections`. Names come from `SimulationState::GetQuadratureFunctionStatePairs`, in registration order; unnamed slots of non-ExaCMech models are written as `State Var <k>`.
- User functions loaded from shared libraries like a UMAT, as plain C++ functions taking `std::shared_ptr<SimulationState>` and a region: `register_function` / `initialize_function` in `[Materials.State_Vars]` (name state variables, with an optional plot flag, and set their initial values for any material model), and `[[PostProcessing.user_functions]]` entries (per-point quantities written as visualization fields or volume averages / integrals, per material and globally). Every function's library and signature is checked at startup.
- `find_package(ExaConstit)` CMake package and `exaconstit_add_user_library()` for building user function libraries against an installed ExaConstit, with an example in `examples/user_functions`.
- `SimulationState::AddQuadratureFunctionStatePair` stops the run for a slice outside the state variables or overlapping another one.
- Volume statistics: `statistics` (mean, variance, std, skewness, excess kurtosis, min, max, or `"all"`) and fixed-bin `histograms` for the built-in volume averages in `[PostProcessing.volume_averages]`, and `statistics` / `histogram` keys for `volume_average` user function entries. Volume-weighted, per material and for the whole sample (combined exactly from the material statistics), written to a new `stats/` folder of the results directory, one file per statistic and one histogram file per quantity.
- `[PostProcessing] output_format = "hdf5"`: all volume averages, volume statistics and histograms, user function averages / integrals and surface quantities of a run in one file, `post_processing.hdf5`, written with Conduit (needs Axom). Each text file is a group and each text column a dataset; a restarted run removes the rows written after its checkpoint, so no step is repeated. `scripts/postprocessing/read_post_processing.py` reads it (h5py).
- Batched UMATs: `interface = "batched"` in `[Materials.Model.UMAT]` calls the UMAT once per region with all of its points (`UmatBatchFunction` in `src/umats/userumat.h`: the UMAT arguments with `NPTS` first and a last `NPTS` dimension on the arrays), in the run's memory space, so a GPU-capable UMAT runs on the GPU and a CPU one can use OpenMP. Examples in C++ and Fortran in `examples/umat_batched`.
- `thread_safe = true` in `[Materials.Model.UMAT]` runs standard (per-point) UMAT calls in an OpenMP loop.

### Changed
- The UMAT model prepares the UMAT arguments and handles its results in device kernels (shared `GradCalc` and new `utilities/kinematics_kernels.hpp`), so GPU runs only copy data for standard (per-point) UMAT calls. `DFGRD0` / `DFGRD1` are computed exactly from the beginning-of-step and current nodes every step instead of from stored reference shape gradients, and the per-point reference shape gradient copy is gone.
- UMAT inputs follow their definitions: `STRAN` is the Eulerian strain at the start of the increment (was at the end), `DSTRAN` the Eulerian strain 1/2 (I - B^-1) of the increment (was 1/2 (I - C^-1)), `CELENT` uses the reference element volume (unchanged for hexahedra, was too small for tetrahedra), and `TEMP` is the material's `temperature` (was 300 K). Results of UMATs that use `STRAN`, `DSTRAN` or `CELENT` change.
- UMAT strains and stress rotation (`[Materials.Model.UMAT]`): `strain_measure = "logarithmic"` (default; exact Hencky strains ln V of `DFGRD0` and of the increment) or `"eulerian"`, and `rotate_stress = true` (default), which rotates `STRESS` and `STRAN` by `DROT` before the call as Abaqus does, so isotropic UMATs written for Abaqus work unchanged; set it false for UMATs that rotate themselves. `DROT` is the exact rotation of the increment (polar decomposition). With the defaults, results of UMATs that rotated the stress themselves or used the strains change.
- UMAT bookkeeping inputs follow Abaqus: `TIME(1)` and `TIME(2)` are the time at the start of the increment (`TIME(2)` was the end), `KSTEP` = 1 and `KINC` the time step number (both were 0), `NOEL` the 1-based global element number and `NPT` the 1-based point number (were 0-based region-local indices), and `CMNAME` the material's name (was empty).
- The UMAT section of `src/options.toml` lists what every UMAT argument holds and which outputs ExaConstit uses.
- UMAT energies `SSE`, `SPD`, `SCD` are kept from step to step (per-region quadrature functions `umat_sse`, `umat_spd`, `umat_scd` with beginning- and end-of-step copies, written to restart files) instead of starting at 0 in every call, so UMATs that accumulate them, for example damage models driven by elastic energy or dissipation, get the accumulated values Abaqus would pass. A restart from a checkpoint written before this change starts them from 0.

### Removed
- The `thermal` UMAT option: it was read but had no effect. Input files that still set it get a warning.

### Fixed
- `load_strategy = "lazy_load"` never loaded the UMAT library, so runs stopped with "UMAT function not available"; with `load_on_setup`, a point asking for a smaller time step left the library loaded.
- The `Crystal Orientations` visualization field divided the quaternions by their squared length instead of their length, so they were not unit quaternions.
- Enabling the `all_state` projection crashed (its vector size was unknown when the visualization fields were created).
- Installed headers kept no directory structure, so headers including e.g. `sim_state/simulation_state.hpp` could not be used from an install.
- The rank-0 gather of global volume averages sent more values than it held from a region root whose region had no valid result.
- The built-in UMAT (`enable_dynamic_loading = false`) could not be found: it was only in the static library, which nothing referenced it from, so it was not linked into `mechanics` and runs stopped with "No built-in UMAT available". It is now compiled into the executable.
- Volume average file headers truncated column names of 18 characters or more and ran them together (e.g. `Equiv_Plastic_Strain`); long names are now written whole.

## [v0.10.1] - 2026-10-21

Requires the same MFEM fork commit as v0.10.0.

### Fixed
- Volume-averaged elastic strains (`avg_elastic_strain_*.txt`) only included the first quadrature points of each region on each rank (as many as the region had elements) and counted the rest as zero, so they were wrong and changed with the number of MPI ranks. Elastic strain averages written by v0.9.0 and v0.10.0 should not be used.
- The `Elastic Strains` visualization field read the quadrature point state variables by element index instead of the element-averaged ones, so its values were wrong and changed with the number of MPI ranks (v0.9.0 and v0.10.0).

## [v0.10.0] - 2026-10-20

Requires the `exaconstit-dev` branch of the ExaConstit MFEM fork (https://github.com/rcarson3/mfem),
tested with commit `74ded90191cacabfa294b974e7d69d11ea84d364` (MFEM v4.9.1 based).
Element assembly (EA) on the GPU requires commit `3cb179f78508c75d47413b231d0893ef8b5a44c0` or later, which applies
EA gradients with `BatchedLinAlg`; with an older fork, GPU EA runs use a wrong Jacobian.

### Added
- Trust-region dogleg nonlinear solver (`nl_solver = "TRDOG"`), ported from SNLS's `SNLSTrDlDenseG`, with options under `[Solvers.NR.trust_region]`.
- `MultTranspose` support for the mechanics nonlinear form integrators (full, PA, EA, and B-bar PA gradients), needed by the TRDOG solver.
- `print_info` option in `[Solvers.NR]` controlling nonlinear solver verbosity (0 = summary only, 1 = per-iteration, 2 = per trust-region trial).
- Unified per-iteration output (`Newton iteration N : ...`) shared by the NR, NRLS, and TRDOG solvers, including the step scale taken each iteration.
- B-bar integration (`integ_model = "BBAR"`) now passes the same volume-averaged kinematics to the material models (ExaCMech and UMAT) as the integrator uses.
- Umpire is used as MFEM's memory pool manager when MFEM is built with Umpire.
- `xtal_light_up` post-processing module supports all eight crystal systems, matching the in-situ light-up.
- `xtal_light_up` Taylor factors now use ExaCMech's equivalent plastic strain rate and work for any number of slip systems.
- New `-lattice_type`, `-lattice_params`, and `-distance_tolerance_deg` options for `fiber_calcs_rank.py`.
- Output directories that are dangling symlinks now have their targets created automatically.
- New calibration / optimization workflow framework (`workflows/exaconstit-calibrate`) with MPI and Flux backends, data smoothers, an SQL results archive, restart support, and a refactored NSGA-III driver.
- Updated Barlat yield-function optimization workflow (`barlat_optimize_v2.py`).
- GPU install builds tune MFEM's batched gemv kernels on the build machine's GPU and rebuild MFEM with the result (`MFEM_GEMV_TUNING` = `AUTO`/`OFF`/header path, `MFEM_GEMV_TUNING_LAUNCHER` for build nodes without a GPU, `MFEM_GEMV_RETUNE`, `MFEM_GEMV_TUNING_ARGS`); the tuning header is saved and reused by later builds.
- `unix_gpu_cuda_workstation_install.sh` and `gpu_cuda_workstation_config.sh` for Linux workstations with an NVIDIA GPU, which detect the CUDA toolkit, compilers, MPI, Python and GPU architecture instead of needing the config edited.
- Per-surface output (`[PostProcessing.surface_quantities]`): for each listed mesh boundary attribute, one file with the current area, the mean displacement, the force (the sum of the internal nodal forces on the surface, i.e. the reaction force), the average normal, and optionally the moment about the surface centroid. Works on hex and tet meshes.

### Changed
- Updated to MFEM v4.9+ (enables newer Hypre versions, which work better with ROCm v7+).
- Dependency versions used by the install scripts and CI: RAJA v2025.12.2, CAMP/Umpire/CHAI v2025.12.0, Hypre v3.1.0, ExaCMech v0.4.3.
- Rejected trust-region trial steps no longer count as nonlinear iterations, so `max_iter` and auto time step control mean the same thing for all solvers.
- The output directory is resolved once on rank 0 and broadcast, so symlinked results directories behave consistently across ranks and concurrent runs.
- The run aborts at startup if the output directory cannot be created, instead of running without writing results.
- Warnings and errors go through the rank 0 MFEM macros, and problems that used to print an error and continue with missing data (unreadable orientation, state variable, grain, or region mapping files, mismatched grain counts or strides, unresolved light-up regions) now abort.
- **Breaking:** `xtal_light_up` Taylor factor functions take the equivalent plastic strain rate as an argument; calls with the old argument order raise an error.
- PyO3-based Rust extensions (`xtal_light_up`, `voxel_coarsen`) updated to PyO3/numpy 0.29 to fix RustSec advisories; they now require Python 3.9+ and Rust 1.88+.
- Boundary condition types use `std::array<bool, 3>` instead of `mfem::Array<bool>`, avoiding GPU memory issues.
- Updated `.clang-format` so header groups are handled correctly by newer clang-format versions.
- EA element matrices are stored in MFEM's `ea_data(trial, test, elem)` order, and ExaCMech's tangent is no longer handled differently for EA runs on the GPU.
- CUDA builds no longer fail to compile post-processing code: the access specifiers of methods holding device lambdas (and the private types those lambdas use) become `public` in CUDA builds only (`src/utilities/cuda_access.hpp`), since nvcc does not allow extended lambdas there.

### Fixed
- Orientation files for non-ExaCMech models (e.g. UMATs) were always read as quaternions and converted; their values are now passed through as is with the stride set by `ori_type` / `ori_stride`, while ExaCMech models still require unit quaternions.
- `ori_state_var_loc = -1` (append orientations to the end of the state variables) wrote the orientations past the end of the state variable array, leaving them unset for UMATs.
- Results differing between multi-GPU and CPU runs on newer MFEM versions, caused by unset velocity field values when applying boundary conditions.
- The `Time.time_type` option was ignored; it now selects the time stepping mode over the section priority order.
- Overlapping DOF constraints between boundary condition options.
- Auto time stepping not exiting when the minimum time step is reached.
- NRLS line search scale dropping below its minimum value.
- Visualization output not following the requested output cadence, and volume averages being written at the visualization cadence.
- Segfaults in post-processing for models without a given variable (zero vector dimension).
- MPI hangs in post-processing when a region does not exist on every rank.
- An `MPI_Barrier` in `CreateOutputFile` only reached by the writing rank.
- Trust-region solver reporting a rejected trial's residual and state after a delta-control failure.
- Duplicate trust-region retries of the same full Newton step after a rejection.
- Trigonal symmetry operators in the in-situ light-up repeated one twofold axis and missed another.
- Regressions in ExaCMech orientation and grain file option handling.
- Umpire's logging conflicting with ExaConstit's logger at program shutdown.
- Install scripts building in `scripts/install` instead of the directory they were run from.
- `voxel_coarsen` `python` feature failing to compile and `polars` feature swapping its delimiter and comment characters.
- Documented projection names corrected to `Element Volumes` and `Element Centroids` in `options.toml`.
- The documented `material_name` and `crystal_plasticity` keys in `[[Materials]]` were ignored (only `name` and `cp` were read); both spellings are now accepted.
- The `avg_*_fname` options in `[PostProcessing.volume_averages]` were ignored, and `enabled = false` did not turn volume averaging off.
- `options.toml` documentation brought up to date, including the new `[Solvers.NR.trust_region]` options.
- Broken link in the README.
- EA applied the transpose of the element Jacobians, so EA Newton residuals differed from full and partial assembly in the 3rd to 4th digit and the trust-region solver's `J^T r` was wrong.
- The transpose PA gradient of `ExaNLFIntegrator` read the wrong entries of the D tensor and contracted it with swapped indices (adjointness off by about 35% with a non-symmetric tangent).
- The B-bar (`ICExaNLFIntegrator`) PA gradient read the 6x6 Voigt tangent as a 3x3x3x3 tensor (reading past the end of the data) and the mesh Jacobians in the wrong layout, giving a wrong Jacobian action.
- Host/device validity bugs that only show up on GPUs with separate memory: velocity self-assignment in `UpdateEndCoords` (diverging multi-GPU runs), `Write()` used where `ReadWrite()` was needed for partial updates of BC vectors, the light-up fiber flags set on the host only, and the fill order in `PartialQuadratureFunction::FillQuadratureFunction`.
- Boundary conditions were never applied when `[BCs.time_info]` was omitted, and the BC option validation read an empty `time_info.cycles`.
- A grain map file with fewer entries than the auto-generated mesh has elements left grain ids of 0 or uninitialized values, leading to out-of-range grain indices; the run now aborts with an error.
- Install scripts assumed Umpire's CMake config directory is under `lib64`, so ExaCMech and ExaConstit could not find Umpire on systems that install into `lib` (Debian, Ubuntu).
- CUDA builds with nvcc as the C++ compiler and g++ as its host compiler failed to configure or link because CMake passed gcc-style options to nvcc; `scripts/install/common/nvcc_cxx_rules_override.cmake` makes CMake emit nvcc-style options.

## [v0.9.0] - 2025-10-31

### Added
- Multi-material support: several ExaCMech and/or UMAT models (regions) can be run together.
- Logging and an expanded post-processing framework with a dedicated post-processing class.
- Light-up support for all major crystal symmetries.
- UMAT loading from a user-provided shared library and function name.
- Selective visualization field output.

### Changed
- Major refactoring of the entire code base, including a central `SimulationState` class and a multi-material `ExaModel` class.
- New option file format and parser; the previous format is still supported, and a conversion script is provided.
- New output layout with headers, time and volume data in all output files, and `basename`-based output directories; `auto_dt_file` removed.
- Greatly expanded documentation.

## [v0.8.2] - 2025-04-11

### Fixed
- Boundary condition manager edge case.

## [v0.8.1] - 2025-04-07

### Fixed
- CMake install issue.

## [v0.8.0] - 2025-03-18

### Added
- Maximum time step option for automatic time stepping.
- Rust-backed post-processing light-up Python script.
- In-situ light-up calculations during simulations.
- Support for more crystal models and ExaCMech model shorthand names.
- Volume-averaged Eulerian strain output.

### Changed
- **Breaking:** Requires ExaCMech v0.4.x.
- **Breaking:** GPU builds require the RAJA Portability Suite (RAJA, Umpire, CHAI).
- **Breaking:** The Python executable must be specified when building the test suite.
- **Breaking:** Minimum CMake version is 3.21 and C++17 is required.
- Updated MFEM fork for newer RAJA versions.
- Legacy ExaCMech material model names are deprecated in favor of shorthand names.
- Auto time stepping retries up to 4 times after a solver failure.
- Fortran is no longer required.

### Removed
- Volume-averaged plastic deformation rate tensor output.

## [v0.7.0] - 2024-01-03

### Added
- HIP support.
- Material parameter optimization and additive manufacturing UQ workflows.
- Powder diffraction style lattice strain post-processing scripts.

### Changed
- **Breaking:** The `CUDA` `rtmodel` option is replaced by `GPU`.

## [v0.6.2] - 2023-03-02

### Fixed
- Boundary condition application.

## [v0.6.1] - 2022-06-17

### Fixed
- Constant strain rate boundary conditions.

## [v0.6.0] - 2022-06-10

### Added
- Light-up quantity output: elastic strain, element centroids, and lattice orientations.
- Automatic time stepping.
- Constant strain rate boundary conditions with a user-defined origin, mixable with velocity boundary conditions.

## [v0.5.0] - 2021-04-04

### Added
- Script converting Neper gmsh output to MFEM meshes.
- Additional volume averages (deformation gradient and, for ExaCMech, plastic work) via `Visualizations.additional_avgs`.

### Fixed
- Subtle bug present in all previous releases (see issue #32).

## [v0.4.1] - 2021-02-08

### Fixed
- Internal compiler error with GCC 6 and 7.

## [v0.4.0] - 2021-01-20

### Added
- Changing essential boundary conditions during a simulation.
- Option to name the average stress output file.
- Unix install example scripts.

### Changed
- **Breaking:** `Time.Auto` renamed to `Time.Fixed` and `Mesh.prefinement` renamed to `Mesh.p_refinement`.
- ExaCMech interface updated for ExaCMech v0.3.2+.
- Python test driver runs in parallel.
- BLT updated to v0.3.6.

## [v0.3.0] - 2020-08-17

### Added
- GPU support through partial assembly and element assembly.
- Incompressible material support through a B-bar formulation.
- ParaView and ADIOS2 data collection output.

### Fixed
- Various bug fixes and performance improvements.
