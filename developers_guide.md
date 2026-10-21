# ExaConstit Developer's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Dependency Version Compatibility](#dependency-version-compatibility)
5. [Codebase Overview](#codebase-overview)
6. [Source Directory Structure](#source-directory-structure)
7. [Key Components](#key-components)
8. [Configuration System](#configuration-system)
9. [Advanced Solver Configuration](#advanced-solver-configuration)
10. [Building and Testing](#building-and-testing)
11. [Development Workflow](#development-workflow)
12. [UMAT Development Resources](#umat-development-resources)
13. [Contributing Guidelines](#contributing-guidelines)

## Introduction

ExaConstit is a high-performance, velocity-based, updated Lagrangian finite element code for nonlinear solid mechanics problems with a focus on micromechanics modeling. Built on the MFEM library, it specializes in crystal plasticity simulations and bulk constitutive property determination for polycrystalline materials.

**Key Features:**
- Velocity-based updated Lagrangian formulation
- Crystal plasticity and micromechanics modeling
- GPU acceleration with CUDA/HIP support
- MPI parallelization for HPC systems
- Integration with ExaCMech material library
- UMAT interface support
- Advanced post-processing capabilities

## Prerequisites

### Required Knowledge
- **C++17**: Modern C++ standards and best practices
- **Finite Element Method (FEM)**: Theory and implementation
- **Solid Mechanics**: Nonlinear mechanics, crystal plasticity
- **Numerical Methods**: Newton-Raphson, Krylov iterative solvers
- **Parallel Computing**: MPI, OpenMP, GPU programming concepts

### System Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, Intel 19+)
- MPI implementation (OpenMPI, MPICH, Intel MPI)
- CMake 3.24 or higher
- Git for version control

## Installation

### Quick Start
For detailed installation instructions, refer to the build scripts in `scripts/install/`:

- **Linux/Unix (Intel CPU)**: `scripts/install/unix_cpu_intel_install.sh`
- **macOS (CPU)**: `scripts/install/unix_cpu_mac_install.sh`
- **GPU (CUDA)**: `scripts/install/unix_gpu_cuda_install.sh`
- **GPU (HIP/AMD)**: `scripts/install/unix_gpu_hip_install.sh`

### Dependencies

**Core Dependencies:**
- **MFEM** (v4.9+, ExaConstit fork): Finite element library with parallel/GPU support
- **ExaCMech** (v0.4.3+): Crystal plasticity constitutive model library
- **RAJA** (≥2025.12.x): Performance portability framework
- **UMPIRE** (≥2025.12.x): (GPU-only) Performance portability framework
- **CHAI** (≥2025.12.x): (GPU-only) Performance portability framework
- **BLT**: LLNL build system
- **SNLS**: Nonlinear solver library

**Optional Dependencies:**
- **ADIOS2**: (MFEM-based) High-performance I/O for visualization
- **Caliper**: Performance profiling
- **Axom** (with Sidre, Conduit and HDF5): Checkpoint/restart support, enabled with `-DAXOM_DIR=<axom install>/lib/cmake` (see `doc/restart.md`)

### Basic Build Process
```bash
git submodule init && git submodule update

# Create build directory
mkdir build && cd build

# Configure
cmake .. \
  -DENABLE_MPI=ON \
  -DENABLE_FORTRAN=OFF \
  -DMFEM_DIR=${MFEM_INSTALL_DIR} \
  -DECMECH_DIR=${EXACMECH_INSTALL_DIR} \
  -DRAJA_DIR=${RAJA_INSTALL_DIR} \
  -DSNLS_DIR=${SNLS_INSTALL_DIR}

# Build
make -j 4
```

## Dependency Version Compatibility

### **MFEM Requirements**
ExaConstit requires a specific MFEM development branch with ExaConstit-specific features:

#### **Current Requirements**
- **Repository**: https://github.com/rcarson3/mfem.git
- **Branch**: `exaconstit-dev`
- **Version Dependencies**:
  - **v0.10.0**: Compatible with MFEM hash `74ded90191cacabfa294b974e7d69d11ea84d364` (MFEM v4.9.1 based)
  - **v0.9.0**: Compatible with MFEM hashes `a6bb7b7c2717e991b52ad72460f212f7aec1173e`
  - **v0.8.0**: Compatible with MFEM hashes `31b42daa3cdddeff04ce3f59befa769b262facd7` or `29a8e15382682babe0f5c993211caa3008e1ec96`
  - **v0.7.0**: Compatible with MFEM hash `78a95570971c5278d6838461da6b66950baea641`
  - **v0.6.0**: Compatible with MFEM hash `1b31e07cbdc564442a18cfca2c8d5a4b037613f0`
  - **v0.5.0**: Required MFEM hash `5ebca1fc463484117c0070a530855f8cbc4d619e`

#### **MFEM Build Requirements**
```bash
# Required dependencies for MFEM
cmake .. \
  -DMFEM_USE_MPI=ON \
  -DMFEM_USE_METIS_5=ON \
  -DMFEM_USE_HYPRE=ON \    # v3.1.0 tested
  -DMFEM_USE_RAJA=ON \     # v2025.12.x+
  -DMFEM_USE_ADIOS2=ON \   # Optional: high-performance I/O
  -DMFEM_USE_ZLIB=ON       # Optional: compressed mesh support
```

**Note**: Future releases will integrate these changes into MFEM master branch, eliminating the need for the development fork.

### **ExaCMech Version Requirements**
- **Repository**: https://github.com/LLNL/ExaCMech.git
- **Version**: v0.4.3+ required (`v0.4.3` tag used by the install scripts and CI)
- **SNLS Dependency**: https://github.com/LLNL/SNLS.git

### **RAJA Portability Suite**
For GPU builds of ExaCMech >= v0.4.3:

#### **Required Components**
- **RAJA**: Performance portability framework
- **Umpire**: Memory management
- **CHAI**: Array abstraction

#### **Version Requirements**
- **Tag**: `v2025.12.x` for all RAJA Portability Suite repositories (see `scripts/install/common/dependency_versions.sh` for the exact tags)
- **Important**: All RAJA suite components (RAJA, Umpire, CHAI) must use matching versions
- **Minimum RAJA**: v2025.12.0
- **Note**: Version mismatch between RAJA components can cause build failures or runtime errors. Umpire must be v2025.12.0 or newer, as it includes a fix (git hash 091305d8ef40aa8f2d75d684fbabeabff2e0c1fc) for a segfault at program shutdown caused by conflicts between Umpire's internal logging and our own.

### **Additional Dependencies**
- **HYPRE**: v3.1.0 tested (algebraic multigrid / various preconditioners)
- **METIS**: Version 5 (mesh partitioning)
- **ADIOS2**: Optional (high-performance parallel I/O)
- **ZLIB**: Optional (compressed mesh and data support)

## Codebase Overview

ExaConstit follows a modular architecture designed for extensibility and performance:

```
ExaConstit/
├── src/                   # Main source code
├── test/                  # Test cases and examples
├── scripts/               # Build scripts and utilities
├── workflows/             # Optimization and UQ workflows
└── cmake/                 # Build system configuration
```

### Design Philosophy
- **Modularity**: Clear separation of concerns between FEM operators, material models, and solvers
- **Performance**: GPU acceleration and memory-efficient algorithms
- **Extensibility**: Plugin architecture for material models and boundary conditions
- **Standards**: Modern C++17 practices and comprehensive documentation

## Source Directory Structure

The `src/` directory contains the core ExaConstit implementation organized into modular components:

### Primary Files
- **`mechanics_driver.cpp`**: Main application entry point and simulation orchestration
- **`system_driver.hpp/cpp`**: Core driver class managing the Newton-Raphson solution process
- **`userumat.h`**: Interface definitions for UMAT material model integration

### Key Directories

#### `boundary_conditions/`
**Purpose**: Boundary condition management and enforcement
- **`BCData.hpp/cpp`**: Data structures for boundary condition storage
- **`BCManager.hpp/cpp`**: Boundary condition application and management

**Key Features**:
- Dirichlet velocity and velocity gradient boundary conditions
- Time-dependent BC scaling and ramping
- Component-wise BC application for selective spatial directions
- Support for multiple BC regions with different behaviors

#### `fem_operators/`
**Purpose**: Finite element operators and integration routines
- **`mechanics_operator.hpp/cpp`**: Nonlinear mechanics operator implementation
- **`mechanics_operator_ext.hpp/cpp`**: Extended operator functionality
- **`mechanics_integrators.hpp/cpp`**: Element-level integration kernels

**Key Features**:
- Element assembly (EA) and partial assembly (PA) modes
- B-bar integration for near-incompressible materials
- GPU-optimized kernel implementations
- Matrix-free operator evaluation

#### `models/`
**Purpose**: Material constitutive model interface and implementations
- **`mechanics_model.hpp/cpp`**: Abstract base class `ExaModel` interface
- **`mechanics_ecmech.hpp/cpp`**: ExaCMech crystal plasticity integration
- **`mechanics_umat.hpp/cpp`**: UMAT interface implementation
- **`mechanics_multi_model.hpp/cpp`**: Multi-region material management

**Supported Models**:
- ExaCMech crystal plasticity (FCC, BCC, HCP)
- User-defined UMAT subroutines
- Multi-material region support

#### `options/`
**Purpose**: Configuration file parsing and option management
- **`option_parser_v2.hpp/cpp`**: Modern TOML-based parser
- **`option_material.cpp`**: Material configuration parsing
- **`option_mesh.cpp`**: Mesh and geometry options
- **`option_boundary_conditions.cpp`**: BC configuration parsing
- **`option_solvers.cpp`**: Linear and nonlinear solver settings
- **`option_time.cpp`**: Time-stepping parameters
- **`option_post_processing.cpp`**: Post-processing configuration
- **`option_restart.cpp`**: Checkpoint/restart configuration
- **`option_enum.cpp`**: Enumeration type conversions
- **`option_util.hpp`**: Utility functions for option parsing

**Features**:
- Backward compatibility with legacy formats
- Hierarchical configuration structure
- Comprehensive validation and error reporting

#### `postprocessing/`
**Purpose**: Output management and field calculations
- **`postprocessing_driver.hpp/cpp`**: Main post-processing orchestration
- **`postprocessing_file_manager.hpp`**: File I/O and directory management
- **`projection_class.hpp/cpp`**: Field projection operations
- **`mechanics_lightup.hpp/cpp`**: Lattice strain calculations

**Capabilities**:
- Volume-averaged stress/strain/deformation gradient
- Lattice strain calculations for diffraction experiments
- Visualization output (VisIt, ParaView, ADIOS2)
- Structured output file organization

#### `sim_state/`
**Purpose**: Simulation state management and field storage
- **`simulation_state.hpp/cpp`**: Central state container class
- **`restart_checkpoint.hpp/cpp`**: Writing checkpoints and restarting from them (requires Axom; see `doc/restart.md`)
- **`exa_sidre_parmesh.hpp/cpp`**: Writing and rebuilding the parallel mesh for checkpoints

**Manages**:
- Finite element spaces and mesh data
- Solution vectors (displacement, velocity)
- Material properties and state variables
- Time-stepping information

#### `solvers/`
**Purpose**: Linear and nonlinear solver implementations
- **`mechanics_solver.hpp/cpp`**: Newton-Raphson solver and variants

**Features**:
- Standard Newton-Raphson
- Newton with line search
- Adaptive step size control
- Device-aware implementations

#### `umat_tests/`
**Purpose**: Example UMAT implementations for testing
- **`umat.f`**: Example Fortran UMAT implementation
- **`umat.cxx`**: Example C++ UMAT implementation
- **`userumat.cxx`**: UMAT loader example
- **`userumat.h`**: UMAT interface definitions

#### `utilities/`
**Purpose**: Helper functions and utility classes
- **`mechanics_log.hpp`**: Logging and performance monitoring
- **`unified_logger.hpp/cpp`**: Unified logging system for all components
- **`mechanics_kernels.hpp/cpp`**: Computational kernels for mechanics operations
- **`assembly_ops.hpp`**: Assembly operation utilities
- **`rotations.hpp`**: Rotation and orientation utilities
- **`strain_measures.hpp`**: Strain computation utilities
- **`dynamic_umat_loader.hpp/cpp`**: Runtime UMAT library loading

**Provides**:
- Performance profiling integration
- Mathematical operations for mechanics
- Debugging and diagnostic tools
- Dynamic loading of user material models

#### `mfem_expt/`
**Purpose**: MFEM extensions and experimental features
- **`partial_qspace.hpp/cpp`**: Partial quadrature space implementations
- **`partial_qfunc.hpp/cpp`**: Partial quadrature function utilities

**Features**:
- Experimental finite element enhancements
- Performance optimizations for specific use cases
- Research and development components

### Organization Principles
- **Modular Design**: Clear separation between components
- **Header/Implementation Pairs**: Consistent `.hpp/.cpp` organization
- **Device Portability**: GPU-aware implementations throughout
- **Template Usage**: Modern C++17 templates for performance
- **Namespace Structure**: `exaconstit::` for internal components

## Key Components

### SystemDriver Class
The `SystemDriver` class orchestrates the entire simulation workflow, managing the Newton-Raphson solution process and coordinating between components.

**Responsibilities**:
- Newton-Raphson nonlinear solution management
- Linear solver and preconditioner setup
- Boundary condition enforcement and updates
- Material model coordination
- Solution advancement

**Key Methods**:
```cpp
void SystemDriver::Solve();           // Main Newton-Raphson solution
void SystemDriver::SolveInit();       // Initial corrector step for BC changes
void SystemDriver::UpdateEssBdr();    // Update essential boundary conditions
void SystemDriver::UpdateVelocity();  // Apply velocity boundary conditions
void SystemDriver::UpdateModel();     // Update material models after convergence
```

### NonlinearMechOperator Class
The finite element operator extending MFEM's NonlinearForm that provides:
- Residual evaluation for Newton-Raphson iterations
- Jacobian computation and assembly
- Essential DOF management
- Support for different assembly strategies (PA/EA/FULL)

### SimulationState Class
Central container managing all simulation data and providing unified access to:
- Mesh and finite element spaces
- Solution fields (velocity, displacement)
- Material properties and state variables
- Quadrature functions for field data
- Time-stepping information

### Material Model Interface
Base class `ExaModel` defines the constitutive model interface:
```cpp
// Main execution method for material model computations
virtual void ModelSetup(nqpts, nelems, space_dim, nnodes, 
                       jacobian, loc_grad, vel) = 0;

// Update state variables after converged solution
virtual void UpdateModelVars() = 0;

// Get material properties for this region
const std::vector<double>& GetMaterialProperties() const;
```

### MultiExaModel Class
Manages multiple material regions within a single simulation:
- Coordinates material model execution across regions
- Routes region-specific data from SimulationState
- Handles heterogeneous material configurations

### PostProcessingDriver Class
Manages all output and post-processing operations:
- Volume averaging calculations (stress, strain, etc.)
- Field projections for visualization
- File output management
- Support for VisIt, ParaView, and ADIOS2

### BCManager Class
Singleton pattern manager for boundary conditions:
- Tracks time-dependent boundary condition changes
- Manages multiple BC types (velocity, velocity gradient)
- Provides BC data to SystemDriver and operators

## Configuration System

ExaConstit uses TOML-based configuration files for all simulation parameters:

### Main Configuration File (`options.toml`)
```toml
basename = "simulation_name"
version = "0.10.0"

[Mesh]
filename = "mesh.mesh"
refine_serial = 0

[Time.Fixed]
dt = 1.0e-3
t_final = 1.0
[Solvers]
  assembly = "ea"
  [Solvers.Krylov]
    rel_tol = 1.0e-12
    abs_tol = 1.0e-30
    linear_solver = "CG"

[Materials]
# Material definitions...

[BCs]
# Boundary condition specifications...
```

### Modular Configuration
- **External material files**: `materials = ["material1.toml", "material2.toml"]`
- **Grain data files**: `grain_file = "grain.txt"`, `orientation_file = "orientations.txt"`

## Advanced Solver Configuration

### **Assembly Methods**
ExaConstit supports multiple finite element assembly strategies optimized for different hardware:

#### **Partial Assembly (PA)**
```toml
[Solvers]
assembly = "PA"
```
- **Memory efficient**: No global matrix formation
- **GPU optimized**: Ideal for GPU acceleration only for very high p-refinement
- **Matrix-free**: Jacobian actions computed on-the-fly
- **Preconditioning**: Currently limited to Jacobi preconditioning

#### **Element Assembly (EA)**
```toml
[Solvers]
assembly = "EA"
```
- **Element-level**: Only element matrices formed
- **Memory balanced**: minimal memory requirements for quadratic or fewer elements
- **GPU compatible**: Supports GPU execution
- **Flexibility**: Suitable for complex material models
- **Preconditioning**: Currently limited to Jacobi preconditioning


#### **Full Assembly**
```toml
[Solvers]
assembly = "FULL"
```
- **Traditional**: Complete global matrix assembly
- **Preconditioning**: Full preconditioner options available
- **Memory intensive**: Requires moderate memory for large problems due to sparse matrix formats
- **CPU optimized**: Best initial set-up for investigating new material models

### **Integration Schemes**

#### **Default Integration**
```toml
[Solvers]
integ_model = "DEFAULT"
```
- **Full integration**: Complete quadrature point evaluation
- **Standard**: Traditional finite element approach
- **Most materials**: Suitable for general material models

#### **B-Bar Integration**
```toml
[Solvers]
integ_model = "BBAR"
```
- **Mixed formulation**: Deviatoric fully integrated and elemental averaged volume contribution 
- **Near-incompressible**: Prevents volumetric locking
- **Advanced**: Based on Hughes-Brezzi formulation (Equation 23)
- **Limitation**: Not compatible with partial assembly

### **Linear Solver Options**

#### **Krylov Methods**
```toml
[Solvers.Krylov]
linear_solver = "GMRES"    # or "cg", "minres"
rel_tol = 1.0e-6
abs_tol = 1.0e-10
max_iter = 1000
```

**GMRES**: General minimal residual
- **Nonsymmetric systems**: Handles general Jacobian matrices
- **Memory**: Requires restart for memory management
- **Robust**: Suitable for challenging material models

**Conjugate Gradient (CG)**: 
- **Symmetric positive definite**: Requires symmetric Jacobian
- **Memory efficient**: Minimal memory requirements
- **Fast convergence**: Optimal for appropriate problems

**MINRES**: Minimal residual for symmetric indefinite
- **Symmetric indefinite**: Handles saddle point problems
- **Specialized**: Useful for constrained problems

#### **Preconditioner Options**
```toml
[Solvers.Krylov]
preconditioner = "JACOBI"   # Assembly-dependent options
```

**Assembly-Dependent Availability:**

For **PA/EA Assembly** (limited to):
- **JACOBI**: Diagonal scaling, GPU-compatible (automatic selection)

For **FULL Assembly** (all options available):
- **JACOBI**: Diagonal scaling
- **AMG**: Algebraic multigrid
- **ILU**: Incomplete LU factorization  
- **L1GS**: ℓ¹-scaled Gauss-Seidel
- **CHEBYSHEV**: Polynomial smoother

**Preconditioner Details:**

**JACOBI** (Diagonal Scaling):
- **Characteristics**: Simple and fast, works everywhere but slow convergence
- **Assembly**: Works with PA, EA, and FULL
- **GPU**: Fully GPU-compatible
- **Use case**: Default for PA/EA assembly, baseline option

**AMG** (Algebraic Multigrid):
- **Characteristics**: Fewer iterations but expensive setup, can fail on some problems
- **Implementation**: HYPRE BoomerAMG
- **Configuration**: Pre-tuned for 3D elasticity
- **Use case**: Large-scale problems with single materials

**ILU** (Incomplete LU Factorization):
- **Characteristics**: Good middle-ground option
- **Implementation**: HYPRE Euclid
- **Use case**: Particularly useful for multi-material systems
- **Try this**: If JACOBI convergence is too slow

**L1GS** (ℓ¹-Scaled Gauss-Seidel):
- **Characteristics**: Advanced smoother
- **Implementation**: HYPRE smoother
- **Use case**: Multi-material systems with contrasting properties
- **Try this**: When materials have very different stiffness values

**CHEBYSHEV** (Chebyshev Polynomial):
- **Characteristics**: Polynomial smoother
- **Implementation**: HYPRE smoother
- **Use case**: Problems with multiple material scales
- **Try this**: For heterogeneous material distributions

**Practical Selection Guidelines:**

For **Single Material Problems**:
- Start with JACOBI (simple, predictable)
- Try AMG if convergence is slow
- Use ILU as a reliable alternative

For **Multi-Material Systems**:
- Start with ILU (good middle-ground)
- Try L1GS for contrasting material properties
- Use CHEBYSHEV for multiple material scales
- AMG may struggle with material interfaces

**Performance Tips**:
- PA/EA assembly automatically uses JACOBI
- If JACOBI convergence is too slow with FULL assembly, try ILU → L1GS → CHEBYSHEV
- AMG has high setup cost but fewer iterations
- Multi-material systems often benefit from experimenting with different preconditioners

### **Nonlinear Solver Configuration**

#### **Newton-Raphson Variants**
```toml
[Solvers.NR]
nonlinear_solver = "NR"           # or "NRLS"
rel_tol = 1.0e-5
abs_tol = 1.0e-10
max_iter = 25
```

**Standard Newton-Raphson**:
- **Full steps**: Always takes complete Newton step
- **Fast convergence**: Quadratic convergence near solution
- **Robustness**: May fail for poor initial guesses

**Newton with Line Search**:
- **Globalization**: Backtracking line search for robustness
- **Convergence**: Improved convergence from poor starting points
- **Cost**: Additional function evaluations per iteration

## Building and Testing

### Build Configuration Options
```bash
# Enable GPU support
-DENABLE_CUDA=ON
-DENABLE_HIP=ON

# Enable specific features
-DENABLE_CALIPER=ON
```

### Running Tests
```bash
# Run example simulations
cd test/data
mpirun -np 4 ../../build/mechanics -opt example.toml
```

### Long Tests
Some tests take too long for CI and are only registered when configured with
`-DEXACONSTIT_ENABLE_LONG_TESTS=ON`. They currently cover the checkpoint/restart code on real
simulations; anyone changing that code must run them locally and make sure they pass:
```bash
ctest -L long --output-on-failure
```
GPU builds also register a GPU version of them (`test_restart_runs_gpu`; `ctest -L gpu` runs
only that one). No test uses more MPI ranks than `EXACONSTIT_TEST_MAX_MPI_TASKS`, which defaults
to the number of physical cores detected at configure time (Open MPI's default number of slots).

### Example Workflows
The `test/data/` directory contains various example cases:
- **Crystal plasticity simulations**
- **Multi-material problems**
- **Complex boundary condition examples**
- **GPU acceleration tests**

## Development Workflow

### Code Organization Best Practices
1. **Header-only utilities**: Place in `utilities/` directory
2. **New material models**: Extend `ExaModel` base class in `models/`
3. **Post-processing features**: Add to `postprocessing/` directory
4. **Configuration options**: Update corresponding `option_*.cpp` files

### Adding New Features

#### New Material Model
1. Create header/source in `models/mechanics_newmodel.hpp/cpp`
2. Inherit from `ExaModel` base class
3. Implement required virtual methods
4. Register every per-point quantity that carries history between steps with `SimulationState::AddQuadratureFunction()` (see [Keeping New State Restartable](#keeping-new-state-restartable))
5. Add configuration parsing support
6. Update `CMakeLists.txt`

#### New Boundary Condition Type
1. Extend `BCManager` class
2. Add parsing support in `option_boundary_conditions.cpp`
3. Update documentation and examples

#### Keeping New State Restartable
Checkpoints (`src/sim_state/restart_checkpoint.*`, described in `doc/restart.md`) save some
state automatically and some only because it is listed explicitly. Any new feature that
carries state from one step to the next must fit one of the cases below, or a restarted run
will silently continue from the wrong state.

**Quadrature point data is saved automatically if it is registered.** Every function in
SimulationState's quadrature-function registry is written and restored, whichever component
registered it; the only exception is `tangent_stiffness`, which is recomputed before use.
- Create history data with `SimulationState::AddQuadratureFunction()`. Data kept in a model's
  own `mfem::Vector` or `PartialQuadratureFunction` is not saved, and nothing warns about it.
  Scratch data recomputed every step can stay private.
- Register it while the objects are being constructed (SimulationState, SystemDriver, the
  models, PostProcessingDriver). The driver restores the checkpoint after all of them exist;
  anything registered later keeps its initial value.
- Register beginning/end-of-step pairs with `AddUpdateVariablePairNames()` as usual; the
  checkpoint then stores an identical end-of-step function as a reference to its partner.
- Data computed once in a constructor is fine if it depends only on the reference
  configuration, the mesh and the options, since a restarted run reproduces those exactly.

**Nodal data is not automatic.** Only the reference and current coordinates and the velocity
are saved; the other SimulationState grid functions are derived from them after a restart. A
new grid function or solution vector that carries history (an additional primal variable,
Lagrange multipliers, and so on) must be added explicitly to `WriteNodalFields()` and to the
binding in `BindFieldsAndQuadratureFunctions()`, with a check that the rebuilt space has the
same size. Store true DOFs where possible.

**Only the volume mesh is saved.** Submeshes, surface meshes and other topologies built on top
of it must be rebuilt in the restarted run from the restored mesh, which is identical to the
original, local numbering included. Anything that cannot be rebuilt that way needs its own
checkpoint data.

**Test it.** `test/test_restart_e2e.cpp` compares a restarted SimulationState with the
original bitwise; add any new state to it, and run the [long tests](#long-tests) for anything
that changes what a simulation evolves.

### Performance Considerations
- **GPU kernels**: Use RAJA for performance portability
- **Memory management**: Follow MFEM memory patterns
- **MPI communication**: Minimize collective operations
- **Assembly strategy**: Choose PA vs EA based on problem size

### Debugging and Profiling
- **Caliper integration**: Built-in performance profiling
- **MFEM debugging**: Use MFEM's debugging capabilities
- **GPU debugging**: CUDA/HIP debugging tools
- **MPI debugging**: TotalView, DDT support

## UMAT Development Resources

### **Interface Requirements**
While UMAT interfaces are traditionally described using Fortran signatures, ExaConstit supports implementation in **Fortran, C++, or C**:

#### **Standard UMAT Signature** (Fortran style)
```fortran
SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD,
 1 RPL,DDSDDT,DRPLDE,DRPLDT,
 2 STRAN,DSTRAN,TIME,DTIME,TEMP,DTEMP,PREDEF,DPRED,CMNAME,
 3 NDI,NSHR,NTENS,NSTATV,PROPS,NPROPS,COORDS,DROT,PNEWDT,
 4 CELENT,DFGRD0,DFGRD1,NOEL,NPT,LAYER,KSPT,KSTEP,KINC)
```

#### **C++ Implementation Example**
```cpp
extern "C" void umat_(double* stress, double* statev, double* ddsdde,
                     double* sse, double* spd, double* scd,
                     // ... additional parameters
                     int* ndi, int* nshr, int* ntens, int* nstatv,
                     double* props, int* nprops,
                     // ... remaining parameters
                     );
```

### **UMAT Development Best Practices**

#### **Memory Management**
- **ExaConstit handles**: State variable allocation and persistence
- **UMAT responsible**: Local variable management within subroutine
- **No dynamic allocation**: Avoid malloc/new within UMAT calls

#### **Thread Safety**
- **No global variables**: UMATs must be thread-safe
- **Local computations**: All calculations using passed parameters
- **State persistence**: Only through provided state variable arrays

#### **Error Handling**
- **Convergence issues**: Set appropriate flags for Newton-Raphson
- **Material failure**: Handle through state variables or stress reduction
- **Numerical stability**: Check for divide-by-zero and overflow conditions

#### **Performance Considerations**
- **CPU execution only**: No current GPU acceleration for UMATs but might be possible in future updates
- **Vectorization**: Ensure compiler optimization is possible
- **Minimal function calls**: Reduce computational overhead within UMAT

### **Development Resources**

#### **Reference Implementations**
- **`src/umat_tests/`**: Example UMAT implementations and conversion guides
- **Template UMATs**: Starting points for custom development

#### **External Resources**
- **NJIT UMAT Collection**: https://web.njit.edu/~sac3/Software.html
- **Academic examples**: Various constitutive models available
- **License considerations**: Verify licensing before use

#### **Build System Integration**
```bash
# Compile UMAT to shared library (Fortran)
gfortran -shared -fPIC -o my_umat.so my_umat.f90

# Compile UMAT (C++)
g++ -shared -fPIC -o my_umat.so my_umat.cpp

# Compile UMAT (C)
gcc -shared -fPIC -o my_umat.so my_umat.c
```

#### **Configuration Integration**
```toml
[Materials.regions.model.UMAT]
library_path = "/path/to/my_umat.so"
num_props = 8
num_state_vars = 12
props = [
    210000.0,  # Young's modulus
    0.3,       # Poisson's ratio
    # ... additional parameters
]
```

## Contributing Guidelines

### Code Standards
- **C++17 compliance**: Use modern C++ features
- **Documentation**: Doxygen-style comments for all public interfaces
- **Testing**: Include test cases for new features
- **Performance**: Maintain GPU and MPI scalability
- **Name Formating**:
   - Function names should be in `PascalCase` for any file but those related to IO (src/options/* and src/utilities/unified_loggers.*) which are `snake_case`.
   - Class / enum names should be in `PascalCase`
   - Enum values should be `UPPER_CASE`
   - Class member variables going forward should be `snake_case` and preferably have a `m_` prefix. However, the `m_` prefix is **not** required if it makes things harder to understand. We're still converting variables over from previous in-consistent naming conventions so if you spot something that needs fixing please do so.
   - Local / function variables going forward should be `snake_case`. Like above we are slowly in the process of converting old code over to this new format so feel free to help out if you can.
   - If doing formatting changes split those into their own commits so it's easier to track changes. Additionally try to change the world all at once and do things in piece meal as it makes it easier to track down where a bug might have been introduced during renaming of things.
- **Code Formating**: We have a `.clang-format` that we make use to enfore a unified coding experience across the code base. An example of how to run the formatter is: `find src -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) ! -path "*/TOML_Reader/*" -exec $CLANG_FORMAT -i {} +` . Note, if you see any changes in the `src/TOML_Reader` directory to revert those changes as that is a TPL that we directly include in the repo and not something we want to update unless directly bringing in the changes from its upstream repo.

### Pull Request Process
1. Fork the repository (if non-LLNL employee)
2. Create feature branch from `exaconstit-dev`
3. Implement changes with tests
4. Ensure all existing tests pass
5. Submit pull request with detailed description

### Licensing
- **BSD-3-Clause license**: All contributions must use this license
- **Third-party code**: Ensure compatible licensing for external dependencies

### Getting Help
- **Primary Developer**: Robert A. Carson (carson16@llnl.gov)
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Refer to MFEM and ExaCMech documentation for underlying libraries

## Additional Resources

### Related Projects
- **ExaCMech**: Crystal plasticity library (https://github.com/LLNL/ExaCMech)
- **MFEM**: Finite element library (https://mfem.org)
- **ExaCA**: Cellular automata for microstructure generation

### Workflows and Applications
- **Optimization workflows**: Multi-objective genetic algorithm parameter optimization
- **UQ workflows**: Uncertainty quantification for additive manufacturing
- **Post-processing tools**: Python scripts for data analysis

### Citation
If using ExaConstit in your research, please cite:
```bibtex
@misc{ exaconstit,
title = {{ExaConstit}},
author = {Carson, Robert A. and Wopschall, Steven R. and Bramwell, Jamie A.},
abstractNote = {The principal purpose of this code is to determine bulk constitutive properties and response of polycrystalline materials. This is a nonlinear quasi-static, implicit solid mechanics code built on the MFEM library based on an updated Lagrangian formulation (velocity based). Within this context, there is flexibility in the type of constitutive model employed, with the code allowing for various UMATs to be interfaced within the code framework or for the use of the ExaCMech library. Using crystal-mechanics-based constitutive models, the code can be used, for example, to compute homogenized response behavior over a polycrystal. },
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20191024.2}},
url = {https://github.com/LLNL/ExaConstit},
doi = {10.11578/dc.20191024.2},
year = {2019},
month = {Aug},
annote = {
   https://www.osti.gov//servlets/purl/1571640
   https://www.osti.gov/biblio/1571640-exaconstit
}
}
```

---

This guide provides a foundation for new developers to understand and contribute to ExaConstit. For specific implementation details, refer to the extensive inline documentation throughout the codebase and the example configurations in `test/data/`.