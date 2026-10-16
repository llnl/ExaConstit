<div align="center">

```
███████╗██╗  ██╗ █████╗  ██████╗ ██████╗ ███╗   ██╗███████╗████████╗██╗████████╗
██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██║╚══██╔══╝
█████╗   ╚███╔╝ ███████║██║     ██║   ██║██╔██╗ ██║███████╗   ██║   ██║   ██║   
██╔══╝   ██╔██╗ ██╔══██║██║     ██║   ██║██║╚██╗██║╚════██║   ██║   ██║   ██║   
███████╗██╔╝ ██╗██║  ██║╚██████╗╚██████╔╝██║ ╚████║███████║   ██║   ██║   ██║   
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝   ╚═╝   
```

**High-Performance Crystal Plasticity & Micromechanics Simulation**

*Velocity-based finite element framework for polycrystalline materials*

[Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Contributing](#contributing)

</div>

---

## What is ExaConstit?

ExaConstit is a cutting-edge, **velocity-based finite element code** designed for high-fidelity simulation of polycrystalline materials. Built on LLNL's MFEM library, it delivers unprecedented performance for crystal plasticity and micromechanics modeling on leadership-class HPC systems.

### Key Applications
- **Crystal Plasticity Simulations** - Grain-level deformation analysis
- **Bulk Constitutive Properties** - Homogenization of polycrystalline materials  
- **Additive Manufacturing** - Process-structure-property relationships
- **Experimental Validation** - Lattice strain calculations for diffraction experiments

## Features

### **Advanced Finite Element Framework**
- **Velocity-Based Formulation** - Updated Lagrangian with superior convergence
- **Multi-Material Support** - Heterogeneous material regions
- **Adaptive Time Stepping** - Automatic timestep control for robustness

### **Crystal Plasticity Modeling**
- **ExaCMech Integration** - Advanced crystal plasticity constitutive models
- **Multi-Crystal Support** - BCC, FCC, and HCP crystal structures  
- **Grain-Level Resolution** - Individual grain orientations and properties
- **State Variable Evolution** - Full history-dependent material behavior

### **High-Performance Computing**
- **GPU Acceleration** - CUDA and HIP support for maximum performance
- **MPI Parallelization** - Scales to tens of thousands of processors
- **Memory Efficiency** - Matrix-free partial assembly algorithms
- **Performance Portability** - RAJA framework for unified CPU/GPU code

### **Material Model Flexibility**
- **ExaCMech Library** - State-of-the-art crystal plasticity models
- **UMAT Interface** - Abaqus-compatible user material subroutines
- **Custom Models** - Extensible architecture for new constitutive laws
- **Multi-Model Regions** - Different materials in different regions

### **Advanced Post-Processing**
- **Visualization Output** - VisIt, ParaView, and ADIOS2 support
- **Volume Averaging** - Macroscopic stress-strain behavior and other useful parameters
- **Lattice Strain Analysis** - In-situ diffraction experiment simulation
- **Python Tools** - Comprehensive analysis and plotting scripts

## Quick Start

### Prerequisites
```bash
# Essential dependencies
MPI implementation (OpenMPI, MPICH, Intel MPI)
MFEM (v4.9+, ExaConstit fork) with parallel/GPU support
ExaCMech (v0.4.3+) crystal plasticity library
RAJA (≥2025.12.x) performance portability
CMake (3.24+)
```

### Installation

ExaConstit provides automated installation scripts for different platforms. For detailed instructions, see [Installation Guide](doc/install.md).

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/LLNL/ExaConstit.git

# Create a separate build directory (recommended)
# This keeps source and build artifacts separate
mkdir -p exaconstit_builds
cd exaconstit_builds
```

#### **Intel CPU Systems (Linux)**
```bash
../ExaConstit/scripts/install/unix_cpu_intel_install.sh
```

#### **macOS Systems**
```bash
../ExaConstit/scripts/install/unix_cpu_mac_install.sh
```

#### **NVIDIA GPU Systems (CUDA)**
```bash
../ExaConstit/scripts/install/unix_gpu_cuda_install.sh
```

#### **AMD GPU Systems (HIP/ROCm)**
```bash
../ExaConstit/scripts/install/unix_gpu_hip_install.sh
```

#### Before First Run

⚠️ **You must customize the build configuration for your system.**

Edit the appropriate config file in `scripts/install/configs/` and update:
- Compiler paths and versions
- MPI installation location
- Python executable path
- Module load commands (HPC systems)

See the [Installation Guide](docs/install.md) for detailed setup instructions.

#### Build Options
```bash
# Clean rebuild
REBUILD=ON ./scripts/install/unix_gpu_cuda_install.sh

# Target specific GPU architecture
CMAKE_GPU_ARCHITECTURES=80 ./scripts/install/unix_gpu_cuda_install.sh

# Adjust parallel jobs
MAKE_JOBS=16 ./scripts/install/unix_cpu_intel_install.sh
```

**Note for MI300A users:** Set `HSA_XNACK=1` before running simulations.

For troubleshooting, manual builds, and advanced configuration, see the [Installation Guide](docs/install.md).

#### **Manual Build**
```bash
# Clone and prepare
git clone https://github.com/LLNL/blt.git cmake/blt
mkdir build && cd build

# Configure
cmake .. \
  -DENABLE_MPI=ON \
  -DMFEM_DIR=${MFEM_INSTALL_DIR} \
  -DECMECH_DIR=${EXACMECH_INSTALL_DIR} \
  -DRAJA_DIR=${RAJA_INSTALL_DIR}

# Build
make -j $(nproc)
```

### First Simulation
```bash
# Run a crystal plasticity example
cd test/data
mpirun -np 4 ../../build/mechanics -opt voce_full.toml

# Generate stress-strain plots
python ../../scripts/postprocessing/macro_stress_strain_plot.py
```

## Examples

### **Crystal Plasticity Simulation**
```toml
# options.toml - Crystal plasticity configuration
grain_file = "grain.txt"
orientation_file = "orientations.txt"

[Mesh]
filename = "polycrystal.mesh"

[Materials]
[[Materials.regions]]
material_name = "titanium_alloy"
mech_type = "ExaCMech"

[Materials.regions.model.ExaCMech]
shortcut = "evptn_HCP_A"

```

### **Post-Processing Workflow**
```bash
# Extract stress-strain data
python scripts/postprocessing/macro_stress_strain_plot.py output/

# Calculate lattice strains (experimental validation)
python scripts/postprocessing/calc_lattice_strain.py \
  --config lattice_strain_config.json

# Generate visualization files
python scripts/postprocessing/adios2_example.py results.bp
```

## Output and Visualization

### **Version 0.10 Updates**
ExaConstit v0.10 focuses on solver robustness and moves to MFEM v4.9+. Highlights:
- **Trust-region dogleg solver**: `nl_solver = "TRDOG"` for problems where Newton-Raphson struggles to converge
- **Solver output control**: the new `print_info` option in `[Solvers.NR]` sets how much the nonlinear solvers print, using one output format for all solvers
- **B-bar consistency**: material models now see the same volume-averaged kinematics as the B-bar integrator
- **Light-up post-processing**: `xtal_light_up` supports all eight crystal systems
- **Calibration workflows**: a new framework in `workflows/exaconstit-calibrate` with MPI and Flux backends
- **Bug fixes**: multi-GPU results, symlinked output directories, time stepping options, and post-processing output cadence

See [CHANGELOG.md](CHANGELOG.md) for the full list of changes.

### **Version 0.9 Output Updates**
ExaConstit v0.9 introduces significant improvements to output management and file organization:

#### **Modern Configuration Support**
- **Legacy compatibility**: Previous option file formats continue to work
- **Conversion utility**: Use our conversion script to migrate to the modern TOML format:
  ```bash
  python scripts/exaconstit_old2new_options.py old_options.toml -o new_options.toml
  ```

#### **Enhanced Output Files** 
- **Headers included**: All simulation output files now contain descriptive headers
- **Time and volume data**: Automatically included in all output files so the auto_dt_file has been removed
- **Improved format**: Enhanced data organization (note: format differs from previous versions)
- **Basename-based directories**: Output location determined by `basename` and `Postprocessing.Projections.output_directory` settings in options file
  ```toml
  # if not provided defaults to option file name
  basename = "exaconstit"  # Creates output sub-directory: exaconstit/
  ```

#### **Advanced Visualization Control**
- **Backward compatibility**: Visualization files remain compatible with previous versions
- **User-friendly naming**: Visualization variable names updated for better clarity  
- **Selective field output**: Specify exactly which fields to save (new capability):
  ```toml
    [PostProcessing.projections]
        # Some of these values are only compatible with ExaCMech
        enabled_projections = ["stress", "von_mises", "volume", "centroid", "dpeff", "elastic_strain"]
        # if set to true then all defaults are outputted by default
        auto_enable_compatible = false
  ```

### **Migration Guide for Existing Users**
- **Existing simulations**: Previous option files work without modification
- **Output processing**: Update post-processing scripts to handle new file headers
- **Directory structure**: Account for new basename-based output organization
- **Visualization workflows**: Existing VisIt/ParaView workflows remain functional

## Advanced Features

### **Mesh Generation & Processing**
- **Auto-Generated Meshes** - From grain ID files
- **Neper Integration** - v4 mesh processing with boundary detection
- **Format Conversion** - VTK to MFEM
- **Boundary Attribute** - Automatic boundary labelling

#### **Mesh Generator Utility**
The `mesh_generator` executable provides flexible mesh creation and conversion:
```bash
# Create MFEM mesh from grain ID file
./mesh_generator --grain_file grains.txt --output polycrystal.mesh

# Convert VTK mesh to MFEM format with boundary attributes
./mesh_generator --vtk_input mesh.vtk --output converted.mesh

# View all options
./mesh_generator --help
```

**Capabilities**:
- **Auto-generated meshes** from grain ID files
- **VTK to MFEM conversion** with automatic boundary attribute generation
- **Boundary Attribute** compatible with ExaConstit requirements

#### **Neper Integration**
**For Neper v4 users**:
```bash
# Generate mesh with face information
neper -M n100-id1.tess -faset 'faces' -format gmsh2.2

# Convert to ExaConstit format
python scripts/meshing/neper_v4_mesh.py input.msh output.mesh
```

**For Neper v2-v3 users**:
```bash
# Convert FEpX format to VTK
python scripts/meshing/fepx2mfem_mesh.py fepx_mesh.txt vtk_mesh.vtk

# Then use mesh_generator for final conversion
./mesh_generator --vtk_input vtk_mesh.vtk --output final.mesh
```

#### **Required Input Files for Crystal Plasticity**
When setting up crystal plasticity simulations, you need (file names can be different):

##### **Essential Files**
- **`grain.txt`**: Element-to-grain ID mapping (one ID per element)
- **`props.txt`**: Material parameters for each grain type/material
- **`state.txt`**: Initial internal state variables (typically zeros)
- **`orientations.txt`**: Crystal orientations (quaternions)
- **`regions.txt`**: Mapping from grain-to-region ID mapping


##### **Mesh Requirements**
- **Format**: MFEM v1.0 or Cubit format
- **Grain IDs**: Must be assigned to element attributes in the mesh
- **Boundary attributes**: Required for boundary condition application

### **Experimental Integration**
- **Lattice Strain Calculations** - Powder diffraction simulation
- **In-Situ Analysis** - Real-time lattice strain monitoring  
- **Microstructure Coupling** - Integration with ExaCA and other tools

#### **Stress-Strain Analysis**
```bash
# Generate macroscopic stress-strain plots
python scripts/postprocessing/macro_stress_strain_plot.py
```

#### **Lattice Strain Analysis**
Simulate powder diffraction experiments with in-situ lattice strain calculations:
```bash
# Extract lattice strain data from ADIOS2 files
python scripts/postprocessing/adios2_extraction.py

# Transform crystal strains to sample coordinates
python scripts/postprocessing/strain_Xtal_to_Sample.py

# Calculate lattice strains for specific HKL directions
python scripts/postprocessing/calc_lattice_strain.py
```

**Enable lattice strain output** in your simulation:
```toml
[Visualizations]
light_up = true  # Enables in-situ lattice strain calculations

# Configure specific HKL directions and parameters in options.toml
```

##### **ADIOS2 Integration**
For large-scale data analysis (recommended for extensive post-processing):
```bash
# Example ADIOS2 data processing
python scripts/postprocessing/adios2_example.py

# Requires MFEM built with ADIOS2 support
```

### **Materials Science Workflows**

#### **Parameter Optimization**
Multi-objective genetic algorithm-based optimization for material parameter identification:
```bash
# Optimize material parameters against experimental data
cd workflows/optimization/
python ExaConstit_NSGA3.py
```

**Features**:
- **Flux integration**: Leverage LLNL's Flux job manager for HPC systems
- **Workstation support**: Simple workflow manager for desktop systems
- **Multi-objective optimization**: Fit multiple experimental datasets simultaneously

#### **Uncertainty Quantification (UQ)**
ExaAM integration for additive manufacturing applications:
```bash
# UQ workflow for process-structure-property relationships
cd workflows/Stage3/pre_main_post_script
python chal_prob_full.py
```

**Applications**:
- **Microstructure-property linkage**: Connect ExaCA microstructures to mechanical properties
- **Part-scale modeling**: Generate data for macroscopic material model parameterization
- **Process optimization**: Optimize additive manufacturing parameters
- **Anisotropic yield surface**: Development from polycrystal simulations

**Academic Reference**: [ExaAM UQ Workflow Paper](https://doi.org/10.1145/3624062.3624103)

## Documentation

### **Getting Started**
- [Developer's Guide](developers_guide.md) - Complete development documentation
- [Configuration Reference](src/options.toml) - All available simulation options

### **Scientific Background**
- **Crystal Plasticity Theory** - Micromechanics fundamentals
- **Finite Element Implementation** - Velocity-based formulation details
- **GPU Acceleration** - Performance optimization strategies

### **Tutorials & Examples**
- **Basic Simulations** - Simple deformation tests
- **Complex Loading** - Cyclic and multiaxial loading
- **Multi-Material Problems** - Composite and layered materials
- **Experimental Validation** - Lattice strain analysis

## Ecosystem & Integration

### **Related LLNL Projects**
- **[ExaCMech](https://github.com/LLNL/ExaCMech)** - Crystal plasticity constitutive models
- **[ExaCA](https://github.com/LLNL/ExaCA)** - Cellular automata code for alloy nucleation and solidification
- **[MFEM](https://mfem.org)** - Finite element methods library
- **ExaAM** - Exascale Computing Project project on additive manufacturing for process-structure-properties calculations

### **Third-Party Tools**
- **Neper** - Polycrystal mesh generation
- **VisIt/ParaView** - Visualization and analysis
- **ADIOS2** - High-performance I/O
- **Python Ecosystem** - NumPy, SciPy, Matplotlib integration

## Performance & Scalability

### **Benchmarks**
- **CPU Performance** - Scales to 1000+ MPI processes
- **GPU Acceleration** - 15-25x speedup on V100 or MI250x/MI300a systems
- **Memory Efficiency** - Matrix-free algorithms reduce memory footprint
- **I/O Performance** - ADIOS2 integration for petascale data management

### **Optimization Features**
- **Partial Assembly** - Matrix-free operator evaluation
- **Device Memory Management** - Automatic host/device transfers
- **Communication Optimization** - Minimal MPI collective operations

## Contributing

We welcome contributions from the materials science and computational mechanics communities!

### **Development**
```bash
# Fork the repository and create a feature branch
git checkout -b feature/amazing-new-capability

# Make your changes with comprehensive tests
# Follow our C++17 coding standards

# Submit a pull request with detailed description
```

### **Contribution Areas**
- **Material Models** - New constitutive relationships
- **Boundary Conditions** - Extended loading capabilities such as Neumann BCs or periodic BCs 
- **Post-Processing** - Analysis and visualization tools
- **Performance** - GPU optimization and scalability
- **Documentation** - Tutorials and examples

### **Getting Help**
- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Technical questions and community support
- **Documentation** - Comprehensive guides and API reference

## License & Citation

ExaConstit is distributed under the **BSD-3-Clause license**. All contributions must be made under this license.

### **Citation**
If you use ExaConstit in your research, please cite the below. Additionally, we would love to be able to point to ExaConstit's use in the literature and elsewhere so feel free to message us with a link to your work as Google Scholar does not always pick up the below citation. We can then list your work among the others that have used our code.

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

### LICENSE

License is under the BSD-3-Clause license. See [LICENSE](LICENSE) file for details. And see also the [NOTICE](NOTICE) file. 

`SPDX-License-Identifier: BSD-3-Clause`

``LLNL-CODE-793434``

## Core Team

### **Lawrence Livermore National Laboratory**
- **Robert A. Carson** (Principal Developer) - carson16@llnl.gov
- **Nathan Barton** - Initial Development
- **Steven R. Wopschall** - Initial Development  
- **Jamie Bramwell** - Initial Development

---

<div align="center">

**Built at Lawrence Livermore National Laboratory**

*Advancing materials science through high-performance computing*

</div>
