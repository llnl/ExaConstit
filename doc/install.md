# ExaConstit Installation Guide

ExaConstit provides a modular build system with automated installation scripts for different platforms and backends. The build system automatically handles all dependencies including RAJA, MFEM, ExaCMech, Hypre, and METIS.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Build System Architecture](#build-system-architecture)
- [First-Time Setup](#first-time-setup)
- [Advanced Configuration](#advanced-configuration)
- [Build Locations and Output](#build-locations-and-output)
- [Troubleshooting](#troubleshooting)
- [Manual Build](#manual-build-advanced-users)

---

## Quick Start

### **1. Download the Repository**
```bash
# Clone the repository
git clone https://github.com/LLNL/ExaConstit.git
```

### **2. Create a Build Directory**
```bash
# Create a separate build directory (recommended)
# This keeps source and build artifacts separate
mkdir -p exaconstit_builds
cd exaconstit_builds
```

**Note:** The build scripts will compile all dependencies in your current working directory. Using a separate build directory outside the source tree is strongly recommended to keep your workspace organized.

### **3. Configure Your System**

⚠️ **Before first run, you must customize the build configuration for your system.**

Edit the appropriate config file in `ExaConstit/scripts/install/configs/` and update:
- Compiler paths and versions
- MPI installation location
- Python executable path
- Module load commands (for HPC systems)

See the [Installation Guide](docs/install.md) for detailed configuration instructions.

### **4. Choose Your Platform**

#### **Intel CPU Systems (Linux)**
```bash
../ExaConstit/scripts/install/unix_cpu_intel_install.sh
```

#### **MacOS Systems**
```bash
../ExaConstit/scripts/install/unix_cpu_mac_install.sh
```

#### **NVIDIA GPU Systems (CUDA)**
```bash
../ExaConstit/scripts/install/unix_gpu_cuda_install.sh
```

For a Linux workstation with an NVIDIA GPU, the CUDA toolkit, the system GCC and a packaged MPI (no environment modules), there is a config that needs no editing. It finds the CUDA toolkit, compilers, MPI and Python from `PATH`, and the GPU architecture from `nvidia-smi`. Any of these can be overridden from the environment, e.g. when the default GCC is newer than the CUDA release supports:
```bash
CUDA_BASE=/opt/cuda-12.9 GCC_VERSION=13 \
  ../ExaConstit/scripts/install/unix_gpu_cuda_workstation_install.sh
```

#### **AMD GPU Systems (HIP/ROCm)**
```bash
../ExaConstit/scripts/install/unix_gpu_hip_install.sh
```

**Note for MI300A users:** Set `HSA_XNACK=1` in your environment before running simulations. This is required due to unified memory requirements and current limitations in MFEM's HIP backend.

---

## Build System Architecture

The installation framework is organized into three components:
```
scripts/install/
├── common/
│   ├── dependency_versions.sh    # Centralized version control
│   ├── preflight_checks.sh       # Validation and utilities
│   └── build_functions.sh        # Shared build logic
├── configs/
│   ├── cpu_intel_config.sh       # Intel compiler configuration
│   ├── cpu_mac_config.sh         # macOS configuration
│   ├── gpu_cuda_config.sh        # NVIDIA CUDA configuration
│   ├── gpu_cuda_workstation_config.sh # NVIDIA CUDA on a Linux workstation (auto-detected)
│   └── gpu_hip_config.sh         # AMD HIP configuration
└── unix_*_install.sh              # Platform-specific entry points
```

- **common/**: Shared build logic used across all platforms
- **configs/**: Platform-specific compiler paths, flags, and settings
- **Entry scripts**: Simple launchers that source the appropriate config and common functions

---

## First-Time Setup

Before running an install script, you'll need to customize the configuration file for your system.

### **Step 1: Edit the Configuration File**

Navigate to the ExaConstit repository and open the appropriate config file in `scripts/install/configs/` with either a built in editor or something like VSCode:
```bash
cd ExaConstit

# For Intel CPU builds
code scripts/install/configs/cpu_intel_config.sh

# For CUDA GPU builds
code scripts/install/configs/gpu_cuda_config.sh

# For HIP/AMD GPU builds
code scripts/install/configs/gpu_hip_config.sh

# For macOS builds
code scripts/install/configs/cpu_mac_config.sh
```

### **Step 2: Update Compiler Paths and Versions**

Each config file has a clearly marked section at the top for system-specific paths. Update these for your environment:

#### **Intel CPU Configuration Example**
```bash
###########################################
# Compiler Versions and Base Paths
###########################################
INTEL_VERSION="2023.2.1-magic"                    # Update to your Intel version
COMPILER_VERSION="intel-${INTEL_VERSION}"
INTEL_BASE="/usr/tce/packages/intel/${COMPILER_VERSION}"  # Update to your path

MPI_IMPL="mvapich2"                               # Or openmpi, mpich, etc.
MPI_VERSION="2.3.7"                               # Update to your MPI version
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}" # update to your path

PYTHON_VERSION="3.12.2"                           # Update to your Python version
PYTHON_BASE="/usr/apps/python-${PYTHON_VERSION}"  # Update to your path
```

**How to find your paths:**
```bash
# Find your compilers
which icc      # Intel C compiler
which icpc     # Intel C++ compiler
which mpicc    # MPI C wrapper
which mpicxx   # MPI C++ wrapper
which python3  # Python executable

# Get version information
icc --version
mpicc --version
python3 --version
```

#### **CUDA GPU Configuration Example**
```bash
###########################################
# Compiler Versions and Base Paths
###########################################
# Host Compiler
CLANG_VERSION="ibm-14.0.5"                        # Update to your Clang version
COMPILER_VERSION="clang-${CLANG_VERSION}"
CLANG_BASE="/usr/tce/packages/clang/${COMPILER_VERSION}"

# CUDA
CUDA_VERSION="11.8.0"                             # Update to your CUDA version
CUDA_BASE="/usr/tce/packages/cuda/cuda-${CUDA_VERSION}" # Update to your CUDA Path

# MPI
MPI_IMPL="spectrum-mpi"                           # Update to your MPI implementation / version / path
MPI_VERSION="rolling-release"
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}"

# Python
PYTHON_VERSION="3.8.2" # Like stated earlier update to your version / path
PYTHON_BASE="/usr/tce/packages/python/python-${PYTHON_VERSION}"
```

**How to find CUDA paths:**
```bash
# Find CUDA installation
which nvcc
nvcc --version

# CUDA is typically at /usr/local/cuda or /usr/local/cuda-11.8
echo $CUDA_HOME  # May already be set
```

#### **HIP/AMD GPU Configuration Example**
```bash
###########################################
# Compiler Versions and Base Paths
###########################################
# ROCm Compiler
# Update all of the below to your own relevant versions / paths / anything specific to your
# system
ROCM_VERSION="6.4.2" 
ROCM_MAGIC_SUFFIX="magic"
COMPILER_VERSION="rocmcc-${ROCM_VERSION}-${ROCM_MAGIC_SUFFIX}"
ROCM_BASE="/usr/tce/packages/rocmcc/${COMPILER_VERSION}"

# MPI - Cray MPICH
MPI_IMPL="cray-mpich"
MPI_VERSION="9.0.1"
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}"

# Python
PYTHON_VERSION="3.9.12"
PYTHON_BASE="/usr/tce/packages/python/python-${PYTHON_VERSION}"
```

**How to find ROCm paths:**
```bash
# Find ROCm installation
which amdclang
which hipcc
hipcc --version

# ROCm is typically at /opt/rocm or /opt/rocm-6.4.2
echo $ROCM_PATH  # May already be set
```

#### **macOS Configuration Example**
```bash
###########################################
# User-Configurable Paths
###########################################
# Homebrew location
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"  # or /usr/local for Intel Macs

# System Clang (usually fine as-is)
CLANG_BASE="/usr/bin"

# MPI installation (REQUIRED: Update this!)
MPI_BASE="${HOME}/local/bin"                      # Update to your MPI location
# Common locations:
#   Homebrew: /opt/homebrew/bin
#   MacPorts: /opt/local/bin
#   Anaconda: ${HOME}/anaconda3/bin

# Python location (REQUIRED: Update this!)
PYTHON_BASE="${HOME}/anaconda3/bin"               # Update to your Python location
# Common locations:
#   Homebrew: /opt/homebrew/bin
#   System: /usr/bin
```

**How to find paths on macOS:**
```bash
# Check architecture
uname -m  # arm64 for Apple Silicon, x86_64 for Intel

# Find MPI (install if missing: brew install open-mpi)
which mpicc
which mpicxx

# Find Python
which python3
python3 --version  # Should be 3.8 or newer

# Check Homebrew prefix
brew --prefix
```

### **Step 3: Update Module Commands (HPC Systems Only)**

If you're on an HPC system with a module system, update the `module load` commands to match your system:
```bash
###########################################
# Module Loading
###########################################
module load intel/2023.2.1-magic    # Update to match your system's modules
module load CMake/3.26.3
module load python/3.12
module list
```

**How to find available modules:**
```bash
module avail          # List all available modules
module avail intel    # Search for Intel modules
module avail cuda     # Search for CUDA modules
module list           # Show currently loaded modules
```

If your system doesn't use modules (like most macOS or personal Linux systems), you can comment out or remove the module commands.

### **Step 4: Run the Install Script**

Once you've customized the config file, run the appropriate install script from your build directory:
```bash
cd ../exaconstit_builds  # Or wherever you want to build
../ExaConstit/scripts/install/unix_cpu_intel_install.sh  # Or appropriate script
```

The script will:
1. Validate your configuration
2. Display a build summary
3. Download and build all dependencies
4. Build ExaConstit
5. Save detailed logs in each component's build directory

**Expected build time:** 10 minutes to 45 minutes depending on system / parallelism / GPU builds or not.

---

## Advanced Configuration

### **Updating Dependency Versions**

All dependency versions are centralized in `common/dependency_versions.sh`:
```bash
# Edit version file
code ExaConstit/scripts/install/common/dependency_versions.sh
```
```bash
# Portability libraries
export CAMP_VER="v2025.12.0"
export RAJA_VER="v2025.12.2"
export UMPIRE_VER="v2025.12.0"
export CHAI_VER="v2025.12.0"

# Material models
export EXACMECH_REPO="https://github.com/LLNL/ExaCMech.git"
export EXACMECH_BRANCH="v0.4.3"     # Tag or branch to check out

# FEM infrastructure
export HYPRE_VER="v3.1.0"
export METIS_VER="5.1.0"

export MFEM_REPO="https://github.com/rcarson3/mfem.git"
export MFEM_BRANCH="exaconstit-dev"  # Change branch if needed

# Main application
export EXACONSTIT_REPO="https://github.com/llnl/ExaConstit.git"
export EXACONSTIT_BRANCH="exaconstit-dev"  # Change branch if needed

# Build standards
export CMAKE_CXX_STANDARD="17"
export CMAKE_BUILD_TYPE="Release"
```

After updating versions, **all** build scripts will automatically use the new versions. No other changes needed.

### **Changing GPU Architecture**

Override the default GPU architecture at runtime:
```bash
# CUDA: Target Ampere A100 instead of default Volta V100
CMAKE_GPU_ARCHITECTURES=80 ./unix_gpu_cuda_install.sh

# HIP: Target MI250X instead of default MI300A
CMAKE_GPU_ARCHITECTURES=gfx90a ./unix_gpu_hip_install.sh
```

Common GPU architectures:

**NVIDIA CUDA:**
- `60` - Pascal (P100)
- `70` - Volta (V100)
- `75` - Turing (RTX 20xx, T4)
- `80` - Ampere (A100)
- `86` - Ampere (RTX 30xx, A40)
- `89` - Ada Lovelace (RTX 40xx, L40)
- `90` - Hopper (H100)

**AMD HIP:**
- `gfx906` - MI50
- `gfx908` - MI100
- `gfx90a` - MI200 series (MI210, MI250)
- `gfx940` - MI300X (compute-only)
- `gfx942` - MI300A (APU)
- `gfx942:xnack+` - MI300A with unified memory support

### **Build Control Options**

Control the build behavior with environment variables:
```bash
# Clean rebuild (removes all build directories and rebuilds from scratch)
REBUILD=ON ./unix_gpu_hip_install.sh

# Force submodule updates (syncs and updates all git submodules)
SYNC_SUBMODULES=ON ./unix_gpu_cuda_install.sh

# Adjust parallel build jobs (default is 4)
MAKE_JOBS=16 ./unix_cpu_intel_install.sh

# Combine multiple options
REBUILD=ON MAKE_JOBS=8 CMAKE_GPU_ARCHITECTURES=80 ./unix_gpu_cuda_install.sh
```

**Available environment variables:**
- `REBUILD` - `ON` to clean and rebuild, `OFF` to reuse existing builds (default: `OFF`)
- `SYNC_SUBMODULES` - `ON` to force submodule sync, `OFF` to skip (default: `OFF`)
- `MAKE_JOBS` - Number of parallel build jobs (default: `4`)
- `CMAKE_GPU_ARCHITECTURES` - GPU architecture target (default varies by platform)
- `MFEM_HIP_ARCHITECTURES` - MFEM-specific HIP arch (HIP only, default: `gfx942`)
- `OPENMP_ON` - Enable OpenMP (default: `OFF`)
- `ENABLE_TESTS_EXACONSTIT` - Build tests (default: `ON`)

### **Using Different Repositories or Branches**

To use a fork or different branch, edit `common/dependency_versions.sh`:
```bash
# Use your fork of MFEM
export MFEM_REPO="https://github.com/YOUR_USERNAME/mfem.git"
export MFEM_BRANCH="my-custom-feature"

# Use development branch of ExaConstit
export EXACONSTIT_BRANCH="develop"

# Use a different ExaCMech repository
export EXACMECH_REPO="https://github.com/YOUR_USERNAME/ExaCMech.git"
export EXACMECH_BRANCH="custom-material-models"
```

To use a specific commit instead of a branch:
```bash
# The build scripts will clone the repo, then manually checkout the commit:
cd mfem && git checkout abc123def456
cd ../exaconstit_builds
# Re-run the build script with REBUILD=ON
```

### **Custom Compiler Flags**

You can add custom flags by editing the config file:
```bash
# In your config file (e.g., configs/gpu_cuda_config.sh)

# Add optimization flags
export CMAKE_CXX_FLAGS="-fPIC -std=c++17 --gcc-toolchain=${GCC_BASE} -O3 -march=native"

# Add debugging symbols
export CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -g"

# Add preprocessor definitions
export CMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -DMY_CUSTOM_DEFINE"
```

---

## Build Locations and Output

By default, all builds install to your build directory:
```
your_build_directory/
├── camp/
│   ├── build_${BUILD_SUFFIX}/        # Build artifacts
│   └── install_${BUILD_SUFFIX}/      # Installed library
├── RAJA/
│   ├── build_${BUILD_SUFFIX}/
│   └── install_${BUILD_SUFFIX}/
├── Umpire/                            # GPU builds only
│   ├── build_${BUILD_SUFFIX}/
│   └── install_${BUILD_SUFFIX}/
├── CHAI/                              # GPU builds only
│   ├── build_${BUILD_SUFFIX}/
│   └── install_${BUILD_SUFFIX}/
├── ExaCMech/
│   ├── build_${BUILD_SUFFIX}/
│   └── install_${BUILD_SUFFIX}/
├── hypre/
│   ├── build_${BUILD_SUFFIX}/
│   └── src/hypre_${BUILD_SUFFIX}/    # Installed library
├── metis-5.1.0/
│   └── install_${BUILD_SUFFIX}/
├── mfem/
│   ├── build_${BUILD_SUFFIX}/
│   └── install_${BUILD_SUFFIX}/
└── ExaConstit/
    ├── build_${BUILD_SUFFIX}/        # Build artifacts
    └── install_dir/                  # Final installation
```

Where `${BUILD_SUFFIX}` is:
- `cpu` for CPU builds
- `cuda` for CUDA/NVIDIA builds
- `hip` for HIP/AMD builds

### **Build Logs**

Build logs are saved in each component's build directory with standardized names:
- `my_<package>_config` - CMake configuration output
- `my_<package>_build` - Compilation output
- `my_<package>_install` - Installation output

Example: To check why RAJA failed to build:
```bash
cd RAJA/build_cuda
less my_raja_build  # View the build log
```

### **Disk Space Requirements**

Typical disk space usage:
- **CPU build**: ~1 GB total
- **GPU build (CUDA/HIP)**: ~2 GB total
  - Includes additional Umpire and CHAI libraries
  - GPU architectures add to binary sizes

---

## Troubleshooting

### **Configuration Issues**

#### **"Module not found" errors**
```
ERROR: Unable to locate a modulefile for 'intel/2023.2.1-magic'
```

**Solution:**
- Check available modules: `module avail intel`
- Update the module version in your config file
- If not using a module system, comment out the `module load` commands

#### **"Compiler not found" errors**
```
CMake Error: CMAKE_C_COMPILER not found
```

**Solution:**
- Verify the compiler path: `ls -la /path/to/compiler`
- Check that the executable exists: `which icc` or `which clang`
- Update the `*_BASE` variable in your config file
- Ensure the compiler is in your `PATH`

#### **"Python not found" errors**
```
Could NOT find Python3 (missing: Python3_EXECUTABLE)
```

**Solution:**
- Verify Python installation: `which python3` or `which python`
- Check Python version (must be 3.8+): `python3 --version`
- Update `PYTHON_BASE` in your config file
- If using Anaconda: `conda activate your_env` before building

#### **MPI errors**
```
Could not find MPI compiler wrappers
```

**Solution:**
- Verify MPI installation: `which mpicc && which mpicxx`
- Update `MPI_BASE` in your config file
- Test MPI: `mpicc --version`
- On macOS, install with: `brew install open-mpi`

### **Build Failures**

#### **Out of memory during compilation**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
- Reduce parallel jobs: `MAKE_JOBS=2 ./unix_*_install.sh`
- Close other applications
- Add swap space if building on a resource-constrained system

#### **Disk space errors**
```
No space left on device
```

**Solution:**
- Check available space: `df -h .`
- Clean previous builds: `REBUILD=ON ./unix_*_install.sh`
- Build in a location with more space
- Remove old build directories

#### **Dependency build fails partway through**

**Solution:**
1. Check the specific log file in the failing component's build directory
2. Common issues:
   - Missing system libraries (install via package manager)
   - Version incompatibilities (check `dependency_versions.sh`)
   - Network issues during git clone (retry with `SYNC_SUBMODULES=ON`)
3. Try a clean rebuild: `REBUILD=ON ./unix_*_install.sh`
4. Build dependencies individually to isolate the issue

#### **Git submodule errors**
```
fatal: unable to access 'https://github.com/...': Failed to connect
```

**Solution:**
- Check network connectivity
- If behind a firewall, configure git proxy
- Clone repositories manually if needed
- Force submodule sync: `SYNC_SUBMODULES=ON ./unix_*_install.sh`

### **Platform-Specific Issues**

#### **macOS: "xcrun: error: invalid active developer path"**

**Solution:**
```bash
xcode-select --install
```

#### **macOS: Missing dependencies**

**Solution:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required tools
brew install cmake
brew install open-mpi
brew install python3
```

#### **macOS: Architecture mismatch (Apple Silicon vs Intel)**

**Solution:**
- Ensure all dependencies are built for the same architecture
- For Apple Silicon, use ARM-native tools: `arch -arm64 brew install ...`
- For Intel compatibility on Apple Silicon: `arch -x86_64 brew install ...`

#### **HPC: Module conflicts**

**Solution:**
```bash
# Clear all modules and reload
# Some HPC will automatically switch things for you
# So only do this if your HPC center advises you to purge modules
module purge
module load intel/2023.2.1-magic
module load cmake/3.26.3
# ... load other required modules
```

#### **HPC: Quota exceeded**

**Solution:**
- Build in your scratch space instead of home directory
- Clean old builds regularly
- Check quota: `quota -s` or `lfs quota -h $HOME`

### **GPU-Specific Issues**

#### **CUDA: "nvcc not found"**

**Solution:**
- Verify CUDA installation: `which nvcc`
- Update `CUDA_BASE` in `gpu_cuda_config.sh`
- Load CUDA module if on HPC: `module load cuda`
- Set `CUDA_HOME` environment variable

#### **CUDA: Architecture mismatch**
```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**Solution:**
- Check your GPU architecture: `nvidia-smi`
- Update `CMAKE_GPU_ARCHITECTURES` to match your hardware
- Common fix: `CMAKE_GPU_ARCHITECTURES=70 ./unix_gpu_cuda_install.sh`

#### **HIP: "amdclang not found"**

**Solution:**
- Verify ROCm installation: `which amdclang`
- Update `ROCM_BASE` in `gpu_hip_config.sh`
- Load ROCm modules: `module load rocm`
- Set `ROCM_PATH` environment variable

#### **HIP: MI300A memory issues**

**Solution:**
- Set unified memory flag: `export HSA_XNACK=1`
- Verify architecture: `CMAKE_GPU_ARCHITECTURES=gfx942:xnack+`
- Check system: `rocminfo | grep xnack`

### **Runtime Issues**

#### **Segmentation fault on startup**

**Possible causes:**
1. Library path issues - Ensure `LD_LIBRARY_PATH` includes all dependency lib directories
2. ABI incompatibility - Rebuild with consistent compiler versions
3. Missing runtime dependencies - Check with `ldd` on the executable

#### **MPI initialization failures**

**Solution:**
```bash
# Test MPI installation
mpirun -np 2 hostname

# Verify MPI library paths
ldd ExaConstit/build_*/mechanics_driver | grep mpi

# Check module environment
module list
```

### **Getting Help**

If you encounter issues not covered here:

1. **Check the build logs**
   - Navigate to the failing component's build directory
   - Review `my_<package>_config`, `my_<package>_build`, or `my_<package>_install`
   - Look for specific error messages

2. **Verify your configuration**
   - Confirm all paths in your config file are correct
   - Test each tool independently: `which compiler`, `mpicc --version`, etc.

3. **Search existing issues**
   - Check the [GitHub Issues](https://github.com/LLNL/ExaConstit/issues) page
   - Search for similar error messages

4. **Open a new issue**
   - Go to [GitHub Issues](https://github.com/LLNL/ExaConstit/issues/new)
   - Include:
     - Your platform and OS version (`uname -a`, `lsb_release -a`, etc.)
     - The config file you're using
     - Relevant sections from error logs
     - Steps you've already tried
     - Output of `module list` (if applicable)

---

## Manual Build (Advanced Users)

If you prefer to build manually or need more control over the build process:

### **Prerequisites**

You'll need to manually build all dependencies first:
1. **CAMP** (v2025.12.0)
2. **RAJA** (v2025.12.2)
3. **Umpire** (v2025.12.0) - GPU builds only
4. **CHAI** (v2025.12.0) - GPU builds only
5. **ExaCMech** (v0.4.3)
6. **Hypre** (v3.1.0)
7. **METIS** (5.1.0)
8. **MFEM** (`exaconstit-dev` branch of https://github.com/rcarson3/mfem; v0.10.0 was tested with commit `3cb179f78508c75d47413b231d0893ef8b5a44c0`)

See `scripts/install/common/build_functions.sh` for the exact build commands and CMake options.

### **Building ExaConstit**

Once all dependencies are built:
```bash
# 1. Clone ExaConstit
git clone https://github.com/LLNL/ExaConstit.git
cd ExaConstit
git checkout the_great_refactoring
git submodule update --init --recursive

# 2. Create build directory
mkdir build && cd build

# 3. Configure (CPU example)
cmake .. \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DENABLE_TESTS=ON \
  -DENABLE_OPENMP=OFF \
  -DENABLE_FORTRAN=OFF \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DMFEM_DIR=${MFEM_INSTALL_DIR}/lib/cmake/mfem/ \
  -DECMECH_DIR=${EXACMECH_INSTALL_DIR}/ \
  -DSNLS_DIR=${EXACMECH_INSTALL_DIR}/ \
  -DRAJA_DIR=${RAJA_INSTALL_DIR}/lib/cmake/raja/ \
  -Dcamp_DIR=${CAMP_INSTALL_DIR}/lib/cmake/camp/ \
  -DCMAKE_BUILD_TYPE=Release

# 4. Build
make -j $(nproc)

# 5. Test
ctest
```

### **GPU Build Options**

For CUDA builds, add:
```bash
cmake .. \
  ... (all the above options) ... \
  -DCMAKE_CXX_COMPILER=${CUDA_ROOT}/bin/nvcc \
  -DCMAKE_CUDA_COMPILER=${CUDA_ROOT}/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=${HOST_CXX_COMPILER} \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_CUDA=ON \
  -DFMT_DIR=${UMPIRE_INSTALL_DIR}/lib64/cmake/fmt \
  -DUMPIRE_DIR=${UMPIRE_INSTALL_DIR}/lib64/cmake/umpire \
  -DCHAI_DIR=${CHAI_INSTALL_DIR}/lib/cmake/chai
```

For HIP builds, add:
```bash
cmake .. \
  ... (all the above options) ... \
  -DCMAKE_CXX_COMPILER=${ROCM_ROOT}/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=${ROCM_ROOT}/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DENABLE_HIP=ON \
  -DFMT_DIR=${UMPIRE_INSTALL_DIR}/lib64/cmake/fmt \
  -DUMPIRE_DIR=${UMPIRE_INSTALL_DIR}/lib64/cmake/umpire \
  -DCHAI_DIR=${CHAI_INSTALL_DIR}/lib/cmake/chai
```

---

## Next Steps

After successful installation:

- **Run the test suite**: `cd ExaConstit/build_*/` then `ctest` or `make test`
- **Try example problems**: See `examples/` directory
- **Read the documentation**: Check the `docs/` folder for detailed usage guides
- **Join the community**: Open issues or discussions on GitHub

For questions about using ExaConstit after installation, see the main [README](../README.md) and documentation in the `docs/` folder.