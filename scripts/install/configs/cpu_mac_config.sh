#!/usr/bin/env bash
# Configuration for Mac CPU builds (Apple Silicon or Intel)

# Build type identification
export BUILD_TYPE="cpu"
export BUILD_SUFFIX="cpu"

###########################################
# User-Configurable Paths
###########################################
# IMPORTANT: Update these paths for your local Mac environment
# These are example paths - you MUST customize them for your system

# Homebrew location (typical paths shown)
# Apple Silicon: /opt/homebrew
# Intel Mac: /usr/local
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

# System Clang (or specify Homebrew LLVM if preferred)
# System Clang is typically fine for macOS
CLANG_BASE="/usr/bin"

# MPI installation location
# Options:
#   - Homebrew: ${HOMEBREW_PREFIX}/bin
#   - MacPorts: /opt/local/bin  
#   - Custom build: ${HOME}/local/bin
#   - Anaconda: ${HOME}/anaconda3/bin
MPI_BASE="${HOME}/local/bin"

# Python location
# Options:
#   - Homebrew: ${HOMEBREW_PREFIX}/bin/python3
#   - Anaconda: ${HOME}/anaconda3/bin/python
#   - System: /usr/bin/python3
PYTHON_BASE="${HOME}/anaconda3/bin"

###########################################
# Compiler Detection
###########################################
# Note: No module system on Mac, so we rely on PATH and explicit settings

# Check if we're on Apple Silicon or Intel
if [[ $(uname -m) == "arm64" ]]; then
  MAC_ARCH="arm64"
  echo "Detected Apple Silicon (ARM64)"
else
  MAC_ARCH="x86_64"
  echo "Detected Intel Mac (x86_64)"
fi

###########################################
# Compilers
###########################################
export CMAKE_C_COMPILER="${CLANG_BASE}/clang"
export CMAKE_CXX_COMPILER="${CLANG_BASE}/clang++"

###########################################
# MPI Wrappers
###########################################
export MPI_C_COMPILER="${MPI_BASE}/mpicc"
export MPI_CXX_COMPILER="${MPI_BASE}/mpicxx"
export MPI_Fortran_COMPILER="${MPI_BASE}/mpifort"

###########################################
# Python
###########################################
export PYTHON_EXECUTABLE="${PYTHON_BASE}/python"

###########################################
# Build Flags
###########################################
# Mac-specific: may need to handle SDK location
# Homebrew libraries are in ${HOMEBREW_PREFIX}/lib
export CMAKE_CXX_FLAGS="-fPIC"
export CMAKE_C_FLAGS="-fPIC"
export CMAKE_EXE_LINKER_FLAGS=""

# Optional: Add Homebrew library paths if needed
# export CMAKE_EXE_LINKER_FLAGS="-L${HOMEBREW_PREFIX}/lib -Wl,-rpath,${HOMEBREW_PREFIX}/lib"

###########################################
# Build Options
###########################################
export OPENMP_ON="OFF"
export ENABLE_TESTS_EXACONSTIT="ON"

# Mac-specific: may want to limit parallelism to avoid overheating
export MAKE_JOBS="${MAKE_JOBS:-$(sysctl -n hw.ncpu)}"

###########################################
# GPU Settings (Not Applicable)
###########################################
export GPU_BACKEND="NONE"
export CMAKE_GPU_COMPILER=""
export CMAKE_GPU_ARCHITECTURES=""
export CMAKE_GPU_FLAGS=""

###########################################
# CHAI Options (CPU build: GPU-specific features disabled)
###########################################
export CHAI_DISABLE_RM="OFF"
export CHAI_THIN_GPU_ALLOCATE="OFF"
export CHAI_ENABLE_PINNED="OFF"
export CHAI_ENABLE_PICK="OFF"
export CHAI_DEBUG="OFF"
export CHAI_ENABLE_GPU_SIMULATION_MODE="OFF"
export CHAI_ENABLE_UM="OFF"
# managed_ptr is required on CPU too: SNLS (batch solvers) and ExaCMech
# use chai::managed_ptr whenever SNLS is built with the RAJA Portability Suite.
export CHAI_ENABLE_MANAGED_PTR="ON"
export CHAI_ENABLE_MANAGED_PTR_ON_GPU="OFF"

###########################################
# Mac-Specific Notes
###########################################
echo "=========================================="
echo "Mac Build Configuration Notes"
echo "=========================================="
echo "Architecture:    ${MAC_ARCH}"
echo "Homebrew prefix: ${HOMEBREW_PREFIX}"
echo ""
echo "IMPORTANT: Verify these paths are correct for your system:"
echo "  Compilers:     ${CLANG_BASE}"
echo "  MPI:           ${MPI_BASE}"
echo "  Python:        ${PYTHON_BASE}"
echo ""
echo "If builds fail, common issues:"
echo "  1. MPI not installed: brew install open-mpi"
echo "  2. CMake too old: brew install cmake"
echo "  3. Wrong Python: Set PYTHON_BASE in this config"
echo "  4. Path issues: Ensure MPI/Python are in your PATH"
echo "=========================================="