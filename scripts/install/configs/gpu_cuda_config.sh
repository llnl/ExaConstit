#!/usr/bin/env bash
# Configuration for CUDA GPU builds

# Build type identification
export BUILD_TYPE="cuda"
export BUILD_SUFFIX="cuda"
export GPU_BACKEND="CUDA"

###########################################
# Compiler Versions and Base Paths
###########################################
# Host Compiler
CLANG_VERSION="ibm-14.0.5"
COMPILER_VERSION="clang-${CLANG_VERSION}"
CLANG_BASE="/usr/tce/packages/clang/${COMPILER_VERSION}"

# GCC for toolchain
GCC_VERSION="11.2.1"
GCC_BASE="/usr/tce/packages/gcc/gcc-${GCC_VERSION}"
GCC_ARCH_SUBDIR="ppc64le-redhat-linux/11"  # Architecture-specific lib path

# CUDA
CUDA_VERSION="11.8.0"
CUDA_BASE="/usr/tce/packages/cuda/cuda-${CUDA_VERSION}"

# MPI
MPI_IMPL="spectrum-mpi"
MPI_VERSION="rolling-release"
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}"

# Python
PYTHON_VERSION="3.8.2"
PYTHON_BASE="/usr/tce/packages/python/python-${PYTHON_VERSION}"

###########################################
# Module Loading
###########################################
module load clang/${CLANG_VERSION}
module load cmake/3.29.2
module load cuda/${CUDA_VERSION}
module list

###########################################
# Compilers
###########################################
export CMAKE_C_COMPILER="${CLANG_BASE}/bin/clang"
export CMAKE_CXX_COMPILER="${CLANG_BASE}/bin/clang++"
export CMAKE_GPU_COMPILER="${CUDA_BASE}/bin/nvcc"

###########################################
# MPI Wrappers
###########################################
export MPI_C_COMPILER="${MPI_BASE}/bin/mpicc"
export MPI_CXX_COMPILER="${MPI_BASE}/bin/mpicxx"
export MPI_Fortran_COMPILER="${MPI_BASE}/bin/mpifort"

###########################################
# Python
###########################################
export PYTHON_EXECUTABLE="${PYTHON_BASE}/bin/python3"

###########################################
# GPU Architecture (Configurable)
###########################################
# Default to Volta (SM_70), can override with environment variable
# Common options: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 90 (Hopper)
export CMAKE_GPU_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES:-70}"

###########################################
# Build Flags
###########################################
export CMAKE_CXX_FLAGS="-fPIC -std=c++17 --gcc-toolchain=${GCC_BASE}"
export CMAKE_C_FLAGS="-fPIC"
export CMAKE_Fortran_FLAGS="-fPIC"
export CMAKE_GPU_FLAGS="-restrict --expt-extended-lambda -Xcompiler --gcc-toolchain=${GCC_BASE} -Xnvlink --suppress-stack-size-warning -std=c++17"

# Linker flags for GCC toolchain integration
GCC_LIB_PATH="${GCC_BASE}/rh/usr/lib/gcc/${GCC_ARCH_SUBDIR}"
export CMAKE_EXE_LINKER_FLAGS="-L${GCC_LIB_PATH} -Wl,-rpath,${GCC_LIB_PATH}"

# BLT-specific flags (used by some dependencies)
export BLT_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"

###########################################
# MFEM Batched Gemv Tuning
###########################################
# AUTO: after building MFEM, tune its batched gemv GPU kernels on this GPU
# and rebuild MFEM with the result (if tuning fails, MFEM keeps its default
# parameters); later builds reuse the saved mfem/gemv_tuning_*.hpp
# (MFEM_GEMV_RETUNE=ON tunes again).
# OFF: MFEM's defaults. Or the path of a tuning header.
# See tune_mfem_gemv in common/build_functions.sh.
export MFEM_GEMV_TUNING="${MFEM_GEMV_TUNING:-AUTO}"
# Command prefix to run the tuning benchmark on a GPU node when building on a
# node without a GPU (empty: run it on the build node), e.g. "lrun -n1" or
# "srun -n1 -G1"
export MFEM_GEMV_TUNING_LAUNCHER="${MFEM_GEMV_TUNING_LAUNCHER:-}"

###########################################
# Build Options
###########################################
export OPENMP_ON="OFF"
export ENABLE_TESTS_EXACONSTIT="ON"
export MAKE_JOBS="${MAKE_JOBS:-4}"

###########################################
# CHAI Options
###########################################
# Conservative settings for V100 GPUs
export CHAI_DISABLE_RM="OFF"           # Keep resource manager enabled
export CHAI_THIN_GPU_ALLOCATE="OFF"    # Use full allocations for stability
export CHAI_ENABLE_PINNED="ON"
export CHAI_ENABLE_PICK="ON"
export CHAI_DEBUG="OFF"
export CHAI_ENABLE_GPU_SIMULATION_MODE="OFF"
export CHAI_ENABLE_UM="ON"
export CHAI_ENABLE_MANAGED_PTR="ON"
export CHAI_ENABLE_MANAGED_PTR_ON_GPU="ON"

###########################################
# CUDA-Specific Build Options
###########################################
# Ensure NVCC uses the correct host compiler
export CUDAHOSTCXX="${CMAKE_CXX_COMPILER}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_BASE}"