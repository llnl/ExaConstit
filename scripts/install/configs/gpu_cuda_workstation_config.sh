#!/usr/bin/env bash
# Configuration for CUDA GPU builds on a Linux workstation: a single machine
# with an NVIDIA GPU, a CUDA toolkit, the system GCC and an MPI installed from
# the distribution's packages (no environment modules).
#
# Everything below is found automatically from PATH and the GPU, and every
# setting can be overridden from the environment, e.g.
#   CUDA_BASE=/opt/cuda-12.9 GCC_VERSION=13 \
#     ./scripts/install/unix_gpu_cuda_workstation_install.sh

# Build type identification
export BUILD_TYPE="cuda"
export BUILD_SUFFIX="cuda"
export GPU_BACKEND="CUDA"

###########################################
# Compiler Versions and Base Paths
###########################################
# CUDA toolkit: the one whose nvcc is on PATH, else /usr/local/cuda
if [ -z "${CUDA_BASE:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_BASE="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  else
    CUDA_BASE="/usr/local/cuda"
  fi
fi

# Host compiler: gcc/g++ from PATH, or gcc-N/g++-N when GCC_VERSION=N is set.
# Each CUDA release supports a limited range of GCC versions, so a system whose
# default GCC is newer than the toolkit allows needs an older one here.
GCC_SUFFIX="${GCC_VERSION:+-${GCC_VERSION}}"
HOST_CC="${HOST_CC:-$(command -v "gcc${GCC_SUFFIX}" || echo "gcc${GCC_SUFFIX}")}"
HOST_CXX="${HOST_CXX:-$(command -v "g++${GCC_SUFFIX}" || echo "g++${GCC_SUFFIX}")}"

# MPI: the wrappers on PATH, or those in ${MPI_BASE}/bin when MPI_BASE is set
mpi_wrapper() {
  if [ -n "${MPI_BASE:-}" ]; then echo "${MPI_BASE}/bin/$1"
  else command -v "$1" || echo "/usr/bin/$1"; fi
}

###########################################
# Environment
###########################################
export PATH="${CUDA_BASE}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_BASE}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# MFEM and ExaConstit use nvcc itself as the C++ compiler; this makes that nvcc
# use the host compiler above rather than whatever g++ is first on PATH
export NVCC_CCBIN="${HOST_CXX}"
# Make the MPI wrappers use the same host compiler, so that everything links
# against one libstdc++ (OpenMPI and MPICH respectively)
export OMPI_CC="${HOST_CC}" OMPI_CXX="${HOST_CXX}"
export MPICH_CC="${HOST_CC}" MPICH_CXX="${HOST_CXX}"

###########################################
# Compilers
###########################################
export CMAKE_C_COMPILER="${HOST_CC}"
export CMAKE_CXX_COMPILER="${HOST_CXX}"
export CMAKE_GPU_COMPILER="${CUDA_BASE}/bin/nvcc"

###########################################
# MPI Wrappers
###########################################
export MPI_C_COMPILER="${MPI_C_COMPILER:-$(mpi_wrapper mpicc)}"
export MPI_CXX_COMPILER="${MPI_CXX_COMPILER:-$(mpi_wrapper mpicxx)}"
export MPI_Fortran_COMPILER="${MPI_Fortran_COMPILER:-$(mpi_wrapper mpifort)}"

###########################################
# Python
###########################################
# Only used to run the test suite, which needs numpy, pandas and toml
export PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python3 || echo python3)}"

###########################################
# GPU Architecture (Configurable)
###########################################
# Default: the compute capability of the first GPU reported by nvidia-smi
# (e.g. 7.0 -> 70). Set CMAKE_GPU_ARCHITECTURES to build for another GPU, or
# when building on a machine without one.
# Common options: 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
if [ -z "${CMAKE_GPU_ARCHITECTURES:-}" ]; then
  CMAKE_GPU_ARCHITECTURES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                             | head -n1 | tr -d '.[:space:]')"
  if [ -z "${CMAKE_GPU_ARCHITECTURES}" ]; then
    echo "ERROR: could not detect the GPU architecture with nvidia-smi;" >&2
    echo "  set CMAKE_GPU_ARCHITECTURES (e.g. 80) and rerun." >&2
    return 1 2>/dev/null || exit 1
  fi
fi
export CMAKE_GPU_ARCHITECTURES

###########################################
# Build Flags
###########################################
# CMAKE_CXX_FLAGS goes to g++ for most dependencies but to nvcc for MFEM and
# ExaConstit (nvcc is their C++ compiler), so it may only hold flags both
# accept: nvcc rejects a bare -fPIC and g++ rejects -Xcompiler. -fPIC is in the
# nvcc-only GPU flags instead.
export CMAKE_CXX_FLAGS="-std=c++17"
export CMAKE_C_FLAGS="-fPIC"
export CMAKE_Fortran_FLAGS="-fPIC"
# -Wno-deprecated-gpu-targets silences the deprecation warning that recent CUDA
# releases print for older architectures
export CMAKE_GPU_FLAGS="-restrict --expt-extended-lambda -Wno-deprecated-gpu-targets -Xcompiler=-fPIC -Xnvlink --suppress-stack-size-warning -std=c++17"

export CMAKE_EXE_LINKER_FLAGS=""

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
# Not needed on a workstation, where the GPU is on the build machine
export MFEM_GEMV_TUNING_LAUNCHER="${MFEM_GEMV_TUNING_LAUNCHER:-}"

###########################################
# Build Options
###########################################
export OPENMP_ON="OFF"
export ENABLE_TESTS_EXACONSTIT="ON"
export MAKE_JOBS="${MAKE_JOBS:-$(nproc 2>/dev/null || echo 4)}"

###########################################
# CHAI Options
###########################################
export CHAI_DISABLE_RM="OFF"
export CHAI_THIN_GPU_ALLOCATE="OFF"
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
