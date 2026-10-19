#!/usr/bin/env bash
# Configuration for Intel CPU builds

# Build type identification
export BUILD_TYPE="cpu"
export BUILD_SUFFIX="cpu"

###########################################
# Compiler Versions and Base Paths
###########################################
INTEL_VERSION="2023.2.1-magic"
COMPILER_VERSION="intel-${INTEL_VERSION}"
INTEL_BASE="/usr/tce/packages/intel/${COMPILER_VERSION}"

MPI_IMPL="mvapich2"
MPI_VERSION="2.3.7"
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}"

PYTHON_VERSION="3.12.2"
PYTHON_BASE="/usr/apps/python-${PYTHON_VERSION}"

###########################################
# Module Loading
###########################################
module load intel/${INTEL_VERSION}
module load cmake/3.26.3
module load python/3.12
module list

###########################################
# Compilers
###########################################
export CMAKE_C_COMPILER="${INTEL_BASE}/bin/icx"
export CMAKE_CXX_COMPILER="${INTEL_BASE}/bin/icpx"
export CMAKE_Fortran_COMPILER="${INTEL_BASE}/bin/ifx"

###########################################
# MPI Wrappers
###########################################
export MPI_C_COMPILER="${MPI_BASE}/bin/mpicc"
export MPI_CXX_COMPILER="${MPI_BASE}/bin/mpicxx"
export MPI_Fortran_COMPILER="${MPI_BASE}/bin/mpifort"

###########################################
# Python
###########################################
export PYTHON_EXECUTABLE="${PYTHON_BASE}/bin/python"

###########################################
# Build Flags
###########################################
export CMAKE_CXX_FLAGS="-fPIC"
export CMAKE_C_FLAGS="-fPIC"
export CMAKE_Fortran_FLAGS="-fPIC"
export CMAKE_EXE_LINKER_FLAGS=""

###########################################
# Build Options
###########################################
export OPENMP_ON="OFF"
export ENABLE_TESTS_EXACONSTIT="ON"
export MAKE_JOBS="${MAKE_JOBS:-4}"

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