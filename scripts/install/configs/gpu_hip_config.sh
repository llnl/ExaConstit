#!/usr/bin/env bash
# Configuration for HIP GPU builds (AMD GPUs)

# Build type identification
export BUILD_TYPE="hip"
export BUILD_SUFFIX="hip"
export GPU_BACKEND="HIP"

###########################################
# Compiler Versions and Base Paths
###########################################
# ROCm Compiler
ROCM_VERSION="6.4.2"
ROCM_MAGIC_SUFFIX="magic"
COMPILER_VERSION="rocmcc-${ROCM_VERSION}-${ROCM_MAGIC_SUFFIX}"
ROCM_BASE="/usr/tce/packages/rocmcc/${COMPILER_VERSION}"

# MPI - Cray MPICH
MPI_IMPL="cray-mpich"
MPI_VERSION="9.0.1"
MPI_COMPILER_VERSION="${MPI_IMPL}-${MPI_VERSION}"
MPI_BASE="/usr/tce/packages/${MPI_IMPL}/${MPI_COMPILER_VERSION}-${COMPILER_VERSION}"

# Cray PE paths for linking
CRAY_MPICH_VERSION="${MPI_VERSION}"
CRAY_LIBFABRIC_VERSION="2.1"
CRAY_PMI_VERSION="6.1.16"
CRAY_PALS_VERSION="1.2.12"

# Python
PYTHON_VERSION="3.9.12"
PYTHON_BASE="/usr/tce/packages/python/python-${PYTHON_VERSION}"

###########################################
# Module Loading
###########################################
module load cmake/3.29.2
module load rocmcc/${ROCM_VERSION}-${ROCM_MAGIC_SUFFIX}
module load rocm/${ROCM_VERSION}
module load ${MPI_IMPL}/${MPI_VERSION}
module list

###########################################
# Compilers
###########################################
export CMAKE_C_COMPILER="${ROCM_BASE}/bin/amdclang"
export CMAKE_CXX_COMPILER="${ROCM_BASE}/bin/amdclang++"
export CMAKE_Fortran_COMPILER="${INTEL_BASE}/bin/amdflang"
export CMAKE_GPU_COMPILER="${ROCM_BASE}/bin/amdclang++"

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
# GPU Architectures (Configurable)
###########################################
# Default to MI300A with xnack+ for unified memory
# Common options:
#   gfx908 (MI100)
#   gfx90a (MI200 series)
#   gfx940 (MI300X - compute only)
#   gfx942 (MI300A - APU with xnack support)
#   gfx942:xnack+ (MI300A with unified memory)
export CMAKE_GPU_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES:-gfx942:xnack+}"

# MFEM has issues with xnack+ in its compilation, so use base arch
export MFEM_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES:-gfx942}"

# Also set AMDGPU_TARGETS for completeness
export AMDGPU_TARGETS="${CMAKE_GPU_ARCHITECTURES}"

###########################################
# Build Flags
###########################################
export CMAKE_CXX_FLAGS="-fPIC -std=c++17 -munsafe-fp-atomics"
export CMAKE_C_FLAGS="-fPIC"
export CMAKE_Fortran_FLAGS="-fPIC"
export CMAKE_GPU_FLAGS="-munsafe-fp-atomics -fgpu-rdc"

###########################################
# MPI Linking Flags (Cray-Specific)
###########################################
# Cray MPICH requires explicit linking to GTL and OFI libraries
MPICH_GTL_LIB="/opt/cray/pe/mpich/${CRAY_MPICH_VERSION}/gtl/lib"
MPICH_OFI_AMD_LIB="/opt/cray/pe/mpich/${CRAY_MPICH_VERSION}/ofi/amd/6.0/lib"

# Runtime library paths for Cray PE
CRAY_LIBFABRIC_LIB="/opt/cray/libfabric/${CRAY_LIBFABRIC_VERSION}/lib64"
CRAY_PMI_LIB="/opt/cray/pe/pmi/${CRAY_PMI_VERSION}/lib"
CRAY_PALS_LIB="/opt/cray/pe/pals/${CRAY_PALS_VERSION}/lib"
ROCM_LLVM_LIB="/opt/rocm-${ROCM_VERSION}/llvm/lib"

# Construct the full MPI linking flags
MPI_CRAY_RPATH_FLAGS="-Wl,-rpath,${CRAY_LIBFABRIC_LIB}:${CRAY_PMI_LIB}:${CRAY_PALS_LIB}:${ROCM_LLVM_LIB}"
MPI_CRAY_LINK_FLAGS="-lxpmem"  # Cray xpmem for shared memory

export CMAKE_EXE_LINKER_FLAGS="-lroctx64 -Wl,-rpath,${MPICH_OFI_AMD_LIB} ${MPI_CRAY_RPATH_FLAGS} -L${MPICH_GTL_LIB} -lmpi_gtl_hsa -Wl,-rpath,${MPICH_GTL_LIB} ${MPI_CRAY_LINK_FLAGS}"

###########################################
# MFEM Batched Gemv Tuning
###########################################
# AUTO: after building MFEM, tune its batched gemv GPU kernels on this GPU
# (if one is reachable) and rebuild MFEM with the result; later builds reuse
# the saved mfem/gemv_tuning_*.hpp (MFEM_GEMV_RETUNE=ON tunes again).
# OFF: MFEM's defaults. Or the path of a tuning header.
# See tune_mfem_gemv in common/build_functions.sh.
export MFEM_GEMV_TUNING="${MFEM_GEMV_TUNING:-AUTO}"
# Command prefix to run the tuning benchmark on a GPU node when building on a
# node without a GPU, e.g. "flux run -n1 -g1"
export MFEM_GEMV_TUNING_LAUNCHER="${MFEM_GEMV_TUNING_LAUNCHER:-}"

###########################################
# Build Options
###########################################
export OPENMP_ON="OFF"
export ENABLE_TESTS_EXACONSTIT="ON"
export MAKE_JOBS="${MAKE_JOBS:-4}"

###########################################
# CHAI Options (MI300-Specific Tuning)
###########################################
# Aggressive settings optimized for MI300A APU architecture
# CHAI_DISABLE_RM=ON: Disable resource manager for APU unified memory
# CHAI_THIN_GPU_ALLOCATE=ON: Use thin allocations for better APU performance
export CHAI_DISABLE_RM="ON"
export CHAI_THIN_GPU_ALLOCATE="ON"
export CHAI_ENABLE_PINNED="ON"
export CHAI_ENABLE_PICK="ON"
export CHAI_DEBUG="OFF"
export CHAI_ENABLE_GPU_SIMULATION_MODE="OFF"
export CHAI_ENABLE_UM="ON"
export CHAI_ENABLE_MANAGED_PTR="ON"
export CHAI_ENABLE_MANAGED_PTR_ON_GPU="ON"

###########################################
# HIP-Specific Build Options
###########################################
export ROCM_PATH="${ROCM_BASE}"
export HIP_PLATFORM="amd"
export HIP_COMPILER="clang"