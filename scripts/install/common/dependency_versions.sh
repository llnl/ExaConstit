#!/usr/bin/env bash
# Central version control for all dependencies

###########################################
# Build infrastructure
###########################################
# BLT lifted out so all RADIUSS-stack packages share a single BLT and stay in sync.
# Each package below is pointed at this via -DBLT_SOURCE_DIR=${BLT_ROOT}.
export BLT_REPO="https://github.com/LLNL/blt.git"
export BLT_VER="v0.7.2"

###########################################
# Portability libraries (RAJA Portability Suite)
###########################################
export CAMP_VER="v2025.12.0"
export RAJA_VER="v2025.12.2"
#export UMPIRE_VER="v2025.09.0"
# For now we need something a little pass the v2025.09.0 release
# for Umpire as we need a small bug fix for any build with Umpire
export UMPIRE_VER="v2025.12.0"
export CHAI_VER="v2025.12.0"

###########################################
# SNLS (lifted out of ExaCMech so it can be built standalone with the
# RAJA Portability Suite and the batch-solver option always enabled)
###########################################
export SNLS_REPO="https://github.com/LLNL/SNLS.git"
export SNLS_VER="v0.4.4"

###########################################
# Restart / checkpoint I/O stack (Axom Sidre)
###########################################
# Sidre requires Conduit, and Conduit's HDF5 relay provides the
# sidre_hdf5 I/O protocol. Stick with the HDF5 1.14 series; the 2.x
# series has caused CMake configuration issues downstream.
export HDF5_REPO="https://github.com/HDFGroup/hdf5.git"
export HDF5_VER="hdf5_1.14.6"
# Conduit v0.9.7 is the version Axom v0.15.0 was released against
# even though we use v0.14.0 below.
export CONDUIT_REPO="https://github.com/LLNL/conduit.git"
export CONDUIT_VER="v0.9.7"

###########################################
# Axom (HPC utility library suite)
###########################################
# Built with core + slic + spin + primal + slam + sidre. Axom will
# eventually consume MFEM as well, which is why build_axom lives in the
# application-stack build file (build_functions_exaconstit.sh) rather
# than the common stack.
export AXOM_REPO="https://github.com/LLNL/axom.git"
export AXOM_VER="v0.14.0"

###########################################
# Material models
###########################################
export EXACMECH_REPO="https://github.com/LLNL/ExaCMech.git"
export EXACMECH_BRANCH="v0.4.3"

###########################################
# FEM infrastructure
###########################################
export HYPRE_VER="v3.1.0"
export METIS_VER="5.1.0"
export METIS_URL="https://mfem.github.io/tpls/metis-${METIS_VER}.tar.gz"

export MFEM_REPO="https://github.com/rcarson3/mfem.git"
export MFEM_BRANCH="exaconstit-dev"

###########################################
# Main application
###########################################
export EXACONSTIT_REPO="https://github.com/llnl/ExaConstit.git"
export EXACONSTIT_BRANCH="exaconstit-dev"

###########################################
# Build standards
###########################################
export CMAKE_CXX_STANDARD="17"
export CMAKE_BUILD_TYPE="Release"
