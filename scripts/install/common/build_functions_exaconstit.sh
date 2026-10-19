#!/usr/bin/env bash
# ExaConstit application-stack build functions: SNLS, ExaCMech, HDF5,
# Conduit, Axom, and ExaConstit. Also defines the top-level
# build_all_dependencies orchestrator.
#
# Depends on the helpers in build_helpers.sh, the common stack defined
# in build_functions_common.sh (BLT, CAMP, RAJA, Umpire, CHAI), and
# MFEM defined in build_functions_mfem.sh.
#
# Axom lives here rather than in the common stack because it will
# eventually depend on MFEM, which puts it logically downstream of the
# MFEM-stack build file and alongside the other application-tier
# packages. HDF5 and Conduit live here because they are only needed
# for Axom's Sidre component (restart / checkpoint I/O).

###########################################
# SNLS
###########################################
# Lifted out of ExaCMech and built standalone with the batch-solver
# option always enabled. Batch solvers require the full RAJA
# Portability Suite (RAJA + Umpire + CHAI + camp); since the common
# stack now builds Umpire and CHAI on every platform, this is uniform
# across CPU and GPU.
build_snls() {
  echo "=========================================="
  echo "Building SNLS"
  echo "=========================================="

  clone_if_missing "${SNLS_REPO}" "${SNLS_VER}" "${BASE_DIR}/SNLS"
  sync_submodules "${BASE_DIR}/SNLS"

  prepare_build_dir "${BASE_DIR}/SNLS/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/SNLS/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DENABLE_FORTRAN=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    # Batch solvers ON everywhere -> needs the full Portability Suite.
    -DUSE_BATCH_SOLVERS=ON
    -DUSE_RAJA_ONLY=OFF
    -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja"
    -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
    -DUMPIRE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire"
    -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai"
    -DFMT_DIR="${FMT_DIR}"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
  fi

  run_with_log my_snls_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_snls_build make -j "${MAKE_JOBS}"
  run_with_log my_snls_install make install

  SNLS_ROOT="${BASE_DIR}/SNLS/install_${BUILD_SUFFIX}"
  export SNLS_ROOT
  echo "SNLS installed to: ${SNLS_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# ExaCMech
###########################################
# Consumes the standalone SNLS instead of its bundled submodule.
# ExaCMech's CMakeLists auto-sets its internal USE_BUILT_SNLS=ON when
# SNLS_DIR is defined, so we only need to pass SNLS_DIR -- no other
# external-SNLS toggle required.
#
# Because the standalone SNLS is built with USE_BATCH_SOLVERS=ON, it
# pulls CHAI / Umpire / fmt into ExaCMech's link line transitively.
# So FMT_DIR / UMPIRE_DIR / CHAI_DIR are passed unconditionally now,
# regardless of whether ExaCMech itself is being built with GPU support.
build_exacmech() {
  echo "=========================================="
  echo "Building ExaCMech"
  echo "=========================================="

  clone_if_missing "${EXACMECH_REPO}" "${EXACMECH_BRANCH}" "${BASE_DIR}/ExaCMech"
  sync_submodules "${BASE_DIR}/ExaCMech"

  prepare_build_dir "${BASE_DIR}/ExaCMech/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/ExaCMech/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DENABLE_MINIAPPS=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    # External SNLS: defining SNLS_DIR is sufficient; ExaCMech sets
    # USE_BUILT_SNLS=ON internally when it sees this variable.
    -DSNLS_DIR="${SNLS_ROOT}/lib/cmake/snls"
    -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja"
    -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
    # SNLS was built with batch solvers, so ExaCMech needs the full
    # Portability Suite resolved transitively even on CPU builds.
    -DFMT_DIR="${FMT_DIR}"
    -DUMPIRE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire"
    -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
  fi

  run_with_log my_ecmech_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_ecmech_build make -j "${MAKE_JOBS}"
  run_with_log my_ecmech_install make install

  ECMECH_ROOT="${BASE_DIR}/ExaCMech/install_${BUILD_SUFFIX}"
  export ECMECH_ROOT
  echo "ExaCMech installed to: ${ECMECH_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# HDF5
###########################################
# Serial HDF5 used by Conduit's relay I/O, which is what Sidre's
# sidre_hdf5 protocol writes through. Sidre's parallel I/O (IOManager)
# writes one HDF5 file per rank group itself, so a parallel HDF5 build
# is not needed. Only the C library is built; Fortran / C++ / examples
# and tests are skipped. Tools stay on so h5dump / h5ls are available
# for inspecting restart files.
build_hdf5() {
  echo "=========================================="
  echo "Building HDF5"
  echo "=========================================="

  clone_if_missing "${HDF5_REPO}" "${HDF5_VER}" "${BASE_DIR}/hdf5"

  prepare_build_dir "${BASE_DIR}/hdf5/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/hdf5/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_STATIC_LIBS=ON
    -DBUILD_TESTING=OFF
    -DHDF5_ENABLE_PARALLEL=OFF
    -DHDF5_BUILD_CPP_LIB=OFF
    -DHDF5_BUILD_FORTRAN=OFF
    -DHDF5_BUILD_JAVA=OFF
    -DHDF5_BUILD_EXAMPLES=OFF
    -DHDF5_BUILD_TOOLS=ON
    -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF
  )

  run_with_log my_hdf5_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_hdf5_build make -j "${MAKE_JOBS}"
  run_with_log my_hdf5_install make install

  HDF5_ROOT="${BASE_DIR}/hdf5/install_${BUILD_SUFFIX}"
  export HDF5_ROOT
  echo "HDF5 installed to: ${HDF5_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# Conduit
###########################################
# Hard dependency of Axom's Sidre component. Built with MPI (Axom
# requires conduit_relay_mpi when AXOM_ENABLE_MPI=ON) and HDF5 relay
# support. Conduit expects HDF5_DIR to be the HDF5 install prefix.
#
# Like Axom, Conduit's CMakeLists lives in src/, so the configure step
# points at ../src.
build_conduit() {
  echo "=========================================="
  echo "Building Conduit"
  echo "=========================================="

  clone_if_missing "${CONDUIT_REPO}" "${CONDUIT_VER}" "${BASE_DIR}/conduit"
  sync_submodules "${BASE_DIR}/conduit"

  prepare_build_dir "${BASE_DIR}/conduit/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/conduit/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DBLT_CXX_STD="c++${CMAKE_CXX_STANDARD}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DBUILD_SHARED_LIBS=OFF
    -DENABLE_TESTS=OFF
    -DENABLE_EXAMPLES=OFF
    -DENABLE_UTILS=OFF
    -DENABLE_DOCS=OFF
    -DENABLE_PYTHON=OFF
    -DENABLE_FORTRAN=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DENABLE_MPI=ON
    -DMPI_C_COMPILER="${MPI_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -DHDF5_DIR="${HDF5_ROOT}"
  )

  run_with_log my_conduit_config cmake ../src "${CMAKE_ARGS[@]}"
  run_with_log my_conduit_build make -j "${MAKE_JOBS}"
  run_with_log my_conduit_install make install

  CONDUIT_ROOT="${BASE_DIR}/conduit/install_${BUILD_SUFFIX}"
  export CONDUIT_ROOT
  echo "Conduit installed to: ${CONDUIT_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# Axom
###########################################
# Built with the core component (always on) plus spin, primal, slam,
# and sidre. Slic is enabled explicitly because spin, sidre and other
# components rely on it for logging. Sidre is what the restart /
# checkpoint logic is built on; it pulls in Conduit (required) and
# HDF5 (for the sidre_hdf5 I/O protocol).
#
# Axom's CMakeLists lives in the src/ subdirectory, so the configure
# step points at ../src rather than ../ like the other packages.
build_axom() {
  echo "=========================================="
  echo "Building Axom"
  echo "=========================================="

  clone_if_missing "${AXOM_REPO}" "${AXOM_VER}" "${BASE_DIR}/axom"
  sync_submodules "${BASE_DIR}/axom"

  prepare_build_dir "${BASE_DIR}/axom/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/axom/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DBLT_CXX_STD="c++${CMAKE_CXX_STANDARD}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    # Disable everything by default, then turn on what we need.
    -DAXOM_ENABLE_ALL_COMPONENTS=OFF
    -DAXOM_ENABLE_SPIN=ON
    -DAXOM_ENABLE_SLIC=ON
    -DAXOM_ENABLE_SIDRE=ON
    -DAXOM_ENABLE_INLET=OFF
    -DAXOM_ENABLE_KLEE=OFF
    -DAXOM_ENABLE_LUMBERJACK=ON
    -DAXOM_ENABLE_MINT=OFF
    -DAXOM_ENABLE_MIR=OFF
    -DAXOM_ENABLE_MULTIMAT=OFF
    -DAXOM_ENABLE_PRIMAL=ON
    -DAXOM_ENABLE_QUEST=OFF
    -DAXOM_ENABLE_SLAM=ON
    # Build settings -- skip everything that isn't the library itself.
    -DAXOM_ENABLE_TESTS=OFF
    -DAXOM_ENABLE_EXAMPLES=OFF
    -DAXOM_ENABLE_TUTORIALS=OFF
    -DAXOM_ENABLE_DOCS=OFF
    -DAXOM_ENABLE_TOOLS=OFF
    -DENABLE_BENCHMARKS=OFF
    -DENABLE_FORTRAN=OFF
    # Parallelism / dependencies
    # AXOM_ENABLE_MPI / AXOM_ENABLE_OPENMP are cmake_dependent_options
    # gated on BLT's ENABLE_MPI / ENABLE_OPENMP, so those must be set
    # too or Axom silently builds without MPI (no Sidre IOManager).
    -DENABLE_MPI=ON
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DAXOM_ENABLE_MPI=ON
    -DAXOM_ENABLE_OPENMP="${OPENMP_ON}"
    -DMPI_C_COMPILER="${MPI_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -DCAMP_DIR="${CAMP_ROOT}"
    -DRAJA_DIR="${RAJA_ROOT}"
    -DUMPIRE_DIR="${UMPIRE_ROOT}"
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
    # Sidre dependencies
    -DCONDUIT_DIR="${CONDUIT_ROOT}"
    -DHDF5_DIR="${HDF5_ROOT}"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    # Spin's GPU paths run through RAJA -> Umpire memory plumbing.
    CMAKE_ARGS+=(
      -DAXOM_ENABLE_${GPU_BACKEND}=ON
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
    )
    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}"
      )
    fi
  fi

  run_with_log my_axom_config cmake ../src "${CMAKE_ARGS[@]}"
  run_with_log my_axom_build make -j "${MAKE_JOBS}"
  run_with_log my_axom_install make install

  AXOM_ROOT="${BASE_DIR}/axom/install_${BUILD_SUFFIX}"
  export AXOM_ROOT
  echo "Axom installed to: ${AXOM_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# ExaConstit
###########################################
# Like ExaCMech, the SNLS-batch transitive deps mean we pass FMT_DIR /
# UMPIRE_DIR / CHAI_DIR unconditionally now (previously GPU-only).
build_exaconstit() {
  echo "=========================================="
  echo "Building ExaConstit"
  echo "=========================================="

  clone_if_missing "${EXACONSTIT_REPO}" "${EXACONSTIT_BRANCH}" "${BASE_DIR}/ExaConstit"
  sync_submodules "${BASE_DIR}/ExaConstit"

  prepare_build_dir "${BASE_DIR}/ExaConstit/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/ExaConstit/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DENABLE_TESTS="${ENABLE_TESTS_EXACONSTIT}"
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DENABLE_FORTRAN=OFF
    -DENABLE_SNLS_V03=ON
    -DCMAKE_INSTALL_PREFIX=../install_dir/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DMFEM_DIR="${MFEM_ROOT}/lib/cmake/mfem"
    -DECMECH_DIR="${ECMECH_ROOT}"
    -DSNLS_DIR="${SNLS_ROOT}/lib/cmake/snls"
    -DAXOM_DIR="${AXOM_ROOT}/lib/cmake"
    -Daxom_DIR="${AXOM_ROOT}/lib/cmake"
    -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja"
    -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
    # SNLS-batch transitive deps (now needed on CPU builds too).
    -DFMT_DIR="${FMT_DIR}"
    -DUMPIRE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire"
    -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai"
  )

  if [ "${BUILD_TYPE}" = "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}"
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
    )
  else
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DENABLE_${GPU_BACKEND}=ON
    )

    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DCMAKE_CUDA_FLAGS="${CMAKE_GPU_FLAGS}"
        -DBLT_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
      )
    elif [ "${GPU_BACKEND}" = "HIP" ]; then
      CMAKE_ARGS+=(
        -DCMAKE_HIP_FLAGS="${CMAKE_GPU_FLAGS}"
      )
    fi
  fi

  run_with_log my_exconstit_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_exconstit_build make -j "${MAKE_JOBS}"

  EXACONSTIT_ROOT="${BASE_DIR}/ExaConstit/install_dir"
  export EXACONSTIT_ROOT
  echo "=========================================="
  echo "ExaConstit build complete!"
  echo "Install prefix: ${EXACONSTIT_ROOT}"
  echo "=========================================="
  cd "${BASE_DIR}"
}

###########################################
# Main orchestration function
###########################################
# Build order honors the dependency graph:
#   1. BLT (header-only build helper, must come first so every
#      downstream package can point at it).
#   2. RAJA Portability Suite: CAMP -> RAJA -> Umpire -> CHAI
#      (Umpire and CHAI now built on every platform).
#   3. MFEM stack: Hypre, METIS, MFEM.
#   4. Application stack: SNLS -> ExaCMech -> HDF5 -> Conduit -> Axom
#      -> ExaConstit. SNLS and ExaCMech come first because the SNLS
#      batch solver path is a hard dependency; HDF5 and Conduit feed
#      Axom's Sidre component; Axom is placed before ExaConstit since
#      ExaConstit consumes it (and Axom will eventually pick up MFEM).
build_all_dependencies() {
  # Common stack
  build_blt
  build_camp
  build_raja
  build_umpire
  build_chai

  # MFEM stack
  build_hypre
  build_metis
  build_mfem

  # Application stack
  build_snls
  build_exacmech
  build_hdf5
  build_conduit
  build_axom
  build_exaconstit
}
