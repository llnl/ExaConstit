#!/usr/bin/env bash
# Common-stack build functions: BLT, CAMP, RAJA, Umpire, CHAI.
#
# These are the shared portability / utility libraries used by both
# the MFEM stack and the ExaConstit application stack. Helpers live
# in build_helpers.sh; the MFEM-stack and application-stack functions
# live in build_functions_mfem.sh and build_functions_exaconstit.sh
# respectively.
#
# Note: Umpire and CHAI are built on every platform now. The batch
# SNLS solvers depend on the full RAJA Portability Suite, and ExaCMech
# transitively links the same set, so making CHAI/Umpire available on
# CPU keeps the dependency graph uniform across CPU and GPU builds.

###########################################
# BLT
###########################################
# BLT is a CMake-only build helper (header / macro / module library).
# It has no compile or install step. We clone it once and point every
# downstream LLNL/RADIUSS package at it via -DBLT_SOURCE_DIR=${BLT_ROOT}.
# This keeps every package on the same BLT version regardless of what
# their bundled submodule happens to point at.
build_blt() {
  echo "=========================================="
  echo "Cloning BLT (${BLT_VER})"
  echo "=========================================="

  clone_if_missing "${BLT_REPO}" "${BLT_VER}" "${BASE_DIR}/blt"

  BLT_ROOT="${BASE_DIR}/blt"
  export BLT_ROOT
  echo "BLT available at: ${BLT_ROOT}"
  echo "Downstream packages will consume it via -DBLT_SOURCE_DIR"
  cd "${BASE_DIR}"
}

###########################################
# CAMP
###########################################
build_camp() {
  echo "=========================================="
  echo "Building CAMP"
  echo "=========================================="

  clone_if_missing "https://github.com/LLNL/camp.git" "${CAMP_VER}" "${BASE_DIR}/camp"
  sync_submodules "${BASE_DIR}/camp"

  prepare_build_dir "${BASE_DIR}/camp/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/camp/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
  fi

  run_with_log my_camp_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_camp_build make -j "${MAKE_JOBS}"
  run_with_log my_camp_install make install

  CAMP_ROOT="${BASE_DIR}/camp/install_${BUILD_SUFFIX}"
  export CAMP_ROOT
  echo "CAMP installed to: ${CAMP_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# RAJA
###########################################
build_raja() {
  echo "=========================================="
  echo "Building RAJA"
  echo "=========================================="

  clone_if_missing "https://github.com/LLNL/RAJA.git" "${RAJA_VER}" "${BASE_DIR}/RAJA"
  sync_submodules "${BASE_DIR}/RAJA"

  prepare_build_dir "${BASE_DIR}/RAJA/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/RAJA/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DRAJA_ENABLE_TESTS=OFF
    -DRAJA_ENABLE_EXAMPLES=OFF
    -DRAJA_ENABLE_BENCHMARKS=OFF
    -DRAJA_ENABLE_REPRODUCERS=OFF
    -DRAJA_ENABLE_EXERCISES=OFF
    -DRAJA_ENABLE_VECTORIZATION=OFF
    -DRAJA_ENABLE_DOCUMENTATION=OFF
    -DRAJA_USE_DOUBLE=ON
    -DRAJA_TIMER=chrono
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DRAJA_USE_BARE_PTR=ON
      )
    fi
  fi

  run_with_log my_raja_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_raja_build make -j "${MAKE_JOBS}"
  run_with_log my_raja_install make install

  RAJA_ROOT="${BASE_DIR}/RAJA/install_${BUILD_SUFFIX}"
  export RAJA_ROOT
  echo "RAJA installed to: ${RAJA_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# Umpire
###########################################
# Built on both CPU and GPU. SNLS's batch solvers depend on Umpire, and
# we want batch solvers available regardless of platform.
build_umpire() {
  echo "=========================================="
  echo "Building Umpire"
  echo "=========================================="

  clone_if_missing "https://github.com/LLNL/Umpire.git" "${UMPIRE_VER}" "${BASE_DIR}/Umpire"
  sync_submodules "${BASE_DIR}/Umpire"

  prepare_build_dir "${BASE_DIR}/Umpire/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/Umpire/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DENABLE_MPI=OFF
    -DUMPIRE_ENABLE_C=OFF
    -DENABLE_FORTRAN=OFF
    -DENABLE_GMOCK=OFF
    -DUMPIRE_ENABLE_IPC_SHARED_MEMORY=OFF
    -DUMPIRE_ENABLE_TOOLS=ON
    -DUMPIRE_ENABLE_BACKTRACE=ON
    -DUMPIRE_ENABLE_BACKTRACE_SYMBOLS=ON
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
  fi

  run_with_log my_umpire_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_umpire_build make -j "${MAKE_JOBS}"
  run_with_log my_umpire_install make install

  UMPIRE_ROOT="${BASE_DIR}/Umpire/install_${BUILD_SUFFIX}"
  export UMPIRE_ROOT

  # Find fmt directory (Umpire vendors fmt and exports a CMake config for it)
  FMT_DIR_CMAKE=$(find "${UMPIRE_ROOT}" -name 'fmtConfig.cmake' -print -quit || true)
  if [ -n "${FMT_DIR_CMAKE}" ]; then
    FMT_DIR=$(dirname "${FMT_DIR_CMAKE}")
  else
    FMT_DIR="${UMPIRE_ROOT}"
  fi
  export FMT_DIR

  echo "Umpire installed to: ${UMPIRE_ROOT}"
  echo "fmt found at: ${FMT_DIR}"
  cd "${BASE_DIR}"
}

###########################################
# CHAI
###########################################
# Built on both CPU and GPU. SNLS's batch solvers consume CHAI's
# ManagedArray and managed_ptr plumbing; on CPU CHAI's GPU-specific
# knobs (pinned, UM, managed_ptr_on_gpu, etc.) are OFF in the platform
# configs, but CHAI_ENABLE_MANAGED_PTR must stay ON since SNLS and
# ExaCMech use chai::managed_ptr on every platform.
build_chai() {
  echo "=========================================="
  echo "Building CHAI"
  echo "=========================================="

  clone_if_missing "https://github.com/LLNL/CHAI.git" "${CHAI_VER}" "${BASE_DIR}/CHAI"
  sync_submodules "${BASE_DIR}/CHAI"

  prepare_build_dir "${BASE_DIR}/CHAI/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/CHAI/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    -DBLT_SOURCE_DIR="${BLT_ROOT}"
    -DENABLE_TESTS=OFF
    -DENABLE_EXAMPLES=OFF
    -DENABLE_DOCS=OFF
    -DENABLE_GMOCK=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DENABLE_MPI=OFF
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_FLAGS="${CMAKE_C_FLAGS}"
    -DCHAI_ENABLE_RAJA_PLUGIN=ON
    -DCHAI_ENABLE_RAJA_NESTED_TEST=OFF
    -DCHAI_THIN_GPU_ALLOCATE="${CHAI_THIN_GPU_ALLOCATE}"
    -DCHAI_ENABLE_PINNED="${CHAI_ENABLE_PINNED}"
    -DCHAI_DISABLE_RM="${CHAI_DISABLE_RM}"
    -DCHAI_ENABLE_PICK="${CHAI_ENABLE_PICK}"
    -DCHAI_DEBUG="${CHAI_DEBUG}"
    -DCHAI_ENABLE_GPU_SIMULATION_MODE="${CHAI_ENABLE_GPU_SIMULATION_MODE}"
    -DCHAI_ENABLE_UM="${CHAI_ENABLE_UM}"
    -DCHAI_ENABLE_MANAGED_PTR="${CHAI_ENABLE_MANAGED_PTR}"
    -DCHAI_ENABLE_MANAGED_PTR_ON_GPU="${CHAI_ENABLE_MANAGED_PTR_ON_GPU}"
    -Dfmt_DIR="${FMT_DIR}"
    -Dumpire_DIR="${UMPIRE_ROOT}"
    -DRAJA_DIR="${RAJA_ROOT}"
    -Dcamp_DIR="${CAMP_ROOT}"
  )

  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
    )
  fi

  run_with_log my_chai_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_chai_build make -j "${MAKE_JOBS}"
  run_with_log my_chai_install make install

  CHAI_ROOT="${BASE_DIR}/CHAI/install_${BUILD_SUFFIX}"
  export CHAI_ROOT
  echo "CHAI installed to: ${CHAI_ROOT}"
  cd "${BASE_DIR}"
}
