#!/usr/bin/env bash
# MFEM-stack build functions: Hypre, METIS, MFEM.
#
# Depends on the helpers in build_helpers.sh and the common stack
# defined in build_functions_common.sh (specifically RAJA / CAMP,
# which MFEM consumes).

###########################################
# Hypre
###########################################
build_hypre() {
  echo "=========================================="
  echo "Building Hypre"
  echo "=========================================="

  if [ ! -d "${BASE_DIR}/hypre" ]; then
    git clone https://github.com/hypre-space/hypre.git --branch "${HYPRE_VER}" --single-branch "${BASE_DIR}/hypre"
  fi

  prepare_build_dir "${BASE_DIR}/hypre/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/hypre/build_${BUILD_SUFFIX}"

  run_with_log my_hypre_config cmake ../src \
    -DCMAKE_INSTALL_PREFIX=../src/hypre_${BUILD_SUFFIX}/ \
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}" \
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}" \
    -DMPI_C_COMPILER="${MPI_C_COMPILER}" \
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

  run_with_log my_hypre_build make -j "${MAKE_JOBS}"
  run_with_log my_hypre_install make install

  HYPRE_ROOT="${BASE_DIR}/hypre/src/hypre_${BUILD_SUFFIX}"
  export HYPRE_ROOT
  echo "Hypre installed to: ${HYPRE_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# METIS
###########################################
build_metis() {
  echo "=========================================="
  echo "Building METIS"
  echo "=========================================="

  if [ ! -d "${BASE_DIR}/metis-${METIS_VER}" ]; then
    curl -o metis-${METIS_VER}.tar.gz "${METIS_URL}"
    tar -xzf metis-${METIS_VER}.tar.gz
    rm metis-${METIS_VER}.tar.gz
  fi

  cd "${BASE_DIR}/metis-${METIS_VER}"

  # METIS doesn't have a proper incremental build, so always clean
  make distclean 2>/dev/null || true

  prepare_build_dir "${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}"

  run_with_log my_metis_config make config \
    prefix="${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}" \
    CC="${CMAKE_C_COMPILER}" \
    CXX="${CMAKE_CXX_COMPILER}"

  run_with_log my_metis_build make -j "${MAKE_JOBS}"
  run_with_log my_metis_install make install

  METIS_ROOT="${BASE_DIR}/metis-${METIS_VER}/install_${BUILD_SUFFIX}"
  export METIS_ROOT
  echo "METIS installed to: ${METIS_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# MFEM
###########################################
build_mfem() {
  echo "=========================================="
  echo "Building MFEM"
  echo "=========================================="

  clone_if_missing "${MFEM_REPO}" "${MFEM_BRANCH}" "${BASE_DIR}/mfem"
  # Don't sync submodules for MFEM to preserve local changes

  prepare_build_dir "${BASE_DIR}/mfem/build_${BUILD_SUFFIX}"
  cd "${BASE_DIR}/mfem/build_${BUILD_SUFFIX}"

  local CMAKE_ARGS=(
    -DMFEM_USE_MPI=YES
    -DMFEM_USE_SIMD=NO
    -DMETIS_DIR="${METIS_ROOT}"
    -DHYPRE_DIR="${HYPRE_ROOT}"
    -DMFEM_USE_RAJA=YES
    -DRAJA_DIR="${RAJA_ROOT}"
    -DRAJA_REQUIRED_PACKAGES="camp"
    -DMFEM_USE_CAMP=ON
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
    -DMFEM_USE_OPENMP="${OPENMP_ON}"
    -DMFEM_USE_ZLIB=YES
    -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX}/
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
    -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
    -DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
    -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )

  if [ "${BUILD_TYPE}" = "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}"
    )
  else
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${CMAKE_GPU_COMPILER}"
      -DMPI_CXX_COMPILER="${MPI_CXX_COMPILER}"
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
      -DMFEM_USE_${GPU_BACKEND}=ON
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    )

    if [ "${GPU_BACKEND}" = "CUDA" ]; then
      CMAKE_ARGS+=(
        -DCMAKE_CUDA_COMPILER="${CMAKE_GPU_COMPILER}"
        -DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CXX_COMPILER}"
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
        -DCMAKE_CUDA_FLAGS="${CMAKE_GPU_FLAGS}"
        -DENABLE_CUDA=ON
      )
    elif [ "${GPU_BACKEND}" = "HIP" ]; then
      CMAKE_ARGS+=(
        -DHIP_ARCH="${MFEM_HIP_ARCHITECTURES}"
        -DCMAKE_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES}"
      )
    fi
  fi

  run_with_log my_mfem_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_mfem_build make -j "${MAKE_JOBS}"
  run_with_log my_mfem_install make install

  MFEM_ROOT="${BASE_DIR}/mfem/install_${BUILD_SUFFIX}"
  export MFEM_ROOT
  echo "MFEM installed to: ${MFEM_ROOT}"
  cd "${BASE_DIR}"
}
