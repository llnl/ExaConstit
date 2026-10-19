#!/usr/bin/env bash
# Common build functions for all ExaConstit dependencies

# Logging wrapper
run_with_log() {
  local log="$1"; shift
  "$@" |& tee "$log"
}

# Clone repository only if missing, initialize submodules on first clone
clone_if_missing() {
  local repo="$1" branch="$2" dest="$3"
  if [ ! -d "$dest/.git" ]; then
    echo "Cloning ${dest}..."
    git clone --branch "$branch" "$repo" "$dest"
    cd "$dest"
    if [ -f .gitmodules ]; then
      git submodule update --init --recursive
    fi
    cd "$BASE_DIR"
  else
    echo "${dest} already exists, skipping clone."
  fi
}

# Optional: force submodule sync when explicitly requested
sync_submodules() {
  local dest="$1"
  if [ "${SYNC_SUBMODULES}" = "ON" ] && [ -f "$dest/.gitmodules" ]; then
    echo "Syncing submodules in ${dest}..."
    cd "$dest"
    git submodule sync --recursive
    git submodule update --init --recursive
    cd "$BASE_DIR"
  fi
}

# Respect REBUILD flag when preparing build directories
prepare_build_dir() {
  local dir="$1"
  if [ "${REBUILD}" = "ON" ]; then
    mkdir -p "$dir"
    rm -rf "$dir"/*
    echo "Cleaned build directory: ${dir}"
  else
    if [ ! -d "$dir" ]; then
      mkdir -p "$dir"
      echo "Created build directory: ${dir}"
    else
      echo "Reusing existing build directory: ${dir}"
    fi
  fi
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
# Umpire (GPU only)
###########################################
build_umpire() {
  if [ "${BUILD_TYPE}" = "cpu" ]; then
    echo "Skipping Umpire (not needed for CPU builds)"
    return 0
  fi
  
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
    -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
    -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
    -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
    -DENABLE_${GPU_BACKEND}=ON
    -Dcamp_DIR="${CAMP_ROOT}/lib/cmake/camp"
  )
  
  run_with_log my_umpire_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_umpire_build make -j "${MAKE_JOBS}"
  run_with_log my_umpire_install make install
  
  UMPIRE_ROOT="${BASE_DIR}/Umpire/install_${BUILD_SUFFIX}"
  export UMPIRE_ROOT
  
  # Find fmt directory
  FMT_DIR_CMAKE=$(find "${UMPIRE_ROOT}" -name 'fmtConfig.cmake' -print -quit || true)
  if [ -n "${FMT_DIR_CMAKE}" ]; then
    FMT_DIR=$(dirname "${FMT_DIR_CMAKE}")
  else
    FMT_DIR="${UMPIRE_ROOT}"
  fi
  export FMT_DIR

  # Find Umpire's CMake config dir (lib64 on RHEL-like systems, lib on Debian/Ubuntu)
  UMPIRE_CMAKE_CONFIG=$(find "${UMPIRE_ROOT}" -name 'umpire-config.cmake' -print -quit || true)
  if [ -n "${UMPIRE_CMAKE_CONFIG}" ]; then
    UMPIRE_CMAKE_DIR=$(dirname "${UMPIRE_CMAKE_CONFIG}")
  else
    UMPIRE_CMAKE_DIR="${UMPIRE_ROOT}/lib64/cmake/umpire"
  fi
  export UMPIRE_CMAKE_DIR

  echo "Umpire installed to: ${UMPIRE_ROOT}"
  echo "fmt found at: ${FMT_DIR}"
  cd "${BASE_DIR}"
}

###########################################
# CHAI (GPU only)
###########################################
build_chai() {
  if [ "${BUILD_TYPE}" = "cpu" ]; then
    echo "Skipping CHAI (not needed for CPU builds)"
    return 0
  fi
  
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
    -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
    -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
    -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
    -DENABLE_${GPU_BACKEND}=ON
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
  
  run_with_log my_chai_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_chai_build make -j "${MAKE_JOBS}"
  run_with_log my_chai_install make install
  
  CHAI_ROOT="${BASE_DIR}/CHAI/install_${BUILD_SUFFIX}"
  export CHAI_ROOT
  echo "CHAI installed to: ${CHAI_ROOT}"
  cd "${BASE_DIR}"
}

###########################################
# ExaCMech
###########################################
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
    -DENABLE_TESTS=OFF
    -DENABLE_MINIAPPS=OFF
    -DENABLE_OPENMP="${OPENMP_ON}"
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
    -DCMAKE_CXX_STANDARD="${CMAKE_CXX_STANDARD}"
    -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja"
    -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
  )
  
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_C_COMPILER="${CMAKE_C_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_FLAGS="${CMAKE_GPU_FLAGS}"
      -DENABLE_${GPU_BACKEND}=ON
      -DFMT_DIR="${FMT_DIR}"
      -DUMPIRE_DIR="${UMPIRE_CMAKE_DIR}"
      -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai"
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

###########################################
# ExaConstit
###########################################
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
    -DMFEM_DIR="${MFEM_ROOT}/lib/cmake/mfem"
    -DECMECH_DIR="${ECMECH_ROOT}"
    -DSNLS_DIR="${ECMECH_ROOT}"
    -DRAJA_DIR="${RAJA_ROOT}/lib/cmake/raja"
    -DCAMP_DIR="${CAMP_ROOT}/lib/cmake/camp"
  )
  
  if [ "${BUILD_TYPE}" = "cpu" ]; then
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${MPI_CXX_COMPILER}"
    )
  else
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS}"
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS}"
      -DCMAKE_${GPU_BACKEND}_COMPILER="${CMAKE_GPU_COMPILER}"
      -DCMAKE_${GPU_BACKEND}_ARCHITECTURES="${CMAKE_GPU_ARCHITECTURES}"
      -DENABLE_${GPU_BACKEND}=ON
      -DFMT_DIR="${FMT_DIR}"
      -DUMPIRE_DIR="${UMPIRE_CMAKE_DIR}"
      -DCHAI_DIR="${CHAI_ROOT}/lib/cmake/chai"
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
build_all_dependencies() {
  build_camp
  build_raja
  build_umpire
  build_chai
  build_exacmech
  build_hypre
  build_metis
  build_mfem
  build_exaconstit
}