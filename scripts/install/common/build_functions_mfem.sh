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
# MFEM batched gemv tuning (GPU builds)
###########################################
# MFEM's native batched matrix-vector kernels (used by element assembly) have
# tuning parameters with defaults per GPU vendor. MFEM's
# config/tune_batched_gemv.py measures them on the GPU at hand and writes a
# header that MFEM is rebuilt with (only one MFEM source file recompiles).
#
# MFEM_GEMV_TUNING:
#   AUTO   (default) tune after MFEM is built and save the header as
#          mfem/gemv_tuning_<suffix>_<arch>.hpp; later builds reuse it
#          (MFEM_GEMV_RETUNE=ON tunes again)
#   OFF    use MFEM's defaults
#   <path> use this tuning header
# MFEM_GEMV_TUNING_LAUNCHER: command prefix that runs the benchmark on a GPU
#   node, e.g. "flux run -n1 -g1" or "srun -n1 -G1", for builds on a node
#   without a GPU. Without it the benchmark runs directly on the build node.
# MFEM_GEMV_TUNING_ARGS: extra arguments for tune_batched_gemv.py.

# Saved tuning header for this build type and GPU architecture
mfem_gemv_tuning_file() {
  local arch="${CMAKE_GPU_ARCHITECTURES}"
  if [ "${GPU_BACKEND}" = "HIP" ]; then
    arch="${MFEM_HIP_ARCHITECTURES:-${CMAKE_GPU_ARCHITECTURES}}"
  fi
  echo "${BASE_DIR}/mfem/gemv_tuning_${BUILD_SUFFIX}_${arch//[^A-Za-z0-9]/_}.hpp"
}

# Tuning header to configure MFEM with (empty for MFEM's defaults)
mfem_gemv_tuning_header() {
  local mode="${MFEM_GEMV_TUNING:-AUTO}"
  if [ "${mode}" = "OFF" ]; then
    echo ""
  elif [ "${mode}" = "AUTO" ]; then
    local hdr
    hdr="$(mfem_gemv_tuning_file)"
    if [ -f "${hdr}" ]; then echo "${hdr}"; else echo ""; fi
  elif [ -f "${mode}" ]; then
    echo "$(cd "$(dirname "${mode}")" && pwd)/$(basename "${mode}")"
  else
    echo "MFEM_GEMV_TUNING: no such file: ${mode}" >&2
    return 1
  fi
}

# Run in the MFEM build directory after MFEM is installed. Never fails the
# build: if tuning is not possible, MFEM keeps its default parameters.
tune_mfem_gemv() {
  [ "${MFEM_GEMV_TUNING:-AUTO}" = "AUTO" ] || return 0
  local hdr script
  hdr="$(mfem_gemv_tuning_file)"
  script="${BASE_DIR}/mfem/config/tune_batched_gemv.py"
  if [ -f "${hdr}" ] && [ "${MFEM_GEMV_RETUNE:-OFF}" != "ON" ]; then
    echo "Using saved batched gemv tuning: ${hdr}"
    return 0
  fi
  if [ ! -f "${script}" ]; then
    echo "WARNING: ${script} not found (older MFEM?); skipping batched gemv tuning."
    return 0
  fi
  local launcher="${MFEM_GEMV_TUNING_LAUNCHER:-}"
  echo "=========================================="
  echo "Tuning MFEM batched gemv kernels"
  echo "=========================================="
  local python="${PYTHON_EXECUTABLE:-python3}"
  [ -x "${python}" ] || python="python3"
  local -a args=(--mfem-dir "${MFEM_ROOT}"
                 --work-dir "${PWD}/tune_batched_gemv"
                 --output "${hdr}.new")
  [ -n "${launcher}" ] && args+=(--launcher "${launcher}")
  local -a extra=()
  read -r -a extra <<< "${MFEM_GEMV_TUNING_ARGS:-}"
  if run_with_log my_mfem_tune "${python}" "${script}" "${args[@]}" \
       ${extra[@]+"${extra[@]}"}; then
    mv "${hdr}.new" "${hdr}"
    run_with_log my_mfem_config_tuned cmake -DMFEM_BATCHED_GEMV_TUNING="${hdr}" .
    run_with_log my_mfem_build_tuned make -j "${MAKE_JOBS}"
    run_with_log my_mfem_install_tuned make install
    echo "MFEM rebuilt with batched gemv tuning: ${hdr}"
  else
    rm -f "${hdr}.new"
    echo "WARNING: batched gemv tuning failed (see my_mfem_tune in $PWD);"
    echo "  MFEM keeps its default parameters."
    if [ -z "${launcher}" ]; then
      echo "  If this node has no GPU, set MFEM_GEMV_TUNING_LAUNCHER"
      echo "  (e.g. \"flux run -n1 -g1\") to run the benchmark on a GPU node."
    fi
  fi
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
        # nvcc is the CXX compiler here; make CMake emit nvcc-style link options
        -DCMAKE_USER_MAKE_RULES_OVERRIDE_CXX="${SCRIPT_DIR}/common/nvcc_cxx_rules_override.cmake"
      )
    elif [ "${GPU_BACKEND}" = "HIP" ]; then
      CMAKE_ARGS+=(
        -DHIP_ARCH="${MFEM_HIP_ARCHITECTURES}"
        -DCMAKE_HIP_ARCHITECTURES="${MFEM_HIP_ARCHITECTURES}"
      )
    fi
    local tuning
    tuning="$(mfem_gemv_tuning_header)"
    CMAKE_ARGS+=(-DMFEM_BATCHED_GEMV_TUNING="${tuning}")
  fi

  run_with_log my_mfem_config cmake ../ "${CMAKE_ARGS[@]}"
  run_with_log my_mfem_build make -j "${MAKE_JOBS}"
  run_with_log my_mfem_install make install

  MFEM_ROOT="${BASE_DIR}/mfem/install_${BUILD_SUFFIX}"
  export MFEM_ROOT
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    tune_mfem_gemv
  fi
  echo "MFEM installed to: ${MFEM_ROOT}"
  cd "${BASE_DIR}"
}
