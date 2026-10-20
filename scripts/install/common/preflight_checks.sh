#!/usr/bin/env bash
# Preflight checks and utility functions

# Resolve BASE_DIR - use current working directory or user-specified location
resolve_base_dir() {
  # Allow user to override BASE_DIR via environment variable
  if [ -n "${BASE_DIR:-}" ]; then
    # User specified BASE_DIR
    if [ ! -d "${BASE_DIR}" ]; then
      echo "ERROR: Specified BASE_DIR does not exist: ${BASE_DIR}" >&2
      echo "Please create the directory first or use a valid path." >&2
      exit 1
    fi
    BASE_DIR=$(cd "${BASE_DIR}" && pwd -P)
    echo "Using user-specified build directory: ${BASE_DIR}"
  else
    # Use current working directory
    BASE_DIR=$(pwd -P)
    echo "Using current directory as build directory: ${BASE_DIR}"
  fi
  
  export BASE_DIR
  
  echo "=========================================="
  echo "Build Configuration:"
  echo "  Base directory: ${BASE_DIR}"
  echo "  All dependencies will be cloned and built here"
  echo "=========================================="
  echo ""
  
  # Optional: warn if running from ExaConstit source tree
  if [[ "${BASE_DIR}" == *"/ExaConstit"* ]]; then
    echo "⚠️  WARNING: You appear to be building inside the ExaConstit source tree."
    echo "   Consider running from a separate build directory:"
    echo "     mkdir -p /scratch/\$USER/exaconstit-build"
    echo "     cd /scratch/\$USER/exaconstit-build"
    echo "     /path/to/ExaConstit/scripts/install/unix_cpu_intel_install.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Build cancelled."
      exit 0
    fi
  fi
}

# Check for required executables and paths
check_required_paths() {
  local missing=0
  for p in "$@"; do
    if [[ "$p" == */bin/* ]]; then
      if [ ! -x "$p" ]; then 
        echo "ERROR: Missing executable: $p" >&2
        missing=1
      fi
    else
      if [ ! -e "$p" ]; then 
        echo "ERROR: Missing path: $p" >&2
        missing=1
      fi
    fi
  done
  if [ "$missing" -ne 0 ]; then 
    echo "ERROR: Required paths missing. Exiting." >&2
    exit 1
  fi
}

# Check for required commands
check_required_commands() {
  local missing=0
  for cmd in "$@"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "ERROR: Required command not found: $cmd" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    echo "ERROR: Required commands missing. Exiting." >&2
    exit 1
  fi
}

# Print build configuration summary
print_build_summary() {
  echo "=========================================="
  echo "ExaConstit Build Configuration"
  echo "=========================================="
  echo "BASE_DIR:        ${BASE_DIR}"
  echo "BUILD_TYPE:      ${BUILD_TYPE}"
  echo "BUILD_SUFFIX:    ${BUILD_SUFFIX}"
  echo "REBUILD:         ${REBUILD}"
  echo "SYNC_SUBMODULES: ${SYNC_SUBMODULES}"
  echo ""
  echo "Compilers:"
  echo "  C:             ${CMAKE_C_COMPILER}"
  echo "  CXX:           ${CMAKE_CXX_COMPILER}"
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    echo "  GPU:           ${CMAKE_GPU_COMPILER}"
    echo "  GPU Arch:      ${CMAKE_GPU_ARCHITECTURES}"
    echo "  Gemv tuning:   ${MFEM_GEMV_TUNING:-AUTO}${MFEM_GEMV_TUNING_LAUNCHER:+ (launcher: ${MFEM_GEMV_TUNING_LAUNCHER})}"
  fi
  echo ""
  echo "MPI Wrappers:"
  echo "  mpicc:         ${MPI_C_COMPILER}"
  echo "  mpicxx:        ${MPI_CXX_COMPILER}"
  echo "  mpifort:       ${MPI_Fortran_COMPILER}"
  echo ""
  echo "Flags:"
  echo "  CXX:           ${CMAKE_CXX_FLAGS}"
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    echo "  GPU:           ${CMAKE_GPU_FLAGS}"
  fi
  echo "  Linker:        ${CMAKE_EXE_LINKER_FLAGS}"
  echo ""
  echo "Key Versions:"
  echo "  CAMP:          ${CAMP_VER}"
  echo "  RAJA:          ${RAJA_VER}"
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    echo "  Umpire:        ${UMPIRE_VER}"
    echo "  CHAI:          ${CHAI_VER}"
  fi
  echo "  Hypre:         ${HYPRE_VER}"
  echo "  MFEM:          ${MFEM_BRANCH}"
  echo "  ExaCMech:      ${EXACMECH_BRANCH}"
  echo "  ExaConstit:    ${EXACONSTIT_BRANCH}"
  echo "=========================================="
}

# Validate configuration before proceeding
validate_configuration() {
  echo "Validating configuration..."
  
  # Check compilers exist
  check_required_paths "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}"
  
  if [ "${BUILD_TYPE}" != "cpu" ]; then
    check_required_paths "${CMAKE_GPU_COMPILER}"
  fi
  
  # Check MPI wrappers
  check_required_paths "${MPI_C_COMPILER}" "${MPI_CXX_COMPILER}" "${MPI_Fortran_COMPILER}"
  
  # Check required commands
  check_required_commands git cmake make curl tar
  
  echo "Configuration validation complete."
}