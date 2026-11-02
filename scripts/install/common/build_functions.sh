#!/usr/bin/env bash
# Meta-loader for the ExaConstit build functions.
#
# The build logic was split into a helpers file and three layer files
# grouped by dependency tier; this file simply sources them in
# dependency order so existing entry-point scripts (unix_*_install.sh)
# keep working unchanged.
#
#   build_helpers.sh               Shared helper functions
#                                  (run_with_log, clone_if_missing,
#                                  sync_submodules, prepare_build_dir).
#   build_functions_common.sh      BLT, CAMP, RAJA, Umpire, CHAI -- the
#                                  shared portability stack.
#   build_functions_mfem.sh        Hypre, METIS, MFEM -- the FEM stack.
#   build_functions_exaconstit.sh  SNLS, ExaCMech, Axom, ExaConstit,
#                                  plus the build_all_dependencies
#                                  orchestrator.

# Resolve our own location so each file sources its sibling.
_BUILD_FUNCTIONS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "${_BUILD_FUNCTIONS_DIR}/build_helpers.sh"
source "${_BUILD_FUNCTIONS_DIR}/build_functions_common.sh"
source "${_BUILD_FUNCTIONS_DIR}/build_functions_mfem.sh"
source "${_BUILD_FUNCTIONS_DIR}/build_functions_exaconstit.sh"

unset _BUILD_FUNCTIONS_DIR
