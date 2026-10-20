#!/usr/bin/env bash
# ExaConstit CUDA build (Linux workstation config)

set -Eeuo pipefail
trap 'echo "Build failed at line $LINENO while running: $BASH_COMMAND" >&2' ERR

# Resolve script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source common infrastructure
source "${SCRIPT_DIR}/common/dependency_versions.sh"
source "${SCRIPT_DIR}/common/preflight_checks.sh"
source "${SCRIPT_DIR}/common/build_functions.sh"

# Resolve BASE_DIR and change to it
resolve_base_dir

# Source configuration
source "${SCRIPT_DIR}/configs/gpu_cuda_workstation_config.sh"

# User-controllable options
export REBUILD="${REBUILD:-OFF}"
export SYNC_SUBMODULES="${SYNC_SUBMODULES:-OFF}"

# Validate and summarize
validate_configuration
print_build_summary

# Build everything
build_all_dependencies

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="