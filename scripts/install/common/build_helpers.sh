#!/usr/bin/env bash
# Shared helper functions used by every build function.
#
# Kept separate from the build_functions_*.sh files so the per-library
# build logic stays focused on CMake invocations rather than the
# logging / cloning / build-dir-prep plumbing.

###########################################
# Logging wrapper
###########################################
run_with_log() {
  local log="$1"; shift
  "$@" |& tee "$log"
}

###########################################
# Clone repository only if missing, initialize submodules on first clone
###########################################
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

###########################################
# Optional: force submodule sync when explicitly requested
###########################################
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

###########################################
# Respect REBUILD flag when preparing build directories
###########################################
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
