#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-eam}"
TEST_BIN="${BUILD_DIR}/test_pair_style"
BASE_YAML="${ROOT_DIR}/unittest/force-styles/tests/atomic-pair-eam.yaml"

# NOTE: the base `eam` style now uses a full neighbor list, which is incompatible
# with the `eam/omp` style (it shares PairEAM but keeps a half-list compute), so
# the OPENMP checks are intentionally out of scope here - only the plain `eam`
# style is validated against the golden fixture.

configure_build() {
  # Configure a focused unit-test build with the packages needed by the EAM style.
  cmake -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D ENABLE_TESTING=ON \
    -D PKG_MANYBODY=ON \
    -D DOWNLOAD_POTENTIALS=OFF \
    -D BUILD_MPI=OFF \
    -D BUILD_SHARED_LIBS=ON
}

cache_has_enabled_bool() {
  local name="$1"
  # Treat common CMake true values as enabled when checking an existing cache.
  grep -Eq "^${name}:BOOL=(ON|TRUE|1)$" "${BUILD_DIR}/CMakeCache.txt"
}

needs_configure() {
  [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]] && return 0
  [[ ! -f "${BUILD_DIR}/Makefile" ]] && return 0
  cache_has_enabled_bool PKG_MANYBODY || return 0
  return 1
}

if needs_configure; then
  configure_build
fi

cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-8}"

export LAMMPS_POTENTIALS="${LAMMPS_POTENTIALS:-${ROOT_DIR}/potentials}"

# Check the (full-list) EAM style against the golden fixture.
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.plain -s
