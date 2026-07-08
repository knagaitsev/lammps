#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-eam}"
TEST_BIN="${BUILD_DIR}/test_pair_style"
BASE_YAML="${ROOT_DIR}/unittest/force-styles/tests/atomic-pair-eam.yaml"
OMP_YAML="${BUILD_DIR}/atomic-pair-eam_explicit_omp.yaml"

configure_build() {
  # Configure a focused unit-test build with the packages needed by the base and OPENMP EAM styles.
  cmake -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D ENABLE_TESTING=ON \
    -D PKG_MANYBODY=ON \
    -D PKG_OPENMP=ON \
    -D DOWNLOAD_POTENTIALS=OFF \
    -D BUILD_MPI=OFF \
    -D BUILD_OMP=ON \
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
  cache_has_enabled_bool PKG_OPENMP || return 0
  cache_has_enabled_bool BUILD_OMP || return 0
  return 1
}

make_explicit_omp_yaml() {
  cp "${BASE_YAML}" "${OMP_YAML}"
  # Convert the base test case to request the explicit /omp style and required package setup.
  perl -0pi \
    -e 's/pair eam\n/pair eam\/omp\n/; s/pre_commands: ! \|\n  variable units index metal/pre_commands: ! |\n  package omp 4\n  variable units index metal/; s/pair_style: eam\n/pair_style: eam\/omp\n/' \
    "${OMP_YAML}"
}

if needs_configure; then
  configure_build
fi

cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-8}"

# Generate temporary YAML variants inside the build tree so source fixtures stay unchanged.
make_explicit_omp_yaml

export LAMMPS_POTENTIALS="${LAMMPS_POTENTIALS:-${ROOT_DIR}/potentials}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"

# Check the original EAM style against the base fixture.
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.plain -s

# Check OPENMP suffix dispatch through the harness' -pk omp 4 -sf omp path.
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.omp -s

# Check that the explicit /omp style name works with the generated fixture.
"${TEST_BIN}" "${OMP_YAML}" \
  --gtest_filter=PairStyle.plain -s
