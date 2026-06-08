#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-opt}"
TEST_BIN="${BUILD_DIR}/test_pair_style"
BASE_YAML="${ROOT_DIR}/unittest/force-styles/tests/mol-pair-lj_long_coul_long.yaml"
OPT_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_opt.yaml"
OMP_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_omp.yaml"

configure_build() {
  # Configure a focused unit-test build with the packages needed by the base, OPT, and OPENMP styles.
  cmake -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D ENABLE_TESTING=ON \
    -D PKG_KSPACE=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_OPT=ON \
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
  # Reconfigure stale build trees that predate the OPENMP additions.
  [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]] && return 0
  cache_has_enabled_bool PKG_OPENMP || return 0
  cache_has_enabled_bool BUILD_OMP || return 0
  return 1
}

make_explicit_opt_yaml() {
  cp "${BASE_YAML}" "${OPT_YAML}"
  # Convert the base test case to request the explicit /opt style name.
  perl -0pi \
    -e 's/pair lj\/long\/coul\/long\n/pair lj\/long\/coul\/long\/opt\n/; s/pair_style: lj\/long\/coul\/long /pair_style: lj\/long\/coul\/long\/opt /' \
    "${OPT_YAML}"
}

make_explicit_omp_yaml() {
  cp "${BASE_YAML}" "${OMP_YAML}"
  # Convert the base test case to request the explicit /omp style and required package setup.
  perl -0pi \
    -e 's/pair lj\/long\/coul\/long\n/pair lj\/long\/coul\/long\/omp\n/; s/pre_commands: ! ""/pre_commands: ! |\n  package omp 4/; s/pair_style: lj\/long\/coul\/long /pair_style: lj\/long\/coul\/long\/omp /' \
    "${OMP_YAML}"
}

if needs_configure; then
  configure_build
fi

cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-8}"

# Generate temporary YAML variants inside the build tree so source fixtures stay unchanged.
make_explicit_opt_yaml
make_explicit_omp_yaml

# Check the original style and the suffix-dispatched OPT style against the base fixture.
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.plain:PairStyle.opt -s

# Check that the explicit /opt style name matches the base fixture expectations.
"${TEST_BIN}" "${OPT_YAML}" \
  --gtest_filter=PairStyle.plain -s

# Check OPENMP suffix dispatch through the harness' -pk omp 4 -sf omp path.
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.omp -s

# Check that the explicit /omp style name works with the generated fixture.
"${TEST_BIN}" "${OMP_YAML}" \
  --gtest_filter=PairStyle.plain -s
