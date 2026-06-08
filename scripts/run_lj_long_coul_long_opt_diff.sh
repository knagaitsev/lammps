#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-opt}"
TEST_BIN="${BUILD_DIR}/test_pair_style"
BASE_YAML="${ROOT_DIR}/unittest/force-styles/tests/mol-pair-lj_long_coul_long.yaml"
OPT_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_opt.yaml"

configure_build() {
  cmake -S "${ROOT_DIR}/cmake" -B "${BUILD_DIR}" \
    -D CMAKE_BUILD_TYPE=Release \
    -D ENABLE_TESTING=ON \
    -D PKG_KSPACE=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_OPT=ON \
    -D DOWNLOAD_POTENTIALS=OFF \
    -D BUILD_MPI=OFF \
    -D BUILD_SHARED_LIBS=ON
}

make_explicit_opt_yaml() {
  cp "${BASE_YAML}" "${OPT_YAML}"
  perl -0pi \
    -e 's/pair lj\/long\/coul\/long\n/pair lj\/long\/coul\/long\/opt\n/; s/pair_style: lj\/long\/coul\/long /pair_style: lj\/long\/coul\/long\/opt /' \
    "${OPT_YAML}"
}

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  configure_build
fi

cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-8}"

make_explicit_opt_yaml

"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.plain:PairStyle.opt -s

"${TEST_BIN}" "${OPT_YAML}" \
  --gtest_filter=PairStyle.plain -s
