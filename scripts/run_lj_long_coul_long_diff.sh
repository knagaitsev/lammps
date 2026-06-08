#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-opt}"
TEST_BIN="${BUILD_DIR}/test_pair_style"
BASE_YAML="${ROOT_DIR}/unittest/force-styles/tests/mol-pair-lj_long_coul_long.yaml"
OPT_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_opt.yaml"
OMP_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_omp.yaml"
DEOPT_TEMPLATE="${ROOT_DIR}/scripts/deopt/pair_lj_long_coul_long_candidate.cpp"
DEOPT_YAML="${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_deopt.yaml"
DEOPT_FUNCTION="${LAMMPS_DEOPT_FUNCTION:-both}"
DEOPT_REFERENCE="${LAMMPS_DEOPT_REFERENCE:-all}"
DEOPT_REQUESTED=0
DEOPT_CANDIDATE_FILE="${LAMMPS_DEOPT_CANDIDATE_FILE:-}"
DEOPT_ENABLE_EVAL=OFF
DEOPT_ENABLE_EVAL_OUTER=OFF

if [[ -n "${LAMMPS_DEOPT_CANDIDATE_FILE:-}" || -n "${LAMMPS_DEOPT_FUNCTION:-}" || -n "${LAMMPS_DEOPT_REFERENCE:-}" ]]; then
  DEOPT_REQUESTED=1
fi

validate_deopt_options() {
  case "${DEOPT_FUNCTION}" in
    eval)
      DEOPT_ENABLE_EVAL=ON
      ;;
    eval_outer)
      DEOPT_ENABLE_EVAL_OUTER=ON
      ;;
    both)
      DEOPT_ENABLE_EVAL=ON
      DEOPT_ENABLE_EVAL_OUTER=ON
      ;;
    *)
      echo "Unsupported LAMMPS_DEOPT_FUNCTION='${DEOPT_FUNCTION}'." >&2
      echo "Use one of: eval, eval_outer, both." >&2
      exit 2
      ;;
  esac

  case "${DEOPT_REFERENCE}" in
    plain|opt|omp|all) ;;
    *)
      echo "Unsupported LAMMPS_DEOPT_REFERENCE='${DEOPT_REFERENCE}'." >&2
      echo "Use one of: plain, opt, omp, all." >&2
      exit 2
      ;;
  esac
}

absolute_path() {
  local path="$1"
  local dir
  local base
  dir="$(dirname "${path}")"
  base="$(basename "${path}")"
  mkdir -p "${dir}"
  printf '%s/%s\n' "$(cd "${dir}" && pwd)" "${base}"
}

prepare_deopt_candidate() {
  [[ "${DEOPT_REQUESTED}" -eq 1 ]] || return 0

  if [[ -z "${DEOPT_CANDIDATE_FILE}" ]]; then
    DEOPT_CANDIDATE_FILE="${BUILD_DIR}/pair_lj_long_coul_long_candidate.cpp"
  fi

  DEOPT_CANDIDATE_FILE="$(absolute_path "${DEOPT_CANDIDATE_FILE}")"

  if [[ ! -f "${DEOPT_CANDIDATE_FILE}" ]]; then
    cp "${DEOPT_TEMPLATE}" "${DEOPT_CANDIDATE_FILE}"
    echo "Created candidate template: ${DEOPT_CANDIDATE_FILE}" >&2
    echo "Implement the requested function(s), then rerun this script." >&2
    exit 2
  fi
}

configure_build() {
  # Configure a focused unit-test build with the packages and optional candidate adapter toggles.
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
    -D BUILD_SHARED_LIBS=ON \
    -D LAMMPS_LJLC_DEOPT_IMPL="${DEOPT_CANDIDATE_FILE}" \
    -D LAMMPS_LJLC_DEOPT_ENABLE_EVAL="${DEOPT_ENABLE_EVAL}" \
    -D LAMMPS_LJLC_DEOPT_ENABLE_EVAL_OUTER="${DEOPT_ENABLE_EVAL_OUTER}"
}

cache_has_enabled_bool() {
  local name="$1"
  # Treat common CMake true values as enabled when checking an existing cache.
  grep -Eq "^${name}:BOOL=(ON|TRUE|1)$" "${BUILD_DIR}/CMakeCache.txt"
}

cache_bool_matches() {
  local name="$1"
  local expected="$2"
  if [[ "${expected}" == "ON" ]]; then
    cache_has_enabled_bool "${name}"
  else
    ! cache_has_enabled_bool "${name}"
  fi
}

cache_value() {
  local name="$1"
  sed -n "s/^${name}:[^=]*=//p" "${BUILD_DIR}/CMakeCache.txt"
}

needs_configure() {
  # Reconfigure stale build trees when required packages or candidate toggles changed.
  [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]] && return 0
  cache_has_enabled_bool PKG_OPENMP || return 0
  cache_has_enabled_bool BUILD_OMP || return 0
  cache_bool_matches LAMMPS_LJLC_DEOPT_ENABLE_EVAL "${DEOPT_ENABLE_EVAL}" || return 0
  cache_bool_matches LAMMPS_LJLC_DEOPT_ENABLE_EVAL_OUTER "${DEOPT_ENABLE_EVAL_OUTER}" || return 0
  [[ "$(cache_value LAMMPS_LJLC_DEOPT_IMPL)" == "${DEOPT_CANDIDATE_FILE}" ]] || return 0
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

make_explicit_deopt_yaml() {
  cp "${BASE_YAML}" "${DEOPT_YAML}"
  # Convert the base test case to request the candidate /deopt style name.
  perl -0pi \
    -e 's/pair lj\/long\/coul\/long\n/pair lj\/long\/coul\/long\/deopt\n/; s/pair_style: lj\/long\/coul\/long /pair_style: lj\/long\/coul\/long\/deopt /' \
    "${DEOPT_YAML}"
}

run_reference_checks() {
  case "${DEOPT_REFERENCE}" in
    plain)
      # Check only the original unsuffixed style before running the candidate.
      "${TEST_BIN}" "${BASE_YAML}" \
        --gtest_filter=PairStyle.plain -s
      ;;
    opt)
      # Check suffix-dispatched OPT plus the explicit /opt style before running the candidate.
      "${TEST_BIN}" "${BASE_YAML}" \
        --gtest_filter=PairStyle.opt -s
      "${TEST_BIN}" "${OPT_YAML}" \
        --gtest_filter=PairStyle.plain -s
      ;;
    omp)
      # Check suffix-dispatched OPENMP plus the explicit /omp style before running the candidate.
      "${TEST_BIN}" "${BASE_YAML}" \
        --gtest_filter=PairStyle.omp -s
      "${TEST_BIN}" "${OMP_YAML}" \
        --gtest_filter=PairStyle.plain -s
      ;;
    all)
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
      ;;
  esac
}

if [[ "${DEOPT_REQUESTED}" -eq 1 ]]; then
  validate_deopt_options
fi
prepare_deopt_candidate

if needs_configure; then
  configure_build
fi

cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-32}"

# Generate temporary YAML variants inside the build tree so source fixtures stay unchanged.
make_explicit_opt_yaml
make_explicit_omp_yaml
make_explicit_deopt_yaml

# Note: reference checks are not actually needed, as the test compares against an existing output file.
# The original input file is: unittest/force-styles/tests/mol-pair-lj_long_coul_long.yaml (referencing `in.fourmol` in the same dir)

# run_reference_checks

if [[ "${DEOPT_REQUESTED}" -eq 1 ]]; then
  # Check the candidate style against the same fixture expectations as the selected reference.
  "${TEST_BIN}" "${DEOPT_YAML}" \
    --gtest_filter=PairStyle.plain -s
fi
