# Pair LJ/Long Coul/Long Deopt Differential Tests

This directory supports experiments where a human or LLM writes a simpler
implementation of the optimized `lj/long/coul/long` kernels and checks it
against LAMMPS' existing force-style differential tests.

The main entry point is:

```bash
scripts/run_lj_long_coul_long_opt_diff.sh
```

The candidate template is:

```bash
scripts/deopt/pair_lj_long_coul_long_candidate.cpp
```

There is also a known-good harness sample:

```bash
scripts/deopt/pair_lj_long_coul_long_candidate_base_delegate.cpp
```

That sample delegates back to the original base pair style. It is useful for
checking that the deopt adapter and test script are wired correctly, but it is
not a deoptimized implementation.

## What Is Being Compared

The script uses LAMMPS' existing force-style unit-test harness. It does not
perform an ad hoc line-by-line comparison between two pair styles. Instead, all
style variants are compared against the same stored expected force, stress, and
energy data from:

```bash
unittest/force-styles/tests/mol-pair-lj_long_coul_long.yaml
```

That YAML fixture is the oracle. It contains:

- the representative input script name: `in.fourmol`
- the pair style command: `lj/long/coul/long long long 8.0`
- pair coefficients
- KSpace setup for `ewald/disp`
- expected initial forces, stress, and energies
- expected post-run forces, stress, and energies
- restart, data-file, `pair_modify nofdotr`, and r-RESPA coverage values used
  by the harness

The referenced input script is loaded from the force-style input directory by
the test harness. The YAML sets:

```yaml
input_file: in.fourmol
```

and `test_pair_style.cpp` loads it through `INPUT_FOLDER`.

When the script compares the original, OPT, OPENMP, and deopt versions, the
meaning is:

- run each implementation on the same representative system
- compare each implementation's force/stress/energy results to the same YAML
  expected data
- if two implementations both match the same expected data within tolerance,
  they are equivalent for this fixture and harness coverage

So, for example, "plain vs OPT" means both `PairStyle.plain` and
`PairStyle.opt` pass against the same oracle. It is a shared-oracle
differential test, not a direct runtime subtraction between two pair styles.

## Reference Styles

The script checks these references:

```bash
"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.plain:PairStyle.opt -s

"${TEST_BIN}" "${OPT_YAML}" \
  --gtest_filter=PairStyle.plain -s

"${TEST_BIN}" "${BASE_YAML}" \
  --gtest_filter=PairStyle.omp -s

"${TEST_BIN}" "${OMP_YAML}" \
  --gtest_filter=PairStyle.plain -s
```

The variants are:

- `PairStyle.plain` with `BASE_YAML`: unsuffixed
  `lj/long/coul/long`.
- `PairStyle.opt` with `BASE_YAML`: the harness runs LAMMPS with `-sf opt`,
  causing suffix dispatch to `lj/long/coul/long/opt`.
- `PairStyle.plain` with `OPT_YAML`: an explicit generated YAML using
  `pair_style: lj/long/coul/long/opt ...`.
- `PairStyle.omp` with `BASE_YAML`: the harness runs LAMMPS with
  `-pk omp 4 -sf omp`, causing suffix dispatch to
  `lj/long/coul/long/omp`.
- `PairStyle.plain` with `OMP_YAML`: an explicit generated YAML using
  `pair_style: lj/long/coul/long/omp ...` and `package omp 4`.

The generated YAML files are written into the build directory, not the source
tree:

```bash
${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_opt.yaml
${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_omp.yaml
${BUILD_DIR}/mol-pair-lj_long_coul_long_explicit_deopt.yaml
```

## Candidate Mode

Candidate mode is enabled by setting any of:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE
LAMMPS_DEOPT_FUNCTION
LAMMPS_DEOPT_REFERENCE
```

Common usage:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE=scripts/deopt/pair_lj_long_coul_long_candidate.cpp \
LAMMPS_DEOPT_FUNCTION=both \
LAMMPS_DEOPT_REFERENCE=plain \
scripts/run_lj_long_coul_long_opt_diff.sh
```

If `LAMMPS_DEOPT_CANDIDATE_FILE` is not set, the script creates a candidate
template in the build directory:

```bash
${BUILD_DIR}/pair_lj_long_coul_long_candidate.cpp
```

and exits so the file can be implemented before rerunning.

`LAMMPS_DEOPT_REFERENCE` controls which known implementation is sanity-checked
before the candidate:

- `plain`: run only the unsuffixed original reference
- `opt`: run suffix-dispatched OPT and explicit `/opt`
- `omp`: run suffix-dispatched OPENMP and explicit `/omp`
- `all`: run all reference checks

`LAMMPS_DEOPT_FUNCTION` controls which candidate function is active:

- `eval`: route normal `compute()` through candidate `eval()`;
  `compute_outer()` falls back to the base style
- `eval_outer`: route r-RESPA outer through candidate `eval_outer()`;
  normal `compute()` falls back to the base style
- `both`: route both paths through the candidate

After the selected reference checks, candidate mode runs:

```bash
"${TEST_BIN}" "${DEOPT_YAML}" \
  --gtest_filter=PairStyle.plain -s
```

`DEOPT_YAML` is generated from the base YAML by replacing the prerequisite and
pair style with:

```yaml
pair lj/long/coul/long/deopt
pair_style: lj/long/coul/long/deopt long long 8.0
```

## Build Configuration

The script configures a focused CMake build in:

```bash
${LAMMPS_DIFF_BUILD_DIR:-/tmp/lammps-diff-opt}
```

with the packages needed for this fixture and its references:

```bash
PKG_KSPACE=ON
PKG_MOLECULE=ON
PKG_OPT=ON
PKG_OPENMP=ON
BUILD_OMP=ON
```

The build target is:

```bash
cmake --build "${BUILD_DIR}" --target test_pair_style -j "${LAMMPS_DIFF_JOBS:-32}"
```

Use `LAMMPS_DIFF_JOBS` to adjust parallelism:

```bash
LAMMPS_DIFF_JOBS=16 scripts/run_lj_long_coul_long_opt_diff.sh
```

The deopt candidate is passed to CMake with:

```bash
LAMMPS_LJLC_DEOPT_IMPL
LAMMPS_LJLC_DEOPT_ENABLE_EVAL
LAMMPS_LJLC_DEOPT_ENABLE_EVAL_OUTER
```

Those compile definitions are scoped to:

```bash
src/KSPACE/pair_lj_long_coul_long_deopt.cpp
```

so the candidate file is included only by the deopt adapter translation unit.

## Code Path Through The Test Harness

The important test harness file is:

```bash
unittest/force-styles/test_pair_style.cpp
```

The path for a plain or explicit style check is:

1. GoogleTest selects a fixture such as `PairStyle.plain`,
   `PairStyle.opt`, or `PairStyle.omp`.
2. `init_lammps()` creates a LAMMPS instance with the fixture-specific command
   line. Examples:
   - plain: no suffix
   - opt: `-sf opt`
   - omp: `-pk omp 4 -sf omp`
3. The harness checks YAML prerequisites with `Info::has_style()`.
4. It executes YAML `pre_commands`.
5. It loads `input_file`, here `in.fourmol`.
6. It executes the YAML `pair_style` command.
7. It applies YAML `pair_coeff` lines.
8. It applies fixed and YAML `post_commands`, including the table and
   `ewald/disp` settings.
9. It runs `run 0 post no`, writes restart/data/coeff files, and compares
   initial forces, stress, and energies to the YAML oracle.
10. `run_lammps()` adds pair-energy computes, runs four MD steps, and compares
    post-run forces, stress, and energies.
11. The harness also exercises restart, write/read data, `pair_modify nofdotr`,
    and r-RESPA paths where supported by the style.

For explicit `/opt`, `/omp`, and `/deopt` YAML files, the fixture is still
`PairStyle.plain`; the pair style name itself contains the suffix.

For suffix-dispatched OPT and OPENMP, the base YAML still says
`lj/long/coul/long`, but the LAMMPS command line suffix causes LAMMPS to choose
the suffixed implementation when the suffixed style exists.

## Deopt Adapter Code Path

The deopt pair style is registered in:

```bash
src/KSPACE/pair_lj_long_coul_long_deopt.h
```

as:

```cpp
PairStyle(lj/long/coul/long/deopt,PairLJLongCoulLongDeopt);
```

`PairLJLongCoulLongDeopt` inherits from the original KSPACE style:

```cpp
class PairLJLongCoulLongDeopt : public PairLJLongCoulLong
```

That means it reuses the original style's setup, coefficients, restart/data
logic, KSpace coupling, tables, and inherited `compute_inner()` /
`compute_middle()` behavior.

The deopt adapter overrides only:

```cpp
void compute(int, int) override;
void compute_outer(int, int) override;
```

When `LAMMPS_LJLC_DEOPT_ENABLE_EVAL` is off, `compute()` simply calls:

```cpp
PairLJLongCoulLong::compute(eflag, vflag);
```

When it is on, `compute()` initializes energy/virial flags and dispatches to
the candidate `eval()`.

Likewise, when `LAMMPS_LJLC_DEOPT_ENABLE_EVAL_OUTER` is off,
`compute_outer()` calls:

```cpp
PairLJLongCoulLong::compute_outer(eflag, vflag);
```

When it is on, `compute_outer()` dispatches to candidate `eval_outer()`.

At the bottom of the adapter:

```cpp
#ifdef LAMMPS_LJLC_DEOPT_IMPL
#include LAMMPS_LJLC_DEOPT_IMPL
#endif
```

The candidate `.cpp` file is textually included into the adapter translation
unit. This is intentional: the candidate functions are template member
functions, so their definitions must be visible where they are instantiated.

## Candidate Function Signatures

Candidate files implement these two template member functions:

```cpp
template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval();

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval_outer();
```

The template parameters mean:

- `EVFLAG`: energy or virial tallying is active
- `EFLAG`: energy tallying is active
- `NEWTON_PAIR`: `force->newton_pair` is active
- `CTABLE`: Coulomb lookup tables are active
- `LJTABLE`: dispersion/LJ lookup tables are active
- `ORDER1`: first-order Ewald term is enabled
- `ORDER6`: sixth-order dispersion Ewald term is enabled

These are compile-time versions of runtime state in the pair style. The deopt
adapter's dispatch chain converts runtime state into the same template
specialization shape used by OPT.

## How The Implementations Differ

The original unoptimized KSPACE style is:

```bash
src/KSPACE/pair_lj_long_coul_long.cpp
src/KSPACE/pair_lj_long_coul_long.h
```

It does not expose `eval()` or `eval_outer()` as separate target functions.
Its main entry points are:

```cpp
void compute(int eflag, int vflag);
void compute_outer(int eflag, int vflag);
```

The OPT style is:

```bash
src/OPT/pair_lj_long_coul_long_opt.cpp
src/OPT/pair_lj_long_coul_long_opt.h
```

It uses no-argument template kernels:

```cpp
template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void eval();

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void eval_outer();
```

The OPENMP style is:

```bash
src/OPENMP/pair_lj_long_coul_long_omp.cpp
src/OPENMP/pair_lj_long_coul_long_omp.h
```

It uses the same conceptual template specialization dimensions, but its
threaded kernels receive a loop range and thread data:

```cpp
template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void eval(int iifrom, int iito, ThrData *const thr);

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void eval_outer(int iifrom, int iito, ThrData *const thr);
```

The deopt style intentionally uses the OPT-shaped serial signature. This keeps
the candidate file simpler and gives the LLM/human a single implementation
shape even when the selected reference is OPENMP.

## Why `eval` And `eval_outer`

`eval()` is the primary optimized normal pair-loop kernel in OPT and OPENMP.
It is the main target for a deoptimization experiment.

`eval_outer()` is the r-RESPA outer-level kernel. The base YAML and harness
exercise r-RESPA coverage, so a candidate can pass normal `compute()` coverage
while still failing the r-RESPA path. This is why the script allows:

```bash
LAMMPS_DEOPT_FUNCTION=eval
LAMMPS_DEOPT_FUNCTION=eval_outer
LAMMPS_DEOPT_FUNCTION=both
```

The inherited inner and middle r-RESPA paths are not targeted here. They remain
the original base implementation in the deopt style.

## Example Commands

Run the default reference checks only:

```bash
scripts/run_lj_long_coul_long_opt_diff.sh
```

Create a candidate template in the build directory:

```bash
LAMMPS_DEOPT_FUNCTION=both scripts/run_lj_long_coul_long_opt_diff.sh
```

Run a candidate against the original unoptimized reference:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE=scripts/deopt/pair_lj_long_coul_long_candidate.cpp \
LAMMPS_DEOPT_REFERENCE=plain \
LAMMPS_DEOPT_FUNCTION=both \
scripts/run_lj_long_coul_long_opt_diff.sh
```

Run a candidate while checking the OPT reference:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE=scripts/deopt/pair_lj_long_coul_long_candidate.cpp \
LAMMPS_DEOPT_REFERENCE=opt \
LAMMPS_DEOPT_FUNCTION=eval \
scripts/run_lj_long_coul_long_opt_diff.sh
```

Run a candidate while checking the OPENMP reference:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE=scripts/deopt/pair_lj_long_coul_long_candidate.cpp \
LAMMPS_DEOPT_REFERENCE=omp \
LAMMPS_DEOPT_FUNCTION=both \
scripts/run_lj_long_coul_long_opt_diff.sh
```

Run the known-good sample candidate:

```bash
LAMMPS_DEOPT_CANDIDATE_FILE=scripts/deopt/pair_lj_long_coul_long_candidate_base_delegate.cpp \
LAMMPS_DEOPT_REFERENCE=plain \
LAMMPS_DEOPT_FUNCTION=both \
scripts/run_lj_long_coul_long_opt_diff.sh
```
