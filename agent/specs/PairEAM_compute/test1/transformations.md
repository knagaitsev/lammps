# Functional extraction of `PairEAM::compute` — transformation log

Branch: `functional-eam-extraction` (from `cs2-test1`).
Target file: `src/MANYBODY/pair_eam.cpp` (the *only* file changed).

---

## T0. Baseline and the correctness envelope

`scripts/run_pair_eam_diff.sh` builds `test_pair_style` (CMake `Release`, `PKG_MANYBODY=ON`,
`PKG_OPENMP=ON`, **`BUILD_MPI=OFF`**, `BUILD_OMP=ON`) and runs three gtest cases against
`unittest/force-styles/tests/atomic-pair-eam.yaml`:

| invocation | style exercised | uses `PairEAM::compute`? |
|---|---|---|
| `--gtest_filter=PairStyle.plain` on the base YAML | `pair_style eam` | **yes** |
| `--gtest_filter=PairStyle.omp` on the base YAML (`-sf omp`) | `eam/omp` | no (`PairEAMOMP::compute`) |
| `--gtest_filter=PairStyle.plain` on the generated `/omp` YAML | `eam/omp` | no |

**Configuration that must keep working:** `pair_style eam` (funcfl `Al_jnp.eam` + `Cu_u3.eam`,
2 types, 32 atoms), single rank (no MPI at all — `BUILD_MPI=OFF`), and **both**
`newton_pair on` and `newton_pair off`: `PairStyle.plain` re-runs the whole fixture with
`newton off` (test_pair_style.cpp:410-438), plus `restart`, `nofdotr` (i.e. `vflag_fdotr == 0`,
so `ev_tally` must produce the virial itself) and `data` variants. `skip_tests: single` — the
`single()` path is not exercised, but `numforce[]`, which `single()` reads, is still reproduced
exactly.

**Tolerance:** `epsilon: 6e-12` in the YAML, applied as a relative comparison
(`5*epsilon` for `run_forces`). Baseline (unmodified code) relative errors vs. the stored
reference are ≤ 1.0e-13, i.e. there are ~60x of headroom. This is the reason accumulation order
may be reassociated (T4/T6): a different-but-equivalent summation order perturbs results at the
1e-16 relative level, far below the tolerance. Measured final errors (see T12) are
indistinguishable from the baseline, so no order-preservation fallback was needed.

**Consequence for scope:** nothing is dropped and no runtime guard was needed. The refactor
handles `newton_pair` on *and* off, keeps the (inert, single-rank) `reverse_comm`/`forward_comm`
calls in place as structural glue, and the derived styles `eam/alloy` and `eam/fs` — which
inherit this `compute()` and differ only in how the spline tables are filled — continue to work
unchanged, because the tables are consumed through the same pure lookups.

---

## T1. The math being extracted (energy → energy + gradient)

The original prompt named only the EAM energy

```
E_i = F_a( rho_bar_i ) + 1/2 sum_{j != i} phi_ab(r_ij),     rho_bar_i = sum_{j != i} rho_b(r_ij)
```

but `PairEAM::compute` also produces the analytic gradient

```
f_i = - grad_{r_i} E_tot = - sum_{j != i} [ F'_a(rho_bar_i) rho_b'(r_ij)
                                          + F'_b(rho_bar_j) rho_a'(r_ij)
                                          + phi_ab'(r_ij) ] rhat_ij
```

(in the source: `psip = fp[i]*rhojp + fp[j]*rhoip + phip`, `fpair = -scale*psip/r`). Per PLAN.md
this pivot — extracting *energy and its gradient* as one core — is the authorized target and is
what is implemented. `f_i` depends on `F'(rho_bar_j)` of the **neighbours**, so the core cannot
be a single map; it is necessarily **two stages** separated by a global barrier over the
`rho_bar`/`fp` arrays.

Final functional form as implemented:

```
rho_bar   = map (\i -> reduce (+) 0 (map (\j -> rho (type j) (type i) (dist i j))
                                         (filter (within_cutoff i) (N i))))
                atoms
(E_F, fp) = unzip (map (\i -> F_embed (type i) (rho_bar i)) owned)
force     = map (\i -> reduce (+) 0 (map (\j -> pair_force i j fp)
                                         (filter (within_cutoff i) (N i))))
                atoms
contribs  = map (pair_contrib fp) (filter within_cutoff half_pairs)
```

---

## T2. Table state → `EamTables` value (purity of the physics functions)

**Rewrite.** The spline tables and scalars (`rdr, rdrho, cutforcesq, rhomax, nr, nrho,
rhor_spline, frho_spline, z2r_spline, type2rhor, type2z2r, type2frho, scale`) are `PairEAM`
members and are invisible to anonymous-namespace functions.

**Chosen option (of the two offered in PLAN.md): the `struct EamTables` bundle.** SETUP copies
the raw pointers/values into an `EamTables` value; every pure function takes
`const EamTables &` as its first argument. No `friend`, no globals, no lambda-capture of `this`.

**Justification.** The tables are written only by `coeff()`/`array2spline()` and are immutable
for the whole duration of `compute()`; a function of `(EamTables, types, r)` is therefore
referentially transparent — same arguments always yield the same result, and nothing is mutated.

---

## T3. Neighbour-list marshalling: half list → per-atom neighbour sets

**Rewrite (SETUP).** The LAMMPS half list is first flattened, in the exact visit order of the
original nested loop, into `std::vector<PairRef> half_pairs` (`NEIGHMASK` stripped). Then

```
for (i,j) in half_pairs:  N(i) += {j}
                          if (newton_pair || j < nlocal):  N(j) += {i}
```

**Why this is semantics-preserving.** The original performs exactly two scatter operations per
half-list entry — one unconditionally onto `i`, one onto `j` guarded by
`(newton_pair || j < nlocal)` — for *both* the density accumulation (`rho[i]` / `rho[j]`) and
the force accumulation (`f[i] +=` / `f[j] -=`). Reversing the direction of an edge and letting
atom `j` gather it is the same set of contributions, because both contributions are functions of
the unordered pair only:

* density: `rho[i] += rho_{tj->ti}(r)` and `rho[j] += rho_{ti->tj}(r)` — the gathered form
  `rho(type_src=type[j], type_dst=type[i], r)` reproduces each side verbatim;
* force: `psip` is invariant under `i<->j` (it swaps `rhoip <-> rhojp`, and `scale`, `type2z2r`
  are symmetric — `pair_eam.cpp:522` sets `scale[j][i] = scale[i][j]`), and `del` flips sign, so
  the gathered contribution on `j` is bit-for-bit the `-delx*fpair` the original scattered.
  `r` is recomputed from `(-delx)^2 + ... == delx^2 + ...`, i.e. bitwise identical.

Carrying the guard `(newton_pair || j < nlocal)` into the *edge insertion* is what makes the
single core body correct for both newton settings: with `newton off` no reverse edge is created
for a ghost `j`, so ghost atoms end up with an empty `N(j)`, zero force and (unused) zero
density — exactly the original behaviour, and the property `virial_fdotr_compute()` relies on
when it sums `f·x` over ghosts.

The result is *not* a symmetric graph in the `newton off` case, and deliberately so: `N(i)` is
defined as "the neighbours whose contribution to atom `i` the original code applied", which is
the mathematical `sum_{j != i}` for every atom that owns a force.

---

## T4. Density loop → `map . reduce . filter` (`eam_core_density`)

**Rewrite.** The nested scatter loop

```c
for ii: for jj: if (rsq < cutforcesq) { rho[i] += ...; if (...) rho[j] += ...; }
```

becomes a gather

```c
fmap([&](int i){ return freduce(plus, 0.0,
                    fmap(rho_term_i, ffilter(within_cutoff_i, N(i)))); },
     findices(nall))
```

**Justification.** After T3 each output element `rho_bar[i]` is a sum over a disjoint,
independently computable input set; a loop whose iterations write disjoint outputs *is* a `map`,
and an accumulation over one atom's neighbours *is* a `reduce`. The only difference from the
original is the **order of summation** of the terms that make up a given `rho_bar[i]` (gathered
in neighbour order rather than interleaved in half-list order). Floating-point addition is
commutative but not associative, so this is a reassociation justified by T0's tolerance budget,
not an identity. Measured effect: below 1e-15 relative.

**Cutoff choice (PLAN.md item 2):** the explicit `ffilter(within_cutoff, ...)` form was chosen
over "make `rho_of_r` return 0 beyond the cutoff". Reason: it is closer to the original control
flow (the tables are *not* zero past `cutforcesq`, they are simply not consulted — evaluating
them there would index `m = MIN(m, nr-1)` off the end of the meaningful range), and it keeps the
`sum_{j != i}` restriction visible in the functional form.

---

## T5. Spline lookups → pure functions

`r_index` / `rho_index` reproduce the original index arithmetic verbatim, including the
asymmetric clamps (`m = MIN(m, nr-1)` for the r-tables, `m = MAX(1, MIN(m, nrho-1))` for the
rho-table, `p = MIN(p, 1.0)`), and `spline_value` / `spline_deriv` are the two cubic evaluations
`((c3 p + c4) p + c5) p + c6` and `(c0 p + c1) p + c2`.

On top of them:

* `rho_of_r(t, src, dst, r)`, `drho_of_r(t, src, dst, r)` — `rhor_spline[type2rhor[src][dst]]`.
* `z2_of_r`, `dz2_of_r` — the tables store **`z2 = r * phi`**, not `phi`.
* `phi_of_r  = z2(r) * (1/r)` and
  `dphi_of_r = z2'(r) * (1/r) - phi(r) * (1/r)`, matching `phip = z2p*recip - phi*recip`
  operand-for-operand (PLAN.md item 4).
* `pair_energy = scale_ab * phi_ab(r)`.

**`scale` placement (PLAN.md item 6).** `phi_of_r`/`dphi_of_r` expose the *unscaled* pair
function; `scale[itype][jtype]` is applied by `pair_energy` and by `pair_fpair` at precisely the
point the original applies it (`evdwl = scale*phi`, `fpair = -scale*psip*recip`). Folding
`scale` into `dphi_of_r` instead would change `scale*(a+b+phip)` into `scale*a + scale*b +
scale*phip`, a needless perturbation; since `scale` is symmetric and constant this is the same
mathematical parameterisation, just evaluated in the original order. The diagonal factor
`scale[t][t]` is folded into `F_embed`.

---

## T6. Embedding loop → `map F_embed` (`eam_core_embedding`)

**Rewrite.** The second original loop becomes `fmap(\i -> F_embed(t, type i, rho_bar i), owned)`
where `owned` is `ilist[0..inum)` as a plain vector.

`F_embed` returns `Embed{energy, deriv, beyond_rhomax}` and folds in the rhomax linear
extrapolation (PLAN.md item 5):

```
F' = spline_deriv(c, p)
F  = spline_value(c, p) + (rho > rhomax ? F' * (rho - rhomax) : 0)
F *= scale[t][t]
```

Note the original computes the extrapolation with the **unscaled** `fp[i]` and scales afterwards;
`F_embed` does the same, and returns `deriv` unscaled because `fp[]` is unscaled in the original.
The `beyond_rhomax` flag is returned as data instead of being written to a captured variable;
the `MPI_Allreduce` + warning stays in TEARDOWN.

The original conditioned the energy branch on `eflag`; the pure function always computes it
(it is cheap and side-effect free) and TEARDOWN applies the `eflag`/`eflag_global`/`eflag_atom`
guards. Identical observable behaviour.

---

## T7. Force loop → two pure maps (`eam_core_forces`)

Stage 2 produces two independent results from the same inputs:

1. `force = map (\i -> reduce vadd 0 (map (pair_force_vec i) (filter within_cutoff (N i)))) atoms`
   — the *gathered* form of the original `f[i] += ; f[j] -=` scatter, valid by T3.
   `pair_force_vec i j = (x_i - x_j) * pair_fpair(...)` with
   `pair_fpair = -scale_ab * (fp_i * rho_b'(r) + fp_j * rho_a'(r) + phi_ab'(r)) * (1/r)`.
2. `pair_contribs = map pair_contrib (filter within_cutoff half_pairs)` — one record per
   *half-list* pair, in the original visit order, carrying `(i, j, evdwl, fpair, delx, dely, delz)`.

Producing both means each pair's scalars are evaluated more than once. That is deliberate: it
keeps the force map a pure gather while giving TEARDOWN everything it needs to call `ev_tally`
with **identical arguments in identical order** — which is what preserves the `nofdotr` virial
and the per-atom energy split bit-for-bit (`ev_tally` itself decides the newton/`j < nlocal`
halving). Performance cost is accepted per PLAN.md.

---

## T8. Comm calls kept as glue between the stages (PLAN.md item 3)

`compute()` is now literally

```
SETUP
  -> eam_core_density              (pure)
  -> [glue] rho[] <- rho_bar ; if (newton_pair) comm->reverse_comm(this)
  -> eam_core_embedding            (pure)
  -> [teardown-a] fp[] <- deriv ; eng_vdwl/eatom += energy
  -> [glue] comm->forward_comm(this) ; embedstep = update->ntimestep
  -> eam_core_forces               (pure)
  -> TEARDOWN
```

The reverse/forward comm calls must sit *between* the stages (the embedding may only be
evaluated on fully summed densities, and stage 2 needs neighbour `fp` values), so each stage is
pure and the comm is setup/teardown for the following stage. The build under test is
`BUILD_MPI=OFF`, so these are identity-like glue here; they are retained unchanged for
structural fidelity and so multi-rank behaviour is not silently broken.

`embedstep` and the `numforce[]` bookkeeping stay in the glue/teardown, as specified.

---

## T9. Accumulation → TEARDOWN

Everything that writes LAMMPS state is now outside the core:

* `f[i] += force[i]` for all `nall` atoms;
* `numforce[i]` recomputed as the number of surviving `pair_contribs` with `i` as first member —
  identical to the original `++numforce[i]` inside the cutoff branch of `i`'s own half list;
* `ev_tally(c.i, c.j, nlocal, newton_pair, eflag ? c.evdwl : 0.0, 0.0, c.fpair, c.delx, c.dely,
  c.delz)` per contribution, in original order. The `eflag ? ... : 0.0` reproduces the original's
  `evdwl` (initialised to `0.0` and only assigned under `if (eflag)`);
* the `beyond_rhomax` `MPI_Allreduce` + warning, and `virial_fdotr_compute()`.

---

## T10. Combinators

`fmap`, `freduce`, `ffilter` are generic file-local templates over `std::vector`, plus
`findices(n)` which materialises the index domain `[0,n)` that the outer maps run over. **These
four functions contain the only loops in the functional core**; no loop remains in any physics
function. Loops that remain in `compute()` itself are exclusively SETUP marshalling and TEARDOWN
scattering (list flattening, `AtomView` construction, writing `f`, `fp`, `numforce`, `ev_tally`).

---

## T11. Changes outside `compute()` — complete list

* **`src/MANYBODY/pair_eam.cpp`**: added `#include <utility>` and `#include <vector>`; added the
  anonymous namespace (combinators, `EamTables`, pure physics, core stages) immediately above
  `PairEAM::compute`.
* **Nothing else.** `pair_eam.h` is untouched, `init_style()` is untouched (the neighbour request
  stays a half list — symmetrisation happens entirely in SETUP, per PLAN.md item 1), no new
  files, no build-system changes, no reformatting of unrelated code. `compute_atomic_energy`,
  `single`, and the comm pack/unpack methods are unchanged.
* **No dropped configuration paths, hence no runtime guards.** Both newton settings, `eflag`/
  `vflag` combinations, `vflag_fdotr` on and off, and the derived `eam/alloy`, `eam/fs` styles
  keep working.

---

## T12. Verification

`./scripts/run_pair_eam_diff.sh` — all three gtest cases pass. `PairStyle.plain` (the case that
exercises `PairEAM::compute`) relative errors, refactored vs. the stored reference:

```
init_forces (newton on)  MaxErr 5.582e-15    init_forces (newton off) MaxErr 6.963e-15
init_stress (newton on)  MaxErr 9.672e-14    init_stress (newton off) MaxErr 6.959e-15
init_energy (newton on)  MaxErr 9.253e-16    init_energy (newton off) MaxErr 1.542e-16
run_forces  (newton on)  MaxErr 8.262e-15    run_forces  (newton off) MaxErr 2.419e-14
run_stress  (newton on)  MaxErr 2.837e-14    run_stress  (newton off) MaxErr 4.808e-14
restart/nofdotr/data     MaxErr <= 9.552e-14
```

versus tolerance `6e-12` (`3e-11` for `run_forces`). The baseline numbers before the refactor are
the same to within a few units in the last digit (e.g. `init_stress (newton on)` 9.672e-14 both
before and after), confirming that the reassociation introduced by the gather form is not the
dominant error term.
