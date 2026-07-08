# Simplifying `PairEAM::compute` to a functional form

Target function: `PairEAM::compute` in `src/MANYBODY/pair_eam.cpp`
Differential test: `scripts/run_pair_eam_diff.sh` (oracle
`unittest/force-styles/tests/atomic-pair-eam.yaml`, relative epsilon `6e-12`).

## Goal

Rewrite `PairEAM::compute` so that its body reads as *setup → functional core →
cleanup*, with the embedded-atom-model core math expressed through generic
`map` / `filter` / `reduce` combinators and pure functions. The EAM per-atom
energy (Wikipedia) is

```
E_i = F_a( sum_{j != i} rho_b(r_ij) )  +  (1/2) sum_{j != i} phi_ab(r_ij)
```

with the target functional shape

```
map( F( reduce (+) (map rho (map distance (filter not_curr_atom all_atoms))) )
     + (1/2) * reduce (+) (map phi (map distance (filter not_curr_atom all_atoms))),
     all_atoms )
```

The rewrite must remain executable and bit-faithful enough to pass the
differential test. It **does not** change `init_style` (the neighbor-list
request), because `eam/omp` inherits it and assumes a *half* list; the two omp
checks in the diff script must keep passing.

Every transformation below is behavior-preserving. The overriding invariant is
**preservation of floating-point accumulation order**: with a half neighbor
list each per-atom accumulator (`rho[i]`, `f[i]`, `eng_vdwl`) is assembled from
the same additions in the same order as the original loops, so results are
bit-identical, not merely within tolerance. (Measured max relative error vs.
the oracle: ~1e-14, well under the 6e-12 budget.)

---

# Transformation 1 — Extract pure spline primitives

The original inlines two Horner polynomials and an index computation in three
places (density, embedding, force). Factor them into pure functions:

- `eam_spline_value(coeff, p)` = `((c3*p + c4)*p + c5)*p + c6` — the cubic
  **value** of a tabulated function in a bin at fractional offset `p`.
- `eam_spline_derivative(coeff, p)` = `(c0*p + c1)*p + c2` — the **derivative**
  (the spline coefficients already carry the `1/grid_spacing` factor).
- `eam_radial_spline_point(r, rdr, nr)` and
  `eam_density_spline_point(rho, rdrho, nrho)` — locate an argument on the `r`
  or `rho` grid, returning a `SplinePoint{index, fraction}`. These reproduce the
  exact clamp arithmetic (`m = MIN(m, n-1)`, and the extra `MAX(1, …)` on the
  density grid).

These are literal extractions of existing expressions, so they are value-exact.

# Transformation 2 — Extract pure EAM physics functions

Name the physics on top of the spline primitives:

- `eam_density_value(source_type, target_type, SplinePoint)` = `rho` — electron
  density deposited by a `source_type` atom onto a `target_type` atom.
- `eam_embedding_energy(type, SplinePoint)` = `F`, and
  `eam_embedding_derivative(type, SplinePoint)` = `F'`.
- `eam_pair_spline_terms(...)` bundles the four raw pair quantities
  (`rho'_(i->j)`, `rho'_(j->i)`, `z2 = r*phi`, `z2'`).
- `eam_pair_force_energy(terms, fp_i, fp_j, r, scale)` computes the per-pair
  scalar force and pair energy, i.e. the analytic gradient of `E` for one pair:
  `phi = z2/r`, `phi' = z2'/r - phi/r`,
  `dE/dr = F'(rho_i) rho'_(j->i) + F'(rho_j) rho'_(i->j) + phi'`,
  `fpair = -scale * (dE/dr) / r`, `pair_energy = scale * phi`.

`phi` is not tabulated directly — LAMMPS stores `z2(r) = r * phi(r)` and
recovers `phi = z2/r`. So the "pure `phi`" is a spline lookup of `z2` divided by
`r`. Operation-for-operation identical to the original.

# Transformation 3 — Introduce order-preserving `map` / `filter` / `reduce`

Add three generic combinators (anonymous namespace, `std::vector`-based):

- `map(xs, f)` — apply `f` to each element, results in input order.
- `filter(xs, keep)` — keep elements satisfying `keep`, in input order.
- `reduce(xs, init, f)` — **left fold**: `f(f(f(init, x0), x1), …)`.

`reduce` is a strict left fold in list order, so `reduce(xs, acc, +)` performs
exactly `((acc + x0) + x1) + …` — the same association as a hand-written
`for` loop that does `acc += x`. This is what lets the combinators replace the
original loops without perturbing floating-point results.

# Transformation 4 — Neighbor selection as `filter ∘ map`

Introduce `in_cutoff_neighbors(i, …)` returning `vector<NeighborContext>`:

```
map(slots,  jj -> RawNeighbor{ j = jj & NEIGHMASK, jtype, rij = xi - xj, rsq })   // map(displacement)
|> filter( nb -> nb.rsq < cutforcesq )                                            // filter(not_curr_atom)
|> map(   nb -> NeighborContext{ j, jtype, rij, r = sqrt(rsq), radial } )          // map(distance)
```

This is the `filter (not_curr_atom / within cutoff)` composed with
`map distance` of the target form, evaluated in neighbor-list order. The cutoff
predicate uses `rsq` (never `sqrt`) exactly as the original `if (rsq < cutforcesq)`.

# Transformation 5 — Density pass as a reduce + Newton scatter

The physical density is `rho_i = sum_{j != i} rho_b(r_ij)`. With a half list the
original visits each edge once and updates *both* endpoints. Split each atom's
contribution into (a) a pure reduce for its own density and (b) a scatter for
the reciprocal contribution:

```
rho[i] = reduce(neighbors, rho[i], (acc, nb) -> acc + eam_density_value(nb.jtype, itype, nb.radial));
for nb in neighbors:
    if (newton_pair || nb.j < nlocal)
        rho[nb.j] += eam_density_value(itype, nb.jtype, nb.radial);
```

`reduce` is seeded with the **current** `rho[i]` (which already holds scatter
contributions from earlier atoms), so it computes `((rho[i] + d0) + d1) + …` in
the same order as the original `rho[i] += …`. The scatter targets `nb.j` are all
distinct from `i` and from each other within one atom's list, so moving the
`rho[j]` updates after the `rho[i]` reduce does not reorder any accumulator's
additions — the result is bit-identical. `reverse_comm` still sums ghost
densities afterward.

The `reduce (+) (map rho neighbors)` half of the target is realized here; the
scatter + `reverse_comm` is the half-neighbor-list realization of the *other*
atoms' `reduce` (see Deviations).

# Transformation 6 — Embedding pass as a `map` over atoms

The `map(… , all_atoms)` and the outer `F(reduce(…))` of the target:

```
for i in atoms:
    density = eam_density_spline_point(rho[i], …);
    fp[i]   = eam_embedding_derivative(itype, density);      // F'(rho_i)
    if eflag:
        phi = eam_embedding_energy(itype, density);          // F(rho_i)
        if rho[i] > rhomax: phi += fp[i] * (rho[i]-rhomax);  // energy-conserving extrapolation
        phi *= scale[itype][itype];
        eng_vdwl += phi;  (and eatom[i] += phi)
```

`fp[i] = F'(rho_i)` is stored for the force pass (it is `dE/d(rho_i)`), then
`forward_comm` publishes it to ghosts. The global energy accumulates the
embedding term `F(rho_i)` per atom in `ilist` order (unchanged).

# Transformation 7 — Force / pair-energy pass as an edge fold

The forces (`-∇E`) and the `(1/2) sum phi` term:

```
for i in atoms:
    neighbors = in_cutoff_neighbors(i, …);
    numforce[i] = neighbors.size();
    for nb in neighbors:
        terms = eam_pair_spline_terms(itype, nb.jtype, nb.radial);
        pe    = eam_pair_force_energy(terms, fp[i], fp[nb.j], nb.r, scale[itype][nb.jtype]);
        apply_pair_force(f, i, nb.j, nb.rij, pe.fpair, newton_pair, nlocal);  // f[i] +=, f[j] -=
        ev_tally(i, nb.j, …, eflag ? pe.pair_energy : 0.0, pe.fpair, nb.rij…);
```

`apply_pair_force` performs the same equal-and-opposite update in neighbor
order, so `f[i]` accumulates identically. `ev_tally` receives the per-pair
`phi(r)` and supplies the `1/2` factor and half-list double-count bookkeeping
for the pair-energy term (the `(1/2) sum phi` half of the target). `numforce[i]`
= number of in-cutoff neighbors, as before (used by `single()`).

# Wrapper (unchanged)

`ev_init(eflag,vflag)`, per-atom array growth, `rho` zeroing (`nall` vs
`nlocal`), the two communications (`reverse_comm` of `rho`, `forward_comm` of
`fp`), `embedstep`, the `rhomax`-exceeded `MPI_Allreduce`/warning, and
`if (vflag_fdotr) virial_fdotr_compute()` are the setup/plumbing that frames the
functional core; they are preserved verbatim.

---

# Deviations from the literal target (what could not be reached, and why)

1. **Forces are the analytic gradient, not the energy.** The target pseudocode
   computes energy `E_i`; the primary output of `compute` is forces `= -∇E`.
   C++ has no autodiff here, so the gradient is written out by hand as the
   per-pair `dE/dr` (`eam_pair_force_energy`). It is presented as a parallel
   functional decomposition (map/fold over the same pure per-pair terms), but it
   is a separate expression from the energy sum.

2. **Density is a half-list scatter-fold, not a per-atom `reduce` over all
   neighbors.** The literal `reduce (+) (map rho neighbors)` needs every atom to
   see *all* its neighbors (a full list). The code uses a half list + Newton
   scatter + `reverse_comm` for efficiency. This was a deliberate choice:
   switching to a full list would (a) reorder the summation and risk the 6e-12
   budget and (b) break `eam/omp`, which shares `PairEAM::init_style` and assumes
   a half list. So `rho_i` is assembled as an edge-fold: atom i's own `reduce`
   plus scatter contributions from every other atom's fold, completed by
   `reverse_comm`. Mathematically the same sum; structurally a scatter, not a
   per-atom `reduce`.

3. **`rho`, `F`, `phi` are cubic-Hermite spline lookups, not closed forms.** The
   "pure functions" evaluate tabulated potentials, and `phi(r) = z2r(r)/r`
   because LAMMPS tabulates `z2 = r*phi`.

4. **MPI communication punctuates the pure core.** `reverse_comm` (complete
   ghost densities) and `forward_comm` (publish `fp` to ghosts) are mandatory
   between the three stages and cannot be folded into a single pure expression.

# Verification

```bash
scripts/run_pair_eam_diff.sh
```

Passes all three checks — `PairStyle.plain` (in-tree `eam`), `PairStyle.omp`
(suffix dispatch), and `PairStyle.plain` on the generated explicit `eam/omp`
fixture — against `atomic-pair-eam.yaml`. Observed max relative error ~1e-14
(budget 6e-12), i.e. bit-faithful.
