# Simplifying `PairEAM::compute` to a functional form (full neighbor list)

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

This version uses a **full neighbor list** (each atom sees all its neighbors) so
that the density is the *literal* per-atom `rho[i] = reduce(+, map(rho, neighbors_i))`
of the target — no half-list scatter, no reverse communication. The rewrite
remains executable and passes the differential test (measured max relative error
vs. the oracle: forces ~1e-14, stress ~3e-14, energy ~6e-15; budget 6e-12).

`eam/omp` is out of scope: it shares `PairEAM` but keeps a half-list `compute`,
so the differential test is scoped to the plain `eam` style.

---

# Transformation 0 — Switch to a full neighbor list

`init_style` requests `NeighConst::REQ_FULL` instead of the default half list, and
the constructor sets `no_virial_fdotr_compute = 1`. Consequences:

- Each **local** atom's density is complete from its own neighbor loop → no
  Newton scatter onto `rho[j]`, and `reverse_comm` of `rho` is removed.
- Each pair is stored **twice** (once in each atom's list). Forces are
  accumulated onto atom `i` only; the reciprocal force is produced when the
  partner atom is visited.
- `fdotr` is invalid when forces touch only local `i`, so the virial is tallied
  per pair via `Pair::ev_tally_full` (`src/pair.cpp:1179`). Setting
  `no_virial_fdotr_compute = 1` makes `ev_setup` route the virial to a per-pair
  global tally (`vflag_global = 1`, `vflag_fdotr = 0`; `src/pair.cpp:965-968`).

# Transformation 1 — Extract pure spline primitives

Factor the inlined Horner polynomials and index arithmetic into pure functions:

- `eam_spline_value(coeff, p)` = `((c3*p + c4)*p + c5)*p + c6` — cubic **value**.
- `eam_spline_derivative(coeff, p)` = `(c0*p + c1)*p + c2` — **derivative** (the
  spline coefficients already carry the `1/grid_spacing` factor).
- `eam_radial_spline_point(r, rdr, nr)` / `eam_density_spline_point(rho, rdrho, nrho)`
  — locate an argument on the `r` / `rho` grid, returning `SplinePoint{index, fraction}`
  with the exact clamp arithmetic.

These are literal extractions, so they are value-exact.

# Transformation 2 — Extract pure EAM physics functions

- `eam_density_value(source_type, target_type, SplinePoint)` = `rho`.
- `eam_embedding_energy(type, SplinePoint)` = `F`;
  `eam_embedding_derivative(type, SplinePoint)` = `F'`.
- `eam_pair_spline_terms(...)` bundles the four raw pair quantities
  (`rho'_(i->j)`, `rho'_(j->i)`, `z2 = r*phi`, `z2'`).
- `eam_pair_force_energy(terms, fp_i, fp_j, r, scale)` computes the per-pair
  scalar force and pair energy, i.e. the analytic gradient of `E` for one pair:
  `phi = z2/r`, `phi' = z2'/r - phi/r`,
  `dE/dr = F'(rho_i) rho'_(j->i) + F'(rho_j) rho'_(i->j) + phi'`,
  `fpair = -scale * (dE/dr) / r`, `pair_energy = scale * phi`.

`phi` is not tabulated directly — LAMMPS stores `z2(r) = r * phi(r)` and recovers
`phi = z2/r`. Operation-for-operation identical to the original.

# Transformation 3 — Introduce generic `map` / `filter` / `reduce`

Three combinators (anonymous namespace, `std::vector`-based):

- `map(xs, f)` — apply `f` to each element, in input order.
- `filter(xs, keep)` — keep elements satisfying `keep`, in input order.
- `reduce(xs, init, f)` — left fold `f(f(f(init, x0), x1), …)`.

# Transformation 4 — Neighbor selection as `filter ∘ map`

`in_cutoff_neighbors(i, …)` returns `vector<NeighborContext>`:

```
map(slots,  jj -> RawNeighbor{ j = jj & NEIGHMASK, jtype, rij = xi - xj, rsq })   // map(displacement)
|> filter( nb -> nb.rsq < cutforcesq )                                            // filter(not_curr_atom)
|> map(   nb -> NeighborContext{ j, jtype, rij, r = sqrt(rsq), radial } )          // map(distance)
```

With a full list this is *all* of atom `i`'s in-cutoff neighbors — the
`filter (not_curr_atom) . map distance` of the target, over `neighbors_i`.

# Transformation 5 — Density + embedding as a single `map` over atoms

Because a full list makes `rho[i]` complete within atom `i`'s own loop, the
density and embedding passes fuse into one map over the local atoms — the outer
`map(… , all_atoms)` and the `F(reduce(map rho …))` of the target:

```
for i in atoms:                                          // map over all_atoms
    neighbors = in_cutoff_neighbors(i, …);
    rho[i] = reduce(neighbors, 0.0, (acc, nb) -> acc + rho(nb.jtype, itype, nb.radial));
                                                          // reduce (+) (map rho (map distance (filter …)))
    fp[i]  = F'(rho[i]);                                  // stored for the force stage; = dE/d(rho_i)
    if eflag:
        phi = F(rho[i]);                                  // F_a( sum_{j!=i} rho_b(r_ij) )
        if rho[i] > rhomax: phi += fp[i] * (rho[i]-rhomax);  // energy-conserving extrapolation
        phi *= scale[itype][itype];
        eng_vdwl += phi;  (and eatom[i] += phi)
```

`forward_comm(fp)` then publishes `fp` to ghost atoms (the force stage reads
`fp[j]` for neighbor `j`, which may be a ghost).

# Transformation 6 — Forces + pair energy as a fold over neighbors

```
for i in atoms:
    neighbors = in_cutoff_neighbors(i, …);
    numforce[i] = neighbors.size();
    for nb in neighbors:                                  // fold over neighbors
        terms = eam_pair_spline_terms(itype, nb.jtype, nb.radial);
        pe    = eam_pair_force_energy(terms, fp[i], fp[nb.j], nb.r, scale[itype][nb.jtype]);
        accumulate_pair_force(f, i, nb.rij, pe.fpair);    // force on i only (full list)
        ev_tally_full(i, eflag ? pe.pair_energy : 0.0, 0.0, pe.fpair, nb.rij…);
```

`accumulate_pair_force` updates `f[i]` only. Because `psip` (hence `fpair`) is
symmetric under `i <-> j` and the displacement flips sign, visiting the partner
pair `(j, i)` produces the equal-and-opposite force on `j`. `ev_tally_full`
tallies half the pair energy `phi(r)` and half the per-pair virial; summed over
the two visits of each pair this gives the full `(1/2) sum phi` energy and the
full virial.

# Wrapper (setup / cleanup)

`ev_init(eflag,vflag)`, per-atom array growth, `forward_comm(fp)`, `embedstep`,
and the `rhomax`-exceeded `MPI_Allreduce`/warning frame the functional core.
(The `rho` zeroing loop, `reverse_comm`, and `virial_fdotr_compute` of the
half-list version are gone.)

---

# Deviations from the literal target (what could not be reached, and why)

1. **Forces are the analytic gradient, not the energy.** The target pseudocode
   computes energy `E_i`; the primary output of `compute` is forces `= -∇E`.
   C++ has no autodiff here, so the gradient is written out by hand as the
   per-pair `dE/dr` (`eam_pair_force_energy`). It is presented as a parallel
   functional decomposition (map/fold over the same pure per-pair terms), but it
   is a separate expression from the energy sum.

2. **Full-list double visit + `ev_tally_full`.** The `(1/2)` in the target is
   realized by tallying half of each pair's energy/virial on each of the pair's
   two visits, and by accumulating the force on `i` only. This matches the target
   `(1/2) sum phi` exactly, but the `1/2` lives in the tally, not as an explicit
   scalar in the readable core.

3. **`rho`, `F`, `phi` are cubic-Hermite spline lookups, not closed forms.** The
   "pure functions" evaluate tabulated potentials, and `phi(r) = z2r(r)/r`
   because LAMMPS tabulates `z2 = r*phi`.

4. **`forward_comm` of `fp` punctuates the core.** The force stage needs
   `fp[j] = F'(rho_j)` for neighbors `j` that may be ghost atoms owned by another
   MPI domain, so the embedding derivatives are communicated to ghosts between
   the two maps. (Unlike the half-list version, no `reverse_comm` of `rho` is
   needed.)

# Verification

```bash
scripts/run_pair_eam_diff.sh
```

Passes the `PairStyle.plain` check (in-tree full-list `eam`) against
`atomic-pair-eam.yaml` — forces, stress, and energies within relative epsilon
6e-12 (5× for run-stage forces), including the restart / nofdotr / data-file
stages. Observed max relative error ~3e-14. `eam/omp` is intentionally not
checked (it shares `PairEAM` but keeps a half-list `compute`).
