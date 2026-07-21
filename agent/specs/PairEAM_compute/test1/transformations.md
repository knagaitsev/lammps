# Simplifying `PairEAM::compute` to the functional-form core math

Target function: `PairEAM::compute` in `src/MANYBODY/pair_eam.cpp`
Differential test: `scripts/run_pair_eam_diff.sh` (oracle
`unittest/force-styles/tests/atomic-pair-eam.yaml`, relative epsilon `6e-12`).

## Goal

Rewrite `PairEAM::compute` so the **main math has no `for` loops** and reads as
the target functional form — a `map` over `all_atoms` of the embedded-atom-model
per-atom energy:

```
map (atom =>
       F( reduce (+) (map rho  (map distance (filter not_curr_atom all_atoms))) )
     + (1/2) * ( reduce (+) (map phi (map distance (filter not_curr_atom all_atoms))) )
    )
    all_atoms
```

A **full neighbor list** makes each atom's density complete from its own
neighbor set, so the per-atom expression above is realized literally. The
rewrite stays executable and passes the (plain) differential test (measured max
relative error: forces ~1e-14, stress ~3e-14, energy ~1e-16; budget 6e-12).

`eam/omp` is out of scope (it shares `PairEAM` but keeps a half-list `compute`),
so the test is scoped to the plain `eam` style.

---

# Transformation 0 — Full neighbor list

`init_style` requests `NeighConst::REQ_FULL`; the constructor sets
`no_virial_fdotr_compute = 1`. Consequences: each local atom's density is
complete from its own loop (no Newton scatter, no `reverse_comm`); each pair is
stored twice, so forces are accumulated onto atom `i` only and the virial is
tallied per pair (`vflag_global`/`vflag_atom` gate the writes; no `fdotr`).

# Transformation 1 — Pure spline primitives

Factor the inlined Horner polynomials and index arithmetic into pure functions:
`eam_spline_value` (cubic value), `eam_spline_derivative` (derivative),
`eam_radial_spline_point` / `eam_density_spline_point` (grid location). Literal
extractions, value-exact.

# Transformation 2 — Pure EAM physics

- `eam_density_value` = `rho`; `eam_embedding_energy` = `F`;
  `eam_embedding_derivative` = `F'`.
- `eam_pair_energy(itype, jtype, radial, r, scale)` = `scale * z2(r)/r` = `phi`
  (LAMMPS tabulates `z2 = r*phi`).
- `eam_pair_spline_terms` + `eam_pair_force_energy` give the per-pair scalar
  force `fpair = -(dE/dr)/r`, where
  `dE/dr = F'(rho_i) rho'_(j->i) + F'(rho_j) rho'_(i->j) + phi'(r)`.

# Transformation 3 — Generic combinators `map` / `filter` / `reduce` / `for_each`

In a sub-namespace `fn` (because `Pair` has an `int *map` member that would
shadow a bare `map` inside `PairEAM` methods):

- `fn::map(xs, f)`, `fn::filter(xs, keep)` — in input order.
- `fn::reduce(xs, init, f)` — left fold `f(f(f(init, x0), x1), …)`.
- `fn::for_each(xs, f)` — the **effectful sibling of `map`**: it drives the outer
  `map` over `all_atoms`, where each atom's result is written to the LAMMPS
  output arrays (`f[i]`, `eng_vdwl`, `eatom[i]`, `virial`). This replaces the raw
  `for (ii …)` atom loops.

# Transformation 4 — Neighbor selection as `filter ∘ map`

`in_cutoff_neighbors(i, …)` = `fn::map(distance) ∘ fn::filter(within cutoff) ∘
fn::map(displacement)` over atom `i`'s list → `vector<NeighborContext>`. This is
the `map distance (filter not_curr_atom …)` of the target, over `neighbors_i`.

# Transformation 5 — Pass 1: the target per-atom energy (`for_each` over atoms)

`const std::vector<int> all_atoms(ilist, ilist + inum)` is the target's
`all_atoms`. Then, with **no raw loop**:

```
for_each(all_atoms, i => {
    nbrs   = in_cutoff_neighbors(i, …)
    rho[i] = reduce(nbrs, 0.0, (acc, nb) -> acc + rho(nb.jtype, itype, nb.radial))     // reduce (+) (map rho …)
    fp[i]  = F'(rho[i])
    if eflag:
        embed = F(rho[i])   (+ rho>rhomax extrapolation, * scale[itype][itype])         // F( reduce(map rho …) )
        pair  = 0.5 * reduce(nbrs, 0.0, (acc, nb) -> acc + phi(itype, nb.jtype, nb.radial, nb.r, scale))
                                                                                          // (1/2)*reduce(map phi …)
        E_i   = embed + pair                                                              // the target per-atom expr
        eng_vdwl += E_i;  eatom[i] += E_i
})
```

Because the pair energy is folded here (not in the force pass), pass 1 is the
**literal target per-atom expression**. Tallying `E_i` to both `eng_vdwl` and
`eatom[i]` keeps the harness invariant `sum(eatom) == eng_vdwl` exact.
`forward_comm(fp)` then publishes `fp` to ghosts.

# Transformation 6 — Pass 2: forces + virial as the analytic gradient (`for_each`)

```
for_each(all_atoms, i => {
    nbrs        = in_cutoff_neighbors(i, …)
    numforce[i] = nbrs.size()
    terms       = map(nbrs, nb -> PairTerm{ nb.rij, fpair(i, nb) })                      // map pair_force
    f[i]       += reduce(terms, {0,0,0}, (a, t) -> a + t.fpair * t.rij)                   // reduce (+) (map pair_force …)
    if evflag:
        vir     = reduce(terms, zero6, (a, t) -> a + half_virial(t.rij, t.fpair))
        virial     += vir      (if vflag_global)
        vatom[i]   += vir      (if vflag_atom)
})
```

The force is the analytic gradient `-dE/dx_i`, expressed as a `reduce` of the
per-pair force vectors (accumulated onto `i` only; the reciprocal comes from
atom `j`'s visit). `half_virial` = `0.5 * {xx,yy,zz,xy,xz,yz} * fpair` reproduces
`Pair::ev_tally_full`'s per-pair virial; summed over each pair's two visits it
gives the full virial. Energy was already tallied in pass 1.

The only remaining literal `for` statements are the fixed 6-component virial
writes (`virial[0..5]`, `vatom[i][0..5]`), written out explicitly — array output,
not part of the core math — and the internal loops of the `fn::` combinators
themselves (the reusable "library").

---

# Deviations from the literal single-`map` target (what could not be reached)

1. **Two passes, not one.** The target is a single `map(atom => …)`. Forces need
   `fp[j] = F'(rho_j)` for every neighbor `j` (possibly a ghost owned by another
   MPI rank), which requires all densities computed and `forward_comm(fp)` before
   any force. So the core is realized as **pass 1 = the target energy map** plus
   **pass 2 = the gradient map**. This split is inherent to a distributed
   many-body force, not a stylistic choice.

2. **Forces are the analytic gradient, not autodiff.** C++ has no autodiff here,
   so `-∇E` is written as the per-pair `dE/dr` and reduced. It is a parallel
   functional expression over the same pure per-pair terms as the energy.

3. **`rho`, `F`, `phi` are cubic-Hermite spline lookups**, not closed forms;
   `phi(r) = z2r(r)/r` because LAMMPS tabulates `z2 = r*phi`.

4. **`for_each` drives `all_atoms`.** The outer `map` is realized by the effectful
   `for_each` (its "result" is the write-back of each atom's density/energy/force
   to the LAMMPS arrays), and `forward_comm(fp)` punctuates the two passes.

# Verification

```bash
scripts/run_pair_eam_diff.sh
```

Passes `PairStyle.plain` against `atomic-pair-eam.yaml` — forces, stress,
`eng_vdwl`, and `sum(eatom) == eng_vdwl` — within relative epsilon 6e-12 (5× for
run-stage forces), including the restart / nofdotr / data-file stages. Observed
max relative error ~3e-14. `eam/omp` is intentionally not checked.
