# Task: Extract the exact functional form of the EAM core math from `PairEAM::compute`

## Mission

Refactor `PairEAM::compute` (`src/MANYBODY/pair_eam.cpp`) so that the function becomes:

```
original_function(...) {
  // SETUP: marshal LAMMPS data structures into plain inputs
  results = functional_core_math(inputs)   // <-- the exact functional form, defined below
  // TEARDOWN: scatter results back into LAMMPS accumulators (f, eng_vdwl, eatom, virial)
}
```

where `functional_core_math` is written purely in terms of `map` / `reduce` / `filter` / pure functions, and the differential test `scripts/run_pair_eam_diff.sh` still passes.

**This is not a code-cleanup task. It is a total-restructuring task.** The setup and teardown may be as large, as slow, and as invasive as necessary. Success is binary: either the exact functional form below is the thing that computes the physics at runtime, or the task is not done.

## Environment, permissions, and git workflow

- **You may install anything.** Missing compilers, CMake versions, MPI, linters, debuggers, numerical-diff tools — install them. `sudo` is authorized when necessary. Never abandon a step because a tool is absent.
- **Python analysis uses `uv`.** If you write analysis scripts (e.g., to diff dump files field-by-field, locate the first diverging quantity, or histogram FP error), manage Python and dependencies with `uv` (`uv venv`, `uv pip install`, or `uv run`). numpy/pandas installs are pre-approved.
- **All work happens on a fresh branch.** Before touching any file: `git checkout -b functional-eam-extraction` (from the current HEAD). Commit after every semantic step with messages naming the transformation (e.g., `T3: symmetrize half neighbor list in setup`). This gives a verifiable, bisectable transformation history that mirrors transformations.md — each commit should correspond to a documented transformation. Never commit to the original branch.
- **Blast radius policy.** All new infrastructure (combinators, pure physics functions, result structs) lives **inside `src/MANYBODY/pair_eam.cpp`** in an anonymous namespace — no new source files, no new headers, no build-system changes. Changes outside `compute()` are allowed only when clearly necessary (e.g., a declaration in `pair_eam.h`, or `init_style` if half-list symmetrization provably fails); make them minimal and record each with its justification in transformations.md. Do not reformat, reorganize, or "improve" code outside the task's needs.
- **Infrastructure building is in scope.** If reaching the target requires real scaffolding — the file-local functional library, a per-pair record type, a debug harness or `uv`-managed Python script that dumps and diffs intermediate stage outputs against the original — build it. Substantial foundational setup is expected, not a deviation. (Throwaway analysis scripts may live under `agent/specs/PairEAM_compute/test1/`; only the production infrastructure is confined to `pair_eam.cpp`.)

## Scope of correctness (deliberately narrow)

The **only** configuration that must work is the one `scripts/run_pair_eam_diff.sh` actually exercises (its input script, potential file, rank count, newton setting, and the base `eam` style). Determine that configuration in step 1 and treat everything else as out of scope:

- MPI multi-rank, `newton_pair` settings other than the tested one, and the derived styles (`eam/alloy`, `eam/fs`, which inherit this `compute()`) do **not** need to keep working on this branch.
- However, do not let unsupported paths fail *silently*. Where the refactor knowingly breaks a path, add a cheap runtime guard (e.g., `error->one(FLERR, "functional-eam-extraction branch: only the diff-tested configuration is supported")`) triggered when that path is entered, and list each dropped path in transformations.md. This is pre-approved and is much better than contorting the functional core to preserve untested paths.
- This narrowing is a feature: use it to simplify. If the diff test runs single-rank, the comm calls are identity-like glue; if it runs newton on, you may specialize the neighbor handling to that case.

## Non-negotiable success criterion

You are done **only** when ALL of the following hold:

1. Every arithmetic operation that contributes to energy or force flows through `functional_core_math` (or the pure functions it composes). No physics arithmetic remains inline in `compute()`.
2. `functional_core_math` and its helpers are written exclusively as compositions of `map`, `reduce`, `filter`, `zip` (if needed), and pure functions. No in-place mutation of shared state inside the core; pure functions take values, return values.
3. `scripts/run_pair_eam_diff.sh` passes.
4. The code compiles and runs with no undefined behavior.
5. `agent/specs/PairEAM_compute/test1/transformations.md` documents every transformation as a verifiable, compiler-style rewrite step.

Do not stop, declare partial success, or leave "future work" notes short of this. Performance regressions of 10x–100x are explicitly acceptable. Memory blowup from materializing neighbor arrays is explicitly acceptable. Restructuring the entire function body is explicitly acceptable.

## The actual core math (pre-analyzed — read this before touching code)

The original prompt gave only the EAM **energy**:

```
E_i = F_a( sum_{j!=i} rho_b(r_ij) ) + (1/2) sum_{j!=i} phi_ab(r_ij)
```

That is correct but incomplete. `PairEAM::compute` also computes **forces**, which are the analytic gradient of the total energy:

```
f_i = - grad_{r_i} E_total
    = - sum_{j!=i} [ F'_a(rho_bar_i) * rho'_b(r_ij)
                   + F'_b(rho_bar_j) * rho'_a(r_ij)
                   + phi'_ab(r_ij) ] * rhat_ij
```

where `rho_bar_i = sum_{j!=i} rho_b(r_ij)` and `rhat_ij = (x_i - x_j)/r_ij`. In the source this appears as `psip = fp[i]*rhojp + fp[j]*rhoip + phip` and `fpair = -scale*psip/r`.

**Do not treat the presence of forces as a deviation requiring escalation.** It is the same core math (energy + its gradient) and it is pre-approved as part of the target. This is the pivot the previous run should have made; it is authorized in advance.

### Critical structural fact

The force on atom `i` depends on `F'(rho_bar_j)` — the embedding derivative of its *neighbors*. Therefore the core math is inherently **two staged maps, not one**:

```
// Stage 1: per-atom density and embedding
rho_bar  = map (\i -> reduce (+) (map (rho . distance i) (neighbors i))) atoms
fp       = map F'  rho_bar          // embedding derivative per atom
E_embed  = map F   rho_bar          // embedding energy per atom (with rhomax extrapolation folded in)

// Stage 2: per-atom pair energy and force (reads the full fp array)
E_pair   = map (\i -> (1/2) * reduce (+) (map (phi . distance i) (neighbors i))) atoms
force    = map (\i -> reduce (vec+) (map (\j -> pair_force i j fp) (neighbors i))) atoms
```

This two-stage structure IS the functional form of EAM-with-forces. Do not try to collapse it into one map; the data dependence forbids it, and that is a property of the math, not a blocker.

### Target functional core (concrete shape to implement in C++)

```cpp
// --- ALL of the following lives inside src/MANYBODY/pair_eam.cpp, in an
//     anonymous namespace above PairEAM::compute (no new files, no new
//     headers; add declarations to pair_eam.h only if truly unavoidable) ---

// --- Functional combinators (file-local, generic) ---
template<class F, class Xs> auto fmap(F f, const Xs& xs);          // map
template<class F, class T, class Xs> T freduce(F f, T init, const Xs& xs);  // fold
template<class P, class Xs> Xs ffilter(P p, const Xs& xs);         // filter

// --- Pure physics functions (wrap the spline tables; a spline IS the
//     numerical representation of the analytic rho/phi/F, so wrapping the
//     table lookup in a pure function of (types, r) is faithful) ---
double rho_of_r (int src_type, int dst_type, double r);   // rho_b(r), 0 beyond cutoff
double phi_of_r (int itype, int jtype, double r);         // phi_ab(r) incl. scale, 0 beyond cutoff
double dphi_of_r(int itype, int jtype, double r);
double drho_of_r(int src_type, int dst_type, double r);
std::pair<double,double> F_embed(int itype, double rho_bar); // (F, F'), rhomax linear
                                                             // extrapolation folded inside

// --- Core math: pure in, pure out ---
struct AtomView { Vec3 x; int type; };
struct PairContrib { int i, j; Vec3 fpair_vec; double phi_half; double r2; ... };
struct EamResult {
  std::vector<double> rho_bar, embed_energy, fp;
  std::vector<double> pair_energy;
  std::vector<Vec3>   force;
  std::vector<PairContrib> pair_contribs;   // enough info for teardown to drive ev_tally/virial
  bool any_beyond_rhomax;
};

EamResult eam_core(const std::vector<AtomView>& atoms,
                   const std::vector<std::vector<int>>& full_neighbors);
```

The bodies of `eam_core` and helpers must read as the pseudocode above: maps over atoms, reduces over neighbors, pure functions of distance and type. Loops are permitted **only inside the generic `fmap`/`freduce`/`ffilter` implementations**, never in the physics.

**Accessing member data from file-local pure functions.** The spline tables, `rdr`, `nr`, `scale`, `cutforcesq`, `rhomax`, etc. are `PairEAM` members, and anonymous-namespace functions cannot see them. Resolve this without polluting purity: bundle the constant table state into a small `struct EamTables { ... }` (raw pointers/values copied in SETUP) passed by const reference to every pure function — or equivalently define the pure functions as lambdas inside `compute()` that close over const table state. Either is pure in the relevant sense: the tables are immutable for the duration of `compute()`, so the functions remain referentially transparent in their true inputs `(types, r)`. State which option you chose in transformations.md. Blanket-`friend` hacks or making members global are not acceptable.

## Pre-authorized transformations (do these; do not ask)

The previous run was too conservative. Each obstacle below has a pre-approved resolution:

1. **Half neighbor list → full per-atom neighbor sets.** The Wikipedia `sum_{j!=i}` needs each atom to see all its neighbors. In SETUP, symmetrize the LAMMPS half list (with `NEIGHMASK` stripped) into `full_neighbors[i]`, covering local atoms (and ghosts as needed for density symmetry — see item 3). Do this by copying/expanding into plain `std::vector`s. Do not change the neighbor request in `init_style` unless symmetrization provably cannot reproduce the half-list semantics; prefer setup-side symmetrization because it leaves the rest of LAMMPS untouched. Specialize to whichever `newton_pair` setting the diff test uses; guard the other setting per the scope section.
2. **`filter not_curr_atom` / cutoff.** Implement the neighbor set as `ffilter(within_cutoff, candidate_pairs)` inside the core, or equivalently make `rho_of_r`/`phi_of_r` return 0 beyond `cutforce` and document the equivalence. Either satisfies the functional form; pick one and state it in transformations.md.
3. **MPI communication (`reverse_comm`, `forward_comm`).** These are distributed-sum plumbing, not physics. Keep them, positioned *between* the two functional stages: SETUP-1 → `eam_core_stage1` (densities, fp, embed energy) → `reverse_comm`(sum ghost densities)/`forward_comm`(distribute fp) glue → `eam_core_stage2` (pair energy, forces) → TEARDOWN. If splitting the core into two pure calls with comm glue between them, that is compliant: each stage is pure; the comm is setup/teardown for stage 2. Only the diff test's rank count must work (per the scope section); if it is single-rank, you may treat the comm calls as inert glue kept for structural fidelity, and you do not need to verify multi-rank behavior.
4. **Spline tables.** `rhor_spline`/`z2r_spline`/`frho_spline` lookups become the bodies of the pure functions. Note in transformations.md: the original stores `z2 = r*phi`; your pure `phi_of_r` should compute `z2(r)/r` and `dphi_of_r` should compute `(z2'(r) - z2(r)/r)/r`, matching `phip = z2p*recip - phi*recip`. Match the original operation order here if the diff test's tolerance turns out to demand it (see item 8).
5. **`rhomax` extrapolation and warning.** Fold the linear extrapolation `F(rho) + F'(rho)*(rho - rhomax)` into `F_embed`. Return `any_beyond_rhomax` in the result; TEARDOWN does the `MPI_Allreduce` + warning.
6. **`scale[itype][jtype]`.** Fold into `phi_of_r`/`dphi_of_r` (it multiplies both `phi` and `fpair` in the original) and into the diagonal embedding term via `scale[t][t]`. Document as a parameter of the pure functions.
7. **`ev_tally` / virial / `eatom` / `eng_vdwl` / `numforce` / `embedstep`.** All accumulation into LAMMPS state is TEARDOWN. Have the core return per-pair records (`pair_contribs`) sufficient for teardown to call `ev_tally` with identical arguments in identical order, and per-atom embed energies for the `eatom`/`eng_vdwl` accumulation. `numforce[i]` is just the neighbor-within-cutoff count — derive it from the core's results in teardown. `embedstep = update->ntimestep` stays in teardown.
8. **Numerical fidelity = the diff script's own tolerance, nothing stricter.** Read `scripts/run_pair_eam_diff.sh` FIRST and extract its comparison method and tolerance. The requirement is to pass that script as written — do not impose bitwise identity on yourself. Write the cleanest functional core the tolerance permits (e.g., you may reassociate sums by using per-atom full-neighbor reduces instead of the original half-list accumulation order). Escalate to order-preservation tactics (matching the original FP operation order inside pure functions, replaying the original half-list (ii, jj) accumulation order in teardown via recorded `pair_contribs`) **only if** the test actually fails and the analysis shows accumulation order is the cause — treat those tactics as a fallback ladder, not the default.

## Escalation protocol (the only reason to stop)

Stop and ask the user **only if** you discover core math in the implementation that is not (a) the EAM energy, (b) its gradient/forces as specified above, or (c) covered by pre-authorizations 1–8. In that single case: state the actual math, give its functional form, and ask whether to pivot. Otherwise run fully autonomously to completion. "This transformation is invasive/risky/large" is never a reason to stop — invasiveness is expected and approved.

## Workflow

1. Create the fresh branch. Read `scripts/run_pair_eam_diff.sh`; run it on the unmodified code to confirm a green baseline and learn the tolerance/ranks. Install anything the build or test needs.
2. Read `pair_eam.h`, `PairEAM::init_style`, comm pack/unpack methods, and the spline setup, so the pure-function wrappers are faithful.
3. Write the combinators and pure functions; unit-check the pure functions against direct table lookups on a few values.
4. Restructure `compute()` into SETUP → stage1 → comm glue → stage2 → TEARDOWN, moving physics into the core incrementally, running the diff test after each mechanical step. Never batch more than one semantic change between test runs.
5. When the diff test fails, diagnose numerically (which quantity, which magnitude of error) — order-of-accumulation and float-op-order are the usual suspects — and fix; do not revert to a conservative structure.
6. Final pass: grep `compute()` for any remaining arithmetic on positions/energies/forces; if any exists outside the core, move it. Then re-run the diff test.
7. Write `agent/specs/PairEAM_compute/test1/transformations.md`: one section per transformation, each stated as a semantics-preserving rewrite with its justification (e.g., "Loop-to-map: a loop with independent iterations writing disjoint outputs is `fmap`"; "Half-list symmetrization: for every (i,j) in the half list, insert j into N(i) and i into N(j); the double-count is compensated by the 1/2 factor / per-pair symmetric application — prove the accumulated totals identical").

## Definition of done — final self-check before reporting

- [ ] `compute()` body is: setup, core call(s), comm glue, teardown — nothing else.
- [ ] Core reads as: `fmap`/`freduce`/`ffilter` + pure functions; no physics loops, no shared mutation.
- [ ] Two-stage structure explicitly reflects the `fp[j]` data dependence.
- [ ] Diff test passes on the final code (paste the passing output in your report).
- [ ] transformations.md complete, including the energy→energy+gradient pivot, the diff test's tolerance and why the chosen accumulation strategy meets it, and every out-of-function change and dropped configuration path.
- [ ] Report states the final functional form in mathematical notation next to the C++ core signature.

If any box is unchecked, you are not done. Continue.

