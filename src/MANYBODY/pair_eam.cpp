// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stephen Foiles (SNL), Murray Daw (SNL)
------------------------------------------------------------------------- */

#include "pair_eam.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "potential_file_reader.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairEAM::PairEAM(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  atomic_energy_enable = 1;
  embedstep = -1;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

  nmax = 0;
  rho = nullptr;
  fp = nullptr;
  numforce = nullptr;
  type2frho = nullptr;

  nfuncfl = 0;
  funcfl = nullptr;

  setfl = nullptr;
  fs = nullptr;

  frho = nullptr;
  rhor = nullptr;
  z2r = nullptr;
  scale = nullptr;

  rhomax = rhomin = 0.0;

  frho_spline = nullptr;
  rhor_spline = nullptr;
  z2r_spline = nullptr;

  // set comm size needed by this Pair

  comm_forward = 1;
  comm_reverse = 1;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairEAM::~PairEAM()
{
  if (copymode) return;

  memory->destroy(rho);
  memory->destroy(fp);
  memory->destroy(numforce);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] type2frho;
    type2frho = nullptr;
    memory->destroy(type2rhor);
    memory->destroy(type2z2r);
    memory->destroy(scale);
  }

  if (funcfl) {
    for (int i = 0; i < nfuncfl; i++) {
      delete [] funcfl[i].file;
      memory->destroy(funcfl[i].frho);
      memory->destroy(funcfl[i].rhor);
      memory->destroy(funcfl[i].zr);
    }
    memory->sfree(funcfl);
    funcfl = nullptr;
  }

  if (setfl) {
    for (int i = 0; i < setfl->nelements; i++) delete [] setfl->elements[i];
    delete [] setfl->elements;
    memory->destroy(setfl->mass);
    memory->destroy(setfl->frho);
    memory->destroy(setfl->rhor);
    memory->destroy(setfl->z2r);
    delete setfl;
    setfl = nullptr;
  }

  if (fs) {
    for (int i = 0; i < fs->nelements; i++) delete [] fs->elements[i];
    delete [] fs->elements;
    memory->destroy(fs->mass);
    memory->destroy(fs->frho);
    memory->destroy(fs->rhor);
    memory->destroy(fs->z2r);
    delete fs;
    fs = nullptr;
  }

  memory->destroy(frho);
  memory->destroy(rhor);
  memory->destroy(z2r);

  memory->destroy(frho_spline);
  memory->destroy(rhor_spline);
  memory->destroy(z2r_spline);
}

/* ----------------------------------------------------------------------
   Functional core of the EAM computation.

   Everything in the anonymous namespace below is file-local infrastructure
   for the "functional extraction" of PairEAM::compute: a tiny set of generic
   combinators (map / reduce / filter), a bundle of the immutable spline
   tables, and the pure physics functions that the core math is expressed in.

   The physics is written exclusively as compositions of fmap / freduce /
   ffilter over pure functions of (types, r).  Loops appear only inside the
   generic combinators, never in the physics.
------------------------------------------------------------------------- */

namespace {

// ---------------------------------------------------------------------
// Generic combinators.  These are the ONLY place a loop is allowed.
// ---------------------------------------------------------------------

template <class F, class T>
auto fmap(F f, const std::vector<T> &xs) -> std::vector<decltype(f(std::declval<const T &>()))>
{
  std::vector<decltype(f(std::declval<const T &>()))> out;
  out.reserve(xs.size());
  for (const T &x : xs) out.push_back(f(x));
  return out;
}

template <class F, class A, class T>
A freduce(F f, A init, const std::vector<T> &xs)
{
  A acc = init;
  for (const T &x : xs) acc = f(acc, x);
  return acc;
}

template <class P, class T>
std::vector<T> ffilter(P p, const std::vector<T> &xs)
{
  std::vector<T> out;
  out.reserve(xs.size());
  for (const T &x : xs) if (p(x)) out.push_back(x);
  return out;
}

// index generator: [0,n) -- the "atoms" domain the outer maps run over
std::vector<int> findices(int n)
{
  std::vector<int> out;
  out.reserve(n);
  for (int i = 0; i < n; ++i) out.push_back(i);
  return out;
}

// ---------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------

struct Vec3 {
  double x, y, z;
};

Vec3 vzero()                            { return Vec3{0.0, 0.0, 0.0}; }
Vec3 vadd(const Vec3 &a, const Vec3 &b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 vsub(const Vec3 &a, const Vec3 &b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 vscale(const Vec3 &a, double s)    { return Vec3{a.x * s, a.y * s, a.z * s}; }
double vnorm2(const Vec3 &a)            { return a.x * a.x + a.y * a.y + a.z * a.z; }

struct AtomView {
  Vec3 x;
  int type;
};

// one entry of the (half) neighbor list, i.e. one interacting pair
struct PairRef {
  int i, j;
};

// per-pair record, everything TEARDOWN needs to drive ev_tally()
struct PairContrib {
  int i, j;
  double evdwl, fpair;
  double delx, dely, delz;
};

// (F, F') of the embedding function, plus the rhomax-extrapolation flag
struct Embed {
  double energy, deriv;
  int beyond_rhomax;
};

// ---------------------------------------------------------------------
// The immutable table state of PairEAM, bundled so that the pure physics
// functions are referentially transparent in their true inputs (types, r).
// Filled once in SETUP; never written by the core.
// ---------------------------------------------------------------------

struct EamTables {
  double rdr, rdrho, cutforcesq, rhomax;
  int nr, nrho;
  double ***rhor_spline, ***frho_spline, ***z2r_spline;
  int **type2rhor, **type2z2r, *type2frho;
  double **scale;
};

// ---------------------------------------------------------------------
// Pure physics.  Each spline table IS the numerical representation of the
// analytic rho / phi / F, so a pure wrapper around the table lookup is a
// faithful rendering of the analytic function.
// ---------------------------------------------------------------------

struct SplineIndex {
  int m;
  double p;
};

// interval + local coordinate for the r-indexed tables (rhor, z2r)
SplineIndex r_index(const EamTables &t, double r)
{
  double p = r * t.rdr + 1.0;
  int m = static_cast<int>(p);
  m = MIN(m, t.nr - 1);
  p -= m;
  p = MIN(p, 1.0);
  return SplineIndex{m, p};
}

// interval + local coordinate for the rho-indexed table (frho)
SplineIndex rho_index(const EamTables &t, double rho)
{
  double p = rho * t.rdrho + 1.0;
  int m = static_cast<int>(p);
  m = MAX(1, MIN(m, t.nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  return SplineIndex{m, p};
}

double spline_value(const double *c, double p) { return ((c[3] * p + c[4]) * p + c[5]) * p + c[6]; }
double spline_deriv(const double *c, double p) { return (c[0] * p + c[1]) * p + c[2]; }

// rho_b(r): density contributed at an atom of type dst by an atom of type src
double rho_of_r(const EamTables &t, int src_type, int dst_type, double r)
{
  const SplineIndex s = r_index(t, r);
  return spline_value(t.rhor_spline[t.type2rhor[src_type][dst_type]][s.m], s.p);
}

// rho_b'(r)
double drho_of_r(const EamTables &t, int src_type, int dst_type, double r)
{
  const SplineIndex s = r_index(t, r);
  return spline_deriv(t.rhor_spline[t.type2rhor[src_type][dst_type]][s.m], s.p);
}

// z2(r) = r*phi(r), the quantity the tables actually store
double z2_of_r(const EamTables &t, int itype, int jtype, double r)
{
  const SplineIndex s = r_index(t, r);
  return spline_value(t.z2r_spline[t.type2z2r[itype][jtype]][s.m], s.p);
}

double dz2_of_r(const EamTables &t, int itype, int jtype, double r)
{
  const SplineIndex s = r_index(t, r);
  return spline_deriv(t.z2r_spline[t.type2z2r[itype][jtype]][s.m], s.p);
}

// phi_ab(r) = z2(r)/r, unscaled
double phi_of_r(const EamTables &t, int itype, int jtype, double r)
{
  return z2_of_r(t, itype, jtype, r) * (1.0 / r);
}

// phi_ab'(r) = (z2'(r) - z2(r)/r)/r, unscaled
double dphi_of_r(const EamTables &t, int itype, int jtype, double r)
{
  const double recip = 1.0 / r;
  return dz2_of_r(t, itype, jtype, r) * recip - phi_of_r(t, itype, jtype, r) * recip;
}

// pair energy actually tallied by LAMMPS: scale_ab * phi_ab(r)
double pair_energy(const EamTables &t, int itype, int jtype, double r)
{
  return t.scale[itype][jtype] * phi_of_r(t, itype, jtype, r);
}

// (F(rho), F'(rho)) with the linear extrapolation beyond rhomax folded in.
// F is returned already multiplied by the diagonal scale factor, matching
// the point at which the original applies it.
Embed F_embed(const EamTables &t, int itype, double rho_bar)
{
  const SplineIndex s = rho_index(t, rho_bar);
  const double *c = t.frho_spline[t.type2frho[itype]][s.m];
  const double deriv = spline_deriv(c, s.p);
  const double raw = spline_value(c, s.p);
  const bool beyond = (rho_bar > t.rhomax);
  const double extrapolated = beyond ? raw + deriv * (rho_bar - t.rhomax) : raw;
  return Embed{extrapolated * t.scale[itype][itype], deriv, beyond ? 1 : 0};
}

// scalar pair force factor: fpair = -scale_ab * psip / r with
// psip = F'(rho_bar_i) rho_b'(r) + F'(rho_bar_j) rho_a'(r) + phi_ab'(r)
double pair_fpair(const EamTables &t, int itype, int jtype, double r, double fp_i, double fp_j)
{
  const double rhoip = drho_of_r(t, itype, jtype, r);    // d(rho at j due to i)
  const double rhojp = drho_of_r(t, jtype, itype, r);    // d(rho at i due to j)
  const double phip = dphi_of_r(t, itype, jtype, r);
  const double psip = fp_i * rhojp + fp_j * rhoip + phip;
  return -t.scale[itype][jtype] * psip * (1.0 / r);
}

// ---------------------------------------------------------------------
// Core math -- stage 1a: rho_bar_i = sum_{j in N(i)} rho_{type_j}(r_ij)
// ---------------------------------------------------------------------

std::vector<double> eam_core_density(const EamTables &t, const std::vector<AtomView> &atoms,
                                     const std::vector<std::vector<int>> &neighbors)
{
  return fmap(
      [&](int i) {
        const AtomView &ai = atoms[i];
        return freduce(
            [&](double acc, int j) {
              return acc + rho_of_r(t, atoms[j].type, ai.type, std::sqrt(vnorm2(vsub(ai.x, atoms[j].x))));
            },
            0.0,
            ffilter([&](int j) { return vnorm2(vsub(ai.x, atoms[j].x)) < t.cutforcesq; }, neighbors[i]));
      },
      findices(static_cast<int>(atoms.size())));
}

// ---------------------------------------------------------------------
// Core math -- stage 1b: per-atom embedding energy and its derivative
// ---------------------------------------------------------------------

std::vector<Embed> eam_core_embedding(const EamTables &t, const std::vector<AtomView> &atoms,
                                      const std::vector<double> &rho_bar,
                                      const std::vector<int> &owned)
{
  return fmap([&](int i) { return F_embed(t, atoms[i].type, rho_bar[i]); }, owned);
}

// ---------------------------------------------------------------------
// Core math -- stage 2: forces (needs fp of the *neighbours*, hence a
// second stage) and the per-pair energy/virial contributions.
// ---------------------------------------------------------------------

struct EamStage2 {
  std::vector<Vec3> force;
  std::vector<PairContrib> pair_contribs;
};

// force contribution on i from a single neighbour j
Vec3 pair_force_vec(const EamTables &t, const std::vector<AtomView> &atoms,
                    const std::vector<double> &fp, int i, int j)
{
  const Vec3 del = vsub(atoms[i].x, atoms[j].x);
  const double r = std::sqrt(vnorm2(del));
  return vscale(del, pair_fpair(t, atoms[i].type, atoms[j].type, r, fp[i], fp[j]));
}

PairContrib pair_contrib(const EamTables &t, const std::vector<AtomView> &atoms,
                         const std::vector<double> &fp, const PairRef &pr)
{
  const Vec3 del = vsub(atoms[pr.i].x, atoms[pr.j].x);
  const double r = std::sqrt(vnorm2(del));
  const int itype = atoms[pr.i].type;
  const int jtype = atoms[pr.j].type;
  return PairContrib{pr.i, pr.j, pair_energy(t, itype, jtype, r),
                     pair_fpair(t, itype, jtype, r, fp[pr.i], fp[pr.j]), del.x, del.y, del.z};
}

EamStage2 eam_core_forces(const EamTables &t, const std::vector<AtomView> &atoms,
                          const std::vector<std::vector<int>> &neighbors,
                          const std::vector<PairRef> &half_pairs, const std::vector<double> &fp)
{
  std::vector<Vec3> force = fmap(
      [&](int i) {
        return freduce(
            vadd, vzero(),
            fmap([&](int j) { return pair_force_vec(t, atoms, fp, i, j); },
                 ffilter([&](int j) { return vnorm2(vsub(atoms[i].x, atoms[j].x)) < t.cutforcesq; },
                         neighbors[i])));
      },
      findices(static_cast<int>(atoms.size())));

  std::vector<PairContrib> contribs =
      fmap([&](const PairRef &pr) { return pair_contrib(t, atoms, fp, pr); },
           ffilter([&](const PairRef &pr) {
             return vnorm2(vsub(atoms[pr.i].x, atoms[pr.j].x)) < t.cutforcesq;
           }, half_pairs));

  return EamStage2{std::move(force), std::move(contribs)};
}

}    // namespace

/* ---------------------------------------------------------------------- */

void PairEAM::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(rho);
    memory->destroy(fp);
    memory->destroy(numforce);
    nmax = atom->nmax;
    memory->create(rho,nmax,"pair:rho");
    memory->create(fp,nmax,"pair:fp");
    memory->create(numforce,nmax,"pair:numforce");
  }

  // ------------------------------------------------------------------
  // SETUP: marshal LAMMPS data structures into plain values
  // ------------------------------------------------------------------

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int newton_pair = force->newton_pair;

  const int inum = list->inum;
  const int *const ilist = list->ilist;
  const int *const numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  const EamTables tables = {rdr, rdrho, cutforcesq, rhomax, nr, nrho,
                            rhor_spline, frho_spline, z2r_spline,
                            type2rhor, type2z2r, type2frho, scale};

  std::vector<AtomView> atoms(nall);
  for (int i = 0; i < nall; i++)
    atoms[i] = AtomView{Vec3{x[i][0], x[i][1], x[i][2]}, type[i]};

  // the half neighbor list, flattened into plain (i,j) pairs, in the exact
  // order the original loops visited them (so ev_tally sees the same order)

  std::vector<PairRef> half_pairs;
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; jj++)
      half_pairs.push_back(PairRef{i, jlist[jj] & NEIGHMASK});
  }

  // symmetrized per-atom neighbor sets: (i,j) puts j into N(i), and i into
  // N(j) exactly when the original applied the reverse contribution
  // (newton_pair || j < nlocal).  Cutoff filtering happens in the core.

  std::vector<std::vector<int>> neighbors(nall);
  for (const PairRef &pr : half_pairs) {
    neighbors[pr.i].push_back(pr.j);
    if (newton_pair || pr.j < nlocal) neighbors[pr.j].push_back(pr.i);
  }

  std::vector<int> owned(ilist, ilist + inum);

  // ------------------------------------------------------------------
  // CORE stage 1a: densities
  // ------------------------------------------------------------------

  const std::vector<double> rho_bar = eam_core_density(tables, atoms, neighbors);

  // GLUE: scatter densities back and let MPI sum the ghost contributions

  for (int i = 0; i < nall; i++) rho[i] = rho_bar[i];
  if (newton_pair) comm->reverse_comm(this);

  // ------------------------------------------------------------------
  // CORE stage 1b: embedding energy and its derivative per owned atom
  // ------------------------------------------------------------------

  std::vector<double> rho_summed(nall);
  for (int i = 0; i < nall; i++) rho_summed[i] = rho[i];

  const std::vector<Embed> embed = eam_core_embedding(tables, atoms, rho_summed, owned);

  // TEARDOWN (embedding part): fp array, energy accumulators

  int beyond_rhomax = 0;
  for (int ii = 0; ii < inum; ii++) {
    const int i = owned[ii];
    fp[i] = embed[ii].deriv;
    if (eflag) {
      if (embed[ii].beyond_rhomax) beyond_rhomax = 1;
      if (eflag_global) eng_vdwl += embed[ii].energy;
      if (eflag_atom) eatom[i] += embed[ii].energy;
    }
  }

  // GLUE: distribute the embedding derivative to the ghosts

  comm->forward_comm(this);
  embedstep = update->ntimestep;

  std::vector<double> fp_all(fp, fp + nall);

  // ------------------------------------------------------------------
  // CORE stage 2: forces and per-pair energy/virial contributions
  // ------------------------------------------------------------------

  const EamStage2 stage2 = eam_core_forces(tables, atoms, neighbors, half_pairs, fp_all);

  // ------------------------------------------------------------------
  // TEARDOWN: scatter results into the LAMMPS accumulators
  // ------------------------------------------------------------------

  for (int i = 0; i < nall; i++) {
    f[i][0] += stage2.force[i].x;
    f[i][1] += stage2.force[i].y;
    f[i][2] += stage2.force[i].z;
  }

  for (int ii = 0; ii < inum; ii++) numforce[owned[ii]] = 0;
  for (const PairContrib &c : stage2.pair_contribs) ++numforce[c.i];

  if (evflag)
    for (const PairContrib &c : stage2.pair_contribs)
      ev_tally(c.i, c.j, nlocal, newton_pair, eflag ? c.evdwl : 0.0, 0.0, c.fpair, c.delx, c.dely,
               c.delz);

  if (eflag && (!exceeded_rhomax)) {
    MPI_Allreduce(&beyond_rhomax, &exceeded_rhomax, 1, MPI_INT, MPI_SUM, world);
    if (exceeded_rhomax) {
      if (comm->me == 0)
        error->warning(FLERR,
                       "A per-atom density exceeded rhomax of EAM potential table - "
                       "a linear extrapolation to the energy was made");
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/*********************************************************************
 * Calculates the atomic energy of atom i
 *********************************************************************/
double PairEAM::compute_atomic_energy(int i, NeighList *neighborList)
{
  double p;
  int m;
  double* coeff;
  double Ei = 0.0;
  double rhoi = 0.0;

  double xi = atom->x[i][0];
  double yi = atom->x[i][1];
  double zi = atom->x[i][2];
  int itype = atom->type[i];

  // loop over all neighbors of the selected atom.

  int* jlist = neighborList->firstneigh[i];
  int jnum = neighborList->numneigh[i];

  for(int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj];

    double delx = xi - atom->x[j][0];
    double dely = yi - atom->x[j][1];
    double delz = zi - atom->x[j][2];
    double rsq = delx*delx + dely*dely + delz*delz;
    if(rsq >= cutforcesq) continue;

    int jtype = atom->type[j];
    double r = sqrt(rsq);

    p = r * rdr + 1.0;
    m = static_cast<int>(p);
    m = MIN(m, nr - 1);
    p -= m;
    p = MIN(p, 1.0);

    // sum pair energy ij
    // divide by 2 to avoid double counting energy

    coeff = z2r_spline[type2z2r[jtype][itype]][m];
    double z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
    Ei += 0.5*z2 / r;

    // sum rho_ij to rho_i
    coeff = rhor_spline[type2rhor[jtype][itype]][m];
    rhoi += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
  }

  // compute the change in embedding energy of atom i.

  p = rhoi * rdrho + 1.0;
  m = static_cast<int>(p);
  m = MAX(1, MIN(m, nrho - 1));
  p -= m;
  p = MIN(p, 1.0);
  coeff = frho_spline[type2frho[itype]][m];
  Ei += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

  return Ei;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairEAM::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  delete[] map;
  map = new int[n+1];
  for (int i = 1; i <= n; i++) map[i] = -1;

  type2frho = new int[n+1];
  memory->create(type2rhor,n+1,n+1,"pair:type2rhor");
  memory->create(type2z2r,n+1,n+1,"pair:type2z2r");
  memory->create(scale,n+1,n+1,"pair:scale");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairEAM::settings(int narg, char **/*arg*/)
{
  if (narg > 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void PairEAM::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 3) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));

  // parse pair of atom types

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct

  int ifuncfl;
  for (ifuncfl = 0; ifuncfl < nfuncfl; ifuncfl++)
    if (strcmp(arg[2],funcfl[ifuncfl].file) == 0) break;

  if (ifuncfl == nfuncfl) {
    nfuncfl++;
    funcfl = (Funcfl *)
      memory->srealloc(funcfl,nfuncfl*sizeof(Funcfl),"pair:funcfl");
    read_file(arg[2]);
    funcfl[ifuncfl].file = utils::strdup(arg[2]);
  }

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (i == j) {
        setflag[i][i] = 1;
        map[i] = ifuncfl;
        atom->set_mass(FLERR,i,funcfl[ifuncfl].mass);
        count++;
      }
      scale[i][j] = 1.0;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();

  neighbor->add_request(this);
  embedstep = -1;

  exceeded_rhomax = 0;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairEAM::init_one(int i, int j)
{
  // single global cutoff = max of cut from all files read in
  // for funcfl could be multiple files
  // for setfl or fs, just one file

  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];

  if (funcfl) {
    cutmax = 0.0;
    for (int m = 0; m < nfuncfl; m++)
      cutmax = MAX(cutmax,funcfl[m].cut);
  } else if (setfl) cutmax = setfl->cut;
  else if (fs) cutmax = fs->cut;

  cutforcesq = cutmax*cutmax;

  return cutmax;
}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void PairEAM::read_file(char *filename)
{
  Funcfl *file = &funcfl[nfuncfl-1];

  // read potential file
  if (comm->me == 0) {
    PotentialFileReader reader(lmp, filename, "eam", unit_convert_flag);

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY,
                                                            unit_convert);
    try {
      reader.skip_line();

      ValueTokenizer values = reader.next_values(2);
      values.next_int(); // ignore
      file->mass = values.next_double();

      values = reader.next_values(5);
      file->nrho = values.next_int();
      file->drho = values.next_double();
      file->nr   = values.next_int();
      file->dr   = values.next_double();
      file->cut  = values.next_double();

      if ((file->nrho <= 0) || (file->nr <= 0) || (file->dr <= 0.0))
        error->one(FLERR,"Invalid EAM potential file");

      memory->create(file->frho, (file->nrho+1), "pair:frho");
      memory->create(file->rhor, (file->nr+1), "pair:rhor");
      memory->create(file->zr, (file->nr+1), "pair:zr");

      reader.next_dvector(&file->frho[1], file->nrho);
      reader.next_dvector(&file->zr[1], file->nr);
      reader.next_dvector(&file->rhor[1], file->nr);

      if (unit_convert) {
        const double sqrt_conv = sqrt(conversion_factor);
        for (int i = 1; i <= file->nrho; ++i)
          file->frho[i] *= conversion_factor;
        for (int j = 1; j <= file->nr; ++j)
          file->zr[j] *= sqrt_conv;
      }
    } catch (TokenizerException &e) {
      error->one(FLERR, e.what());
    }
  }

  MPI_Bcast(&file->mass, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&file->nrho, 1, MPI_INT, 0, world);
  MPI_Bcast(&file->drho, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&file->nr, 1, MPI_INT, 0, world);
  MPI_Bcast(&file->dr, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, world);

  if (comm->me != 0) {
    memory->create(file->frho, (file->nrho+1), "pair:frho");
    memory->create(file->rhor, (file->nr+1), "pair:rhor");
    memory->create(file->zr, (file->nr+1), "pair:zr");
  }

  MPI_Bcast(&file->frho[1], file->nrho, MPI_DOUBLE, 0, world);
  MPI_Bcast(&file->zr[1], file->nr, MPI_DOUBLE, 0, world);
  MPI_Bcast(&file->rhor[1], file->nr, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void PairEAM::file2array()
{
  int i,j,k,m,n;
  int ntypes = atom->ntypes;
  double sixth = 1.0/6.0;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  int active;
  double rmax;
  dr = drho = rmax = rhomax = 0.0;

  for (int i = 0; i < nfuncfl; i++) {
    active = 0;
    for (j = 1; j <= ntypes; j++)
      if (map[j] == i) active = 1;
    if (active == 0) continue;
    Funcfl *file = &funcfl[i];
    dr = MAX(dr,file->dr);
    drho = MAX(drho,file->drho);
    rmax = MAX(rmax,(file->nr-1) * file->dr);
    rhomax = MAX(rhomax,(file->nrho-1) * file->drho);
  }

  // set nr,nrho from cutoff and spacings

  nr = std::lround(rmax/dr);
  nrho = std::lround(rhomax/drho);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array

  nfrho = nfuncfl + 1;
  memory->destroy(frho);
  memory->create(frho,nfrho,nrho+1,"pair:frho");

  // interpolate each file's frho to a single grid and cutoff

  double r,p,cof1,cof2,cof3,cof4;

  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *file = &funcfl[i];
    for (m = 1; m <= nrho; m++) {
      r = (m-1)*drho;
      p = r/file->drho + 1.0;
      k = static_cast<int>(p);
      k = MIN(k,file->nrho-2);
      k = MAX(k,2);
      p -= k;
      p = MIN(p,2.0);
      cof1 = -sixth*p*(p-1.0)*(p-2.0);
      cof2 = 0.5*(p*p-1.0)*(p-2.0);
      cof3 = -0.5*p*(p+1.0)*(p-2.0);
      cof4 = sixth*p*(p*p-1.0);
      frho[n][m] = cof1*file->frho[k-1] + cof2*file->frho[k] +
        cof3*file->frho[k+1] + cof4*file->frho[k+2];
    }
    n++;
  }

  // add extra frho of zeroes for non-EAM types to point to (pair hybrid)
  // this is necessary b/c fp is still computed for non-EAM atoms

  for (m = 1; m <= nrho; m++) frho[nfrho-1][m] = 0.0;

  // type2frho[i] = which frho array (0 to nfrho-1) each atom type maps to
  // if atom type doesn't point to file (non-EAM atom in pair hybrid)
  // then map it to last frho array of zeroes

  for (i = 1; i <= ntypes; i++)
    if (map[i] >= 0) type2frho[i] = map[i];
    else type2frho[i] = nfrho-1;

  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  nrhor = nfuncfl;
  memory->destroy(rhor);
  memory->create(rhor,nrhor,nr+1,"pair:rhor");

  // interpolate each file's rhor to a single grid and cutoff

  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *file = &funcfl[i];
    for (m = 1; m <= nr; m++) {
      r = (m-1)*dr;
      p = r/file->dr + 1.0;
      k = static_cast<int>(p);
      k = MIN(k,file->nr-2);
      k = MAX(k,2);
      p -= k;
      p = MIN(p,2.0);
      cof1 = -sixth*p*(p-1.0)*(p-2.0);
      cof2 = 0.5*(p*p-1.0)*(p-2.0);
      cof3 = -0.5*p*(p+1.0)*(p-2.0);
      cof4 = sixth*p*(p*p-1.0);
      rhor[n][m] = cof1*file->rhor[k-1] + cof2*file->rhor[k] +
        cof3*file->rhor[k+1] + cof4*file->rhor[k+2];
    }
    n++;
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  for (i = 1; i <= ntypes; i++)
    for (j = 1; j <= ntypes; j++)
      type2rhor[i][j] = map[i];

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  nz2r = nfuncfl*(nfuncfl+1)/2;
  memory->destroy(z2r);
  memory->create(z2r,nz2r,nr+1,"pair:z2r");

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff
  // final z2r includes unit conversion of 27.2 eV/Hartree and 0.529 Ang/Bohr

  double zri,zrj;

  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *ifile = &funcfl[i];
    for (j = 0; j <= i; j++) {
      Funcfl *jfile = &funcfl[j];

      for (m = 1; m <= nr; m++) {
        r = (m-1)*dr;

        p = r/ifile->dr + 1.0;
        k = static_cast<int>(p);
        k = MIN(k,ifile->nr-2);
        k = MAX(k,2);
        p -= k;
        p = MIN(p,2.0);
        cof1 = -sixth*p*(p-1.0)*(p-2.0);
        cof2 = 0.5*(p*p-1.0)*(p-2.0);
        cof3 = -0.5*p*(p+1.0)*(p-2.0);
        cof4 = sixth*p*(p*p-1.0);
        zri = cof1*ifile->zr[k-1] + cof2*ifile->zr[k] +
          cof3*ifile->zr[k+1] + cof4*ifile->zr[k+2];

        p = r/jfile->dr + 1.0;
        k = static_cast<int>(p);
        k = MIN(k,jfile->nr-2);
        k = MAX(k,2);
        p -= k;
        p = MIN(p,2.0);
        cof1 = -sixth*p*(p-1.0)*(p-2.0);
        cof2 = 0.5*(p*p-1.0)*(p-2.0);
        cof3 = -0.5*p*(p+1.0)*(p-2.0);
        cof4 = sixth*p*(p*p-1.0);
        zrj = cof1*jfile->zr[k-1] + cof2*jfile->zr[k] +
          cof3*jfile->zr[k+1] + cof4*jfile->zr[k+2];

        z2r[n][m] = 27.2*0.529 * zri*zrj;
      }
      n++;
    }
  }

  // type2z2r[i][j] = which z2r array (0 to nz2r-1) each type pair maps to
  // set of z2r arrays only fill lower triangular Nelement matrix
  // value = n = sum over rows of lower-triangular matrix until reach irow,icol
  // swap indices when irow < icol to stay lower triangular
  // if map = -1 (non-EAM atom in pair hybrid):
  //   type2z2r is not used by non-opt
  //   but set type2z2r to 0 since accessed by opt

  int irow,icol;
  for (i = 1; i <= ntypes; i++) {
    for (j = 1; j <= ntypes; j++) {
      irow = map[i];
      icol = map[j];
      if (irow == -1 || icol == -1) {
        type2z2r[i][j] = 0;
        continue;
      }
      if (irow < icol) {
        irow = map[j];
        icol = map[i];
      }
      n = 0;
      for (m = 0; m < irow; m++) n += m + 1;
      n += icol;
      type2z2r[i][j] = n;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairEAM::array2spline()
{
  rdr = 1.0/dr;
  rdrho = 1.0/drho;

  memory->destroy(frho_spline);
  memory->destroy(rhor_spline);
  memory->destroy(z2r_spline);

  memory->create(frho_spline,nfrho,nrho+1,7,"pair:frho");
  memory->create(rhor_spline,nrhor,nr+1,7,"pair:rhor");
  memory->create(z2r_spline,nz2r,nr+1,7,"pair:z2r");

  for (int i = 0; i < nfrho; i++)
    interpolate(nrho,drho,frho[i],frho_spline[i]);

  for (int i = 0; i < nrhor; i++)
    interpolate(nr,dr,rhor[i],rhor_spline[i]);

  for (int i = 0; i < nz2r; i++)
    interpolate(nr,dr,z2r[i],z2r_spline[i]);
}

/* ---------------------------------------------------------------------- */

void PairEAM::interpolate(int n, double delta, double *f, double **spline)
{
  for (int m = 1; m <= n; m++) spline[m][6] = f[m];

  spline[1][5] = spline[2][6] - spline[1][6];
  spline[2][5] = 0.5 * (spline[3][6]-spline[1][6]);
  spline[n-1][5] = 0.5 * (spline[n][6]-spline[n-2][6]);
  spline[n][5] = spline[n][6] - spline[n-1][6];

  for (int m = 3; m <= n-2; m++)
    spline[m][5] = ((spline[m-2][6]-spline[m+2][6]) +
                    8.0*(spline[m+1][6]-spline[m-1][6])) / 12.0;

  for (int m = 1; m <= n-1; m++) {
    spline[m][4] = 3.0*(spline[m+1][6]-spline[m][6]) -
      2.0*spline[m][5] - spline[m+1][5];
    spline[m][3] = spline[m][5] + spline[m+1][5] -
      2.0*(spline[m+1][6]-spline[m][6]);
  }

  spline[n][4] = 0.0;
  spline[n][3] = 0.0;

  for (int m = 1; m <= n; m++) {
    spline[m][2] = spline[m][5]/delta;
    spline[m][1] = 2.0*spline[m][4]/delta;
    spline[m][0] = 3.0*spline[m][3]/delta;
  }
}

/* ---------------------------------------------------------------------- */

double PairEAM::single(int i, int j, int itype, int jtype,
                       double rsq, double /*factor_coul*/, double /*factor_lj*/,
                       double &fforce)
{
  int m;
  double r,p,rhoip,rhojp,z2,z2p,recip,phi,phip,psip;
  double *coeff;

  if (!numforce)
    error->all(FLERR,"EAM embedding data required for this calculation is missing");

  if ((comm->me == 0) && (embedstep != update->ntimestep)) {
    error->warning(FLERR,"EAM embedding data not computed for this time step ");
    embedstep = update->ntimestep;
  }

  if (numforce[i] > 0) {
    p = rho[i]*rdrho + 1.0;
    m = static_cast<int>(p);
    m = MAX(1,MIN(m,nrho-1));
    p -= m;
    p = MIN(p,1.0);
    coeff = frho_spline[type2frho[itype]][m];
    phi = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
    if (rho[i] > rhomax) phi += fp[i] * (rho[i]-rhomax);
    phi *= 1.0/static_cast<double>(numforce[i]);
  } else phi = 0.0;

  r = sqrt(rsq);
  p = r*rdr + 1.0;
  m = static_cast<int>(p);
  m = MIN(m,nr-1);
  p -= m;
  p = MIN(p,1.0);

  coeff = rhor_spline[type2rhor[itype][jtype]][m];
  rhoip = (coeff[0]*p + coeff[1])*p + coeff[2];
  coeff = rhor_spline[type2rhor[jtype][itype]][m];
  rhojp = (coeff[0]*p + coeff[1])*p + coeff[2];
  coeff = z2r_spline[type2z2r[itype][jtype]][m];
  z2p = (coeff[0]*p + coeff[1])*p + coeff[2];
  z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

  recip = 1.0/r;
  phi += z2*recip;
  phip = z2p*recip - phi*recip;
  psip = fp[i]*rhojp + fp[j]*rhoip + phip;
  fforce = -psip*recip;

  return phi;
}

/* ---------------------------------------------------------------------- */

int PairEAM::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = fp[j];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairEAM::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) fp[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

int PairEAM::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = rho[i];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairEAM::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairEAM::memory_usage()
{
  double bytes = (double)maxeatom * sizeof(double);
  bytes += (double)maxvatom*6 * sizeof(double);
  bytes += (double)2 * nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   swap fp array with one passed in by caller
------------------------------------------------------------------------- */

void PairEAM::swap_eam(double *fp_caller, double **fp_caller_hold)
{
  double *tmp = fp;
  fp = fp_caller;
  *fp_caller_hold = tmp;

  // skip warning about out-of-sync timestep, since we already warn in the caller
  embedstep = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void *PairEAM::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"scale") == 0) return (void *) scale;

  return nullptr;
}

/* ----------------------------------------------------------------------
   peratom requests from FixPair
   return ptr to requested data
   also return ncol = # of quantites per atom
     0 = per-atom vector
     1 or more = # of columns in per-atom array
   return NULL if str is not recognized
---------------------------------------------------------------------- */

void *PairEAM::extract_peratom(const char *str, int &ncol)
{
  if (strcmp(str,"rho") == 0) {
    ncol = 0;
    return (void *) rho;
  } else if (strcmp(str,"fp") == 0) {
    ncol = 0;
    return (void *) fp;
  }

  return nullptr;
}
