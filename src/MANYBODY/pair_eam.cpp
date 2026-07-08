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
#include "neigh_request.h"
#include "potential_file_reader.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   Functional-form helper library for the EAM core math.

   The embedded-atom-model energy of atom i is
       E_i = F_a( sum_{j!=i} rho_b(r_ij) )  +  (1/2) sum_{j!=i} phi_ab(r_ij)
   (F = embedding function, rho = electron density, phi = pair potential).

   The helpers below express PairEAM::compute() as a small number of pure
   functions (rho, F, phi and their derivatives are cubic-Hermite spline
   lookups) combined by generic, order-preserving map / filter / reduce
   combinators.  The combinators build and fold their results in neighbor-list
   order so the floating-point accumulation order is identical to the original
   hand-written loops (required to stay within the differential-test epsilon).
------------------------------------------------------------------------- */

namespace {

// --- generic, order-preserving combinators --------------------------------

template <class T, class F>
auto map(const std::vector<T> &xs, F f) -> std::vector<decltype(f(xs[0]))>
{
  std::vector<decltype(f(xs[0]))> out;
  out.reserve(xs.size());
  for (const T &x : xs) out.push_back(f(x));
  return out;
}

template <class T, class Pred>
std::vector<T> filter(const std::vector<T> &xs, Pred keep)
{
  std::vector<T> out;
  for (const T &x : xs)
    if (keep(x)) out.push_back(x);
  return out;
}

// left fold: reduce(xs, init, f) = f(f(f(init, x0), x1), ...) in list order
template <class T, class A, class F>
A reduce(const std::vector<T> &xs, A init, F f)
{
  A acc = init;
  for (const T &x : xs) acc = f(acc, x);
  return acc;
}

// --- value types ----------------------------------------------------------

struct Vec3 {
  double x;
  double y;
  double z;
};

// a spline argument located in bin `index` at fractional offset `fraction`
struct SplinePoint {
  int index;
  double fraction;
};

// spline terms needed to build the per-pair force and pair energy
struct EAMPairSplineTerms {
  double density_derivative_i;             // rho'_(i->j)(r)
  double density_derivative_j;             // rho'_(j->i)(r)
  double pair_potential_times_r;           // z2(r) = r * phi(r)
  double pair_potential_times_r_derivative;// z2'(r)
};

// an in-cutoff neighbor j of some atom i, with the geometry it induces
struct NeighborContext {
  int j;
  int jtype;
  Vec3 rij;
  double r;
  SplinePoint radial;
};

// scalar force magnitude/r and the (scaled) pair energy for one pair
struct PairForceEnergy {
  double fpair;
  double pair_energy;
};

// --- pure geometry --------------------------------------------------------

Vec3 atom_position(double **x, int i)
{
  return {x[i][0], x[i][1], x[i][2]};
}

Vec3 displacement(const Vec3 &xi, const Vec3 &xj)
{
  return {xi.x - xj.x, xi.y - xj.y, xi.z - xj.z};
}

double dot(const Vec3 &a, const Vec3 &b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

int neighbor_atom(int packed_neighbor)
{
  return packed_neighbor & NEIGHMASK;
}

// --- pure spline evaluation -----------------------------------------------

// cubic value: ((c3*p + c4)*p + c5)*p + c6
double eam_spline_value(const double *coeff, double p)
{
  return ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
}

// analytic derivative (already divided by grid spacing): (c0*p + c1)*p + c2
double eam_spline_derivative(const double *coeff, double p)
{
  return (coeff[0]*p + coeff[1])*p + coeff[2];
}

SplinePoint eam_radial_spline_point(double r, double rdr, int nr)
{
  double p = r*rdr + 1.0;
  int m = static_cast<int>(p);
  m = MIN(m,nr-1);
  p -= m;
  p = MIN(p,1.0);
  return {m,p};
}

SplinePoint eam_density_spline_point(double density, double rdrho, int nrho)
{
  double p = density*rdrho + 1.0;
  int m = static_cast<int>(p);
  m = MAX(1,MIN(m,nrho-1));
  p -= m;
  p = MIN(p,1.0);
  return {m,p};
}

// --- pure EAM physics (rho, F, F', phi) -----------------------------------

// rho: electron density deposited by a `source_type` atom onto a `target_type`
// atom at radial spline point `r`
double eam_density_value(double ***rhor_spline, int **type2rhor, int source_type,
                         int target_type, const SplinePoint &r)
{
  return eam_spline_value(rhor_spline[type2rhor[source_type][target_type]][r.index], r.fraction);
}

// F': derivative of the embedding energy at density point `density`
double eam_embedding_derivative(double ***frho_spline, int *type2frho, int type,
                                const SplinePoint &density)
{
  return eam_spline_derivative(frho_spline[type2frho[type]][density.index], density.fraction);
}

// F: embedding energy at density point `density`
double eam_embedding_energy(double ***frho_spline, int *type2frho, int type,
                            const SplinePoint &density)
{
  return eam_spline_value(frho_spline[type2frho[type]][density.index], density.fraction);
}

EAMPairSplineTerms eam_pair_spline_terms(double ***rhor_spline, double ***z2r_spline,
                                         int **type2rhor, int **type2z2r, int itype, int jtype,
                                         const SplinePoint &r)
{
  const double density_derivative_i =
      eam_spline_derivative(rhor_spline[type2rhor[itype][jtype]][r.index], r.fraction);
  const double density_derivative_j =
      eam_spline_derivative(rhor_spline[type2rhor[jtype][itype]][r.index], r.fraction);
  const double *z = z2r_spline[type2z2r[itype][jtype]][r.index];
  const double pair_potential_times_r_derivative = eam_spline_derivative(z, r.fraction);
  const double pair_potential_times_r = eam_spline_value(z, r.fraction);
  return {density_derivative_i, density_derivative_j, pair_potential_times_r,
          pair_potential_times_r_derivative};
}

// per-pair scalar force (fpair) and pair energy, i.e. the analytic gradient of
// the EAM energy for one pair.  phi = z2/r, phi' = z2'/r - phi/r, and
// psip = dE/dr = F'(rho_i) rho'_(j->i) + F'(rho_j) rho'_(i->j) + phi'.
PairForceEnergy eam_pair_force_energy(const EAMPairSplineTerms &terms, double fp_i, double fp_j,
                                      double r, double scale)
{
  const double recip = 1.0/r;
  const double phi = terms.pair_potential_times_r*recip;
  const double phip = terms.pair_potential_times_r_derivative*recip - phi*recip;
  const double psip = fp_i*terms.density_derivative_j + fp_j*terms.density_derivative_i + phip;
  return {-scale*psip*recip, scale*phi};
}

// --- neighbor selection: filter to within-cutoff, map to NeighborContext ---

// filter(not_curr_atom / within cutoff) . map(distance) over atom i's half list
std::vector<NeighborContext> in_cutoff_neighbors(int i, double **x, int *type, int *jlist,
                                                 int jnum, double cutforcesq, double rdr, int nr)
{
  struct RawNeighbor {
    int j;
    int jtype;
    Vec3 rij;
    double rsq;
  };

  const Vec3 xi = atom_position(x,i);

  std::vector<int> slots(jnum);
  for (int jj = 0; jj < jnum; jj++) slots[jj] = jj;

  const std::vector<RawNeighbor> raw = map(slots, [&](int jj) {
    const int j = neighbor_atom(jlist[jj]);
    const Vec3 rij = displacement(xi, atom_position(x,j));
    return RawNeighbor{j, type[j], rij, dot(rij,rij)};
  });

  const std::vector<RawNeighbor> within =
      filter(raw, [&](const RawNeighbor &nb) { return nb.rsq < cutforcesq; });

  return map(within, [&](const RawNeighbor &nb) {
    const double r = sqrt(nb.rsq);
    return NeighborContext{nb.j, nb.jtype, nb.rij, r, eam_radial_spline_point(r, rdr, nr)};
  });
}

// full neighbor list: accumulate the pair force onto atom i only; the reciprocal
// force on j is produced when atom j is visited with i as its neighbor
void accumulate_pair_force(double **f, int i, const Vec3 &rij, double fpair)
{
  f[i][0] += rij.x*fpair;
  f[i][1] += rij.y*fpair;
  f[i][2] += rij.z*fpair;
}

}    // namespace

/* ---------------------------------------------------------------------- */

PairEAM::PairEAM(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  atomic_energy_enable = 1;

  // full neighbor list: each atom sees all its neighbors, so the virial is
  // tallied per pair (via ev_tally_full) rather than through fdotr
  no_virial_fdotr_compute = 1;

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

/* ---------------------------------------------------------------------- */

void PairEAM::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  int beyond_rhomax = 0;

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

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // -------------------------------------------------------------------------
  // Stage 1+2 - density and embedding:  map over atoms
  //
  //     rho_i = sum_{j != i} rho_beta(r_ij)     (full list: reduce over all
  //                                              of atom i's neighbors)
  //     fp_i  = F'(rho_i),   E += F(rho_i)
  //
  // With a full neighbor list each atom's density is complete from its own
  // neighbor loop, so density and embedding fuse into a single map over atoms
  // (no Newton scatter, no reverse communication).  If rho > rhomax (close
  // approach of two atoms) it exceeds the table, so a linear term conserves
  // energy.
  // -------------------------------------------------------------------------

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i];
    const std::vector<NeighborContext> neighbors =
        in_cutoff_neighbors(i, x, type, firstneigh[i], numneigh[i], cutforcesq, rdr, nr);

    rho[i] = reduce(neighbors, 0.0, [&](double acc, const NeighborContext &nb) {
      return acc + eam_density_value(rhor_spline, type2rhor, nb.jtype, itype, nb.radial);
    });

    const SplinePoint density = eam_density_spline_point(rho[i], rdrho, nrho);
    fp[i] = eam_embedding_derivative(frho_spline, type2frho, itype, density);
    if (eflag) {
      double phi = eam_embedding_energy(frho_spline, type2frho, itype, density);
      if (rho[i] > rhomax) {
        phi += fp[i] * (rho[i]-rhomax);
        beyond_rhomax = 1;
      }
      phi *= scale[itype][itype];
      if (eflag_global) eng_vdwl += phi;
      if (eflag_atom) eatom[i] += phi;
    }
  }

  // communicate derivative of embedding function to ghost atoms
  // (the force stage reads fp[j] for neighbor j, which may be a ghost)

  comm->forward_comm(this);
  embedstep = update->ntimestep;

  // -------------------------------------------------------------------------
  // Stage 3 - forces + pair energy:  fold over each atom's neighbors
  //
  // For each pair the scalar force is fpair = -(dE/dr)/r, where
  //   dE/dr = F'(rho_i) rho'_(j->i) + F'(rho_j) rho'_(i->j) + phi'(r).
  // With a full list the force is accumulated onto atom i only (the reciprocal
  // force is produced when atom j is visited with i as its neighbor).
  // ev_tally_full tallies half the pair energy phi(r) and half the per-pair
  // virial; summed over the two visits of each pair this yields the full
  // (1/2) sum phi energy and the full virial.
  // -------------------------------------------------------------------------

  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];
    const int itype = type[i];
    const std::vector<NeighborContext> neighbors =
        in_cutoff_neighbors(i, x, type, firstneigh[i], numneigh[i], cutforcesq, rdr, nr);
    numforce[i] = static_cast<int>(neighbors.size());

    for (const NeighborContext &nb : neighbors) {
      const EAMPairSplineTerms terms = eam_pair_spline_terms(
          rhor_spline, z2r_spline, type2rhor, type2z2r, itype, nb.jtype, nb.radial);
      const PairForceEnergy pe =
          eam_pair_force_energy(terms, fp[i], fp[nb.j], nb.r, scale[itype][nb.jtype]);

      accumulate_pair_force(f, i, nb.rij, pe.fpair);

      const double evdwl = eflag ? pe.pair_energy : 0.0;
      if (evflag)
        ev_tally_full(i, evdwl, 0.0, pe.fpair, nb.rij.x, nb.rij.y, nb.rij.z);
    }
  }

  if (eflag && (!exceeded_rhomax)) {
    MPI_Allreduce(&beyond_rhomax, &exceeded_rhomax, 1, MPI_INT, MPI_SUM, world);
    if (exceeded_rhomax) {
      if (comm->me == 0)
        error->warning(FLERR,
                       "A per-atom density exceeded rhomax of EAM potential table - "
                       "a linear extrapolation to the energy was made");
    }
  }
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

  neighbor->add_request(this, NeighConst::REQ_FULL);
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
