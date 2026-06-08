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

#include "pair_lj_long_coul_long_deopt.h"

#include "force.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairLJLongCoulLongDeopt::PairLJLongCoulLongDeopt(LAMMPS *lmp) : PairLJLongCoulLong(lmp)
{
  respa_enable = 1;
}

/* ---------------------------------------------------------------------- */

void PairLJLongCoulLongDeopt::compute(int eflag, int vflag)
{
#ifdef LAMMPS_LJLC_DEOPT_ENABLE_EVAL
  ev_init(eflag,vflag);
  dispatch_candidate<false>();
#else
  PairLJLongCoulLong::compute(eflag, vflag);
#endif
}

/* ---------------------------------------------------------------------- */

void PairLJLongCoulLongDeopt::compute_outer(int eflag, int vflag)
{
#ifdef LAMMPS_LJLC_DEOPT_ENABLE_EVAL_OUTER
  ev_init(eflag,vflag);
  dispatch_candidate<true>();
#else
  PairLJLongCoulLong::compute_outer(eflag, vflag);
#endif
}

/* ---------------------------------------------------------------------- */

template <const bool OUTER>
void PairLJLongCoulLongDeopt::dispatch_candidate()
{
  const int order1 = ewald_order & (1 << 1);
  const int order6 = ewald_order & (1 << 6);

  if (order6) {
    if (order1) dispatch_tables<OUTER,1,1>();
    else dispatch_tables<OUTER,0,1>();
  } else {
    if (order1) dispatch_tables<OUTER,1,0>();
    else dispatch_tables<OUTER,0,0>();
  }
}

template <const bool OUTER, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::dispatch_tables()
{
  if (ncoultablebits) {
    if (ndisptablebits) dispatch_ev<OUTER,1,1,ORDER1,ORDER6>();
    else dispatch_ev<OUTER,1,0,ORDER1,ORDER6>();
  } else {
    if (ndisptablebits) dispatch_ev<OUTER,0,1,ORDER1,ORDER6>();
    else dispatch_ev<OUTER,0,0,ORDER1,ORDER6>();
  }
}

template <const bool OUTER, const int CTABLE, const int LJTABLE, const int ORDER1,
          const int ORDER6>
void PairLJLongCoulLongDeopt::dispatch_ev()
{
  if (evflag) {
    if (eflag_global || eflag_atom) dispatch_newton<OUTER,1,1,CTABLE,LJTABLE,ORDER1,ORDER6>();
    else dispatch_newton<OUTER,1,0,CTABLE,LJTABLE,ORDER1,ORDER6>();
  } else {
    dispatch_newton<OUTER,0,0,CTABLE,LJTABLE,ORDER1,ORDER6>();
  }
}

template <const bool OUTER, const int EVFLAG, const int EFLAG, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::dispatch_newton()
{
  if (force->newton_pair) dispatch_eval<OUTER,EVFLAG,EFLAG,1,CTABLE,LJTABLE,ORDER1,ORDER6>();
  else dispatch_eval<OUTER,EVFLAG,EFLAG,0,CTABLE,LJTABLE,ORDER1,ORDER6>();
}

template <const bool OUTER, const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
          const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::dispatch_eval()
{
  if constexpr (OUTER) eval_outer<EVFLAG,EFLAG,NEWTON_PAIR,CTABLE,LJTABLE,ORDER1,ORDER6>();
  else eval<EVFLAG,EFLAG,NEWTON_PAIR,CTABLE,LJTABLE,ORDER1,ORDER6>();
}

#ifdef LAMMPS_LJLC_DEOPT_IMPL
#include LAMMPS_LJLC_DEOPT_IMPL
#endif
