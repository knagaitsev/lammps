/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/long/coul/long/deopt,PairLJLongCoulLongDeopt);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_LONG_COUL_LONG_DEOPT_H
#define LMP_PAIR_LJ_LONG_COUL_LONG_DEOPT_H

#include "pair_lj_long_coul_long.h"

namespace LAMMPS_NS {

class PairLJLongCoulLongDeopt : public PairLJLongCoulLong {
 public:
  PairLJLongCoulLongDeopt(class LAMMPS *);
  void compute(int, int) override;
  void compute_outer(int, int) override;

 protected:
  template <const bool OUTER>
  void dispatch_candidate();

  template <const bool OUTER, const int ORDER1, const int ORDER6>
  void dispatch_tables();

  template <const bool OUTER, const int CTABLE, const int LJTABLE, const int ORDER1,
            const int ORDER6>
  void dispatch_ev();

  template <const bool OUTER, const int EVFLAG, const int EFLAG, const int CTABLE,
            const int LJTABLE, const int ORDER1, const int ORDER6>
  void dispatch_newton();

  template <const bool OUTER, const int EVFLAG, const int EFLAG, const int NEWTON_PAIR,
            const int CTABLE, const int LJTABLE, const int ORDER1, const int ORDER6>
  void dispatch_eval();

  template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
            const int LJTABLE, const int ORDER1, const int ORDER6>
  void eval();

  template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
            const int LJTABLE, const int ORDER1, const int ORDER6>
  void eval_outer();
};

}    // namespace LAMMPS_NS

#endif
#endif
