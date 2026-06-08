// Known-good sample candidate for pair_style lj/long/coul/long/deopt.
//
// This is intentionally not a deoptimized replacement kernel. It delegates to
// the original base style while preserving the active energy and virial request
// bits, which makes it useful for verifying the harness and candidate plumbing.

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval()
{
  int eflag = ENERGY_NONE;
  if (eflag_global) eflag |= ENERGY_GLOBAL;
  if (eflag_atom) eflag |= ENERGY_ATOM;
  if (eflag_only) eflag |= ENERGY_ONLY;

  int vflag = VIRIAL_NONE;
  if (vflag_global) vflag |= VIRIAL_PAIR;
  if (vflag_fdotr) vflag |= VIRIAL_FDOTR;
  if (vflag_atom) vflag |= VIRIAL_ATOM;
  if (cvflag_atom) vflag |= VIRIAL_CENTROID;

  PairLJLongCoulLong::compute(eflag, vflag);
}

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval_outer()
{
  int eflag = ENERGY_NONE;
  if (eflag_global) eflag |= ENERGY_GLOBAL;
  if (eflag_atom) eflag |= ENERGY_ATOM;
  if (eflag_only) eflag |= ENERGY_ONLY;

  int vflag = VIRIAL_NONE;
  if (vflag_global) vflag |= VIRIAL_PAIR;
  if (vflag_fdotr) vflag |= VIRIAL_FDOTR;
  if (vflag_atom) vflag |= VIRIAL_ATOM;
  if (cvflag_atom) vflag |= VIRIAL_CENTROID;

  PairLJLongCoulLong::compute_outer(eflag, vflag);
}
