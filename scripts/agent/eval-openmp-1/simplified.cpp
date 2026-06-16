#include "atom.h"
#include "comm.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "kspace.h"
#include "math_extra.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathExtra;
using namespace EwaldConst;

void PairLJLongCoulLongDeopt::compute_candidate(int /*eflag*/, int /*vflag*/)
{
  // TODO: replace this stub with a simplified implementation equivalent to reference.cpp.
}
