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

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval()
{
  // TODO
}
