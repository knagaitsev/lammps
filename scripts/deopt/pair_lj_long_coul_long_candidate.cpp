// Candidate implementations for pair_style lj/long/coul/long/deopt.
//
// The test script configures LAMMPS to include this file in the deopt adapter.
// Implement eval, eval_outer, or both depending on LAMMPS_DEOPT_FUNCTION.
// Remove the #error line from each function you implement.

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval()
{
// #error "Implement PairLJLongCoulLongDeopt::eval() in the candidate file."
}

template <const int EVFLAG, const int EFLAG, const int NEWTON_PAIR, const int CTABLE,
          const int LJTABLE, const int ORDER1, const int ORDER6>
void PairLJLongCoulLongDeopt::eval_outer()
{
// #error "Implement PairLJLongCoulLongDeopt::eval_outer() in the candidate file."
}
