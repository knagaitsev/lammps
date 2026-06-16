Your objective is to write a simplified version of a particular piece of reference code, without breaking the functionality or semantics of the simplified code that you generate. The objective is to produce code that has effectively deoptimized the original code, reducing the code to the essential math that is needed while maintaining correctness of the code.

Once the simplified code is generated and passes differential tests, a human expert will review the code, using the reference math from the LAMMPS documentation to ensure the code actually implements the math. This is the reason that the code should be simplified as much as possible, with no concern for performance.

You are prohibited from making calls to functions or importing new things, outside of what the reference code already calls and imports.

Reference code (already verified to pass correctness checks): scripts/agent/eval-plain-1/reference.cpp

Reference documentation: doc/src/pair_lj_long.rst

Simplified code path where you should implement the TODO. This is the only file you are allowed to modify: scripts/agent/eval-plain-1/simplified.cpp

Differential test that must pass for the simplified code to be considered valid: LAMMPS_DEOPT_CANDIDATE_FILE=scripts/agent/eval-plain-1/simplified.cpp LAMMPS_DEOPT_REFERENCE=plain LAMMPS_DEOPT_FUNCTION=eval scripts/run_lj_long_coul_long_diff.sh
