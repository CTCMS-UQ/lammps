
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(test3/tally,ComputeTest3Tally);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEST3_TALLY_H
#define LMP_COMPUTE_TEST3_TALLY_H

#include "compute.h"

namespace LAMMPS_NS {


class ComputeTest3Tally : public Compute {

 public:
  ComputeTest3Tally(class LAMMPS *lmp, int narg, char **arg) : Compute(lmp,narg,arg) {}
  ~ComputeTest3Tally() override {}

  void init() override {}

  double compute_scalar() override {return 0.0;}
  void compute_peratom() override {}

  void pair_setup_callback(int, int) override {}


 private:
  bigint did_setup;
};
}    // namespace LAMMPS_NS

#endif
#endif
