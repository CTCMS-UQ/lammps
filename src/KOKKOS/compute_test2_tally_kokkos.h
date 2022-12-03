/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing Author: Stephen Sanderson (UQ)
------------------------------------------------------------------------- */

#ifdef KOKKOS_TALLY_CLASS
// clang-format off
TallyStyle(ComputeTest2TallyKokkos<DeviceType>)
// clang-format on
#elif defined(COMPUTE_CLASS)
// clang-format off
ComputeStyle(test2/tally/kk,ComputeTest2TallyKokkos<LMPDeviceType>);
ComputeStyle(test2/tally/kk/device,ComputeTest2TallyKokkos<LMPDeviceType>);
ComputeStyle(test2/tally/kk/host,ComputeTest2TallyKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEST2_TALLY_KOKKOS_H
#define LMP_COMPUTE_TEST2_TALLY_KOKKOS_H

#include "compute.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct ComputeTest2TallyEV {
  F_FLOAT sum_epair;

  KOKKOS_INLINE_FUNCTION
  ComputeTest2TallyEV() : sum_epair(0.0) {}

  KOKKOS_INLINE_FUNCTION
  void operator+=(const ComputeTest2TallyEV &rhs) {
    sum_epair += rhs.sum_epair;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile ComputeTest2TallyEV &rhs) volatile {
    sum_epair += rhs.sum_epair;
  }
};

template<class DeviceType> class ComputeTest2TallyKokkos;

// Functor type that provides an ev_tally function to be called per pair.
// Needs to be header only for linking purposes
template<class DeviceType>
struct ComputeTest2TallyFunctor {
  void init_step(class ComputeTest2TallyKokkos<DeviceType> *c_ptr) {
    // NOTE: could set properties using c_ptr->members if needed (e.g. a dup of some Kokkos view)
  }
  void consolidate(class ComputeTest2TallyKokkos<DeviceType> *c_ptr, const ComputeTest2TallyEV &ev) {
    // pass ev result back to compute
    c_ptr->ev = ev;
  }

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(ComputeTest2TallyEV &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
    const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    // TODO: tally things here in `tally`.
    // Could also change values by following member pointers,
    // but would need to store a value per pair or use atomics.

    // Tally epair as an easily tested example
    tally.sum_epair += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(newton_pair||(j<nlocal)))?1.0:0.5)*evdwl;
  }

};

template<class DeviceType>
class ComputeTest2TallyKokkos : public Compute {

 public:
  /* - Members required for kokkos tally framework ------------------ */
  // struct to handle tallying
  typedef struct ComputeTest2TallyFunctor<DeviceType> tally_functor;

  // Extra data to be reduced over during pair calculation
  typedef ComputeTest2TallyEV tally_type;

  // Called once per step to deduce required PairComputeFunctor type
  unsigned long tally_mask() override;
  /* --------------------------------------------------------------- */

  ComputeTest2TallyKokkos(class LAMMPS *, int, char **);
  ~ComputeTest2TallyKokkos() override;

  void init() override;

  void pair_setup_callback(int, int) override;

  double compute_scalar() override;

 private:
  bigint did_setup;
  tally_type ev;
  friend struct ComputeTest2TallyFunctor<DeviceType>;
};

}    // namespace LAMMPS_NS

#endif
#endif
