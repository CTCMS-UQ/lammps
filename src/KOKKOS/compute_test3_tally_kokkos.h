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
TallyStyle(ComputeTest3TallyKokkos<DeviceType>)
// clang-format on
#elif defined(COMPUTE_CLASS)
// clang-format off
ComputeStyle(test3/tally/kk,ComputeTest3TallyKokkos<LMPDeviceType>);
ComputeStyle(test3/tally/kk/device,ComputeTest3TallyKokkos<LMPDeviceType>);
ComputeStyle(test3/tally/kk/host,ComputeTest3TallyKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_TEST3_TALLY_KOKKOS_H
#define LMP_COMPUTE_TEST3_TALLY_KOKKOS_H

#include "compute.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

struct ComputeTest3TallyEV {
  F_FLOAT sum_epair;

  KOKKOS_INLINE_FUNCTION
  ComputeTest3TallyEV() : sum_epair(0.0) {}

  KOKKOS_INLINE_FUNCTION
  void operator+=(const ComputeTest3TallyEV &rhs) {
    sum_epair += rhs.sum_epair;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const volatile ComputeTest3TallyEV &rhs) volatile {
    sum_epair += rhs.sum_epair;
  }
};

template<class DeviceType> class ComputeTest3TallyKokkos;

// Functor type that provides an ev_tally function to be called per pair.
// Needs to be header only for linking purposes
template<class DeviceType>
struct ComputeTest3TallyFunctor {
  inline void init_step(class ComputeTest3TallyKokkos<DeviceType> *c_ptr) {
    // NOTE: could set properties using c_ptr->members if needed (e.g. a dup of some Kokkos view)
  }
  inline void consolidate(class ComputeTest3TallyKokkos<DeviceType> *c_ptr, const ComputeTest3TallyEV &ev) {
    // pass ev result back to compute
    c_ptr->ev = ev;
  }

  template<int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(ComputeTest3TallyEV &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
    const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    // TODO: tally things here in `tally`.
    // Could also change values by following member pointers,
    // but would need to store a value per pair or use atomics.

    // Tally epair as an easily tested example
    // Multiply by 3 to differentiate from test/tally/kk
    tally.sum_epair += 3.0*(((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(newton_pair||(j<nlocal)))?1.0:0.5)*evdwl;
  }

};

template<class DeviceType>
class ComputeTest3TallyKokkos : public Compute {

 public:
  /* - Members required for kokkos tally framework ------------------ */
  // struct to handle tallying
  typedef struct ComputeTest3TallyFunctor<DeviceType> tally_functor;

  // Extra data to be reduced over during pair calculation
  typedef ComputeTest3TallyEV tally_type;

  // Called once per step to deduce required PairComputeFunctor type
  unsigned long tally_mask() override;
  /* --------------------------------------------------------------- */

  ComputeTest3TallyKokkos(class LAMMPS *, int, char **);
  ~ComputeTest3TallyKokkos() override;

  void init() override;

  void pair_setup_callback(int, int) override;

  double compute_scalar() override;

 private:
  bigint did_setup;
  tally_type ev;
  friend struct ComputeTest3TallyFunctor<DeviceType>;
};

}    // namespace LAMMPS_NS

#endif
#endif
