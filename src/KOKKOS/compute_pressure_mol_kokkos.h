/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef KOKKOS_TALLY_CLASS
// clang-format off
TallyStyle(ComputePressureMolKokkos<DeviceType>)
// clang-format on
#elif defined(COMPUTE_CLASS)
// clang-format off
ComputeStyle(pressure/mol/kk,ComputePressureMolKokkos<LMPHostType>);
ComputeStyle(pressure/mol/kk/device,ComputePressureMolKokkos<LMPDeviceType>);
ComputeStyle(pressure/mol/kk/host,ComputePressureMolKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_PRESSURE_MOL_KOKKOS_H
#define LMP_COMPUTE_PRESSURE_MOL_KOKKOS_H

#include "compute_pressure_mol.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType>
class ComputePressureMolKokkos : public ComputePressureMol {
 public:
  struct TallyVirial {
    F_FLOAT virial[9];

    KOKKOS_INLINE_FUNCTION
    TallyVirial() {for (int i = 0; i < 9; ++i) virial[i] = 0.0;}

    KOKKOS_INLINE_FUNCTION
    void operator+=(const TallyVirial &rhs) {
      for (int i = 0; i < 9; ++i) virial[i] += rhs.virial[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile TallyVirial &rhs) volatile {
      for (int i = 0; i < 9; ++i) virial[i] += rhs.virial[i];
    }
  };

  struct TallyFunctor {
    typename ArrayTypes<DeviceType>::t_float_2d_randomread k_com_peratom;

    inline void init_step(ComputePressureMolKokkos *c_ptr) {
      k_com_peratom = c_ptr->k_com_peratom.template view<DeviceType>();
    }

    inline void consolidate(ComputePressureMolKokkos *c_ptr, const TallyVirial &tally) {
      for (int i = 0; i < 9; ++i)
        c_ptr->pair_virial[i] = tally.virial[i];
      printf("virial:\n  [ %g\t%g\t%g\n    %g\t%g\t%g\n    %g\t%g\t%g ]\n",
          tally.virial[0], tally.virial[3], tally.virial[4],
          tally.virial[6], tally.virial[1], tally.virial[5],
          tally.virial[7], tally.virial[8], tally.virial[3]);
    }

    template<int NEIGHFLAG>
    KOKKOS_INLINE_FUNCTION
    void ev_tally(TallyVirial &tally,
      const int &i, const int &j, const int &nlocal, const int &newton_pair,
      const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
    {
      // No virial contribution if i and j both not owned (can this ever occur?)
      if (i >= nlocal && j >= nlocal) return;

      double delcom[3];
      double scale = ((NEIGHFLAG==HALF||NEIGHFLAG==HALFTHREAD) && (newton_pair||(j<nlocal)) ) ? 1.0 : 0.5;
      for (int d = 0; d < 3; ++d)
        delcom[d] = scale * (k_com_peratom(i,d) - k_com_peratom(j,d));

      tally.virial[0] += delcom[0]*delx*fpair;
      tally.virial[1] += delcom[1]*dely*fpair;
      tally.virial[2] += delcom[2]*delz*fpair;
      tally.virial[3] += delcom[0]*dely*fpair;
      tally.virial[4] += delcom[0]*delz*fpair;
      tally.virial[5] += delcom[1]*delz*fpair;
      tally.virial[6] += delcom[1]*delx*fpair;
      tally.virial[7] += delcom[2]*delx*fpair;
      tally.virial[8] += delcom[2]*dely*fpair;
    }
  };

  typedef TallyFunctor tally_functor;
  typedef TallyVirial tally_type;
  TALLY_MASK tally_mask() override;

  ComputePressureMolKokkos(LAMMPS *, int, char **);
  void init() override;
  void pair_setup_callback(int, int) override;
  void pair_tally_callback(int, int, int, int,
      double, double, double, double, double, double) override {}

 protected:
  // TODO: class FixPropertyMolKokkos *molpropKK;
  typename ArrayTypes<DeviceType>::tdual_float_2d k_com_peratom;

};

}    // namespace LAMMPS_NS

#endif
#endif
