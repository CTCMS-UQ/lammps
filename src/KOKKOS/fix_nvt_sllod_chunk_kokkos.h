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

#ifdef FIX_CLASS
// clang-format off
FixStyle(nvt/sllod/kk,FixNVTSllodKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/kk/device,FixNVTSllodKokkos<LMPDeviceType>);
FixStyle(nvt/sllod/kk/host,FixNVTSllodKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_NVT_SLLOD_KOKKOS_H
#define LMP_FIX_NVT_SLLOD_KOKKOS_H

#include "fix_nh_kokkos.h"
#include "kokkos_few.h"
#include "kokkos_type.h"

// clang-format off
namespace LAMMPS_NS {

struct TagFixNVTSllodKokkos_temp1{};
struct TagFixNVTSllodKokkos_temp2{};

template<class DeviceType>
class FixNVTSllodChunkKokkos : public FixNHKokkos<DeviceType> {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNVTSllodKokkos(class LAMMPS *, int, char **);
  ~FixNVTSllodChunkKokkos();

  void init() override;
  void setup(int) override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodChunk_temp1, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNVTSllodChunk_temp2, const int& i) const;

 private:
  int nondeformbias;
  int nchunk, maxchunk;
  int kickflag;           // Whether to apply the streaming profile on init
  
  typename AT::t_v_array vcmall;
  typename AT::t_v_array vcm;
  typename AT::t_float_1d massproc;
  typename AT::t_float_1d masstotal;

  void nh_v_temp() override;
  void vcm_thermal_compute();

  class ComputeChunkAtom *cchunk;
  class ComputeVCMChunk *cvcm;

 protected:
  typename AT::t_x_array x;
  typename AT::t_v_array v;
  typename AT::t_f_array_const f;
  typename AT::t_float_1d mass;
  typename AT::t_float_1d rmass;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
  typename AT::t_imageint_1d image;


  Few<double, 6> d_h_two;

  class DomainKokkos *domainKK;
  class AtomKokkos *atomKK;
};

}    // namespace LAMMPS_NS

#endif
#endif

