// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pressure_mol_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain_kokkos.h"
#include "error.h"
#include "fix_property_mol.h" // TODO: _kokkos.h"
#include "memory_kokkos.h"
#include "modify.h"
#include "pair_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputePressureMolKokkos<DeviceType>::ComputePressureMolKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputePressureMol(lmp, narg, arg)
{
  kokkosable = 1;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = X_MASK | MOLECULE_MASK | IMAGE_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePressureMolKokkos<DeviceType>::init()
{
  ComputePressureMol::init();
  // TODO
  // molpropKK = dynamic_cast<FixPropertyMolKokkos*>(modify->get_fix_by_id(id_molprop));
  // if (molpropKK == nullptr)
  //   error->all(FLERR, "Compute pressure/mol/kk requires a fix property/mol/kk variant");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
TALLY_MASK  ComputePressureMolKokkos<DeviceType>::tally_mask() {
  if (did_setup != update->ntimestep) return 0;
  return s_TALLY_MASK<DeviceType>::template get_tally_mask<ComputePressureMolKokkos>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePressureMolKokkos<DeviceType>::pair_setup_callback(int eflag, int vflag) {
  if (did_setup == update->ntimestep || !matchstep(update->ntimestep)) return;
  did_setup = update->ntimestep;

  for (int d = 0; d < 9; d++)
    pair_virial[d] = 0.0;

  // Make sure CoM is up to date TODO
  // if (molpropKK->com_step != update->ntimestep)
  //   molpropKK->com_compute();
  if (molprop->com_step != update->ntimestep)
    molprop->com_compute();

  // Pre-compute CoM positions for each atom
  if (atomKK->nmax > nmax) {
    nmax = atomKK->nmax;
    memoryKK->grow_kokkos(k_com_peratom, com_peratom, nmax, 3, "pressure/mol:com_peratom");
  }
  auto molecule = atomKK->k_molecule.view<DeviceType>();
  auto x = atomKK->k_x.view<DeviceType>();
  auto image = atomKK->k_image.view<DeviceType>();
  auto view_com_peratom = k_com_peratom.template view<DeviceType>();
  // auto com = molpropKK->k_com.view<DeviceType>();
  auto com = molprop->com; // TODO: swap to above
  int triclinic = domain->triclinic;
  auto prd = Few<double,3>(domain->prd);
  auto h = Few<double,6>(domain->h);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType>(0,atomKK->nlocal+atomKK->nghost),
    KOKKOS_LAMBDA(const int &i) {
      tagint m = molecule(i)-1;
      if (m < 0) {
        view_com_peratom(i,0) = x(i,0);
        view_com_peratom(i,1) = x(i,1);
        view_com_peratom(i,2) = x(i,2);
      } else {
        Few<double,3> xcm;
        xcm[0] = com[m][0];
        xcm[1] = com[m][1];
        xcm[2] = com[m][2];
        // CoM is stored in unwrapped coordinates. Need to map to same image as atom
        imageint ix = (2*IMGMAX - (image(i) & IMGMASK)) & IMGMASK;
        imageint iy = (2*IMGMAX - (image(i) >> IMGBITS & IMGMASK)) & IMGMASK;
        imageint iz = (2*IMGMAX - (image(i) >> IMG2BITS)) & IMGMASK;
        auto unwrap = DomainKokkos::unmap(prd, h, triclinic, xcm, ix | (iy << IMGBITS) | (iz << IMG2BITS) );
        view_com_peratom(i,0) = unwrap[0];
        view_com_peratom(i,1) = unwrap[1];
        view_com_peratom(i,2) = unwrap[2];
        printf("CoM[%d]: [%g, %g, %g]\n", i, unwrap[0], unwrap[1], unwrap[2]);
      }
    });
  k_com_peratom.template modify<DeviceType>();
}

namespace LAMMPS_NS {
template class ComputePressureMolKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputePressureMolKokkos<LMPHostType>;
#endif
}
