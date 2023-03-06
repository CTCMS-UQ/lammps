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

#include "fix_property_mol_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "domain.h"
#include "domain_kokkos.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "memory_kokkos.h"
#include "update.h"
#include "kokkos_few.h"
#include "kokkos.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;


/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::FixPropertyMolKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixPropertyMol(lmp, narg, arg)
{
  kokkosable = 1;
  // Now check if any of the mass, COM or VCM flags have been requested by the base class's constructor
  // and call the relevant memory allocation functions. Set to zero first so the Kokkos registration
  // functions actually do stuff
  if(mass_flag)
  {
    mass_flag = 0;
    request_mass();
  }
  if(com_flag)
  {
    com_flag = 0;
    request_com();
  }
  if(vcm_flag)
  {
    vcm_flag = 0;
    request_vcm();
  }
  
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixPropertyMolKokkos<DeviceType>::~FixPropertyMolKokkos()
{
  if (copymode) return;
  if (mass_flag)
  {
    memoryKK->destroy_kokkos(k_mass, mass);
    memoryKK->destroy_kokkos(k_massproc, massproc);
  }
  if(com_flag)
  {
    memoryKK->destroy_kokkos(k_com, com);
    memoryKK->destroy_kokkos(k_comproc, comproc);
  }
  if(vcm_flag)
  {
    memoryKK->destroy_kokkos(k_vcm, vcm);
    memoryKK->destroy_kokkos(k_vcmproc, vcmproc);
    memoryKK->destroy_kokkos(k_ke_singles, ke_singles);
    memoryKK->destroy_kokkos(k_keproc, keproc);
  }
} 

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixPropertyMolKokkos<DeviceType>::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_com() {
  if (com_flag) return;
  com_flag = 1;
  request_mass();
  memoryKK->create_kokkos(k_com, com, molmax, "property/mol:com");
  memoryKK->create_kokkos(k_comproc, comproc, molmax, "property/mol:comproc");
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_vcm() {
  if (vcm_flag) return;
  vcm_flag = 1;
  request_mass();
  memoryKK->create_kokkos(k_vcm, vcm, molmax, "property/mol:vcm");
  memoryKK->create_kokkos(k_vcmproc, vcmproc, molmax, "property/mol:vcmproc");
  memoryKK->create_kokkos(k_ke_singles, ke_singles, 6, "property/mol:ke_singles");
  memoryKK->create_kokkos(k_keproc, keproc, 6, "property/mol:keproc");
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::request_mass() {
  if (mass_flag) return;
  mass_flag = 1;
  memoryKK->create_kokkos(k_mass, mass, molmax, "property/mol:mass");
  memoryKK->create_kokkos(k_massproc, massproc, molmax, "property/mol:massproc");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::init()
{
  // Error if system doesn't track molecule ids.
  // Check here since atom_style could change before run.

  if (!atom->molecule_flag)
    error->all(FLERR,
        "Fix property/mol when atom_style does not define a molecule attribute");
}

/* ----------------------------------------------------------------------
   Need to calculate mass and CoM before main setup() calls since those
   could rely on the memory being allocated (e.g. for virial tallying)
---------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::setup_pre_force(int /*vflag*/) {
  dynamic_group = group->dynamic[igroup];
  grow_permolecule();

  // com_compute also computes mass if dynamic_group is set
  // so no need to call mass_compute in that case
  if (mass_flag && !dynamic_group) mass_compute();
  if (com_flag) com_compute();
  if (vcm_flag) vcm_compute();
  if (mass_flag) count_molecules();
}

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::setup_pre_force_respa(int vflag, int ilevel) {
  if (ilevel == 0) setup_pre_force(vflag);
}


/* ----------------------------------------------------------------------
   Calculate number of molecules and grow permolecule arrays if needed.
   Grows to the maximum of previous max. mol id + grow_by and new max. mol id
   if either is larger than nmax.
   Returns true if the number of molecules (max. mol id) changed.
------------------------------------------------------------------------- */

template<class DeviceType>
bool FixPropertyMolKokkos<DeviceType>::grow_permolecule(int grow_by) {
  // Calculate maximum molecule id
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  tagint maxone = -1;
  for (int i = 0; i < nlocal; i++)
    if (molecule[i] > maxone) maxone = molecule[i];
  tagint maxall;
  MPI_Allreduce(&maxone, &maxall, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (maxall > MAXSMALLINT)
    error->all(FLERR, "Molecule IDs too large for fix property/mol");

  tagint old_molmax = molmax;
  tagint new_size = molmax + grow_by;
  molmax = maxall;
  new_size = MAX(molmax, new_size);

  // Grow arrays as needed
  if (nmax < new_size) {
    nmax = new_size;
    if (mass_flag)
    {
      memoryKK->grow_kokkos(k_mass, mass, nmax, "property/mol:mass");
      memoryKK->grow_kokkos(k_massproc, massproc, nmax, "property/mol:massproc");
    }
    if (com_flag)
    {
      memoryKK->grow_kokkos(k_com, com, nmax, "property/mol:com");
      memoryKK->grow_kokkos(k_comproc, comproc, nmax, "property/mol:comproc");
    }
    if (vcm_flag)
    {
      memoryKK->grow_kokkos(k_vcm, vcm, nmax, "property/mol:vcm");
      memoryKK->grow_kokkos(k_vcmproc, vcmproc, nmax, "property/mol:vcmproc");
    }
  }

  size_array_rows = static_cast<int>(molmax);
  return old_molmax != molmax;
}


/* ----------------------------------------------------------------------
   Count the number of molecules with non-zero mass.
   Mass of molecules is only counted from atoms in the group, so count is
   the number of molecules in the group.
------------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::count_molecules() {
  count_step = update->ntimestep;
  nmolecule = 0;
  k_mass.template sync<DeviceType>();
  d_mass = k_mass.view<DeviceType>();

  {
    // Local copy of mass for LAMBDA capture
    auto l_mass = this->d_mass;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,molmax), LAMMPS_LAMBDA (const int m, int &rcount) {
      if (l_mass[m] > 0.0)
        rcount += 1;
    }, nmolecule);
  }
}


/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::mass_compute() {
  
  // Kokkos stuff
  atomKK->sync(execution_space,datamask_read);

  atomKK->k_mass.sync<DeviceType>();
  atom_mask = atomKK->k_mask.view<DeviceType>();
  atom_type = atomKK->k_type.view<DeviceType>();
  atom_molID = atomKK->k_molecule.view<DeviceType>();

  // Only instantiate the necessary mass view
  if (atomKK->rmass) {
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }

  d_mass = k_mass.view<DeviceType>();
  d_massproc = k_massproc.view<DeviceType>();

  tagint nlocal = atomKK->nlocal;

  mass_step = update->ntimestep;
  // Grow arrays if the molecules might have changed this step
  if (dynamic_mols) grow_permolecule();
  if (molmax == 0) return;

  // Zero arrays
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_massproc_zero>(0, molmax), *this);
  copymode = 0;

  // Now set up the scatter view to track massproc so we can reduce into the elements without races
  // TODO EVK: I don't think we need the full helper functions and scaffolding, since our synchronisations
  // don't depend on the type of neighbour list. The LAMMPS scatter view helpers are designed to work with
  // force or force-style views, where we need to worry about simultaneous accesses for half neighbour lists
  // but this isn't a problem here. Could be cleaner to just let Kokkos decide whether to duplicate.
  // Could we just check whether LMP_USE_ATOMICS is set?
  neighflag = lmp->kokkos->neighflag;
  // Don't think we need anything on top of the standard stuff for determining whether to duplicate
  if(neighflag == HALF)
    need_dup = std::is_same<typename NeedDup<HALF,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else if (neighflag == HALFTHREAD)
    need_dup = std::is_same<typename NeedDup<HALFTHREAD,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else
    need_dup = std::is_same<typename NeedDup<FULL,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;

  if(need_dup) {
    dup_massproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_massproc);
  } else {
    ndup_massproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_massproc);
  }
  // Template based on whether to use rmass
  copymode = 1;
  if (atomKK->rmass) {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<HALF, 1> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<HALFTHREAD, 1> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<FULL, 1> >(0, nlocal), *this);
    }
  } else {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<HALF, 0> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<HALFTHREAD, 0> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_mass_compute<FULL, 0> >(0, nlocal), *this);
    }
  }
  copymode = 0;
  
  // Collect results from the scatter view
  if(need_dup) {
    Kokkos::Experimental::contribute(d_massproc, dup_massproc);
  } else {
    Kokkos::Experimental::contribute(d_massproc, ndup_massproc);
  }

  k_massproc.sync<DeviceType>();
  MPI_Allreduce(k_massproc.h_view.data(), k_mass.h_view.data(), molmax, MPI_DOUBLE, MPI_SUM, world);
  k_mass.sync<DeviceType>();
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_massproc_zero, const int& m) const
{
  d_massproc[m] = 0.0;
}

template<class DeviceType>
template<int NEIGHFLAG, int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_mass_compute<NEIGHFLAG, RMASS>, const int&i) const
{
  auto v_massproc = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_massproc),decltype(ndup_massproc)>::get(dup_massproc,ndup_massproc);
  auto a_massproc = v_massproc.template access<AtomicDup_v<NEIGHFLAG,DeviceType> >();
  if (RMASS) {
    if (groupbit & atom_mask[i]) {
      tagint m = atom_molID[i]-1;
      if (m < 0) return;
      a_massproc[m] += atom_rmass[i];
    }
  } else {
    if (groupbit & atom_mask[i]) {
      tagint m = atom_molID[i]-1;
      if (m < 0) return;
      a_massproc[m] += atom_mass[atom_type[i]];
    } 
  }
}

/* ----------------------------------------------------------------------
   Calculate center of mass of each molecule in unwrapped coords
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::com_compute() {
  atomKK->sync(execution_space,datamask_read);
  com_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;

  int nlocal = atomKK->nlocal;
  // Kokkos variables
  atom_molID = atomKK->k_molecule.view<DeviceType>();
  atom_type = atomKK->k_type.view<DeviceType>();
  if(atomKK->rmass)
  {
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }
  atom_x = atomKK->k_x.view<DeviceType>();
  atom_image = atomKK->k_image.view<DeviceType>();

  d_com = k_com.view<DeviceType>();
  d_comproc = k_comproc.view<DeviceType>();

  // Zero the arrays
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_comproc_zero>(0, molmax), *this);
  copymode = 0;

  if (recalc_mass) {
    mass_compute();
  }

  // Get the box properties from domainKK so we can remap inside the kernel
  prd = Few<double, 3>(domain->prd);
  h = Few<double, 6>(domain->h);
  triclinic = domain->triclinic;

  // Now we need a scatter view to reduce into, may or may not need to be duplicated
  if(neighflag == HALF)
    need_dup = std::is_same<typename NeedDup<HALF,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else if (neighflag == HALFTHREAD)
    need_dup = std::is_same<typename NeedDup<HALFTHREAD,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else
    need_dup = std::is_same<typename NeedDup<FULL,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;

  //dup_comproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_comproc);
  if(need_dup)
  {
    dup_comproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_comproc);
  } else {
    ndup_comproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_comproc);
  }
  copymode = 1;
  if (atomKK->rmass) {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<HALF, 1> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<HALFTHREAD, 1> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<FULL, 1> >(0, nlocal), *this);
    }
  } else {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<HALF, 0> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<HALFTHREAD, 0> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_compute<FULL, 0> >(0, nlocal), *this);
    }
  }

  if(need_dup){
    Kokkos::Experimental::contribute(d_comproc, dup_comproc);
  } else {
    Kokkos::Experimental::contribute(d_comproc, ndup_comproc);
  }

  k_comproc.sync<DeviceType>();
  MPI_Allreduce(k_comproc.h_view.data(),k_com.h_view.data(),3*molmax,MPI_DOUBLE,MPI_SUM,world);
  k_com.sync<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_com_scale>(0, molmax), *this);
  copymode = 0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_comproc_zero, const int& m) const
{
  d_comproc(m, 0) = 0.0;
  d_comproc(m, 1) = 0.0;
  d_comproc(m, 2) = 0.0;
}

template<class DeviceType>
template<int NEIGHFLAG, int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_com_compute<NEIGHFLAG, RMASS>, const int& i) const
{
  auto v_comproc = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_comproc),decltype(ndup_comproc)>::get(dup_comproc,ndup_comproc);
  auto a_comproc = v_comproc.template access<AtomicDup_v<NEIGHFLAG,DeviceType> >();
  if(groupbit & atom_mask[i])
  {
    // TODO EVK: Refactor, too much copy-paste
    if(RMASS){
      tagint m = atom_molID[i]-1;
      if(m < 0) return;
      double massone = atom_rmass[i];
      // Need to unwrap the coords. Make a Kokkos Few first to interface with domainKK
      Few<double, 3> x_i;
      x_i[0] = atom_x(i, 0);
      x_i[1] = atom_x(i, 1);
      x_i[2] = atom_x(i, 2);
      auto unwrap = DomainKokkos::unmap(prd, h, triclinic, x_i, atom_image[i]);
      a_comproc(m, 0) += unwrap[0] * massone;
      a_comproc(m, 1) += unwrap[1] * massone;
      a_comproc(m, 2) += unwrap[2] * massone;
    } else {
      tagint m = atom_molID[i]-1;
      if(m < 0) return;
      double massone = atom_mass[atom_type[i]];
      // Need to unwrap the coords. Make a Kokkos Few first to interface with domainKK
      Few<double, 3> x_i;
      x_i[0] = atom_x(i, 0);
      x_i[1] = atom_x(i, 1);
      x_i[2] = atom_x(i, 2);
      auto unwrap = DomainKokkos::unmap(prd, h, triclinic, x_i, atom_image[i]);
      a_comproc(m, 0) += unwrap[0] * massone;
      a_comproc(m, 1) += unwrap[1] * massone;
      a_comproc(m, 2) += unwrap[2] * massone;
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_com_scale, const int &m) const
{
  if (d_mass[m] > 0.0) {
    d_com(m, 0) /= d_mass[m];
    d_com(m, 1) /= d_mass[m];
    d_com(m, 2) /= d_mass[m];
  } else {
    d_com(m, 0) = 0.0; 
    d_com(m, 1) = 0.0;
    d_com(m, 2) = 0.0;
  }
}

/* ----------------------------------------------------------------------
   Calculate center of mass velocity of each molecule 
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */
template<class DeviceType>
void FixPropertyMolKokkos<DeviceType>::vcm_compute()
{
  vcm_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;

  int nlocal = atomKK->nlocal;
  // Kokkos variables
  atom_molID = atomKK->k_molecule.view<DeviceType>();
  atom_type = atomKK->k_type.view<DeviceType>();
  atom_mask = atomKK->k_mask.view<DeviceType>();
  if(atomKK->rmass)
  {
    atom_rmass = atomKK->k_rmass.view<DeviceType>();
  } else {
    atom_mass = atomKK->k_mass.view<DeviceType>();
  }
  atom_x = atomKK->k_x.view<DeviceType>();
  atom_v = atomKK->k_v.view<DeviceType>();
  atom_image = atomKK->k_image.view<DeviceType>();

  d_vcm = k_vcm.view<DeviceType>();
  d_vcmproc = k_vcmproc.view<DeviceType>();
  d_keproc = k_keproc.view<DeviceType>();

  // Zero the arrays
  Kokkos::deep_copy(d_keproc, 0.0);
  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcmproc_zero>(0, molmax), *this);
  copymode = 0;

  // Recalculate mass if it's changed
  if (recalc_mass) {
    mass_compute();
  }

  // Scatter view to reduce into in VCM functor
  if(neighflag == HALF)
    need_dup = std::is_same<typename NeedDup<HALF,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else if (neighflag == HALFTHREAD)
    need_dup = std::is_same<typename NeedDup<HALFTHREAD,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;
  else
    need_dup = std::is_same<typename NeedDup<FULL,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;

  if(need_dup)
  {
    dup_vcmproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vcmproc);
    dup_keproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_keproc);
  } else {
    ndup_vcmproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vcmproc);
    ndup_keproc = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_keproc);
  }
  copymode = 1;
  if (atomKK->rmass) {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<HALF, 1> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<HALFTHREAD, 1> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<FULL, 1> >(0, nlocal), *this);
    }
  } else {
    if (neighflag == HALF) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<HALF, 0> >(0, nlocal), *this);
    } else if (neighflag == HALFTHREAD) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<HALFTHREAD, 0> >(0, nlocal), *this);
    } else if (neighflag == FULL) {
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_compute<FULL, 0> >(0, nlocal), *this);
    }
  }

  if(need_dup){
    Kokkos::Experimental::contribute(d_vcmproc, dup_vcmproc);
    Kokkos::Experimental::contribute(d_keproc, dup_keproc);
  } else {
    Kokkos::Experimental::contribute(d_vcmproc, ndup_vcmproc);
    Kokkos::Experimental::contribute(d_keproc, ndup_keproc);
  }

  k_vcmproc.sync<DeviceType>();
  k_keproc.sync<DeviceType>();

  MPI_Allreduce(k_vcmproc.h_view.data(),k_vcm.h_view.data(),3*molmax,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(k_keproc.h_view.data(),k_ke_singles.h_view.data(), 6, MPI_DOUBLE, MPI_SUM, world);
  k_vcm.sync<DeviceType>();
  k_ke_singles.sync<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPropertyMol_vcm_scale>(0, molmax), *this);
  copymode = 0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_vcmproc_zero, const int& m) const
{
  d_vcmproc(m, 0) = 0.0;
  d_vcmproc(m, 1) = 0.0;
  d_vcmproc(m, 2) = 0.0;
}

template<class DeviceType>
template<int NEIGHFLAG, int RMASS>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_vcm_compute<NEIGHFLAG, RMASS>, const int& i) const
{
  auto v_vcmproc = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_vcmproc),decltype(ndup_vcmproc)>::get(dup_vcmproc,ndup_vcmproc);
  auto a_vcmproc = v_vcmproc.template access<AtomicDup_v<NEIGHFLAG,DeviceType> >();
  auto v_keproc = ScatterViewHelper<NeedDup_v<NEIGHFLAG,DeviceType>,decltype(dup_keproc),decltype(ndup_keproc)>::get(dup_keproc,ndup_keproc);
  auto a_keproc = v_keproc.template access<AtomicDup_v<NEIGHFLAG,DeviceType> >();
  if(RMASS){
    if (groupbit & atom_mask[i]) {
      double massone = atom_rmass[i];
      tagint m = atom_molID[i]-1;
      if (m < 0) {
        a_keproc[0] += atom_v(i, 0)*atom_v(i, 0)*massone;
        a_keproc[1] += atom_v(i, 1)*atom_v(i, 1)*massone;
        a_keproc[2] += atom_v(i, 2)*atom_v(i, 2)*massone;
        a_keproc[3] += atom_v(i, 0)*atom_v(i, 1)*massone;
        a_keproc[4] += atom_v(i, 0)*atom_v(i, 2)*massone;
        a_keproc[5] += atom_v(i, 1)*atom_v(i, 2)*massone;
      } else {
        a_vcmproc(m, 0) += atom_v(i, 0) * massone;
        a_vcmproc(m, 1) += atom_v(i, 1) * massone;
        a_vcmproc(m, 2) += atom_v(i, 2) * massone;
      }
    }
  } else {
    if (groupbit & atom_mask[i]) {
      double massone = atom_mass[atom_type[i]];
      tagint m = atom_molID[i]-1;
      if (m < 0) {
        a_keproc[0] += atom_v(i, 0)*atom_v(i, 0)*massone;
        a_keproc[1] += atom_v(i, 1)*atom_v(i, 1)*massone;
        a_keproc[2] += atom_v(i, 2)*atom_v(i, 2)*massone;
        a_keproc[3] += atom_v(i, 0)*atom_v(i, 1)*massone;
        a_keproc[4] += atom_v(i, 0)*atom_v(i, 2)*massone;
        a_keproc[5] += atom_v(i, 1)*atom_v(i, 2)*massone;
      } else {
        a_vcmproc(m, 0) += atom_v(i, 0) * massone;
        a_vcmproc(m, 1) += atom_v(i, 1) * massone;
        a_vcmproc(m, 2) += atom_v(i, 2) * massone;
      }
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixPropertyMolKokkos<DeviceType>::operator()(TagFixPropertyMol_vcm_scale, const int& m) const
{
  if(d_mass[m] > 0.0)
  {
    d_vcm(m, 0) /= d_mass[m];
    d_vcm(m, 1) /= d_mass[m];
    d_vcm(m, 2) /= d_mass[m];
  } else {
    d_vcm(m, 0) = 0.0;
    d_vcm(m, 1) = 0.0;
    d_vcm(m, 2) = 0.0;
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

template<class DeviceType>
double FixPropertyMolKokkos<DeviceType>::memory_usage()
{
  double bytes = 0.0;
  if (mass_flag) bytes += nmax * 2 * sizeof(double);
  if (com_flag)  bytes += nmax * 6 * sizeof(double);
  if (vcm_flag)  bytes += nmax * 6 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   basic array output
------------------------------------------------------------------------- */

template<class DeviceType>
double FixPropertyMolKokkos<DeviceType>::compute_array(int imol, int col)
{
  if (imol > static_cast<int>(molmax))
    error->all(FLERR, fmt::format(
      "Cannot request info for molecule {} from fix property/mol (molmax = {})",
      imol, molmax));

  if (col == 3) {
    // Mass requested
    if (!mass_flag)
      error->all(FLERR, "This fix property/mol does not calculate mass");
    if (dynamic_group && mass_step != update->ntimestep) mass_compute();
    return mass[imol];
  } else {
    // CoM requested
    if (!com_flag)
      error->all(FLERR, "This fix property/mol does not calculate CoM");
    if (com_step != update->ntimestep) com_compute();
    return com[imol][col];
  }
}

namespace LAMMPS_NS {
template class FixPropertyMolKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixPropertyMolKokkos<LMPHostType>;
#endif
}

