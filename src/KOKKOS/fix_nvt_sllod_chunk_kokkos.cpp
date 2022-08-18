// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Emily Kahl (Uni. of QLD, e.kahl@uq.edu.au)
------------------------------------------------------------------------- */

#include "fix_nvt_sllod_kokkos.h"

#include "atom.h"
#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_deform_kokkos.h"
#include "group.h"
#include "kokkos_few.h"
#include "math_extra.h"
#include "memory_kokkos.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixNVTSllodChunkKokkos<DeviceType>::FixNVTSllodChunkKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixNHKokkos<DeviceType>(lmp, narg, arg)
{
  // Cast atomKK to AtomKokkos type from this->atom
  atomKK = (AtomKokkos *) atom;
  this->kokkosable = 1;
  this->domainKK = (DomainKokkos *) domain;
  
  if (!this->tstat_flag)
    this->error->all(FLERR,"Temperature control must be used with fix nvt/sllod/chunk");
  if (this->pstat_flag)
    this->error->all(FLERR,"Pressure control can not be used with fix nvt/sllod/chunk");

  // default values

  if (this->mtchain_default_flag) this->mtchain = 1;

  this->kickflag = 0;

  int iarg = 3;

  while (iarg < narg) {
    if (strcmp(arg[iarg++], "kick")==0) {
      if (iarg >= narg) error->all(FLERR,"Invalid fix nvt/sllod/chunk command");
      if (strcmp(arg[iarg], "yes")==0) {
        kickflag = 1;
      } else if (strcmp(arg[iarg], "no")==0) {
        kickflag = 0;
      } else error->all(FLERR,"Invalid fix nvt/sllod/chunk command");
      ++iarg;
    }
  }

  // create a new compute temp style
  // id = fix-ID + temp

  this->id_temp = utils::strdup(std::string(id) + "_temp");
  this->modify->add_compute(fmt::format("{} {} temp/deform/kk",
                                  this->id_temp,group->names[igroup]));
  this->tcomputeflag = 1;
  this->maxchunk = 0;
/*  
  vcm = nullptr;
  vcmall = nullptr;
  masstotal = nullptr;
  massproc = nullptr;
*/

}

/* ---------------------------------------------------------------------- */
FixNVTSllodChunkKokkos::~FixNVTSllodChunkKokkos() {
  memoryKK->destroy_kokkos(vcm);
  memoryKK->destroy_kokkos(vcmall);
  memoryKK->destroy_kokkos(massproc);
  memoryKK->destroy_kokkos(masstotal);
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVTSllodChunkKokkos<DeviceType>::init()
{
  FixNHKokkos<DeviceType>::init();

  if (!temperature->tempbias)
    error->all(FLERR,"Temperature for fix nvt/sllod/chunk does not have a bias");

  nondeformbias = 0;
  if (strcmp(temperature->style,"temp/deform/chunk") != 0) nondeformbias = 1;

  // check fix deform remap settings

  int i;
  for (i = 0; i < modify->nfix; i++)
    if (strncmp(modify->fix[i]->style,"deform",6) == 0) {
      if (((FixDeform *) modify->fix[i])->remapflag != Domain::V_REMAP)
        error->all(FLERR,"Using fix nvt/sllod/chunk with inconsistent fix deform "
                   "remap option");
      break;
    }
  if (i == this->modify->nfix)
    error->all(FLERR,"Using fix nvt/sllod/chunk with no fix deform defined");
  
  // Chunk compute
  if(idchunk == nullptr)
    error->all(FLERR,"fix nvt/sllod/chunk does not use chunk/atom compute");
  int icompute = modify->find_compute(idchunk);
  if (icompute < 0)
    error->all(FLERR,"Chunk/atom compute does not exist for "
               "fix nvt/sllod/chunk");
  cchunk = dynamic_cast<ComputeChunkAtom *>( modify->compute[icompute]);
  if (strcmp(cchunk->style,"chunk/atom") != 0)
    error->all(FLERR,"fix nvt/sllod/chunk does not use chunk/atom compute");

  // Chunk VCM compute
  if(idchunk == nullptr)
    error->all(FLERR,"fix nvt/sllod/chunk does not use vcm/chunk compute");
  icompute = modify->find_compute(idvcm);
  if (icompute < 0)
    error->all(FLERR,"vcm/chunk compute does not exist for "
               "fix nvt/sllod/chunk");
  cvcm = dynamic_cast<ComputeVCMChunk *>( modify->compute[icompute]);
  if (strcmp(cvcm->style,"vcm/chunk") != 0)
    error->all(FLERR," does not use vcm/chunk compute");
}

/* ----------------------------------------------------------------------
   perform half-step scaling of velocities
-----------------------------------------------------------------------*/


void FixNVTSllodChunkKokkos::setup(int vflag) {
  FixNHKokkos::setup(vflag);

  // Apply kick if necessary
  if (this->kickflag) {
    // Call remove_bias first to calculate biases
    atomKK->sync(this->temperature->execution_space,this->temperature->datamask_read);
    this->temperature->compute_scalar();
    atomKK->modified(this->temperature->execution_space,this->temperature->datamask_modify);

    this->temperature->remove_bias_all();

    // Restore twice to apply streaming profile
    this->temperature->restore_bias_all();
    this->temperature->restore_bias_all();

    // Don't kick again if multi-step run
    this->kickflag = 0;
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixNVTSllodChunkKokkos<DeviceType>::nh_v_temp(){
  // remove and restore bias = streaming velocity = Hrate*lamda + Hratelo
  // thermostat thermal velocity only
  // vdelu = SLLOD correction = Hrate*Hinv*vthermal
  // for non temp/deform BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias

 // (Original) if (nondeformbias) temperature->compute_scalar();
  if (nondeformbias) {
    atomKK->sync(this->temperature->execution_space,this->temperature->datamask_read);
    this->temperature->compute_scalar();
    atomKK->modified(this->temperature->execution_space,this->temperature->datamask_modify);
  }
  // Remove bias from all atoms at once to avoid re-calculating the COM positions
  this->temperature->remove_bias_all();
  
  // Use molecular/chunk centre-of-mass velocity when calculating SLLOD correction
  vcm_thermal_compute();
  
  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;
  int index;
  
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>():
  rmass = atomKK->k_rmass.view<DeviceType>():
          
  double massone;
  int nlocal = atomKK->nlocal;   
  
  if(this->igroup == atomKK->firstgroup) nlocal = atomKK->nfirst;
  
  double h_two[6];
  MathExtra::multiply_shape_shape(this->domain->h_rate,this->domain->h_inv,h_two);

  d_h_two = Few<double, 6>(h_two);
  
  if (vdelu.extent(0) < atomKK->nmax)
    vdelu = typename AT::t_v_array(Kokkos::NoInit("nvt/sllod/kk:vdelu"), atomKK->nmax);

  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodChunk_temp1>(0,nlocal),*this);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixNVTSllodChunk_temp2>(0,nlocal),*this);
  this->copymode = 0;

  this->temperature->restore_bias_all();

}
/* calculate COM thermal velocity. 
 * Pre: atom velocities should have streaming bias removed
 *      COM positions should already be computed when removing biases
 */
void FixNVTSllodChunk::vcm_thermal_compute() {
  int index;
  double massone;

  // compute chunk/atom assigns atoms to chunk IDs
  // extract ichunk index vector from compute
  // ichunk = 1 to Nchunk for included atoms, 0 for excluded atoms
  nchunk = cchunk->setup_chunks();
  cchunk->compute_ichunk();
  int *ichunk = cchunk->ichunk;

  if (nchunk > maxchunk) {
    maxchunk = nchunk;
    memoryKK->destroy_kokkos(this->vcm)
    memoryKK->destroy_kokkos(this->vcm);
    memoryKK->destroy_kokkos(this->vcmall);
    memoryKK->destroy_kokkos(this->massproc);
    memoryKK->destroy_kokkos(this->masstotal);
    memoryKK->create_kokkos(this->vcm,maxchunk,3,"nvt/sllod/chunk:vcm");
    memoryKK->create_kokkos(this->vcmall,maxchunk,3,"nvt/sllod/chunk:vcmall");
    memoryKK->create_kokkos(this->massproc,maxchunk,"nvt/sllod/chunk:massproc");
    memoryKK->create_kokkos(this->masstotal,maxchunk,"nvt/sllod/chunk:masstotal");
  }

  // zero local per-chunk values

  Kokkos::parallel_for (nchunk, KOKKOS_LAMBDA (int i)) {
    this->vcm[i][0] = this->vcm[i][1] = this->vcm[i][2] = 0.0;
    this->massproc[i] = 0.0;
  }

  // compute COM and VCM for each chunk
  
  v = atomKK->k_v.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mass = atomKK->k_mass.view<DeviceType>():
  rmass = atomKK->k_rmass.view<DeviceType>():               
  image = atomKK->k_image.view<DeviceType>();

  int nlocal = atomKK->nlocal;

  int xbox, ybox, zbox;
  double v_adjust[3];
  
  Kokkos::parallel_for(nlocal, KOKKOS_LAMBDA (int i)){
    if (mask[i] & groupbit) {
      index = ichunk[i]-1;
      if (index < 0) continue;
      if (rmass) {
        massone = rmass[i];
      } else {
        massone = mass[type[i]];
      }
      // Adjust the velocity to reflect the thermal velocity 
      this->vcm(index,0) += v(i,0) * massone;
      this->vcm(index,1) += v(i,1) * massone;
      this->vcm(index,2) += v(i,2) * massone;
      this->massproc(index) += massone;
    }
  }
  
  // Not sure how to handle the MPI ALL Reduce yet
  MPI_Allreduce(&vcm[0][0],&vcmall[0][0],3*nchunk,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(massproc,masstotal,nchunk,MPI_DOUBLE,MPI_SUM,world);

  Kokkos::parallel_for(nchunk, KOKKOS_LAMBDA (int i)) {
    if (this->masstotal(i) > 0.0) {
      this->vcmall(i,0) /= masstotal(i);
      this->vcmall(i,1) /= masstotal(i);
      this->vcmall(i,2) /= masstotal(i);
    } else {
      this->vcmall(i,0) = this->vcmall(i,1) = this->vcmall(i,2) = 0.0;
    }
  }
}


template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodChunkKokkos<DeviceType>::operator()(TagFixNVTSllodChunk_temp1, const int &i) const {
  if (mask[i] & this->groupbit) {
    int index = ichunk[i]-1;
    if( index < 0 ) continue;
    vdelu(i,0) = d_h_two[0]*vcmall(index,0) + d_h_two[5]*vcmall(index,1) + d_h_two[4]*vcmall(index,2);
    vdelu(i,1) = d_h_two[1]*vcmall(index,1) + d_h_two[3]*vcmall(index,2);
    vdelu(i,2) = d_h_two[2]*vcmall(index,2);
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void FixNVTSllodChunkKokkos<DeviceType>::operator()(TagFixNVTSllodChunk_temp2, const int &i) const {
  if (mask[i] & this->groupbit) {
    int index = ichunk[i]-1;
    if( index < 0 ) continue;
    
    v(i,0) = v(i,0) - vcmall(index,0) + vcmall(index,0)*this->factor_eta - this->dthalf*vdelu(i,0);
    v(i,1) = v(i,1) - vcmall(index,1) + vcmall(index,1)*this->factor_eta - this->dthalf*vdelu(i,1);
    v(i,2) = v(i,2) - vcmall(index,2) + vcmall(index,2)*this->factor_eta - this->dthalf*vdelu(i,2);

  }
}

namespace LAMMPS_NS {
template class FixNVTSllodChunkKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixNVTSllodChunkKokkos<LMPHostType>;
#endif
}
