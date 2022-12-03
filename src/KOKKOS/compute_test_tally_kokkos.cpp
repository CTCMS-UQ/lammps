#include "compute_test_tally_kokkos.h"

#include "error.h"
#include "force.h"
#include "pair_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

template<class DeviceType>
ComputeTestTallyKokkos<DeviceType>::ComputeTestTallyKokkos(class LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp,narg,arg), ev()
{
  if (narg != 3) error->all(FLERR,"Illegal compute test/tally command");
  kokkosable = 1;
  scalar_flag = 1;
  extscalar = 1;
}

template<class DeviceType>
ComputeTestTallyKokkos<DeviceType>::~ComputeTestTallyKokkos() {
  if (force && force->pair) force->pair->del_tally_callback(this);
}

template<class DeviceType>
void ComputeTestTallyKokkos<DeviceType>::init() {
  if (force->pair == nullptr)
    error->all(FLERR, "Trying to use compute test/tally without a pair style");
  else
    force->pair->add_tally_callback(this);
}

template<class DeviceType>
double ComputeTestTallyKokkos<DeviceType>::compute_scalar() {
  invoked_scalar = update->ntimestep;
  scalar = ev.sum_epair;
  return scalar;
}

template<class DeviceType>
void ComputeTestTallyKokkos<DeviceType>::pair_setup_callback(int eflag, int) {
  // NOTE: only updating value on eflag steps as example
  if (eflag) did_setup = update->ntimestep;
  else did_setup = -1;
}

template<class DeviceType>
TALLY_MASK ComputeTestTallyKokkos<DeviceType>::tally_mask() {
  // Can't have body of this function in header since including
  // s_TALLY_MASK from pair_kokkos.h in the header would create a dependency loop.
  // NOTE: can return 0 if tallying not needed this timestep
  return did_setup == -1 ? 0 :
    s_TALLY_MASK<DeviceType>::template get_tally_mask<ComputeTestTallyKokkos<DeviceType>>();
}

namespace LAMMPS_NS {
template class ComputeTestTallyKokkos<LMPDeviceType>;
template class ComputeTestTallyFunctor<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeTestTallyKokkos<LMPHostType>;
template class ComputeTestTallyFunctor<LMPHostType>;
#endif
}
