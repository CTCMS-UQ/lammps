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

#include "compute_pressure_mol.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "fix_property_mol.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "update.h"

#include <cctype>
#include <cstring>
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePressureMol::ComputePressureMol(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), id_temp(nullptr), id_molprop(nullptr),
  molprop(nullptr), pstyle(nullptr), pair_ptr(nullptr), com_peratom(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute pressure/mol command");
  if (igroup) error->all(FLERR,"Compute pressure/mol must use group all");

  scalar_flag = vector_flag = 1;
  size_vector = 9;
  extscalar = 0;
  extvector = 0;
  pressflag = 1;
  timeflag = 1;

  id_molprop = utils::strdup(arg[3]);
  if (strcmp(arg[4],"NULL") == 0) id_temp = nullptr;
  else id_temp = utils::strdup(arg[4]);

  // process optional args
  pairhybridflag = 0;
  if (narg == 5) {
    keflag = 1;
    pairflag = 1;
    kspaceflag = 1;
  } else {
    keflag = 0;
    pairflag = 0;
    kspaceflag = 0;
    int iarg = 5;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"ke") == 0) keflag = 1;
      else if (strcmp(arg[iarg],"pair/hybrid") == 0) {
        if (lmp->suffix)
          pstyle = utils::strdup(fmt::format("{}/{}",arg[++iarg],lmp->suffix));
        else
          pstyle = utils::strdup(arg[++iarg]);

        nsub = 0;

        if (narg > iarg) {
          if (isdigit(arg[iarg][0])) {
            nsub = utils::inumeric(FLERR,arg[iarg],false,lmp);
            ++iarg;
            if (nsub <= 0)
              error->all(FLERR,"Illegal compute pressure/mol command");
          }
        }

        // check if pair style with and without suffix exists

        pair_ptr = (Pair *) force->pair_match(pstyle,1,nsub);
        if (!pair_ptr && lmp->suffix) {
          pstyle[strlen(pstyle) - strlen(lmp->suffix) - 1] = '\0';
          pair_ptr = (Pair *) force->pair_match(pstyle,1,nsub);
        }

        if (!pair_ptr)
          error->all(FLERR,"Unrecognized pair style in compute pressure/mol command");

        pairhybridflag = 1;
      }
      else if (strcmp(arg[iarg],"pair") == 0) pairflag = 1;
      else if (strcmp(arg[iarg],"kspace") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg],"virial") == 0) {
        pairflag = 1;
        kspaceflag = 1;
      }
      else error->all(FLERR,"Illegal compute pressure command");
      iarg++;
    }
  }

  // error check
  if (keflag && id_temp == nullptr)
    error->all(FLERR,"Compute pressure/mol requires temperature ID "
               "to include kinetic energy");

  // pairflag + pairhybridflag would mean double-counting
  if (pairflag && pairhybridflag)
    error->all(FLERR,"The 'pairhybrid' option is incompatible with the 'pair' "
        "and 'virial' options for compute pressure/mol");

  vector = new double[size_vector];
  nmax = -1;
}

/* ---------------------------------------------------------------------- */

ComputePressureMol::~ComputePressureMol()
{
  if (force && force->pair) pair_ptr->del_tally_callback(this);
  if (com_peratom) memory->destroy(com_peratom);
  delete [] id_temp;
  delete [] id_molprop;
  delete [] vector;
  delete [] pstyle;
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::init()
{
  if (pairflag || pairhybridflag) {
    if (force->pair == nullptr)
      error->all(FLERR, "Trying to use compute pressure/mol without pair style");
  }

  // set temperature compute, must be done in init()
  // fixes could have changed or compute_modify could have changed it
  if (keflag) {
    int icompute = modify->find_compute(id_temp);
    if (icompute < 0)
      error->all(FLERR,"Could not find compute pressure temperature ID");
    if (modify->compute[icompute]->tempflag == 0)
      error->all(FLERR,"Compute pressure temperature ID does not "
                 "compute temperature");
    temperature = modify->compute[icompute];
  }

  // recheck if pair style with and without suffix exists
  if (pairhybridflag) {
    pair_ptr = (Pair *) force->pair_match(pstyle,1,nsub);
    if (!pair_ptr && lmp->suffix) {
      strcat(pstyle,"/");
      strcat(pstyle,lmp->suffix);
      pair_ptr = (Pair *) force->pair_match(pstyle,1,nsub);
    }

    if (!pair_ptr)
      error->all(FLERR,"Unrecognized pair style in compute pressure/mol command");

    auto ph = dynamic_cast<PairHybrid *>( force->pair);
    ph->no_virial_fdotr_compute = 1;
  } else {
    pair_ptr = force->pair;
  }
  if (pairflag || pairhybridflag)
    pair_ptr->add_tally_callback(this);

  // find fix property/mol
  molprop = dynamic_cast<FixPropertyMol*>(modify->get_fix_by_id(id_molprop));
  if (molprop == nullptr)
    error->all(FLERR, "Compute pressure/mol could not find a fix property/mol with id {}", id_molprop);
  if (igroup != molprop->igroup)
    error->all(FLERR, "Compute pressure/mol must be defined for the same group as fix nvt/sllod/mol");
  // Make sure CoM can be computed
  molprop->request_com();

  // Make sure ghost atoms have image flags.
  // Needed to map CoM to same image as ghost atoms.
  if (!comm->ghost_imageflags)
    error->all(FLERR, "Fix property/mol requires image flags to be communicated. "
        "Use comm_modify image yes to enable this.");

  // Warn about possible incompatibilities
  if (comm->me == 0) {
    // Check for tally callback issues
    if ((pairflag || pairhybridflag) &&
        pair_ptr->single_enable == 0 ||
        pair_ptr->manybody_flag)
      error->warning(FLERR,"Compute pressure/mol used with incompatible pair style");

    // Check for fixes that contribute to the virial (currently not handled)
    for (auto &ifix : modify->get_fix_list()) {
      if (ifix->thermo_virial)
        error->warning(FLERR,
            "Compute pressure/mol does not account for fix virial "
            "contributions. This warning can be safely ignored for fixes that "
            "only contribute intramolecular forces");
    }

    // kspace not yet supported
    if (kspaceflag && force->kspace)
      error->warning(FLERR,"Compute pressure/mol does not yet handle kspace forces");
  }
  did_setup = -1;

  boltz = force->boltz;
  nktv2p = force->nktv2p;
  dimension = domain->dimension;

  // flag Kspace contribution separately, since not summed across procs
  if (kspaceflag && force->kspace) {
    kspace_virial = force->kspace->virial;
  } else kspace_virial = nullptr;
}

/* ----------------------------------------------------------------------
   compute total pressure, averaged over Pxx, Pyy, Pzz
------------------------------------------------------------------------- */

double ComputePressureMol::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  if (did_setup != invoked_scalar || update->vflag_global != invoked_scalar)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  // invoke temperature if it hasn't been already

  if (keflag) {
    if (temperature->invoked_scalar != update->ntimestep)
      temperature->compute_scalar();
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(3,3);
    if (keflag)
      scalar = (temperature->dof * boltz * temperature->scalar +
                virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
    else
      scalar = (virial[0] + virial[1] + virial[2]) / 3.0 * inv_volume * nktv2p;
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(2,2);
    if (keflag)
      scalar = (temperature->dof * boltz * temperature->scalar +
                virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
    else
      scalar = (virial[0] + virial[1]) / 2.0 * inv_volume * nktv2p;
  }

  return scalar;
}

/* ----------------------------------------------------------------------
   compute pressure tensor
   assume KE tensor has already been computed
------------------------------------------------------------------------- */

void ComputePressureMol::compute_vector()
{
  invoked_vector = update->ntimestep;
  if (did_setup != invoked_vector || update->vflag_global != invoked_vector)
    error->all(FLERR,"Virial was not tallied on needed timestep");

  if (force->kspace && kspace_virial && force->kspace->scalar_pressure_flag)
    error->all(FLERR,"Must use 'kspace_modify pressure/scalar no' for "
               "tensor components with kspace_style msm");

  int i;
  double ke_tensor[9];
  if (keflag) {
    // invoke temperature if it hasn't been already
    if (temperature->invoked_vector != update->ntimestep)
      temperature->compute_vector();

    // The kinetic energy tensor is symmetric by definition,
    // but we still need the full 9 elements,
    // so copy them and duplicate as necessary
    double *temp_tensor = temperature->vector;
    for(i=0; i < 6; i++)
      ke_tensor[i] = temp_tensor[i];
    ke_tensor[6] = temp_tensor[3];
    ke_tensor[7] = temp_tensor[4];
    ke_tensor[8] = temp_tensor[5];
  }

  if (dimension == 3) {
    inv_volume = 1.0 / (domain->xprd * domain->yprd * domain->zprd);
    virial_compute(9,3);
    if (keflag) {
      for (int i = 0; i < 9; i++)
        vector[i] = (ke_tensor[i] + virial[i]) * inv_volume * nktv2p;
    } else {
      for (int i = 0; i < 9; i++)
        vector[i] = virial[i] * inv_volume * nktv2p;
    }
  } else {
    inv_volume = 1.0 / (domain->xprd * domain->yprd);
    virial_compute(4,2);

    if (keflag) {
      vector[0] = (ke_tensor[0] + virial[0]) * inv_volume * nktv2p;
      vector[1] = (ke_tensor[1] + virial[1]) * inv_volume * nktv2p;
      vector[2] = (ke_tensor[3] + virial[3]) * inv_volume * nktv2p;
      vector[3] = (ke_tensor[6] + virial[6]) * inv_volume * nktv2p;
    } else {
      vector[0] = virial[0] * inv_volume * nktv2p;
      vector[1] = virial[1] * inv_volume * nktv2p;
      vector[2] = virial[3] * inv_volume * nktv2p;
      vector[3] = virial[6] * inv_volume * nktv2p;
    }
    vector[4] = vector[5] = vector[6] = vector[7] = vector[8] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::virial_compute(int n, int ndiag)
{
  int i,j;
  double v[9],*vcomponent;

  for (i = 0; i < n; i++) v[i] = 0.0;

  // sum contributions to virial from pair forces

  if (pairflag || pairhybridflag)
    for (i = 0; i < n; i++) v[i] += pair_virial[i];

  // sum virial across procs

  MPI_Allreduce(v,virial,n,MPI_DOUBLE,MPI_SUM,world);

  // KSpace virial contribution is already summed across procs
  // TODO(SS): calculate this correctly (probably need a kspace_virial_mol)

  if (kspace_virial)
    for (i = 0; i < n; i++) virial[i] += kspace_virial[i];

  // LJ long-range tail correction, only if pair contributions are included
  // TODO(SS): Check that this is correct

  if ((pairflag || pairhybridflag) && pair_ptr->tail_flag)
    for (i = 0; i < ndiag; i++) virial[i] += pair_ptr->ptail * inv_volume;
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::reset_extra_compute_fix(const char *id_new)
{
  delete [] id_temp;
  id_temp = utils::strdup(id_new);
}

/* ---------------------------------------------------------------------- */

void ComputePressureMol::pair_setup_callback(int eflag, int vflag) {
  if (did_setup == update->ntimestep || !matchstep(update->ntimestep)) return;
  did_setup = update->ntimestep;

  for (int d = 0; d < 9; d++)
    pair_virial[d] = 0.0;

  // Make sure CoM is up to date
  if (molprop->com_step != update->ntimestep)
    molprop->com_compute();

  // Pre-compute CoM positions for each atom
  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(com_peratom, nmax, 3, "pressure/mol:com_peratom");
  }
  tagint * molecule = (molprop->use_mpiallreduce) ? atom->molecule : molprop->molnum;
  for (int i = 0; i < atom->nlocal + atom->nghost; ++i) {
    tagint m = molecule[i]-1;
    if (m < 0) {
      com_peratom[i][0] = atom->x[i][0];
      com_peratom[i][1] = atom->x[i][1];
      com_peratom[i][2] = atom->x[i][2];
    } else {
      com_peratom[i][0] = molprop->com[m][0];
      com_peratom[i][1] = molprop->com[m][1];
      com_peratom[i][2] = molprop->com[m][2];
      // CoM is stored in unwrapped coordinates. Need to map to same image as atom
      imageint ix = (2*IMGMAX - (atom->image[i] & IMGMASK)) & IMGMASK;
      imageint iy = (2*IMGMAX - (atom->image[i] >> IMGBITS & IMGMASK)) & IMGMASK;
      imageint iz = (2*IMGMAX - (atom->image[i] >> IMG2BITS)) & IMGMASK;
      domain->unmap(com_peratom[i], ix | (iy << IMGBITS) | (iz << IMG2BITS));
    }
  }
}

/* ----------------------------------------------------------------------
   tally molecular virials into global accumulator
   have delx, dely, delz and fpair (which gives fx, fy, fz)
   get delcomx, delcomy, delcomz (molecule centre-of-mass separation)
   from atom->property_molecule
------------------------------------------------------------------------- */

void ComputePressureMol::pair_tally_callback(int i, int j, int nlocal,
    int newton_pair, double evdwl, double ecoul, double fpair,
    double delx, double dely, double delz)
{
  // Virial does not need to be tallied if we didn't do setup this step
  if (did_setup != update->ntimestep) return;

  // No virial contribution if i and j both not owned
  if (i >= nlocal && j >= nlocal) return;

  double delcom[3], v[9];
  double scale = (newton_pair || (i < nlocal && j < nlocal)) ? 1.0 : 0.5;
  for (int d = 0; d < 3; d++) {
    delcom[d] = scale * (com_peratom[i][d] - com_peratom[j][d]);
  }

  v[0] = delcom[0]*delx*fpair;
  v[1] = delcom[1]*dely*fpair;
  v[2] = delcom[2]*delz*fpair;
  v[3] = delcom[0]*dely*fpair;
  v[4] = delcom[0]*delz*fpair;
  v[5] = delcom[1]*delz*fpair;
  v[6] = delcom[1]*delx*fpair;
  v[7] = delcom[2]*delx*fpair;
  v[8] = delcom[2]*dely*fpair;

  for (int d = 0; d < 9; ++d)
    pair_virial[d] += v[d];
}
