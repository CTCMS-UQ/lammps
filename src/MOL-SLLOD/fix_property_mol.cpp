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

#include "fix_property_mol.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPropertyMol::FixPropertyMol(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), mass(nullptr), com(nullptr), massproc(nullptr), comproc(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal fix property/atom command");

  int iarg = 3;

  mass_flag = 0;
  com_flag = 0;

  dynamic_group_allow = 1;
  dynamic_group = group->dynamic[igroup];
  dynamic_mols = 0;
  use_mpiallreduce = false;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "mass") == 0) {
      request_mass();
      iarg++;
    } else if (strcmp(arg[iarg], "com") == 0) {
      request_com();
      iarg++;
    } else if (strcmp(arg[iarg], "dynamic") == 0) {
      if (++iarg >= narg)
        error->all(FLERR,
                   "Illegal fix property/mol command. "
                   "Expected value after 'dynamic' keyword");
      dynamic_mols = utils::logical(FLERR, arg[iarg], false, lmp);
      iarg++;
    } else if (strcmp(arg[iarg], "use_mpiallreduce") == 0) {
      if (++iarg >= narg)
        error->all(FLERR,
                   "Illegal fix property/mol command. "
                   "Expected value after 'use_mpiallreduce' keyword");
      use_mpiallreduce = utils::logical(FLERR, arg[iarg], false, lmp);
      iarg++;
    } else
      error->all(FLERR, "Illegal fix property/mol command");
  }

  nmax = 0;
  molmax = 1;
  nmolecule = 0;

  com_step = -1;
  mass_step = -1;
  count_step = -1;
  preneigh_step = -1;

  array_flag = 1;
  size_array_cols = 4;
  size_array_rows_variable = 1;

  recvcounts = new int[comm->nprocs];
  displs = new int[comm->nprocs];
}

/* ---------------------------------------------------------------------- */

FixPropertyMol::~FixPropertyMol()
{
  for (auto &item : permolecule) mem_destroy(item);
  delete[] recvcounts;
  delete[] displs;
}

/* ---------------------------------------------------------------------- */

int FixPropertyMol::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  mask |= PRE_NEIGHBOR;
  return mask;
}

void FixPropertyMol::request_com()
{
  if (com_flag) return;
  com_flag = 1;
  request_mass();
  register_permolecule("property/mol:com", &com, Atom::DOUBLE, 3);
  register_permolecule("property/mol:comproc", &comproc, Atom::DOUBLE, 3);
}

void FixPropertyMol::request_mass()
{
  if (mass_flag) return;
  mass_flag = 1;
  register_permolecule("property/mol:mass", &mass, Atom::DOUBLE, 0);
  register_permolecule("property/mol:massproc", &massproc, Atom::DOUBLE, 0);
}

/* ----------------------------------------------------------------------
   allocate a per-molecule array which will be grown automatically.
   This should be called with the *address* of the pointer to the
   allocated memory:

   double *arr;
   register_permolecule("arr", &arr, Atom::DOUBLE, 0);
   destroy_permolecule(&arr);
---------------------------------------------------------------------- */

void FixPropertyMol::register_permolecule(std::string name, void *address, int datatype, int cols)
{
  if (address == nullptr) return;

  for (auto &item : permolecule) {
    if (address == item.address) return;
  }
  permolecule.emplace_back(PerMolecule{name, address, datatype, cols});
  if (nmax > 0) {
    auto &item = permolecule.back();
    mem_create(item);
  }
}

/* ----------------------------------------------------------------------
   de-allocate a per-molecule array. This should be called with the
   *address* of the pointer to the allocated memory:

   double *arr;
   register_permolecule("arr", &arr, Atom::DOUBLE, 0);
   destroy_permolecule(&arr);
---------------------------------------------------------------------- */
void FixPropertyMol::destroy_permolecule(void *address)
{
  auto item = permolecule.begin();
  while (item != permolecule.end()) {
    if (item->address == address) {
      mem_destroy(*item);
      item = permolecule.erase(item);
    } else
      ++item;
  }
}

/* ---------------------------------------------------------------------- */

void FixPropertyMol::init()
{
  // Error if system doesn't track molecule ids.
  // Check here since atom_style could change before run.

  if (!atom->molecule_flag)
    error->all(FLERR, "Fix property/mol when atom_style does not define a molecule attribute");
}

/* ----------------------------------------------------------------------
   Need to calculate mass and CoM before main setup() calls since those
   could rely on the memory being allocated (e.g. for virial tallying)
---------------------------------------------------------------------- */
void FixPropertyMol::setup_pre_force(int /*vflag*/)
{
  dynamic_group = group->dynamic[igroup];
  grow_permolecule();

  // com_compute also computes mass if dynamic_group is set
  // so no need to call mass_compute in that case
  if (mass_flag && !dynamic_group) mass_compute();
  if (com_flag) com_compute();
  if (mass_flag) count_molecules();
}

void FixPropertyMol::setup_pre_force_respa(int vflag, int ilevel)
{
  if (ilevel == 0) setup_pre_force(vflag);
}

/* ----------------------------------------------------------------------
   Calculate number of molecules and grow permolecule arrays if needed.
   Grows to the maximum of previous max. mol id + grow_by and new max. mol id
   if either is larger than nmax.
   Returns true if the number of molecules (max. mol id) changed.
------------------------------------------------------------------------- */

bool FixPropertyMol::grow_permolecule(int grow_by)
{
  // Calculate maximum molecule id
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  tagint maxone = -1;
  for (int i = 0; i < nlocal; i++)
    if (molecule[i] > maxone) maxone = molecule[i];
  tagint maxall;
  MPI_Allreduce(&maxone, &maxall, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (maxall > MAXSMALLINT) error->all(FLERR, "Molecule IDs too large for fix property/mol");

  tagint old_molmax = molmax;
  tagint new_size = molmax + grow_by;
  molmax = maxall;
  new_size = MAX(molmax, new_size);

  // Grow arrays as needed
  if (nmax < new_size) {
    nmax = new_size;
    for (auto &item : permolecule) mem_grow(item);
  }

  size_array_rows = static_cast<int>(molmax);
  return old_molmax != molmax;
}

/* ----------------------------------------------------------------------
   Count the number of molecules with non-zero mass.
   Mass of molecules is only counted from atoms in the group, so count is
   the number of molecules in the group.
------------------------------------------------------------------------- */
void FixPropertyMol::count_molecules()
{
  count_step = update->ntimestep;
  nmolecule = 0;
  if (use_mpiallreduce) {
    for (tagint m = 0; m < molmax; ++m)
      if (mass[m] > 0.0) ++nmolecule;
  } else {
    for (auto const &m : owned_mols)
      if (mass[m] > 0.0) ++nmolecule;
    MPI_Allreduce(MPI_IN_PLACE, &nmolecule, 1, MPI_LMP_TAGINT, MPI_SUM, world);
  }
}

/* ----------------------------------------------------------------------
   Update total mass of each molecule
------------------------------------------------------------------------- */

void FixPropertyMol::mass_compute()
{
  mass_step = update->ntimestep;
  if (dynamic_mols) grow_permolecule();
  if (molmax == 0) return;
  double massone;
  for (tagint m = 0; m < molmax; ++m) massproc[m] = 0.0;

  for (int i = 0; i < atom->nlocal; ++i) {
    if (groupbit & atom->mask[i]) {
      tagint m = atom->molecule[i] - 1;
      if (m < 0) continue;
      if (atom->rmass)
        massone = atom->rmass[i];
      else
        massone = atom->mass[atom->type[i]];
      massproc[m] += massone;
    }
  }
  if (use_mpiallreduce) {
    MPI_Allreduce(massproc, mass, molmax, MPI_DOUBLE, MPI_SUM, world);
  } else {
    if (dynamic_mols || dynamic_group) pre_neighbor();
    memset(mass, 0, molmax * sizeof(double));
    for (auto m : local_mols) mass[m] = massproc[m];
    int b = 0;
    for (auto m : send_mols) buffer[buffer_mylo + b++] = mass[m];
    MPI_Allgatherv(MPI_IN_PLACE, send_size, MPI_DOUBLE, &(buffer.front()), recvcounts, displs,
                   MPI_DOUBLE, world);
    for (auto lookup : buffer_ghost_lookup) mass[lookup.second] += buffer[lookup.first];
  }
}

/* ----------------------------------------------------------------------
   Calculate center of mass of each molecule in unwrapped coords
   Also update molecular mass if group is dynamic
------------------------------------------------------------------------- */

void FixPropertyMol::com_compute()
{
  com_step = update->ntimestep;
  // Recalculate mass if number of molecules (max. mol id) changed, or if
  // group is dynamic
  bool recalc_mass = dynamic_group;
  if (dynamic_mols) recalc_mass |= grow_permolecule();
  if (molmax == 0) return;

  int nlocal = atom->nlocal;
  tagint *molecule = atom->molecule;

  int *type = atom->type;
  double *amass = atom->mass;
  double *rmass = atom->rmass;
  double **x = atom->x;
  double **v = atom->v;
  double massone, unwrap[3];

  for (int m = 0; m < molmax; ++m) {
    comproc[m][0] = 0.0;
    comproc[m][1] = 0.0;
    comproc[m][2] = 0.0;
  }

  if (recalc_mass) {
    mass_step = update->ntimestep;
    for (tagint m = 0; m < molmax; ++m) massproc[m] = 0.0;
  }

  for (int i = 0; i < nlocal; ++i) {
    if (groupbit & atom->mask[i]) {
      tagint m = molecule[i] - 1;
      if (m < 0) continue;
      if (rmass)
        massone = rmass[i];
      else
        massone = amass[type[i]];

      // NOTE: if FP error becomes a problem here in long-running
      //       simulations, could maybe do something clever with
      //       image flags to reduce it, but MPI makes that difficult,
      //       and it would mean needing to store image flags for CoM
      domain->unmap(x[i], atom->image[i], unwrap);
      comproc[m][0] += unwrap[0] * massone;
      comproc[m][1] += unwrap[1] * massone;
      comproc[m][2] += unwrap[2] * massone;
      if (recalc_mass) massproc[m] += massone;
    }
  }

  if (use_mpiallreduce) {
    MPI_Allreduce(&comproc[0][0], &com[0][0], 3 * molmax, MPI_DOUBLE, MPI_SUM, world);
  } else {
    if (dynamic_mols || dynamic_group) pre_neighbor();
    memset(&com[0][0], 0, 3 * molmax * sizeof(double));
    for (int d = 0; d < 3; d++) {
      for (auto m : local_mols) com[m][d] = comproc[m][d];
      int b = 0;
      for (auto m : send_mols) buffer[buffer_mylo + b++] = com[m][d];
      MPI_Allgatherv(MPI_IN_PLACE, send_size, MPI_DOUBLE, &(buffer.front()), recvcounts, displs,
                     MPI_DOUBLE, world);
      for (auto lookup : buffer_ghost_lookup) com[lookup.second][d] += buffer[lookup.first];
    }
  }
  if (recalc_mass) {
    if (use_mpiallreduce) {
      MPI_Allreduce(massproc, mass, molmax, MPI_DOUBLE, MPI_SUM, world);
    } else {
      if (dynamic_mols || dynamic_group) pre_neighbor();
      memset(mass, 0, molmax * sizeof(double));
      for (auto m : local_mols) mass[m] = massproc[m];
      int b = 0;
      for (auto m : send_mols) buffer[buffer_mylo + b++] = mass[m];
      MPI_Allgatherv(MPI_IN_PLACE, send_size, MPI_DOUBLE, &(buffer.front()), recvcounts, displs,
                     MPI_DOUBLE, world);
      for (auto lookup : buffer_ghost_lookup) mass[lookup.second] += buffer[lookup.first];
    }
  }

  for (int m = 0; m < molmax; ++m) {
    // Some molecule ids could be skipped (not assigned atoms)
    if (mass[m] > 0.0) {
      com[m][0] /= mass[m];
      com[m][1] /= mass[m];
      com[m][2] /= mass[m];
    } else {
      com[m][0] = com[m][1] = com[m][2] = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixPropertyMol::memory_usage()
{
  double bytes = 0.0;
  if (mass_flag) bytes += nmax * 2 * sizeof(double);
  if (com_flag) bytes += nmax * 6 * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   basic array output
------------------------------------------------------------------------- */

double FixPropertyMol::compute_array(int imol, int col)
{
  if (imol > static_cast<int>(molmax))
    error->all(
        FLERR,
        fmt::format("Cannot request info for molecule {} from fix property/mol (molmax = {})", imol,
                    molmax));

  if (col == 3) {
    // Mass requested
    if (!mass_flag) error->all(FLERR, "This fix property/mol does not calculate mass");
    if (dynamic_group && mass_step != update->ntimestep) mass_compute();
    return mass[imol];
  } else {
    // CoM requested
    if (!com_flag) error->all(FLERR, "This fix property/mol does not calculate CoM");
    if (com_step != update->ntimestep) com_compute();
    return com[imol][col];
  }
}

/* ----------------------------------------------------------------------
   pre-neighbor mpi rendezvous
------------------------------------------------------------------------- */

void FixPropertyMol::pre_neighbor()
{
  if (use_mpiallreduce) return;
  if (update->ntimestep <= preneigh_step) return;
  preneigh_step = update->ntimestep;

  local_mols.clear();
  ghost_mols.clear();
  send_mols.clear();
  buffer_ghost_lookup.clear();

  int const nlocal = atom->nlocal;
  int const nghost = atom->nghost;
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;

  for (int i = 0; i < nlocal; ++i) {
    if (!(mask[i] & groupbit)) continue;
    local_mols.insert(molecule[i] - 1);
  }

  for (int i = nlocal; i < nlocal + nghost; ++i) {
    if (!(mask[i] & groupbit)) continue;
    if (atom->map(atom->tag[i]) < nlocal) continue;    // remove ghosts of locals
    ghost_mols.insert(molecule[i] - 1);
  }

  // gather global list of ghost_mols
  send_size = ghost_mols.size();
  MPI_Allgather(&send_size, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  displs[0] = 0;
  buffer_size = recvcounts[0];
  buffer_mylo = 0;            // defaults for proc 0
  buffer_myhi = send_size;    // defaults for proc 0
  for (int p = 1; p < comm->nprocs; p++) {
    displs[p] = displs[p - 1] + recvcounts[p - 1];
    buffer_size += recvcounts[p];
    if (p == comm->me) {
      buffer_mylo = displs[p];
      buffer_myhi = displs[p] + send_size;
    }
  }
  buffer.resize(buffer_size);
  int b = 0;
  for (auto m : ghost_mols) { buffer[buffer_mylo + b++] = ubuf(m).d; }
  MPI_Allgatherv(MPI_IN_PLACE, send_size, MPI_DOUBLE, &(buffer.front()), recvcounts, displs,
                 MPI_DOUBLE, world);

  // check which mols I should send
  for (int b = 0; b < buffer_mylo; ++b) {
    tagint m = ubuf(buffer[b]).i;
    auto found = local_mols.find(m);
    if (found != local_mols.end()) send_mols.insert(m);
  }
  for (int b = buffer_myhi; b < buffer_size; ++b) {
    tagint m = ubuf(buffer[b]).i;
    auto found = local_mols.find(m);
    if (found != local_mols.end()) send_mols.insert(m);
  }

  // gather global list of send_mols
  send_size = send_mols.size();
  MPI_Allgather(&send_size, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  displs[0] = 0;
  buffer_size = recvcounts[0];
  buffer_mylo = 0;            // defaults for proc 0
  buffer_myhi = send_size;    // defaults for proc 0
  for (int p = 1; p < comm->nprocs; p++) {
    displs[p] = displs[p - 1] + recvcounts[p - 1];
    buffer_size += recvcounts[p];
    if (p == comm->me) {
      buffer_mylo = displs[p];
      buffer_myhi = displs[p] + send_size;
    }
  }
  buffer.resize(buffer_size);
  b = 0;
  for (auto m : send_mols) { buffer[buffer_mylo + b++] = ubuf(m).d; }
  MPI_Allgatherv(MPI_IN_PLACE, send_size, MPI_DOUBLE, &(buffer.front()), recvcounts, displs,
                 MPI_DOUBLE, world);

  owned_mols = local_mols;
  // check which mols I should lookup and own
  for (int b = 0; b < buffer_mylo; ++b) {
    tagint m = ubuf(buffer[b]).i;
    auto found = ghost_mols.find(m);
    if (found != ghost_mols.end()) buffer_ghost_lookup[b] = m;
    if (m % 2 == 0) {
      auto found = owned_mols.find(m);
      if (found != owned_mols.end()) owned_mols.erase(m);
    }
  }
  for (int b = buffer_myhi; b < buffer_size; ++b) {
    tagint m = ubuf(buffer[b]).i;
    auto found = ghost_mols.find(m);
    if (found != ghost_mols.end()) buffer_ghost_lookup[b] = m;
    if (m % 2 == 1) {
      auto found = owned_mols.find(m);
      if (found != owned_mols.end()) owned_mols.erase(m);
    }
  }
}

/* ----------------------------------------------------------------------
   memory handling for permolecule data
------------------------------------------------------------------------- */

template <typename T> inline void FixPropertyMol::mem_create_impl(PerMolecule &item)
{
  if (item.cols == 0)
    memory->create(*(T **) item.address, nmax, item.name.c_str());
  else if (item.cols > 0)
    memory->create(*(T ***) item.address, nmax, item.cols, item.name.c_str());
}
void FixPropertyMol::mem_create(PerMolecule &item)
{

  if (item.datatype == Atom::DOUBLE)
    mem_create_impl<double>(item);
  else if (item.datatype == Atom::INT)
    mem_create_impl<int>(item);
  else if (item.datatype == Atom::BIGINT)
    mem_create_impl<bigint>(item);
}

template <typename T> inline void FixPropertyMol::mem_grow_impl(PerMolecule &item)
{
  if (item.cols == 0)
    memory->grow(*(T **) item.address, nmax, item.name.c_str());
  else if (item.cols > 0)
    memory->grow(*(T ***) item.address, nmax, item.cols, item.name.c_str());
}
void FixPropertyMol::mem_grow(PerMolecule &item)
{
  if (item.datatype == Atom::DOUBLE)
    mem_grow_impl<double>(item);
  else if (item.datatype == Atom::INT)
    mem_grow_impl<int>(item);
  else if (item.datatype == Atom::BIGINT)
    mem_grow_impl<bigint>(item);
}

template <typename T> inline void FixPropertyMol::mem_destroy_impl(PerMolecule &item)
{
  if (item.cols == 0)
    memory->destroy(*(T **) item.address);
  else if (item.cols > 0)
    memory->destroy(*(T ***) item.address);
}
void FixPropertyMol::mem_destroy(PerMolecule &item)
{
  if (item.datatype == Atom::DOUBLE)
    mem_destroy_impl<double>(item);
  else if (item.datatype == Atom::INT)
    mem_destroy_impl<int>(item);
  else if (item.datatype == Atom::BIGINT)
    mem_destroy_impl<bigint>(item);
}