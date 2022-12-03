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

#ifdef PAIR_CLASS

#else

// clang-format off
#ifndef LMP_PAIR_KOKKOS_H
#define LMP_PAIR_KOKKOS_H

#include "compute.h"
#include "Kokkos_Macros.hpp"
#include "pair.h"               // IWYU pragma: export
#include "neighbor_kokkos.h"
#include "neigh_list_kokkos.h"
#include "Kokkos_Vectorization.hpp"
#include "Kokkos_ScatterView.hpp"

#include "style_tally_kokkos.h"

namespace LAMMPS_NS {

template<int Table>
struct CoulLongTable {
  enum {DoTable = Table};
};

// Tags for doing coulomb calculations or not
// They facilitate function overloading, since
// partial template specialization of member functions is not allowed
struct CoulTag {};
struct NoCoulTag {};

template<int FLAG>
struct DoCoul {
  typedef NoCoulTag type;
};

template<>
struct DoCoul<1> {
  typedef CoulTag type;
};

// Assign unique bit mask to each tally compute style or combination of styles.
// This will overflow if too many kokkos tally computes, but could be extended
// to a larger data type if needed.
// Specialise based on DeviceType so that lookup fails (returns 0) for computes
// with DeviceType different to pair style.
template<class DeviceType, class T, class ... Ts>
struct TallyMaskLookup {
  template<class U, std::size_t i = 0, std::enable_if_t<std::is_void<U>::value,bool> = true>
  constexpr static TALLY_MASK get_mask() {
    return 0;
  }

  template<class U, std::size_t i = 0, std::enable_if_t<!(sizeof...(Ts) > 0) && !std::is_void<U>::value,bool> = true>
  constexpr static TALLY_MASK get_mask() {
    static_assert(8*sizeof(TALLY_MASK) >= i, "Too many kokkos tally computes. Try increasing the size of TALLY_MASK.");
    if (std::is_same<T,U>::value) return 1 << i;
    return 0;
  }

  template<class U, std::size_t i = 0, std::enable_if_t<(sizeof...(Ts) > 0) && !std::is_void<U>::value,bool> = true>
  constexpr static TALLY_MASK get_mask() {
    static_assert(8*sizeof(TALLY_MASK) >= i, "Too many kokkos tally computes. Try increasing the size of TALLY_MASK.");
    if (std::is_same<T,U>::value) return 1 << i;
    return TallyMaskLookup<DeviceType,Ts...>::template get_mask<U,i+1>();
  }
};

template<class DeviceType>
struct s_TALLY_MASK {
  // TallyMaskLookup instantiation containing all kokkos tally computes
  typedef TallyMaskLookup<DeviceType
#define TallyStyle(Class) ,Class
#define KOKKOS_TALLY_CLASS
#include "style_tally_kokkos.h"
#undef KOKKOS_TALLY_CLASS
#undef TallyStyle
    > Lookup;

  template<class T, class ... Ts>
  static constexpr TALLY_MASK get_tally_mask(std::enable_if_t<(sizeof...(Ts)>0),bool> = true) {
    return Lookup::template get_mask<T>() | get_tally_mask<Ts...>();
  }
  template<class T, class ... Ts>
  static constexpr TALLY_MASK get_tally_mask(std::enable_if_t<!(sizeof...(Ts)>0),bool> = true) {
    return Lookup::template get_mask<T>();
  }
};

// Wrapper to get handle EV_FLOAT and FEV_FLOAT types in case of TallyStyle = void
// T should be some instantiation of TallyFunctor
template<class T> struct fev_wrapper {
  typedef typename T::ev_value_type ev_value_type;
  typedef typename T::fev_value_type fev_value_type;

  static KOKKOS_INLINE_FUNCTION
  void tally_add(ev_value_type &ev, const fev_value_type &fev) {
    ev.tally += fev.tally;
  }
};
template<> struct fev_wrapper<void> {
  typedef EV_FLOAT ev_value_type;
  typedef FEV_FLOAT fev_value_type;

  static KOKKOS_INLINE_FUNCTION
  void tally_add(ev_value_type &, const fev_value_type &) {}
};

// Combine multiple tally styles and pass through callbacks.
// Data can be linked to a particular compute through the init_step function,
// and the result of the pair reduction is passed back to the relevant
// compute with the consolidate function.
// Templated over DeviceType so that lookup fails for computes not on the
// same device as the pair style.
template<class DeviceType, class First, class ... Rest>
struct TallyFunctor {
  typedef s_EV_FLOAT<First, Rest...> ev_value_type;
  typedef s_FEV_FLOAT<First, Rest...> fev_value_type;

  typename First::tally_functor first;
  TallyFunctor<DeviceType, Rest...> rest;

  TallyFunctor() : first(), rest() {}

  inline void init_step(Compute* c_ptr, const TALLY_MASK &mask) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.init_step(static_cast<First*>(c_ptr));
    else rest.init_step(c_ptr,mask);
  }

  template<class EVType>
  inline void consolidate(Compute* c_ptr, const TALLY_MASK &mask, const EVType &ev) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.consolidate(static_cast<First*>(c_ptr), ev.template get<First>());
    else rest.consolidate(c_ptr,mask,ev);
  }

  template<int NEIGHFLAG, class EVType>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVType &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
    const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    first.template ev_tally<NEIGHFLAG>(tally.first,i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
    rest.template ev_tally<NEIGHFLAG>(tally.rest,i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
  }
};

template<class DeviceType, class First>
struct TallyFunctor<DeviceType, First> {
  typedef s_EV_FLOAT<First> ev_value_type;
  typedef s_FEV_FLOAT<First> fev_value_type;

  typename First::tally_functor first;

  TallyFunctor() : first() {}

  inline void init_step(Compute* c_ptr, const TALLY_MASK &mask) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.init_step(static_cast<First*>(c_ptr));
  }

  template<class EVType>
  inline void consolidate(Compute* c_ptr, const TALLY_MASK &mask, EVType ev) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.consolidate(static_cast<First*>(c_ptr), ev.template get<First>());
  }

  template<int NEIGHFLAG, class EVType>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVType &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
    const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    first.template ev_tally<NEIGHFLAG>(tally.first,i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
  }
};

// Specialise for voids to avoid errors. Can maybe remove this if 'void,void>'
// tail can be removed when building TallyCombinator
template<class DeviceType, class First>
struct TallyFunctor<DeviceType,First,void> {
  typedef s_EV_FLOAT<First,void> ev_value_type;
  typedef s_FEV_FLOAT<First,void> fev_value_type;

  typename First::tally_functor first;

  TallyFunctor() : first() {}

  inline void init_step(Compute* c_ptr, const TALLY_MASK &mask) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.init_step(static_cast<First*>(c_ptr));
  }

  template<class EVType>
  inline void consolidate(Compute* c_ptr, const TALLY_MASK &mask, EVType ev) {
    if ((mask & s_TALLY_MASK<DeviceType>::template get_tally_mask<First>()) == mask)
      first.consolidate(static_cast<First*>(c_ptr), ev.template get<First>());
  }

  template<int NEIGHFLAG, class EVType>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVType &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
    const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    first.template ev_tally<NEIGHFLAG>(tally.first,i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
  }
};

template<class DeviceType>
struct TallyFunctor<DeviceType,void> {
  typedef s_EV_FLOAT<void> ev_value_type;
  typedef s_FEV_FLOAT<void> fev_value_type;

  TallyFunctor() {}

  inline void init_step(Compute* c_ptr, const TALLY_MASK &mask) {}

  template<class EVType>
  inline void consolidate(Compute* c_ptr, const TALLY_MASK &mask, EVType ev) {}

  template<int NEIGHFLAG, class EVType>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVType &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const double &evdwl, const double &ecoul, const double &fpair,
    const double &delx, const double &dely, const double &delz) const
  {
  }
};

template<class DeviceType>
struct TallyFunctor<DeviceType,void,void> {
  typedef s_EV_FLOAT<void,void> ev_value_type;
  typedef s_FEV_FLOAT<void,void> fev_value_type;

  TallyFunctor() {}

  inline void init_step(Compute* c_ptr, const TALLY_MASK &mask) {}

  template<class EVType>
  inline void consolidate(Compute* c_ptr, const TALLY_MASK &mask, EVType ev) {}

  template<int NEIGHFLAG, class EVType>
  KOKKOS_INLINE_FUNCTION
  void ev_tally(EVType &tally,
    const int &i, const int &j, const int &nlocal, const int &newton_pair,
    const double &evdwl, const double &ecoul, const double &fpair,
    const double &delx, const double &dely, const double &delz) const
  {
  }
};


//Specialisation for Neighborlist types Half, HalfThread, Full
template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, class Specialisation = void, class TallyStyle = void>
struct PairComputeFunctor  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  constexpr static bool TALLYFLAG = !std::is_void<TallyStyle>::value;

  // Reduction type, contains evdwl, ecoul, virial[6], and extra data for TallyStyle
  typedef typename fev_wrapper<TallyStyle>::ev_value_type ev_value_type;
  typedef typename fev_wrapper<TallyStyle>::fev_value_type fev_value_type;
  typedef ev_value_type value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  using KKDeviceType = typename KKDevice<device_type>::value;
  using DUP = typename NeedDup<NEIGHFLAG,device_type>::value;

  // The force array is atomic for Half/Thread neighbor style
  //Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > f;
  KKScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_f;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  //Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > eatom;
  KKScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,KKDeviceType,KKScatterSum,DUP> dup_eatom;

  //Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > vatom;
  KKScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_vatom;



  NeighListKokkos<device_type> list;

  template<class T, std::enable_if_t<!std::is_void<T>::value,bool> = true>
  constexpr static T wrap_tally_style(T* t_ptr) {return *t_ptr;}
  template<class T, std::enable_if_t<std::is_void<T>::value,bool> = true>
  constexpr static T* wrap_tally_style(T* t_ptr) {return nullptr;}

  template<class T, class T_FEV, std::enable_if_t<!std::is_void<T>::value,bool> = true>
  inline static void wrap_ev_tally(const T &t, T_FEV &ev,
      const int &i, const int &j, const int& nlocal, const int&newton_pair,
      const F_FLOAT &evdwl, const F_FLOAT &ecoul, const F_FLOAT &fpair,
      const F_FLOAT &delx, const F_FLOAT &dely, const F_FLOAT &delz)
  {
    t.template ev_tally<NEIGHFLAG>(ev.tally,i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
  }
  template<class T, class T_FEV, std::enable_if_t<std::is_void<T>::value,bool> = true>
  inline static void wrap_ev_tally(T* t_ptr, T_FEV&,
      const int&, const int&, const int&, const int&,
      const F_FLOAT&, const F_FLOAT&, const F_FLOAT&,
      const F_FLOAT&, const F_FLOAT&, const F_FLOAT&) {}

  typename std::conditional<TALLYFLAG, TallyStyle, void*>::type t;

  PairComputeFunctor(PairStyle* c_ptr,
                     NeighListKokkos<device_type>* list_ptr,
                     TallyStyle *t_ptr = nullptr):
  c(*c_ptr),list(*list_ptr),t(wrap_tally_style(t_ptr)) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
    dup_f     = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_vatom);
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctor() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute() {
    Kokkos::Experimental::contribute(c.f, dup_f);

    if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
  }

  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel
  template<int EVFLAG, int NEWTON_PAIR>
  KOKKOS_FUNCTION
  ev_value_type compute_item(const int& ii,
                        const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    ev_value_type ev;
    const int i = list.d_ilist[ii];
    const X_FLOAT xtmp = c.x(i,0);
    const X_FLOAT ytmp = c.x(i,1);
    const X_FLOAT ztmp = c.x(i,2);
    const int itype = c.type(i);

    const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
    const int jnum = list.d_numneigh[i];

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

        const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG || TALLYFLAG) {
          F_FLOAT evdwl = 0.0;
          if (c.eflag) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*evdwl;
          }

          if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl,fpair,delx,dely,delz);
          if (TALLYFLAG) wrap_ev_tally(t,ev,i,j,c.nlocal,NEWTON_PAIR,evdwl,0.0,fpair,delx,dely,delz);
        }
      }

    }

    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;

    return ev;
  }

  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel
  template<int EVFLAG, int NEWTON_PAIR>
  KOKKOS_FUNCTION
  ev_value_type compute_item(const int& ii,
                        const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    ev_value_type ev;
    const int i = list.d_ilist[ii];
    const X_FLOAT xtmp = c.x(i,0);
    const X_FLOAT ytmp = c.x(i,1);
    const X_FLOAT ztmp = c.x(i,2);
    const int itype = c.type(i);
    const F_FLOAT qtmp = c.q(i);

    const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
    const int jnum = list.d_numneigh[i];

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
          fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
          fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag) {
            if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype))) {
              evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*evdwl;
            }
            if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype))) {
              ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              ev.ecoul += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*ecoul;
            }
          }

          if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
          if (TALLYFLAG) wrap_ev_tally(t,ev,i,j,c.nlocal,NEWTON_PAIR,evdwl,ecoul,fpair,delx,dely,delz);
        }
      }
    }

    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;

    return ev;
  }

  // Use TeamPolicy, assume Newton off, Full Neighborlist, and no energy/virial
  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel
  KOKKOS_FUNCTION
  void compute_item_team(typename Kokkos::TeamPolicy<device_type>::member_type team,
                         const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    const int inum = team.league_size();
    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      t_scalar3<double> fsum;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, t_scalar3<double>& ftmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

          ftmp.x += delx*fpair;
          ftmp.y += dely*fpair;
          ftmp.z += delz*fpair;
        }

      },fsum);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        f(i,0) += fsum.x;
        f(i,1) += fsum.y;
        f(i,2) += fsum.z;
      });

    });
  }

  // Use TeamPolicy, assume Newton off, Full Neighborlist, and no energy/virial
  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel
  KOKKOS_FUNCTION
  void compute_item_team(typename Kokkos::TeamPolicy<device_type>::member_type team,
                         const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    const int inum = team.league_size();
    const int atoms_per_team = team.team_size();
    int firstatom = team.league_rank()*atoms_per_team;
    int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);
      const F_FLOAT qtmp = c.q(i);

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      t_scalar3<double> fsum;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, t_scalar3<double>& ftmp) {
        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          F_FLOAT fpair = F_FLOAT();

          if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
            fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
            fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

          ftmp.x += delx*fpair;
          ftmp.y += dely*fpair;
          ftmp.z += delz*fpair;
        }
      },fsum);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
      f(i,0) += fsum.x;
      f(i,1) += fsum.y;
      f(i,2) += fsum.z;
      });
    });
  }


  // Use TeamPolicy, assume Newton off, Full Neighborlist, and energy/virial
  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel
  KOKKOS_FUNCTION
  ev_value_type compute_item_team_ev(typename Kokkos::TeamPolicy<device_type>::member_type team,
                                const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    ev_value_type ev;

    const int inum = team.league_size();
    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      fev_value_type fev;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, fev_value_type& fev_tmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

          fev_tmp.f[0] += delx*fpair;
          fev_tmp.f[1] += dely*fpair;
          fev_tmp.f[2] += delz*fpair;

          F_FLOAT evdwl = 0.0;
          if (c.eflag) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            fev_tmp.evdwl += 0.5*evdwl;
          }
          if (c.vflag_either) {
            fev_tmp.v[0] += 0.5*delx*delx*fpair;
            fev_tmp.v[1] += 0.5*dely*dely*fpair;
            fev_tmp.v[2] += 0.5*delz*delz*fpair;
            fev_tmp.v[3] += 0.5*delx*dely*fpair;
            fev_tmp.v[4] += 0.5*delx*delz*fpair;
            fev_tmp.v[5] += 0.5*dely*delz*fpair;
          }
          if (TALLYFLAG) wrap_ev_tally(t,fev_tmp,i,j,c.nlocal,c.newton_pair,evdwl,0.0,fpair,delx,dely,delz);
        }
      },fev);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        f(i,0) += fev.f[0];
        f(i,1) += fev.f[1];
        f(i,2) += fev.f[2];

        if (c.eflag_global)
          ev.evdwl += fev.evdwl;

        if (c.eflag_atom)
          d_eatom(i) += fev.evdwl;

        if (c.vflag_global) {
          ev.v[0] += fev.v[0];
          ev.v[1] += fev.v[1];
          ev.v[2] += fev.v[2];
          ev.v[3] += fev.v[3];
          ev.v[4] += fev.v[4];
          ev.v[5] += fev.v[5];
        }

        if (c.vflag_atom) {
          d_vatom(i,0) += fev.v[0];
          d_vatom(i,1) += fev.v[1];
          d_vatom(i,2) += fev.v[2];
          d_vatom(i,3) += fev.v[3];
          d_vatom(i,4) += fev.v[4];
          d_vatom(i,5) += fev.v[5];
        }
        fev_wrapper<TallyStyle>::tally_add(ev,fev);
      });
    });
    return ev;
  }

  // Use TeamPolicy, assume Newton off, Full Neighborlist, and energy/virial
  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel
  KOKKOS_FUNCTION
  ev_value_type compute_item_team_ev(typename Kokkos::TeamPolicy<device_type>::member_type team,
                                const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    ev_value_type ev;

    const int inum = team.league_size();
    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);
      const F_FLOAT qtmp = c.q(i);

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      fev_value_type fev;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, fev_value_type& fev_tmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          F_FLOAT fpair = F_FLOAT();

          if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
            fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
            fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

          fev_tmp.f[0] += delx*fpair;
          fev_tmp.f[1] += dely*fpair;
          fev_tmp.f[2] += delz*fpair;

          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag) {
            if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype))) {
              evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              fev_tmp.evdwl += 0.5*evdwl;
            }
            if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype))) {
              ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              fev_tmp.ecoul += 0.5*ecoul;
            }
          }
          if (c.vflag_either) {
            fev_tmp.v[0] += 0.5*delx*delx*fpair;
            fev_tmp.v[1] += 0.5*dely*dely*fpair;
            fev_tmp.v[2] += 0.5*delz*delz*fpair;
            fev_tmp.v[3] += 0.5*delx*dely*fpair;
            fev_tmp.v[4] += 0.5*delx*delz*fpair;
            fev_tmp.v[5] += 0.5*dely*delz*fpair;
          }
          if (TALLYFLAG) wrap_ev_tally(t,fev_tmp,i,j,c.nlocal,c.newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
        }
      },fev);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        f(i,0) += fev.f[0];
        f(i,1) += fev.f[1];
        f(i,2) += fev.f[2];

        if (c.eflag_global) {
          ev.evdwl += fev.evdwl;
          ev.ecoul += fev.ecoul;
        }

        if (c.eflag_atom)
          d_eatom(i) += fev.evdwl + fev.ecoul;

        if (c.vflag_global) {
          ev.v[0] += fev.v[0];
          ev.v[1] += fev.v[1];
          ev.v[2] += fev.v[2];
          ev.v[3] += fev.v[3];
          ev.v[4] += fev.v[4];
          ev.v[5] += fev.v[5];
        }

        if (c.vflag_atom) {
          d_vatom(i,0) += fev.v[0];
          d_vatom(i,1) += fev.v[1];
          d_vatom(i,2) += fev.v[2];
          d_vatom(i,3) += fev.v[3];
          d_vatom(i,4) += fev.v[4];
          d_vatom(i,5) += fev.v[5];
        }
        fev_wrapper<TallyStyle>::tally_add(ev,fev);
      });
    });
    return ev;
  }

  KOKKOS_INLINE_FUNCTION
    void ev_tally(ev_value_type &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int EFLAG = c.eflag;
    const int NEWTON_PAIR = c.newton_pair;
    const int VFLAG = c.vflag_either;

    if (EFLAG) {
      if (c.eflag_atom) {
        const E_FLOAT epairhalf = 0.5 * epair;
        if (NEWTON_PAIR || i < c.nlocal) a_eatom[i] += epairhalf;
        if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) a_eatom[j] += epairhalf;
      }
    }

    if (VFLAG) {
      const E_FLOAT v0 = delx*delx*fpair;
      const E_FLOAT v1 = dely*dely*fpair;
      const E_FLOAT v2 = delz*delz*fpair;
      const E_FLOAT v3 = delx*dely*fpair;
      const E_FLOAT v4 = delx*delz*fpair;
      const E_FLOAT v5 = dely*delz*fpair;

      if (c.vflag_global) {
        if (NEIGHFLAG!=FULL) {
          if (NEWTON_PAIR) {
            ev.v[0] += v0;
            ev.v[1] += v1;
            ev.v[2] += v2;
            ev.v[3] += v3;
            ev.v[4] += v4;
            ev.v[5] += v5;
          } else {
            if (i < c.nlocal) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }
            if (j < c.nlocal) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }
          }
        } else {
          ev.v[0] += 0.5*v0;
          ev.v[1] += 0.5*v1;
          ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3;
          ev.v[4] += 0.5*v4;
          ev.v[5] += 0.5*v5;
        }
      }

      if (c.vflag_atom) {
        if (NEWTON_PAIR || i < c.nlocal) {
          a_vatom(i,0) += 0.5*v0;
          a_vatom(i,1) += 0.5*v1;
          a_vatom(i,2) += 0.5*v2;
          a_vatom(i,3) += 0.5*v3;
          a_vatom(i,4) += 0.5*v4;
          a_vatom(i,5) += 0.5*v5;
        }
        if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) {
          a_vatom(j,0) += 0.5*v0;
          a_vatom(j,1) += 0.5*v1;
          a_vatom(j,2) += 0.5*v2;
          a_vatom(j,3) += 0.5*v3;
          a_vatom(j,4) += 0.5*v4;
          a_vatom(j,5) += 0.5*v5;
        }
      }
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    if (c.newton_pair) compute_item<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    else compute_item<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    if (c.newton_pair)
      energy_virial += compute_item<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    else
      energy_virial += compute_item<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<device_type>::member_type& team) const {
    compute_item_team(team,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<device_type>::member_type& team, value_type &energy_virial) const {
    energy_virial += compute_item_team_ev(team,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }
};


// Filter out Neighflags which are not supported for PairStyle
// The enable_if clause will invalidate the last parameter of the function, so that
// a match is only achieved, if PairStyle supports the specific neighborlist variant.
// This uses the fact that failure to match template parameters is not an error.
// By having the enable_if with a ! and without it, exactly one of the functions
// pair_compute_neighlist will match - either the dummy version
// or the real one further below.
template<class PairStyle, unsigned NEIGHFLAG, class Specialisation>
EV_FLOAT pair_compute_neighlist (PairStyle* fpair, typename std::enable_if<!((NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0), NeighListKokkos<typename PairStyle::device_type>*>::type list) {
  EV_FLOAT ev;
  (void) fpair;
  (void) list;
  printf("ERROR: calling pair_compute with invalid neighbor list style: requested %i  available %i \n",NEIGHFLAG,PairStyle::EnabledNeighFlags);
  return ev;
}

template<class DeviceType, class FunctorStyle>
int GetTeamSize(FunctorStyle& KOKKOS_GPU_ARG(functor), int KOKKOS_GPU_ARG(inum),
                int KOKKOS_GPU_ARG(reduce_flag), int team_size, int KOKKOS_GPU_ARG(vector_length)) {

#ifdef LMP_KOKKOS_GPU
    int team_size_max;

    if (reduce_flag)
      team_size_max = Kokkos::TeamPolicy<DeviceType>(inum,Kokkos::AUTO).team_size_max(functor,Kokkos::ParallelReduceTag());
    else
      team_size_max = Kokkos::TeamPolicy<DeviceType>(inum,Kokkos::AUTO).team_size_max(functor,Kokkos::ParallelForTag());

    if (team_size*vector_length > team_size_max)
      team_size = team_size_max/vector_length;
#else
    team_size = 1;
#endif
    return team_size;
}

// Submit ParallelFor for NEIGHFLAG=HALF,HALFTHREAD,FULL
template<class PairStyle, unsigned NEIGHFLAG, class Specialisation>
EV_FLOAT pair_compute_neighlist (PairStyle* fpair, typename std::enable_if<(NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0, NeighListKokkos<typename PairStyle::device_type>*>::type list) {

  if (!fpair->lmp->kokkos->neigh_thread_set)
    if (list->inum <= 16384 && NEIGHFLAG == FULL)
      fpair->lmp->kokkos->neigh_thread = 1;

  if (fpair->atom->ntypes > MAX_TYPES_STACKPARAMS)
    return PairStyle::template pair_compute_functor<NEIGHFLAG,false,Specialisation>
      ::functor_type::pair_compute_neighlist(fpair,list);
  return PairStyle::template pair_compute_functor<NEIGHFLAG,true,Specialisation>
    ::functor_type::pair_compute_neighlist(fpair,list);
}

template<class PairStyle, class Specialisation>
EV_FLOAT pair_compute (PairStyle* fpair, NeighListKokkos<typename PairStyle::device_type>* list) {
  EV_FLOAT ev;
  if (fpair->neighflag == FULL) {
    ev = pair_compute_neighlist<PairStyle,FULL,Specialisation> (fpair,list);
  } else if (fpair->neighflag == HALFTHREAD) {
    ev = pair_compute_neighlist<PairStyle,HALFTHREAD,Specialisation> (fpair,list);
  } else if (fpair->neighflag == HALF) {
    ev = pair_compute_neighlist<PairStyle,HALF,Specialisation> (fpair,list);
  }
  return ev;
}

template<class DeviceType>
struct PairVirialFDotRCompute {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  typename AT::t_x_array_const_um x;
  typename AT::t_f_array_const_um f;
  const int offset;

  PairVirialFDotRCompute(  typename AT::t_x_array_const_um x_,
  typename AT::t_f_array_const_um f_,
  const int offset_):x(x_),f(f_),offset(offset_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j, value_type &energy_virial) const {
    const int i = j + offset;
    energy_virial.v[0] += f(i,0)*x(i,0);
    energy_virial.v[1] += f(i,1)*x(i,1);
    energy_virial.v[2] += f(i,2)*x(i,2);
    energy_virial.v[3] += f(i,1)*x(i,0);
    energy_virial.v[4] += f(i,2)*x(i,0);
    energy_virial.v[5] += f(i,2)*x(i,1);
  }
};

template<class PairStyle>
void pair_virial_fdotr_compute(PairStyle* fpair) {
  EV_FLOAT virial;
  if (fpair->neighbor->includegroup == 0) {
    int nall = fpair->atom->nlocal + fpair->atom->nghost;
    Kokkos::parallel_reduce(nall,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,0),virial);
  } else {
    Kokkos::parallel_reduce(fpair->atom->nfirst,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,0),virial);
    EV_FLOAT virial_ghost;
    Kokkos::parallel_reduce(fpair->atom->nghost,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,fpair->atom->nlocal),virial_ghost);
    virial+=virial_ghost;
  }
  fpair->vflag_fdotr = 0;
  fpair->virial[0] = virial.v[0];
  fpair->virial[1] = virial.v[1];
  fpair->virial[2] = virial.v[2];
  fpair->virial[3] = virial.v[3];
  fpair->virial[4] = virial.v[4];
  fpair->virial[5] = virial.v[5];
}


// Pair styles should mark DeclPairComputeFunctor as a typedef for pair_compute_functor
// Do this with a macro (defined later) to also handle other required friend definitions
template<class PairStyle, int NEIGHFLAG, bool STACKPARAMS, class Specialisation>
struct DeclPairComputeFunctorImpl;


// Wrap functions inside struct to make friend declaration possible in pair styles
struct kokkos_tally {
  // Called when at least one tally compute this timestep
  template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation, class TallyCombo>
  inline static EV_FLOAT pair_compute_neighlist_tally(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    typename TallyCombo::tally_computes::ev_value_type ev;
    typename TallyCombo::tally_computes tally_compute_pack{};
    TALLY_MASK masks[fpair->num_tally_compute];
    for (int i = 0; i < fpair->num_tally_compute; ++i) {
      // Call pair_setup_callback first so compute can decide if it's needed this step.
      // Can return 0 as mask to skip the step.
      fpair->list_tally_compute[i]->pair_setup_callback(fpair->eflag, fpair->vflag);
      masks[i] = fpair->list_tally_compute[i]->tally_mask();
      if (masks[i]) tally_compute_pack.init_step(fpair->list_tally_compute[i], masks[i]);
    }
    typename TallyCombo::pair_compute_functor ff{fpair,list,&tally_compute_pack};
    if (fpair->lmp->kokkos->neigh_thread) {
      int vector_length = 8;
      int atoms_per_team = GetTeamSize<typename PairStyle::device_type>(ff, list->inum, (fpair->eflag || fpair->vflag), atoms_per_team, vector_length);
      Kokkos::TeamPolicy<typename PairStyle::device_type,Kokkos::IndexType<int> > policy(list->inum,atoms_per_team,vector_length);
      Kokkos::parallel_reduce(policy,ff,ev);
    } else {
      Kokkos::parallel_reduce(list->inum,ff,ev);
      ff.contribute();
    }
    for (int i = 0; i < fpair->num_tally_compute; ++i)
      if (masks[i]) tally_compute_pack.consolidate(fpair->list_tally_compute[i], masks[i], ev);
    return EV_FLOAT(std::move(ev));
  }

  // Called when no tally computes this timestep
  template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation>
  inline static EV_FLOAT pair_compute_neighlist(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    EV_FLOAT ev;
    PairComputeFunctor<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation> ff{fpair,list};
    if (fpair->lmp->kokkos->neigh_thread) {
      int vector_length = 8;
      int atoms_per_team = GetTeamSize<typename PairStyle::device_type>(ff, list->inum, (fpair->eflag || fpair->vflag), atoms_per_team, vector_length);
      Kokkos::TeamPolicy<typename PairStyle::device_type,Kokkos::IndexType<int> > policy(list->inum,atoms_per_team,vector_length);
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(policy,ff,ev);
      else                              Kokkos::parallel_for(policy,ff);
    } else {
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(list->inum,ff,ev);
      else                              Kokkos::parallel_for(list->inum,ff);
      ff.contribute();
    }
    return ev;
  }
};

// Recursive lookup table for all combinations of tally computes.
// Limitation of 1 instance at runtime per compute type on a given timestep.
// Maybe possible to relax this for specific computes with a clever associated tally functor
template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation, class T, class U, class ... Ts>
struct TallyCombinator {
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,void> OnlyT;
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,U> TU;
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,Ts...> TTs;
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,U,Ts...> UTs;

  typedef TallyFunctor<typename PairStyle::device_type,T,U,Ts...> tally_computes;
  typedef struct PairComputeFunctor<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,tally_computes> pair_compute_functor;
  static constexpr TALLY_MASK bitmask = s_TALLY_MASK<typename PairStyle::device_type>::template get_tally_mask<T,U,Ts...>();

  inline static EV_FLOAT pair_compute_neighlist(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    // Build mask for tally computes this timestep
    TALLY_MASK mask = 0;
    for (int i = 0; i < fpair->num_tally_compute; ++i)
      mask |= fpair->list_tally_compute[i]->tally_mask();
    // If none, call the non-tally pair functor, otherwise look up the correct tally pair functor
    if (mask) return pair_compute_neighlist_impl(fpair,list,mask);
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }

  // Never called with mask == 0
  inline static EV_FLOAT pair_compute_neighlist_impl(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list,
      const TALLY_MASK &mask)
  {
    // If I match the mask, call my pair_compute_neighlist implementation.
    if (mask == bitmask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TallyCombinator>(fpair,list);
    if ((mask & OnlyT::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,OnlyT>(fpair,list);
    if ((mask & TU::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TU>(fpair,list);
    if ((mask & TTs::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TTs>(fpair,list);
    if ((mask & UTs::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,UTs>(fpair,list);

    // If no exact match, find a combinator that does match and look up pair functor through it.
    if (mask & TTs::bitmask) return TTs::pair_compute_neighlist_impl(fpair,list,mask);
    if (mask & UTs::bitmask) return UTs::pair_compute_neighlist_impl(fpair,list,mask);

    // Fall back to non-tally functor if no match can be found (should never happen)
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }
};

template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation, class T, class U>
struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,U> {
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,void> OnlyT;
  typedef struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,U> TU;

  typedef TallyFunctor<typename PairStyle::device_type,T,U> tally_computes;
  typedef struct PairComputeFunctor<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,tally_computes> pair_compute_functor;
  static constexpr TALLY_MASK bitmask = s_TALLY_MASK<typename PairStyle::device_type>::template get_tally_mask<T,U>();

  inline static EV_FLOAT pair_compute_neighlist(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    TALLY_MASK mask = 0;
    for (int i = 0; i < fpair->num_tally_compute; ++i)
      mask |= fpair->list_tally_compute[i]->tally_mask();
    if (mask) return pair_compute_neighlist_impl(fpair,list,mask);
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }

  // Never called with mask == 0
  inline static EV_FLOAT pair_compute_neighlist_impl(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list,
      const TALLY_MASK &mask)
  {
    // Get exact match or fall back to no tally
    if (mask == bitmask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TallyCombinator>(fpair,list);
    if ((mask & OnlyT::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,OnlyT>(fpair,list);
    if ((mask & TU::bitmask) == mask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TU>(fpair,list);
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }
};

// Specialise for voids to avoid compile error.
// Can maybe remove if 'void,void>' tail can be avoided.
template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation, class T>
struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,T,void> {

  typedef TallyFunctor<typename PairStyle::device_type,T,void> tally_computes;
  typedef struct PairComputeFunctor<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,tally_computes> pair_compute_functor;
  static constexpr TALLY_MASK bitmask = s_TALLY_MASK<typename PairStyle::device_type>::template get_tally_mask<T>();

  inline static EV_FLOAT pair_compute_neighlist(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    TALLY_MASK mask = 0;
    for (int i = 0; i < fpair->num_tally_compute; ++i)
      mask |= fpair->list_tally_compute[i]->tally_mask();
    if (mask == bitmask) return pair_compute_neighlist_impl(fpair,list,mask);
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }

  inline static EV_FLOAT pair_compute_neighlist_impl(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list,
      const TALLY_MASK &mask)
  {
    // If I match the mask, call my pair_compute_neighlist implementation.
    // If not, something went wrong so just call the non-tally functor
    if (mask == bitmask) return kokkos_tally::pair_compute_neighlist_tally<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,TallyCombinator>(fpair,list);
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }
};

template<class PairStyle,int NEIGHFLAG,bool STACKPARAMS,class Specialisation>
struct TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,void,void> {
  typedef TallyFunctor<typename PairStyle::device_type,void,void> tally_computes;
  typedef struct PairComputeFunctor<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,tally_computes> pair_compute_functor;
  static constexpr TALLY_MASK bitmask = 0;

  inline static EV_FLOAT pair_compute_neighlist(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list)
  {
    return pair_compute_neighlist_impl(fpair,list,bitmask);
  }

  inline static EV_FLOAT pair_compute_neighlist_impl(
      PairStyle* fpair,
      NeighListKokkos<typename PairStyle::device_type>* list,
      const TALLY_MASK &mask)
  {
    return kokkos_tally::pair_compute_neighlist<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation>(fpair,list);
  }
};

// Create recursive TallyCombinator type with all possible PairComputeFunctor types.
// This acts as a lookup table by storing (at compile time) the bit mask
// associated with each combination of tally computes.
template<class PairStyle, int NEIGHFLAG, bool STACKPARAMS, class Specialisation=void>
struct DeclPairComputeFunctor {
  typedef typename PairStyle::device_type DeviceType;
  typedef TallyCombinator<PairStyle,NEIGHFLAG,STACKPARAMS,Specialisation,
#define KOKKOS_TALLY_CLASS
#define TallyStyle(Class) Class,
#include "style_tally_kokkos.h"
#undef KOKKOS_TALLY_CLASS
#undef TallyStyle
  void,void> functor_type;

  // Finish with two voids to make sure template is always satisfied even if no tally computes exist.
  // Could maybe reduce duplication elsewhere if this requirement can be removed.
};

}

#endif
#endif

