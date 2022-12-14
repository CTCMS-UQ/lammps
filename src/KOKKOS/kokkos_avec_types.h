#ifndef KOKKOS_AVEC_TYPES_H
#define KOKKOS_AVEC_TYPES_H

#include "kokkos.h"
#include "atom_masks.h"
#include "domain.h"

namespace LAMMPS_NS {

/* --------------------------------------------------------------------------
   Macro to generate maybe type for comm of each atom vec type.
   Empty for unneeded atom vecs based on MASK.
   Store ref to host view instead of copy to avoid reference counting overhead.
 --------------------------------------------------------------------------- */
#define GenerateAvecMaybe(view_name,MY_MASK,extra_cond)                       \
template<class Avec, unsigned long MASK, class T,                             \
  unsigned int DIM2=0, bool _enabled = ((MASK & MY_MASK) > 0) && extra_cond>  \
struct Maybe_ ## view_name;                                                   \
                                                                              \
template<class Avec, unsigned long MASK, class T>                             \
struct Maybe_ ## view_name<Avec,MASK,T,0,true> {                              \
  static_assert((MASK&MY_MASK)>0 && extra_cond, "Maybe_" #view_name           \
      " should not explicitly specify the _enabled parameter");               \
  typedef typename T::value_type value_type;                                  \
  T& _view;                                                                   \
  Maybe_ ## view_name(Avec* avec) : _view(avec->view_name) {}                 \
  inline void pack(double *buf, int &m, const int j) {                        \
    buf[m++] = ubuf(_view(j)).d;                                              \
  }                                                                           \
                                                                              \
  inline void unpack(const int i, const double *buf, int &m) {                \
    if (std::is_integral<value_type>::value)                                  \
      _view(i) = (value_type) ubuf(buf[m++]).i;                               \
    else _view(i) = buf[m++];                                                 \
  }                                                                           \
                                                                              \
};                                                                            \
                                                                              \
template<class Avec, unsigned long MASK, class T, unsigned int DIM2>          \
struct Maybe_ ## view_name<Avec,MASK,T,DIM2,true> {                           \
  static_assert((MASK&MY_MASK)>0 && (DIM2>0) && extra_cond, "Maybe_"          \
    #view_name " should not explicitly specify the _enabled parameter");      \
  typedef typename T::value_type value_type;                                  \
  T& _view;                                                                   \
  Maybe_ ## view_name(Avec* avec) : _view(avec->view_name) {}                 \
  inline void pack(double *buf, int &m, const int j) {                        \
    for (unsigned int d = 0; d < DIM2; ++d)                                   \
      buf[m++] = ubuf(_view(j,d)).d;                                          \
  }                                                                           \
                                                                              \
  inline void unpack(const int i, const double *buf, int &m) {                \
    if (std::is_integral<value_type>::value) {                                \
      for (unsigned int d = 0; d < DIM2; ++d)                                 \
        _view(i,d) = (value_type) ubuf(buf[m++]).i;                           \
    } else {                                                                  \
      for (unsigned int d = 0; d < DIM2; ++d)                                 \
        _view(i,d) = buf[m++];                                                \
    }                                                                         \
  }                                                                           \
};                                                                            \
                                                                              \
template<class Avec, unsigned long MASK, class T, unsigned int DIM2>          \
struct Maybe_ ## view_name<Avec,MASK,T,DIM2,false> {                          \
  static_assert(!((MASK&MY_MASK)>0 && extra_cond), "Maybe_" #view_name        \
      " should not explicitly specify the _enabled parameter");               \
  Maybe_ ## view_name(Avec*) {}                                               \
  inline void pack(double *buf, int &m, const int j) {}                       \
  inline void unpack(const int i, const double *buf, int &m) {}               \
};

/* --------------------------------------------------------------------------
   Maybe types wrapped in struct for simple friend declaration in KOKKOS
   atom vec styles
 -------------------------------------------------------------------------- */
struct AvecKokkosTypes {
/* --------------------------------------------------------------------------
   Generate maybe types. Only handling comm and border for now.
   Some masks alias, so use atom vec class type as extra filter.
 -------------------------------------------------------------------------- */
  GenerateAvecMaybe(h_tag,      TAG_MASK,     true);  // full, atomic, charge, bond, spin, angle, sphere
  GenerateAvecMaybe(h_type,     TYPE_MASK,    true);  // full, atomic, charge, bond, spin, angle, sphere
  GenerateAvecMaybe(h_mask,     MASK_MASK,    true);  // full, atomic, charge, bond, spin, angle, sphere
  GenerateAvecMaybe(h_q,        Q_MASK,       true);  // full, atomic, charge, bond
  GenerateAvecMaybe(h_molecule, MOLECULE_MASK,true);  // full, bond, angle
  GenerateAvecMaybe(h_sp,       SP_MASK,      (std::is_same<Avec,class AtomVecSpinKokkos>::value));   // spin
  GenerateAvecMaybe(h_radius,   RADIUS_MASK,  (std::is_same<Avec,class AtomVecSphereKokkos>::value)); // sphere
  GenerateAvecMaybe(h_rmass,    RMASS_MASK,   (std::is_same<Avec,class AtomVecSphereKokkos>::value)); // sphere
  GenerateAvecMaybe(h_dpdTheta, DPDTHETA_MASK,(std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_uCond,    UCOND_MASK,   (std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_uMech,    UMECH_MASK,   (std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_uChem,    UCHEM_MASK,   (std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_uCG,      UCG_MASK,     (std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_uCGnew,   UCGNEW_MASK,  (std::is_same<Avec,class AtomVecDPDKokkos>::value));    // dpd
  GenerateAvecMaybe(h_omega,    OMEGA_MASK,   (std::is_same<Avec,class AtomVecHybridKokkos>::value)); // hybrid
  GenerateAvecMaybe(h_angmom,   ANGMOM_MASK,  (std::is_same<Avec,class AtomVecHybridKokkos>::value)); // hybrid

/* --------------------------------------------------------------------------
   Specialise for x
 -------------------------------------------------------------------------- */
  template<unsigned long MASK, bool _enabled = ((MASK & X_MASK) > 0)>
  struct Maybe_h_x;

  template<unsigned long MASK>
  struct Maybe_h_x<MASK,true> {
    ArrayTypes<LMPHostType>::t_x_array& _view;
    template<class Avec>
    Maybe_h_x(Avec* avec) : _view(avec->h_x) {}

    inline void pack(double *buf, int &m, const int j) {
      buf[m++] = _view(j,0);
      buf[m++] = _view(j,1);
      buf[m++] = _view(j,2);
    }

    inline void pack(double *buf, int &m, const int j,
        const double &dx, const double &dy, const double &dz)
    {
      buf[m++] = _view(j,0) + dx;
      buf[m++] = _view(j,1) + dy;
      buf[m++] = _view(j,2) + dz;
    }

    inline void unpack(const int i, const double *buf, int &m) {
      _view(i,0) = buf[m++];
      _view(i,1) = buf[m++];
      _view(i,2) = buf[m++];
    }
  };

  template<unsigned long MASK>
  struct Maybe_h_x<MASK,false> {
    template<class Avec> Maybe_h_x(Avec*) {}
    inline void pack(double *buf, int &m, const int j) {}
    inline void pack(double *buf, int &m, const int j,
        const double&, const double&, const double&) {}
    inline void unpack(const int i, const double *buf, int &m) {}
  };

/* --------------------------------------------------------------------------
   Specialise for v
 -------------------------------------------------------------------------- */
  template<unsigned long MASK, bool _enabled = ((MASK & V_MASK) > 0)>
  struct Maybe_h_v;

  template<unsigned long MASK>
  struct Maybe_h_v<MASK,true> {
    ArrayTypes<LMPHostType>::t_v_array& _view;
    template<class Avec>
    Maybe_h_v(Avec* avec) : _view(avec->h_v) {}

    inline void pack(double *buf, int &m, const int j) {
      buf[m++] = _view(j,0);
      buf[m++] = _view(j,1);
      buf[m++] = _view(j,2);
    }

    inline void pack(double *buf, int &m, const int j,
        const double &dvx, const double &dvy, const double &dvz)
    {
      buf[m++] = _view(j,0) + dvx;
      buf[m++] = _view(j,1) + dvy;
      buf[m++] = _view(j,2) + dvz;
    }

    inline void unpack(const int i, const double *buf, int &m) {
      _view(i,0) = buf[m++];
      _view(i,1) = buf[m++];
      _view(i,2) = buf[m++];
    }
  };

  template<unsigned long MASK>
  struct Maybe_h_v<MASK,false> {
    template<class Avec>
    Maybe_h_v(Avec*) {}
    inline void pack(double *buf, int &m, const int j) {}
    inline void pack(double *buf, int &m, const int j,
        const double&, const double&, const double&) {}
    inline void unpack(const int i, const double *buf, int &m) {}
  };

/* --------------------------------------------------------------------------
   Specialise for image
 -------------------------------------------------------------------------- */
  template<unsigned long MASK, bool _enabled = ((MASK & IMAGE_MASK) > 0)>
  struct Maybe_h_image;

  template<unsigned long MASK>
  struct Maybe_h_image<MASK,true> {
    ArrayTypes<LMPHostType>::t_imageint_1d& _view;
    template<class Avec>
    Maybe_h_image(Avec* avec) : _view(avec->h_image) {}

    inline void pack(double *buf, int &m, const int j) {
      buf[m++] = ubuf(_view(j)).d;
    }

    inline void pack(double *buf, int &m, const int j,
        const int &dix, const int &diy, const int &diz)
    {
      imageint xi = (_view(j) & IMGMASK) - dix;
      imageint yi = ((_view(j) >> IMGBITS) & IMGMASK) - diy;
      imageint zi = (_view(j) >> IMG2BITS) - diz;
      imageint img = (xi & IMGMASK) |
        ((yi & IMGMASK) << IMGBITS) |
        ((zi & IMGMASK) << IMG2BITS);
      buf[m++] = ubuf(img).d;
    }

    inline void unpack(const int i, const double *buf, int &m) {
      _view(i) = (imageint) ubuf(buf[m++]).i;
    }
  };

  template<unsigned long MASK>
  struct Maybe_h_image<MASK,false> {
    template<class Avec>
    Maybe_h_image(Avec*) {}
    inline void pack(double *buf, int &m, const int j) {}
    inline void pack(double *buf, int &m, const int j,
        const int&, const int&, const int&) {}
    inline void unpack(const int i, const double *buf, int &m) {}
  };

/* --------------------------------------------------------------------------
   Handle classic comm/border with atom vecs included/excluded as needed
   based on MASK.
 -------------------------------------------------------------------------- */
  template<unsigned long MASK, class Avec>
  struct AtomVecKokkos_Classic {
    typedef ArrayTypes<LMPHostType> HAT;
    Maybe_h_x<MASK> _x;
    Maybe_h_v<MASK> _v;
    Maybe_h_tag<Avec,MASK,HAT::t_tagint_1d> _tag;
    Maybe_h_type<Avec,MASK,HAT::t_int_1d> _type;
    Maybe_h_mask<Avec,MASK,HAT::t_int_1d> _mask;
    Maybe_h_image<MASK> _image;
    Maybe_h_q<Avec,MASK,HAT::t_float_1d> _q;
    Maybe_h_molecule<Avec,MASK,HAT::t_tagint_1d> _molecule;
    Maybe_h_sp<Avec,MASK,HAT::t_sp_array,4> _sp;
    Maybe_h_radius<Avec,MASK,HAT::t_float_1d> _radius;
    Maybe_h_rmass<Avec,MASK,HAT::t_float_1d> _rmass;
    Maybe_h_dpdTheta<Avec,MASK,HAT::t_efloat_1d> _dpdTheta;
    Maybe_h_uCond<Avec,MASK,HAT::t_efloat_1d> _uCond;
    Maybe_h_uMech<Avec,MASK,HAT::t_efloat_1d> _uMech;
    Maybe_h_uChem<Avec,MASK,HAT::t_efloat_1d> _uChem;
    Maybe_h_uCG<Avec,MASK,HAT::t_efloat_1d> _uCG;
    Maybe_h_uCGnew<Avec,MASK,HAT::t_efloat_1d> _uCGnew;
    Maybe_h_omega<Avec,MASK,HAT::t_v_array,3> _omega;
    Maybe_h_omega<Avec,MASK,HAT::t_v_array,3> _angmom;

    Avec* _avec;

    AtomVecKokkos_Classic(Avec* avec)
      : _x(avec), _v(avec), _tag(avec), _type(avec), _mask(avec),
      _image(avec), _q(avec), _molecule(avec), _sp(avec), _radius(avec),
      _rmass(avec), _dpdTheta(avec), _uCond(avec), _uMech(avec), _uChem(avec),
      _uCG(avec), _uCGnew(avec), _omega(avec), _angmom(avec), _avec(avec)
    {}

    inline void pack(double *buf, int &m, const int *list, const int n) {
      for (int i = 0; i < n; i++) {
        int j = list[i];
        _x.pack(buf,m,j);
        _v.pack(buf,m,j);
        _tag.pack(buf,m,j);
        _type.pack(buf,m,j);
        _mask.pack(buf,m,j);
        _image.pack(buf,m,j);
        _q.pack(buf,m,j);
        _molecule.pack(buf,m,j);
        _sp.pack(buf,m,j);
        _radius.pack(buf,m,j);
        _rmass.pack(buf,m,j);
        _dpdTheta.pack(buf,m,j);
        _uCond.pack(buf,m,j);
        _uMech.pack(buf,m,j);
        _uChem.pack(buf,m,j);
        _uCG.pack(buf,m,j);
        _uCGnew.pack(buf,m,j);
        _omega.pack(buf,m,j);
        _angmom.pack(buf,m,j);
      }
    }

    template<bool BORDERFLAG>
    inline void pack_pbc(double *buf, int &m, const int *list,
        const int n, const int* pbc)
    {
      double dx, dy, dz;
      if (_avec->domain->triclinic == 0) {
        dx = pbc[0]*_avec->domain->xprd;
        dy = pbc[1]*_avec->domain->yprd;
        dz = pbc[2]*_avec->domain->zprd;
      } else if (BORDERFLAG) {
        // pack_border
        dx = pbc[0];
        dy = pbc[1];
        dz = pbc[2];
      } else {
        // pack_comm
        dx = pbc[0]*_avec->domain->xprd + pbc[5]*_avec->domain->xy + pbc[4]*_avec->domain->xz;
        dy = pbc[1]*_avec->domain->yprd + pbc[3]*_avec->domain->yz;
        dz = pbc[2]*_avec->domain->zprd;
      }
      for (int i = 0; i < n; i++) {
        int j = list[i];
        _x.pack(buf,m,j,dx,dy,dz);
        _v.pack(buf,m,j);
        _tag.pack(buf,m,j);
        _type.pack(buf,m,j);
        _mask.pack(buf,m,j);
        _image.pack(buf,m,j,pbc[0],pbc[1],pbc[2]);
        _q.pack(buf,m,j);
        _molecule.pack(buf,m,j);
        _sp.pack(buf,m,j);
        _radius.pack(buf,m,j);
        _rmass.pack(buf,m,j);
        _dpdTheta.pack(buf,m,j);
        _uCond.pack(buf,m,j);
        _uMech.pack(buf,m,j);
        _uChem.pack(buf,m,j);
        _uCG.pack(buf,m,j);
        _uCGnew.pack(buf,m,j);
        _omega.pack(buf,m,j);
        _angmom.pack(buf,m,j);
      }
    }

    template<bool BORDERFLAG>
    inline void pack_deform(double *buf, int &m, const int *list,
        const int n, const int* pbc)
    {
      const int deform_groupbit = _avec->deform_groupbit;
      const int* mask = _avec->mask;
      double dx, dy, dz, dvx, dvy, dvz;
      if (_avec->domain->triclinic == 0) {
        dx = pbc[0]*_avec->domain->xprd;
        dy = pbc[1]*_avec->domain->yprd;
        dz = pbc[2]*_avec->domain->zprd;
      } else if (BORDERFLAG) {
        // pack_border
        dx = pbc[0];
        dy = pbc[1];
        dz = pbc[2];
      } else {
        // pack_comm
        dx = pbc[0]*_avec->domain->xprd + pbc[5]*_avec->domain->xy + pbc[4]*_avec->domain->xz;
        dy = pbc[1]*_avec->domain->yprd + pbc[3]*_avec->domain->yz;
        dz = pbc[2]*_avec->domain->zprd;
      }
      dvx = pbc[0]*_avec->h_rate[0] + pbc[5]*_avec->h_rate[5] + pbc[4]*_avec->h_rate[4];
      dvy = pbc[1]*_avec->h_rate[1] + pbc[3]*_avec->h_rate[3];
      dvz = pbc[2]*_avec->h_rate[2];
      for (int i = 0; i < n; i++) {
        int j = list[i];
        _x.pack(buf,m,j,dx,dy,dz);
        if (mask[i] & deform_groupbit) _v.pack(buf,m,j,dvx,dvy,dvz);
        else _v.pack(buf,m,j);
        _tag.pack(buf,m,j);
        _type.pack(buf,m,j);
        _mask.pack(buf,m,j);
        _image.pack(buf,m,j,pbc[0],pbc[1],pbc[2]);
        _q.pack(buf,m,j);
        _molecule.pack(buf,m,j);
        _sp.pack(buf,m,j);
        _radius.pack(buf,m,j);
        _rmass.pack(buf,m,j);
        _dpdTheta.pack(buf,m,j);
        _uCond.pack(buf,m,j);
        _uMech.pack(buf,m,j);
        _uChem.pack(buf,m,j);
        _uCG.pack(buf,m,j);
        _uCGnew.pack(buf,m,j);
        _omega.pack(buf,m,j);
        _angmom.pack(buf,m,j);
      }
    }

    inline void unpack(const int first, const int last, const double *buf, int &m) {
      for (int i = first; i < last; i++) {
        _x.unpack(i,buf,m);
        _v.unpack(i,buf,m);
        _tag.unpack(i,buf,m);
        _type.unpack(i,buf,m);
        _mask.unpack(i,buf,m);
        _image.unpack(i,buf,m);
        _q.unpack(i,buf,m);
        _molecule.unpack(i,buf,m);
        _sp.unpack(i,buf,m);
        _radius.unpack(i,buf,m);
        _rmass.unpack(i,buf,m);
        _dpdTheta.unpack(i,buf,m);
        _uCond.unpack(i,buf,m);
        _uMech.unpack(i,buf,m);
        _uChem.unpack(i,buf,m);
        _uCG.unpack(i,buf,m);
        _uCGnew.unpack(i,buf,m);
        _omega.unpack(i,buf,m);
        _angmom.unpack(i,buf,m);
      }
    }
  };


  template<unsigned long MASK, class Avec, bool BORDERFLAG>
  static inline void AvecKokkos_pack_impl(Avec* avec, int& m, const int n,
      const int* list, double* buf, const int pbc_flag, const int* pbc)
  {
    if (pbc_flag == 0) {
      if (avec->comm_images == 0)
        AtomVecKokkos_Classic<MASK,Avec>(avec).pack(buf,m,list,n);
      else
        AtomVecKokkos_Classic<MASK|IMAGE_MASK,Avec>(avec).pack(buf,m,list,n);
    } else {
      if (avec->comm_images == 0)
        AtomVecKokkos_Classic<MASK,Avec>(avec).template pack_pbc<BORDERFLAG>(
            buf,m,list,n,pbc);
      else
        AtomVecKokkos_Classic<MASK|IMAGE_MASK,Avec>(avec).template pack_pbc<BORDERFLAG>(
            buf,m,list,n,pbc);
    }
  }

  template<unsigned long MASK, class Avec, bool BORDERFLAG>
  static inline void AvecKokkos_pack_vel_impl(Avec* avec, int& m, const int n,
      const int* list, double* buf, const int pbc_flag, const int* pbc)
  {
    if (pbc_flag == 0 || !avec->deform_vremap)
      return AvecKokkos_pack_impl<MASK,Avec,BORDERFLAG>(avec,m,n,list,buf,pbc_flag,pbc);
    if (avec->comm_images == 0)
      AtomVecKokkos_Classic<MASK,Avec>(avec).template pack_deform<BORDERFLAG>(buf,m,list,n,pbc);
    else
      AtomVecKokkos_Classic<MASK|IMAGE_MASK,Avec>(avec).template pack_deform<BORDERFLAG>(buf,m,list,n,pbc);
  }

};

/* --------------------------------------------------------------------------
   Interface wrapper functions
 -------------------------------------------------------------------------- */
template<unsigned long MASK, class Avec>
inline void AvecKokkos_pack_border(Avec* avec, int& m, const int n,
    const int* list, double* buf, const int pbc_flag, const int* pbc)
{
  AvecKokkosTypes::AvecKokkos_pack_impl<MASK,Avec,true>(
      avec,m,n,list,buf,pbc_flag,pbc);
}

template<unsigned long MASK, class Avec>
inline void AvecKokkos_pack_comm(Avec* avec, int& m, const int n,
    const int* list, double* buf, const int pbc_flag, const int* pbc)
{
  AvecKokkosTypes::AvecKokkos_pack_impl<MASK,Avec,false>(
      avec,m,n,list,buf,pbc_flag,pbc);
}

template<unsigned long MASK, class Avec>
inline void AvecKokkos_pack_border_vel(Avec* avec, int& m, const int n,
    const int* list, double* buf, const int pbc_flag, const int* pbc)
{
  AvecKokkosTypes::AvecKokkos_pack_vel_impl<MASK,Avec,true>(
      avec,m,n,list,buf,pbc_flag,pbc);
}

template<unsigned long MASK, class Avec>
inline void AvecKokkos_pack_comm_vel(Avec* avec, int& m, const int n,
    const int* list, double* buf, const int pbc_flag, const int* pbc)
{
  AvecKokkosTypes::AvecKokkos_pack_vel_impl<MASK,Avec,false>(
      avec,m,n,list,buf,pbc_flag,pbc);
}

template<unsigned long MASK, class Avec>
inline void AvecKokkos_unpack(Avec* avec, const int first,
    const int last, const double* buf, int &m)
{
  if (avec->comm_images == 0)
    AvecKokkosTypes::AtomVecKokkos_Classic<MASK,Avec>(avec).unpack(first,last,buf,m);
  else
    AvecKokkosTypes::AtomVecKokkos_Classic<MASK|IMAGE_MASK,Avec>(avec).unpack(first,last,buf,m);
}

} // namespace LAMMPS_NS

#endif
