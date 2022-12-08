// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_ATOM_VEC_KOKKOS_H
#define LMP_ATOM_VEC_KOKKOS_H

#include "atom_vec.h"           //  IWYU pragma: export

#include "kokkos_type.h"
#include <type_traits>

namespace LAMMPS_NS {

union d_ubuf {
  double d;
  int64_t i;
  KOKKOS_INLINE_FUNCTION
  d_ubuf(double arg) : d(arg) {}
  KOKKOS_INLINE_FUNCTION
  d_ubuf(int64_t arg) : i(arg) {}
  KOKKOS_INLINE_FUNCTION
  d_ubuf(int arg) : i(arg) {}
};

class AtomVecKokkos : public AtomVec {
 public:
  AtomVecKokkos(class LAMMPS *);

  bigint roundup(bigint) override;
  int pack_comm(int, int *, double *, int, int *) override;
  int pack_comm_vel(int, int *, double *, int, int *) override;
  void unpack_comm(int, int, double *) override;
  void unpack_comm_vel(int, int, double *) override;
  int pack_reverse(int, int, double *) override;
  void unpack_reverse(int, int *, double *) override;
  void data_vel(int, const std::vector<std::string> &) override;
  void pack_vel(double **) override;
  void write_vel(FILE *, int, double **) override;

  virtual void sync(ExecutionSpace space, unsigned int mask) = 0;
  virtual void modified(ExecutionSpace space, unsigned int mask) = 0;
  virtual void sync_overlapping_device(ExecutionSpace space, unsigned int mask) = 0;

  virtual int
    pack_comm_self(const int &n, const DAT::tdual_int_2d &list,
                   const int & iswap, const int nfirst,
                   const int &pbc_flag, const int pbc[]);

  virtual int
    pack_comm_self_fused(const int &n, const DAT::tdual_int_2d &list,
                         const DAT::tdual_int_1d &sendnum_scan,
                         const DAT::tdual_int_1d &firstrecv,
                         const DAT::tdual_int_1d &pbc_flag,
                         const DAT::tdual_int_2d &pbc,
                         const DAT::tdual_int_1d &g2l);

  virtual int
    pack_comm_kokkos(const int &n, const DAT::tdual_int_2d &list,
                     const int & iswap, const DAT::tdual_xfloat_2d &buf,
                     const int &pbc_flag, const int pbc[]);

  virtual void
    unpack_comm_kokkos(const int &n, const int &nfirst,
                       const DAT::tdual_xfloat_2d &buf);

  virtual int
    pack_comm_vel_kokkos(const int &n, const DAT::tdual_int_2d &list,
                         const int & iswap, const DAT::tdual_xfloat_2d &buf,
                         const int &pbc_flag, const int pbc[]);

  virtual void
    unpack_comm_vel_kokkos(const int &n, const int &nfirst,
                           const DAT::tdual_xfloat_2d &buf);

  virtual int
    unpack_reverse_self(const int &n, const DAT::tdual_int_2d &list,
                      const int & iswap, const int nfirst);

  virtual int
    pack_reverse_kokkos(const int &n, const int &nfirst,
                        const DAT::tdual_ffloat_2d &buf);

  virtual void
    unpack_reverse_kokkos(const int &n, const DAT::tdual_int_2d &list,
                          const int & iswap, const DAT::tdual_ffloat_2d &buf);

  virtual int
    pack_border_kokkos(int n, DAT::tdual_int_2d k_sendlist,
                       DAT::tdual_xfloat_2d buf,int iswap,
                       int pbc_flag, int *pbc, ExecutionSpace space) = 0;

  virtual void
    unpack_border_kokkos(const int &n, const int &nfirst,
                         const DAT::tdual_xfloat_2d &buf,
                         ExecutionSpace space) = 0;

  virtual int
    pack_border_vel_kokkos(int /*n*/, DAT::tdual_int_2d /*k_sendlist*/,
                           DAT::tdual_xfloat_2d /*buf*/,int /*iswap*/,
                           int /*pbc_flag*/, int * /*pbc*/, ExecutionSpace /*space*/) { return 0; }

  virtual void
    unpack_border_vel_kokkos(const int &/*n*/, const int & /*nfirst*/,
                             const DAT::tdual_xfloat_2d & /*buf*/,
                             ExecutionSpace /*space*/) {}

  virtual int
    pack_exchange_kokkos(const int &nsend, DAT::tdual_xfloat_2d &buf,
                         DAT::tdual_int_1d k_sendlist,
                         DAT::tdual_int_1d k_copylist,
                         ExecutionSpace space, int dim, X_FLOAT lo, X_FLOAT hi) = 0;

  virtual int
    unpack_exchange_kokkos(DAT::tdual_xfloat_2d &k_buf, int nrecv,
                           int nlocal, int dim, X_FLOAT lo, X_FLOAT hi,
                           ExecutionSpace space) = 0;


  int no_comm_vel_flag,no_border_vel_flag;
  int no_comm_image_flag,no_border_image_flag;

 protected:

  HAT::t_x_array h_x;
  HAT::t_v_array h_v;
  HAT::t_f_array h_f;
  HAT::t_imageint_1d h_image;

  class CommKokkos *commKK;
  size_t buffer_size;
  void* buffer;

  #ifdef LMP_KOKKOS_GPU
  template<class ViewType>
  Kokkos::View<typename ViewType::data_type,
               typename ViewType::array_layout,
               LMPPinnedHostType,
               Kokkos::MemoryTraits<Kokkos::Unmanaged> >
  create_async_copy(const ViewType& src) {
    typedef Kokkos::View<typename ViewType::data_type,
                 typename ViewType::array_layout,
                 typename std::conditional<
                   std::is_same<typename ViewType::execution_space,LMPDeviceType>::value,
                   LMPPinnedHostType,typename ViewType::memory_space>::type,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> > mirror_type;
    if (buffer_size == 0) {
       buffer = Kokkos::kokkos_malloc<LMPPinnedHostType>(src.span());
       buffer_size = src.span();
    } else if (buffer_size < src.span()) {
       buffer = Kokkos::kokkos_realloc<LMPPinnedHostType>(buffer,src.span());
       buffer_size = src.span();
    }
    return mirror_type(buffer, src.d_view.layout());
  }

  template<class ViewType>
  void perform_async_copy(ViewType& src, unsigned int space) {
    typedef Kokkos::View<typename ViewType::data_type,
                 typename ViewType::array_layout,
                 typename std::conditional<
                   std::is_same<typename ViewType::execution_space,LMPDeviceType>::value,
                   LMPPinnedHostType,typename ViewType::memory_space>::type,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> > mirror_type;
    if (buffer_size == 0) {
       buffer = Kokkos::kokkos_malloc<LMPPinnedHostType>(src.span()*sizeof(typename ViewType::value_type));
       buffer_size = src.span();
    } else if (buffer_size < src.span()) {
       buffer = Kokkos::kokkos_realloc<LMPPinnedHostType>(buffer,src.span()*sizeof(typename ViewType::value_type));
       buffer_size = src.span();
    }
    mirror_type tmp_view((typename ViewType::value_type*)buffer, src.d_view.layout());

    if (space == Device) {
      Kokkos::deep_copy(LMPHostType(),tmp_view,src.h_view),
      Kokkos::deep_copy(LMPHostType(),src.d_view,tmp_view);
      src.clear_sync_state();
    } else {
      Kokkos::deep_copy(LMPHostType(),tmp_view,src.d_view),
      Kokkos::deep_copy(LMPHostType(),src.h_view,tmp_view);
      src.clear_sync_state();
    }
  }
  #else
  template<class ViewType>
  void perform_async_copy(ViewType& src, unsigned int space) {
    if (space == Device)
      src.template sync<LMPDeviceType>();
    else
      src.template sync<LMPHostType>();
  }
  #endif
};

template<class DeviceType, int PBC_FLAG, bool IMG_FLAG>
struct AtomVecKokkos_PackImageMaybe;

template<class DeviceType, int PBC_FLAG>
struct AtomVecKokkos_PackImageMaybe<DeviceType, PBC_FLAG, true> {
  typedef ArrayTypes<DeviceType> AT;
  const typename AT::t_imageint_1d_randomread _image;
  int _dix, _diy, _diz;

  AtomVecKokkos_PackImageMaybe(
      const typename AT::t_imageint_1d_randomread& image,
      const int& dix, const int& diy, const int& diz)
    : _image(image),
    _dix(dix), _diy(diy), _diz(diz) {}

  template<int IDX>
  KOKKOS_INLINE_FUNCTION
  void pack_buf(const typename AT::t_xfloat_2d& buf,
      const int& i, const int& j) const
  {
    if (PBC_FLAG == 0) {
      buf(i,IDX) = d_ubuf(_image(j)).d;
    } else {
      imageint xi = (_image(j) & IMGMASK) - _dix;
      imageint yi = ((_image(j) >> IMGBITS) & IMGMASK) - _diy;
      imageint zi = (_image(j) >> IMG2BITS) - _diz;
      imageint img = (xi & IMGMASK) |
        ((yi & IMGMASK) << IMGBITS) |
        ((zi & IMGMASK) << IMG2BITS);
      buf(i,IDX) = d_ubuf(img).d;
    }
  }
};

template<class DeviceType, int PBC_FLAG>
struct AtomVecKokkos_PackImageMaybe<DeviceType,PBC_FLAG,false> {
  typedef ArrayTypes<DeviceType> AT;
  AtomVecKokkos_PackImageMaybe(
      const typename AT::t_imageint_1d_randomread&,
      const int&, const int&, const int&) {}
  template<int>
  KOKKOS_INLINE_FUNCTION
  void pack_buf(const typename AT::t_xfloat_2d&,
      const int&, const int&) const {}
};

template<class DeviceType, bool IMG_FLAG>
struct AtomVecKokkos_UnpackImageMaybe;

template<class DeviceType>
struct AtomVecKokkos_UnpackImageMaybe<DeviceType,true> {
  typedef ArrayTypes<DeviceType> AT;
  const typename AT::t_imageint_1d _image;

  AtomVecKokkos_UnpackImageMaybe(const typename AT::t_imageint_1d& image)
    : _image(image) {}

  template<int IDX>
  KOKKOS_INLINE_FUNCTION
  void unpack_buf(const typename AT::t_xfloat_2d_const& _buf,
      const int& i, const int& j) const
  {
    _image(i) = (imageint) d_ubuf(_buf(j,IDX)).i;
  }
};

template<class DeviceType>
struct AtomVecKokkos_UnpackImageMaybe<DeviceType,false> {
  typedef ArrayTypes<DeviceType> AT;
  AtomVecKokkos_UnpackImageMaybe(const typename AT::t_imageint_1d& image) {}
  template<int>
  KOKKOS_INLINE_FUNCTION
  void unpack_buf(const typename AT::t_xfloat_2d_const&,
      const int&, const int&) const {}
};

template<class DeviceType, int PBC_FLAG, bool IMG_FLAG>
struct AtomVecKokkos_PackImageSelfMaybe;

template<class DeviceType, int PBC_FLAG>
struct AtomVecKokkos_PackImageSelfMaybe<DeviceType, PBC_FLAG, true> {
  typedef ArrayTypes<DeviceType> AT;
  const typename AT::t_imageint_1d_randomread _image;
  const typename AT::t_imageint_1d _imagew;
  int _dix, _diy, _diz;

  AtomVecKokkos_PackImageSelfMaybe(
      const typename AT::tdual_imageint_1d& image,
      const int& dix, const int& diy, const int& diz)
    : _image(image.template view<DeviceType>()),
      _imagew(image.template view<DeviceType>()),
      _dix(dix), _diy(diy), _diz(diz) {}

  KOKKOS_INLINE_FUNCTION
  void pack_buf(const int& i, const int& j) const {
    if (PBC_FLAG == 0) {
      _imagew(i) = (imageint) d_ubuf(_image(j)).i;
    } else {
      imageint xi = (_image(j) & IMGMASK) - _dix;
      imageint yi = ((_image(j) >> IMGBITS) & IMGMASK) - _diy;
      imageint zi = (_image(j) >> IMG2BITS) - _diz;
      _imagew(i) = (xi & IMGMASK)
        | ((yi & IMGMASK) << IMGBITS)
        | ((zi & IMGMASK) << IMG2BITS);
    }
  }
};

template<class DeviceType, int PBC_FLAG>
struct AtomVecKokkos_PackImageSelfMaybe<DeviceType,PBC_FLAG,false> {
  typedef ArrayTypes<DeviceType> AT;
  AtomVecKokkos_PackImageSelfMaybe(const typename AT::tdual_imageint_1d&,
      const imageint&, const imageint&, const imageint&) {}
  KOKKOS_INLINE_FUNCTION
  void pack_buf(const int&, const int&) const {}
};

}

#endif

