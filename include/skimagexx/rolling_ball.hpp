#pragma once
#include <cstddef>
#include <vector>
#include <skimagexx/convert.hpp>
namespace skimagexx {
// Caller retains valid numeric storage for shape and signed byte strides.
// Rank1..32; custom kernel is contiguous C-order, positive same-rank extents.
// float/double preserve working precision; input conversion and user protocols
// are boundary policy. Output always owns contiguous C-order storage.
struct RollingBallGeometry {
    std::vector<std::ptrdiff_t> shape, strides, kernel_shape;
};
// Complete image -> background operation. Null custom_kernel generates the
// ball internally using the admitted geometry primitive; radius is ignored
// for a supplied kernel. No caller-composed padding/difference/traversal.
template<class T> std::vector<T> rolling_ball(const T*,const RollingBallGeometry&,
    const T* custom_kernel=nullptr,double radius=100.,bool nansafe=false,int workers=0);
// Storage descriptor for the complete dtype-preserving operation below.
// half uses uint16 storage. swap reverses each stored item relative to host
// byte order; it does not change scalar type or numeric scaling.
struct RollingBallStorage {
    const void* data=nullptr;
    PixelType type=PixelType::float64;
    bool half=false, swap=false;
};
struct RollingBallResult {
    std::vector<unsigned char> bytes;
    std::vector<std::ptrdiff_t> shape;
    PixelType type=PixelType::float64;
    bool half=false, swap=false;
    int captured_workers=1;
};
// Complete image -> owning background with literal astype precision policy:
// half/f32 work in f32; bool/integers/f64 in f64; result returns input storage
// dtype and byte order. The optional custom kernel is contiguous C-order with
// g.kernel_shape; its own dtype/order is converted internally. Source pointer
// lifetime/valid span belongs to the caller; no returned bytes borrow it.
// Integer results require finite representable truncated values (other unsafe
// NumPy cast/FP/ISA domains remain unsupported). Half assumes nearest rounding.
RollingBallResult rolling_ball(RollingBallStorage,const RollingBallGeometry&,
    RollingBallStorage custom_kernel={},double radius=100.,bool nansafe=false,int workers=0);
namespace detail {
// Prepared public boundary: NumPy supplies cast/kernel-subtraction policy.
// This operation still owns padding, stable min-plus traversal and output.
template<class T> std::vector<T> rolling_ball_from_difference(
    const T*,const RollingBallGeometry&,const T* difference,bool nansafe,int workers);
}
extern template std::vector<float> rolling_ball(const float*,const RollingBallGeometry&,const float*,double,bool,int);
extern template std::vector<double> rolling_ball(const double*,const RollingBallGeometry&,const double*,double,bool,int);
}
