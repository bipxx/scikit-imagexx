#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {
// Private materialized spatial stages; not a whole-loop/convergence provider.
// NumPy scalar arithmetic, energy reductions, dtype conversion, allocation,
// warning/callback policy, channel ordering and return aliases stay in the shell.
inline constexpr const char* tv_chambolle_interface = "tv-chambolle-stages-v1";
std::uint32_t tv_chambolle_capabilities() noexcept; // F32=1,F64=2,div=4,grad=8,norm=16,dual=32.
enum class TvScalarType { float32, float64 };
struct TvConstView {
    const void* data=nullptr;
    TvScalarType type=TvScalarType::float64;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides; // Bytes; logical element0 pointer.
};
struct TvView {
    void* data=nullptr;
    TvScalarType type=TvScalarType::float64;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
};
struct TvConstPlanes {
    const void* data=nullptr;
    TvScalarType type=TvScalarType::float64;
    std::vector<std::size_t> shape; // Spatial shape only, rank D in [1,4].
};
struct TvPlanes {
    void* data=nullptr;
    TvScalarType type=TvScalarType::float64;
    std::vector<std::size_t> shape;
};
// Planes are native-endian C contiguous (D,)+shape. Spatial views permit
// arbitrary byte strides and unaligned input/output, including negative reads.
// Writable views must be provably non-self-overlapping. Every input/output
// byte interval must be disjoint, except p is explicitly updated in dual_into.
// Null pointers are accepted only for a zero-size spatial shape. Validate
// shape/type/overflow/alias domains before any write. No workspace allocation.
// Element arithmetic rounds to the storage type after each original stage,
// requires FP contraction disabled in baseline and AVX2 translation units.
// These typed primitives do not dispatch NumPy FP handlers: exceptional or
// stateful policies need the shell's explicit materialized provider path.
void tv_chambolle_divergence_into(const TvConstPlanes& p,const TvView& d);
void tv_chambolle_gradient_into(const TvConstView& out,const TvPlanes& g);
// norm=sqrt(sum_a(T(g[a]*g[a]))) in ascending axis order with T rounding.
// A rank<=4 slow-axis fold only, not a general NumPy spatial sum replacement.
void tv_chambolle_norm_into(const TvConstPlanes& g,const TvView& norm);
// Denominator is already norm*(tau/weight)+1 from the shell's ordered scalar
// and typed-array providers. tau_work is rounded once to T before tau*g;
// then separate T multiply, T subtraction and T divide, never fused.
void tv_chambolle_dual_into(const TvPlanes& p,const TvConstPlanes& g,
                           const TvConstView& denominator,double tau_work);
}

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
// Private contiguous leaves; return completed prefix, never read/write tails.
std::size_t tv_divergence(const void*,void*,std::size_t,std::size_t,std::size_t,std::size_t,
                         const std::size_t*,const std::size_t*,bool);
std::size_t tv_gradient(const void*,void*,std::size_t,std::size_t,bool);
std::size_t tv_norm(const void*,void*,std::size_t,std::size_t,std::size_t,bool);
std::size_t tv_dual(void*,const void*,const void*,std::size_t,std::size_t,
                    std::size_t,std::size_t,double,bool);
}
#endif
