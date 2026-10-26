#pragma once
#include <complex>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace skimagexx::detail {
// Private projection stage: native aligned C-contiguous disjoint buffers.
// calc has shape; powers is (shape[axis],K); output replaces shape[axis] byK.
// Caller certifies allocations. Empty output reads nothing; empty reduction
// writes zero. Coordinate powers/source dtype promotion remain frontend stages.
#define SKIMAGEXX_MOMENTS_DECLARE(T) \
unsigned moments_project(const T*,const std::vector<std::size_t>&,const T*, \
                         std::size_t K,std::size_t axis,T*,int workers)
SKIMAGEXX_MOMENTS_DECLARE(float);
SKIMAGEXX_MOMENTS_DECLARE(double);
SKIMAGEXX_MOMENTS_DECLARE(std::complex<float>);
SKIMAGEXX_MOMENTS_DECLARE(std::complex<double>);
#undef SKIMAGEXX_MOMENTS_DECLARE

template<class T> struct MomentsValue {using real=T;static constexpr bool complex=false;};
template<class R> struct MomentsValue<std::complex<R>> {using real=R;static constexpr bool complex=true;};
template<class T> inline T moments_one(const T* calc,const T* powers,std::size_t n,
                                      std::size_t K,std::size_t inner,std::size_t index) {
    const auto rest=index%inner,k=(index/inner)%K,outer=index/(inner*K);
    T sum{};
    for(std::size_t j=0;j<n;++j) {
        const auto a=calc[(outer*n+j)*inner+rest],b=powers[j*K+k];
        if constexpr(MomentsValue<T>::complex) {
            // Unconjugated complex dot, with explicit component operations.
            const auto real=a.real()*b.real()-a.imag()*b.imag();
            const auto imag=a.real()*b.imag()+a.imag()*b.real();
            sum=T(sum.real()+real,sum.imag()+imag);
        } else sum+=a*b;
    }
    return sum;
}
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
#define SKIMAGEXX_MOMENTS_DECLARE(T) \
void moments_project(const T*,const T*,std::size_t n,std::size_t K,std::size_t inner, \
                     T*,std::size_t begin,std::size_t end)
SKIMAGEXX_MOMENTS_DECLARE(float);
SKIMAGEXX_MOMENTS_DECLARE(double);
SKIMAGEXX_MOMENTS_DECLARE(std::complex<float>);
SKIMAGEXX_MOMENTS_DECLARE(std::complex<double>);
#undef SKIMAGEXX_MOMENTS_DECLARE
}
#endif
