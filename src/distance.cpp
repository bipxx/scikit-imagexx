#include "skimagexx/distance.hpp"
#include "skimagexx/image.hpp"
#include "distance_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <limits>

namespace skimagexx {
namespace {
void validate_distance(std::size_t n,PixelType type,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/std::max(pixel_size(type),sizeof(double));
    const auto magnitude=[](std::ptrdiff_t v){return v<0 ? std::size_t(-(v+1))+1 : std::size_t(v);};
    const auto p=magnitude(pixels),c=magnitude(channels);
    if(n>limit || c>limit/2 || (n>1 && p>(limit-2*c)/(n-1)))
        throw std::overflow_error("distance strides overflow");
}
template<class T> void execute_distance(const void* a,const void* b,T* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    const auto ar=detail::distance_reader<T>(at),br=detail::distance_reader<T>(bt);
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    const bool vectorized=has_avx2();
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        const auto* first=static_cast<const unsigned char*>(a)+std::ptrdiff_t(begin)*ap*ab;
        const auto* second=static_cast<const unsigned char*>(b)+std::ptrdiff_t(begin)*bp*bb;
        std::size_t consumed=0;
#ifdef SKIMAGEXX_AVX2
        if(vectorized) consumed=simd::delta_e76(first,second,out+begin,end-begin,at,bt,ap,ac,bp,bc);
#else
        (void)vectorized;
#endif
        for(auto j=consumed;j<end-begin;++j)
            out[begin+j]=detail::distance_pixel(first+std::ptrdiff_t(j)*ap*ab,second+std::ptrdiff_t(j)*bp*bb,ar,br,ac,bc);
    });
}
}
void delta_e76_into(const void* a,const void* b,void* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    validate_distance(n,at,ap,ac); validate_distance(n,bt,bp,bc);
    // Validate even empty arrays, without dereferencing their buffers.
    (void)detail::distance_reader<double>(at); (void)detail::distance_reader<double>(bt);
    if(!n) return;
    if(!a || !b || !out) throw std::invalid_argument("distance buffers must be nonnull");
    if(at==PixelType::float32 && bt==PixelType::float32)
        execute_distance(a,b,static_cast<float*>(out),n,at,bt,ap,ac,bp,bc);
    else execute_distance(a,b,static_cast<double*>(out),n,at,bt,ap,ac,bp,bc);
}
}
