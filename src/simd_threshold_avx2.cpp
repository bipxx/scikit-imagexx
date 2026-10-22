#include "simd.hpp"
#include <immintrin.h>
#include <algorithm>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
namespace skimagexx::simd {
namespace {
unsigned first_bit(unsigned mask) {
#if defined(_MSC_VER)
    unsigned long index;_BitScanForward(&index,mask);return unsigned(index);
#elif defined(__GNUC__) || defined(__clang__)
    return unsigned(__builtin_ctz(mask));
#else
    unsigned index=0;while(!(mask&1)) {mask>>=1;++index;}return index;
#endif
}
template<class T> std::size_t run(const T* image,std::size_t count,const T* edges,std::size_t bins,std::uint64_t* out) {
    const __m256d low=_mm256_set1_pd(double(edges[0]));
    const __m256d scale=_mm256_set1_pd(double(bins)/(double(edges[bins])-double(edges[0])));
    alignas(16) int indices[4];std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m256d values;
        if constexpr(sizeof(T)==4)values=_mm256_cvtps_pd(_mm_loadu_ps(image+i));
        else values=_mm256_loadu_pd(image+i);
        const auto estimates=_mm256_mul_pd(_mm256_sub_pd(values,low),scale);
        _mm_store_si128(reinterpret_cast<__m128i*>(indices),_mm256_cvttpd_epi32(estimates));
        for(int j=0;j<4;++j) {
            const auto v=image[i+j];
            if(!(v>=edges[0] && v<=edges[bins]))continue;
            auto b=indices[j]<0 ? std::size_t(0) : std::min(std::size_t(indices[j]),bins-1);
            while(b && v<edges[b])--b;
            while(b+1<bins && v>=edges[b+1])++b;
            ++out[b];
        }
    }
    return i;
}
}
std::size_t histogram_edges(const float* p,std::size_t n,const float* e,std::size_t b,std::uint64_t* o) {return run(p,n,e,b,o);}
std::size_t histogram_edges(const double* p,std::size_t n,const double* e,std::size_t b,std::uint64_t* o) {return run(p,n,e,b,o);}
std::size_t minimum_histogram_peaks(const float* hist,std::size_t count,std::size_t* indices) {
    std::size_t i=0,peaks=0;bool rising=true;
    for(;i+8<count;i+=8) {
        const auto left=_mm256_loadu_ps(hist+i),right=_mm256_loadu_ps(hist+i+1);
        unsigned up=unsigned(_mm256_movemask_ps(_mm256_cmp_ps(right,left,_CMP_GT_OQ)));
        unsigned down=unsigned(_mm256_movemask_ps(_mm256_cmp_ps(right,left,_CMP_LT_OQ)));
        // Ignore equal pairs and repeated signs. Carry direction over packet
        // boundaries so a plateau is resolved at its last, rather than first,
        // bin. No floating arithmetic is reordered by these comparisons.
        for(;;) {
            const unsigned wanted=rising ? down : up;
            if(!wanted)break;
            const unsigned lane=first_bit(wanted);
            if(rising) {
                if(peaks<2)indices[peaks]=i+lane;
                if(++peaks==3)return peaks;
            }
            rising=!rising;
            const unsigned remaining=0xffu<<(lane+1);
            up&=remaining;down&=remaining;
        }
    }
    for(;i+1<count;++i) {
        if(rising && hist[i+1]<hist[i]) {
            if(peaks<2)indices[peaks]=i;
            if(++peaks==3)return peaks;
            rising=false;
        } else if(!rising && hist[i+1]>hist[i])rising=true;
    }
    return peaks;
}
}
