#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t multiotsu_scores(float first,const float* tail,const float* middle,
        const float* fixed,std::size_t fixed_count,float* out,std::size_t count) {
    const auto initial=_mm256_set1_ps(first);std::size_t i=0;
    for(;i+8<=count;i+=8) {
        auto sum=_mm256_add_ps(initial,_mm256_loadu_ps(tail+i));
        for(std::size_t j=0;j<fixed_count;++j)sum=_mm256_add_ps(sum,_mm256_set1_ps(fixed[j]));
        _mm256_storeu_ps(out+i,_mm256_add_ps(sum,_mm256_loadu_ps(middle+i)));
    }
    return i;
}
}
