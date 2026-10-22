#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t multiotsu_two_scores(const float* mass,const float* moment,const float* first,
        float* out,std::size_t count,float total_mass,float total_moment) {
    const auto wtotal=_mm256_set1_ps(total_mass),stotal=_mm256_set1_ps(total_moment);
    const auto zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1.f);std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto w=_mm256_sub_ps(wtotal,_mm256_loadu_ps(mass+i));
        const auto s=_mm256_sub_ps(stotal,_mm256_loadu_ps(moment+i));
        const auto positive=_mm256_cmp_ps(w,zero,_CMP_GT_OQ);
        const auto divisor=_mm256_blendv_ps(one,w,positive);
        const auto right=_mm256_and_ps(positive,_mm256_div_ps(_mm256_mul_ps(s,s),divisor));
        _mm256_storeu_ps(out+i,_mm256_add_ps(_mm256_loadu_ps(first+i),right));
    }
    return i;
}
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
