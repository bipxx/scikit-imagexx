#include "simd.hpp"
#include "skimagexx/threshold.hpp"
#include "threshold_detail.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd {
std::size_t threshold_logs(const float* input,float* output,std::size_t count) {
    const auto ones=_mm256_set1_ps(1.0f);
    const auto minimum=_mm256_set1_ps(std::numeric_limits<float>::min());
    const auto infinity=_mm256_set1_ps(std::numeric_limits<float>::infinity());
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto value=_mm256_loadu_ps(input+i);
        const auto normal=_mm256_and_ps(_mm256_cmp_ps(value,minimum,_CMP_GE_OQ),_mm256_cmp_ps(value,infinity,_CMP_LT_OQ));
        if(_mm256_movemask_ps(normal)!=255) {
            for(std::size_t j=0;j<8;++j)output[i+j]=skimagexx::threshold_log(input[i+j],true);
            continue;
        }
        const auto bits=_mm256_castps_si256(value);
        auto exponent=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(bits,23),_mm256_set1_epi32(126)));
        auto x=_mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(bits,_mm256_set1_epi32(0x7fffff)),_mm256_set1_epi32(126<<23)));
        const auto lower=_mm256_cmp_ps(x,_mm256_set1_ps(detail::threshold_log_sqrt_half),_CMP_LE_OQ);
        x=_mm256_blendv_ps(x,_mm256_add_ps(x,x),lower);
        exponent=_mm256_blendv_ps(exponent,_mm256_sub_ps(exponent,ones),lower);
        x=_mm256_sub_ps(x,ones);
        auto numerator=_mm256_set1_ps(detail::threshold_log_p[5]);
        auto denominator=_mm256_set1_ps(detail::threshold_log_q[5]);
        for(int j=4;j>=0;--j) {
            numerator=_mm256_fmadd_ps(numerator,x,_mm256_set1_ps(detail::threshold_log_p[j]));
            denominator=_mm256_fmadd_ps(denominator,x,_mm256_set1_ps(detail::threshold_log_q[j]));
        }
        const auto logarithm=_mm256_fmadd_ps(exponent,_mm256_set1_ps(detail::threshold_log_ln2),_mm256_div_ps(numerator,denominator));
        _mm256_storeu_ps(output+i,logarithm);
    }
    return i;
}
}
