#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
namespace {
void store16(std::uint16_t* out,__m256i values) {
    const auto packed=_mm_packus_epi32(_mm256_castsi256_si128(values),_mm256_extracti128_si256(values,1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),packed);
}
}
std::size_t clahe_quantize(const std::uint16_t* input,std::uint16_t* out,std::size_t count,
        std::uint16_t low,std::uint16_t high,std::size_t width) {
    const auto minimum=_mm256_set1_pd(double(low)),range=_mm256_set1_pd(double(high-low));
    const auto scale=_mm256_set1_pd(16383.);
    const auto divisor=_mm256_set1_ps(float(width));
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto values=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i)));
        const auto first=_mm256_cvtepi32_pd(_mm256_castsi256_si128(values));
        const auto second=_mm256_cvtepi32_pd(_mm256_extracti128_si256(values,1));
        const auto convert=[&](__m256d v) {
            const auto normalized=_mm256_mul_pd(_mm256_div_pd(_mm256_sub_pd(v,minimum),range),scale);
            return _mm256_cvttpd_epi32(_mm256_round_pd(normalized,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC));
        };
        const auto integers=_mm256_inserti128_si256(_mm256_castsi128_si256(convert(first)),convert(second),1);
        // Rounded levels are in [0,16383]; float division followed by truncation
        // gives the exact integer quotient throughout this bounded range.
        store16(out+i,_mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(integers),divisor)));
    }
    return i;
}
std::size_t clahe_line(const std::uint16_t* input,std::uint16_t* out,std::size_t count,
        const std::int32_t* const* maps,const double* factors,std::size_t ndim,
        std::size_t offset,std::size_t kernel) {
    const auto one=_mm256_set1_pd(1.),denominator=_mm256_set1_pd(double(kernel));
    const auto lanes=_mm256_setr_pd(0.,1.,2.,3.);const auto edges=std::size_t(1)<<ndim;
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto indices=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i)));
        const auto fx0=_mm256_div_pd(_mm256_add_pd(_mm256_set1_pd(double(offset+i)),lanes),denominator);
        const auto fx1=_mm256_div_pd(_mm256_add_pd(_mm256_set1_pd(double(offset+i+4)),lanes),denominator);
        auto sum=_mm256_setzero_ps();
        for(std::size_t edge=0;edge<edges;++edge) {
            auto c0=edge&1 ? fx0 : _mm256_sub_pd(one,fx0),c1=edge&1 ? fx1 : _mm256_sub_pd(one,fx1);
            for(std::size_t d=0;d+1<ndim;++d) {
                const auto coefficient=_mm256_set1_pd(factors[edge*(ndim-1)+d]);
                c0=_mm256_mul_pd(c0,coefficient);c1=_mm256_mul_pd(c1,coefficient);
            }
            const auto mapped=_mm256_i32gather_epi32(maps[edge],indices,4);
            const auto a=_mm256_cvtepi32_pd(_mm256_castsi256_si128(mapped));
            const auto b=_mm256_cvtepi32_pd(_mm256_extracti128_si256(mapped,1));
            const auto first=_mm256_cvtpd_ps(_mm256_mul_pd(a,c0)),second=_mm256_cvtpd_ps(_mm256_mul_pd(b,c1));
            const auto contribution=_mm256_insertf128_ps(_mm256_castps128_ps256(first),second,1);
            sum=_mm256_add_ps(sum,contribution);
        }
        store16(out+i,_mm256_cvttps_epi32(sum));
    }
    return i;
}
}
