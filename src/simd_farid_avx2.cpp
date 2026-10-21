#include "simd.hpp"
#include <immintrin.h>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd {
namespace {
__m256d load(const double* p) {return _mm256_loadu_pd(p);}
__m256d load(const float* p) {return _mm256_cvtps_pd(_mm_loadu_ps(p));}
__m256d load(const std::uint8_t* p) {
    std::uint32_t bytes;std::memcpy(&bytes,p,4);
    return _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(bytes)))),_mm256_set1_pd(1.0/255.0));
}
void store(float* output,__m256d value,bool replace,bool square) {
    auto result=_mm256_cvtpd_ps(value);
    if(square)result=_mm_mul_ps(result,result);
    if(!replace)result=_mm_add_ps(_mm_loadu_ps(output),result);
    _mm_storeu_ps(output,result);
}
void store(double* output,__m256d value,bool replace,bool square) {
    if(square)value=_mm256_mul_pd(value,value);
    if(!replace)value=_mm256_add_pd(_mm256_loadu_pd(output),value);
    _mm256_storeu_pd(output,value);
}
template<class Input,class Output> std::size_t row(const void* const* sources,const std::ptrdiff_t* shifts,
    const double* weights,std::size_t taps,Output* output,std::size_t width,std::size_t x,std::size_t end,
    double cval,bool replace,bool square) {
    const auto constant=_mm256_set1_pd(cval);
    // Four independent accumulators hide the dependent-add latency without
    // reassociating any individual output's lexicographic coefficient sequence.
    for(;x+16<=end && x+18<=width;x+=16) {
        auto a=_mm256_setzero_pd(),b=a,c=a,d=a;
        for(std::size_t tap=0;tap<taps;++tap) {
            const auto weight=_mm256_set1_pd(weights[tap]);
            const auto* source=static_cast<const Input*>(sources[tap]);
            if(source) {
                const auto* p=source+std::ptrdiff_t(x)+shifts[tap];
                a=_mm256_add_pd(a,_mm256_mul_pd(load(p),weight));
                b=_mm256_add_pd(b,_mm256_mul_pd(load(p+4),weight));
                c=_mm256_add_pd(c,_mm256_mul_pd(load(p+8),weight));
                d=_mm256_add_pd(d,_mm256_mul_pd(load(p+12),weight));
            } else {
                const auto value=_mm256_mul_pd(constant,weight);
                a=_mm256_add_pd(a,value);b=_mm256_add_pd(b,value);
                c=_mm256_add_pd(c,value);d=_mm256_add_pd(d,value);
            }
        }
        store(output+x,a,replace,square);store(output+x+4,b,replace,square);
        store(output+x+8,c,replace,square);store(output+x+12,d,replace,square);
    }
    for(;x+4<=end && x+6<=width;x+=4) {
        auto sum=_mm256_setzero_pd();
        for(std::size_t tap=0;tap<taps;++tap) {
            const auto* source=static_cast<const Input*>(sources[tap]);
            const auto value=source ? load(source+std::ptrdiff_t(x)+shifts[tap]) : constant;
            sum=_mm256_add_pd(sum,_mm256_mul_pd(value,_mm256_set1_pd(weights[tap])));
        }
        store(output+x,sum,replace,square);
    }
    return x;
}
}
std::size_t farid_row(const void* const* rows,const std::ptrdiff_t* shifts,const double* weights,std::size_t taps,
    void* output,PixelType type,std::size_t width,std::size_t begin,std::size_t end,double cval,bool replace,bool square) {
    if(type==PixelType::float32)return row<float>(rows,shifts,weights,taps,static_cast<float*>(output),width,begin,end,cval,replace,square);
    if(type==PixelType::float64)return row<double>(rows,shifts,weights,taps,static_cast<double*>(output),width,begin,end,cval,replace,square);
    return row<std::uint8_t>(rows,shifts,weights,taps,static_cast<double*>(output),width,begin,end,cval,replace,square);
}
std::size_t farid_finish(float* output,std::size_t count,double divisor) {
    std::size_t i=0;const auto denominator=_mm256_set1_ps(float(divisor));
    for(;i+8<=count;i+=8)_mm256_storeu_ps(output+i,_mm256_div_ps(_mm256_sqrt_ps(_mm256_loadu_ps(output+i)),denominator));
    return i;
}
std::size_t farid_finish(double* output,std::size_t count,double divisor) {
    std::size_t i=0;const auto denominator=_mm256_set1_pd(divisor);
    for(;i+4<=count;i+=4)_mm256_storeu_pd(output+i,_mm256_div_pd(_mm256_sqrt_pd(_mm256_loadu_pd(output+i)),denominator));
    return i;
}
}
