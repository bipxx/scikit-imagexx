#include "simd.hpp"
#include "simd_math.hpp"
#include "skimagexx/gabor.hpp"
#include "gabor_math_detail.hpp"
#include <immintrin.h>
#include <cmath>

namespace skimagexx::simd { namespace {
// Reduce to [-pi/4,pi/4]. The scalar slow path handles large arguments whose
// reduction needs libm's wider precision. No FMA or reassociated image sums.
void sincos4(__m256d phase,__m256d& sine,__m256d& cosine) {
    alignas(32) double v[4];_mm256_store_pd(v,phase);
    bool ordinary=true;for(auto x:v)ordinary=ordinary && std::abs(x)<100000.;
    if(!ordinary){double s[4],c[4];for(int i=0;i<4;++i){s[i]=std::sin(v[i]);c[i]=std::cos(v[i]);}
        sine=_mm256_loadu_pd(s);cosine=_mm256_loadu_pd(c);return;}
    const auto k=_mm256_round_pd(_mm256_mul_pd(phase,_mm256_set1_pd(0.63661977236758134308)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    auto r=_mm256_sub_pd(phase,_mm256_mul_pd(k,_mm256_set1_pd(1.57079632673412561417)));
    r=_mm256_sub_pd(r,_mm256_mul_pd(k,_mm256_set1_pd(6.07710050650619224932e-11)));
    const auto square=_mm256_mul_pd(r,r);
    constexpr double sc[]={1,-1./6,1./120,-1./5040,1./362880,-1./39916800,1./6227020800,-1./1307674368000,1./355687428096000};
    constexpr double cc[]={1,-1./2,1./24,-1./720,1./40320,-1./3628800,1./479001600,-1./87178291200,1./20922789888000};
    auto s=_mm256_set1_pd(sc[8]),c=_mm256_set1_pd(cc[8]);
    for(int i=7;i>=0;--i){s=_mm256_add_pd(_mm256_set1_pd(sc[i]),_mm256_mul_pd(square,s));c=_mm256_add_pd(_mm256_set1_pd(cc[i]),_mm256_mul_pd(square,c));}
    s=_mm256_mul_pd(s,r);
    const auto quadrant=_mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k));
    const auto odd=_mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(quadrant,_mm256_set1_epi64x(1)),_mm256_set1_epi64x(1)));
    const auto sign_s=_mm256_castsi256_pd(_mm256_slli_epi64(_mm256_and_si256(quadrant,_mm256_set1_epi64x(2)),62));
    const auto sign_c=_mm256_castsi256_pd(_mm256_slli_epi64(_mm256_and_si256(_mm256_add_epi64(quadrant,_mm256_set1_epi64x(1)),_mm256_set1_epi64x(2)),62));
    sine=_mm256_xor_pd(_mm256_blendv_pd(s,c,odd),sign_s);
    cosine=_mm256_xor_pd(_mm256_blendv_pd(c,s,odd),sign_c);
}
}
namespace {
std::size_t kernel_row(void* output,bool single,std::size_t start,std::size_t width,double y,const GaborParameters& p,bool accurate) {
    const auto ct=_mm256_set1_pd(p.cosine),st=_mm256_set1_pd(p.sine);
    const auto ys=_mm256_set1_pd(y*p.sine),yc=_mm256_set1_pd(y*p.cosine);
    const auto sx=_mm256_set1_pd(p.sigma_x_squared),sy=_mm256_set1_pd(p.sigma_y_squared);
    const auto frequency=_mm256_set1_pd(p.angular_frequency),offset=_mm256_set1_pd(p.offset);
    std::size_t i=0;
    for(;i+4<=width;i+=4){
        const auto x=double(i)-double(width/2);const auto coord=_mm256_setr_pd(x,x+1,x+2,x+3);
        const auto rx=_mm256_add_pd(_mm256_mul_pd(coord,ct),ys);
        const auto ry=_mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(),coord),st),yc);
        const auto exponent=_mm256_mul_pd(_mm256_set1_pd(-.5),_mm256_add_pd(_mm256_div_pd(_mm256_mul_pd(rx,rx),sx),_mm256_div_pd(_mm256_mul_pd(ry,ry),sy)));
        const auto phase=_mm256_add_pd(_mm256_mul_pd(frequency,rx),offset);
        __m256d magnitude,s,c;
        if(accurate){
            alignas(32) double ex[4],ph[4],mag[4],sn[4],cs[4];
            _mm256_store_pd(ex,exponent);_mm256_store_pd(ph,phase);
            for(int j=0;j<4;++j){mag[j]=std::exp(ex[j]);sn[j]=std::sin(ph[j]);cs[j]=std::cos(ph[j]);}
            magnitude=_mm256_load_pd(mag);s=_mm256_load_pd(sn);c=_mm256_load_pd(cs);
        }else{magnitude=math::exp4(exponent);sincos4(phase,s,c);}
        s=_mm256_mul_pd(magnitude,s);c=_mm256_mul_pd(magnitude,c);
        if(single){
            const auto norm=_mm_set1_ps(float(p.normalization));
            const auto real=_mm_mul_ps(_mm256_cvtpd_ps(c),norm),imag=_mm_mul_ps(_mm256_cvtpd_ps(s),norm);
            auto* target=static_cast<float*>(output)+2*(start+i);
            _mm_storeu_ps(target,_mm_unpacklo_ps(real,imag));_mm_storeu_ps(target+4,_mm_unpackhi_ps(real,imag));
        }else{
            s=_mm256_mul_pd(s,_mm256_set1_pd(p.normalization));c=_mm256_mul_pd(c,_mm256_set1_pd(p.normalization));
            const auto lo=_mm256_unpacklo_pd(c,s),hi=_mm256_unpackhi_pd(c,s);
            auto* target=static_cast<double*>(output)+2*(start+i);
            _mm256_storeu_pd(target,_mm256_permute2f128_pd(lo,hi,0x20));
            _mm256_storeu_pd(target+4,_mm256_permute2f128_pd(lo,hi,0x31));
        }
    }
    return i;
}
}
std::size_t gabor_kernel_row(void* output,bool single,std::size_t start,std::size_t width,double y,const GaborParameters& p) {
    return kernel_row(output,single,start,width,y,p,false);
}
std::size_t gabor_kernel_accurate_row(void* output,bool single,std::size_t start,std::size_t width,double y,const GaborParameters& p) {
    return kernel_row(output,single,start,width,y,p,true);
}
}
