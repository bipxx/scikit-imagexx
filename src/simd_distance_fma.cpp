#include "simd.hpp"
#include "distance_svml.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
namespace {
struct SvmlVector {
    using R=__m256d;
    static R c(double x){return _mm256_set1_pd(x);}
    static R mul(R x,R y){return _mm256_mul_pd(x,y);}
    static R fma(R x,R y,R z){return _mm256_fmadd_pd(x,y,z);}
};
}
// Called only after the AVX2/FMA runtime check. No AVX-512 instructions needed.
void distance_svml_angles(const double* x,const double* y,double* output) {
    for(int j=0;j<4;++j) {
        const auto ax=std::abs(x[j]),ay=std::abs(y[j]);
        if(ax<0x1p-500 || ay<0x1p-500 || ax>0x1p500 || ay>0x1p500 || !std::isfinite(ax+ay)) {
            for(int i=0;i<4;++i){const auto h=detail::svml_atan2(y[i],x[i]);output[i]=h+(h<0?0x1.921fb54442d18p+2:0);}
            return;
        }
    }
    const auto vx=_mm256_loadu_pd(x),vy=_mm256_loadu_pd(y),sign=SvmlVector::c(-0.);
    const auto ax=_mm256_andnot_pd(sign,vx),ay=_mm256_andnot_pd(sign,vy);
    const auto swap=_mm256_cmp_pd(ay,ax,_CMP_GT_OQ);
    const auto numerator=_mm256_blendv_pd(ay,_mm256_xor_pd(ax,sign),swap);
    const auto denominator=_mm256_blendv_pd(ax,ay,swap);
    auto z=detail::svml_atan_polynomial<SvmlVector>(_mm256_div_pd(numerator,denominator));
    z=_mm256_add_pd(z,_mm256_and_pd(swap,SvmlVector::c(0x1.921fb54442d18p+0)));
    z=_mm256_or_pd(_mm256_andnot_pd(sign,z),_mm256_and_pd(sign,vx));
    z=_mm256_add_pd(z,_mm256_and_pd(_mm256_cmp_pd(vx,SvmlVector::c(0),_CMP_LE_OQ),SvmlVector::c(0x1.921fb54442d18p+1)));
    z=_mm256_or_pd(_mm256_andnot_pd(sign,z),_mm256_and_pd(sign,vy));
    z=_mm256_add_pd(z,_mm256_and_pd(_mm256_cmp_pd(z,SvmlVector::c(0),_CMP_LT_OQ),SvmlVector::c(0x1.921fb54442d18p+2)));
    _mm256_storeu_pd(output,z);
}
}
