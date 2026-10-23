#include "frangi_detail.hpp"
#include "frangi_exp.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
// Same NumPy BSD-licensed P5/Q2 coefficients and rounded operations as the
// portable helper. Power-of-two scaling is exact in the ordinary exponent
// range; exceptional/subnormal inputs retain the portable implementation.
void frangi_exp_float4(const float* input,float* output){
 const auto original=_mm_loadu_ps(input);
 const auto ordinary=_mm_and_ps(_mm_cmpge_ps(original,_mm_set1_ps(-87.f)),_mm_cmple_ps(original,_mm_setzero_ps()));
 const auto value=_mm_blendv_ps(_mm_setzero_ps(),original,ordinary);
 auto quadrant=_mm_mul_ps(value,_mm_set1_ps(1.4426950408889634074f));
 quadrant=_mm_add_ps(quadrant,_mm_set1_ps(0x1.8p23f));quadrant=_mm_sub_ps(quadrant,_mm_set1_ps(0x1.8p23f));
 auto x=_mm_fmadd_ps(quadrant,_mm_set1_ps(-6.93145752e-1f),value);
 x=_mm_fmadd_ps(quadrant,_mm_set1_ps(-1.42860677e-6f),x);x=_mm_fmadd_ps(quadrant,_mm_setzero_ps(),x);
 auto p=_mm_fmadd_ps(_mm_set1_ps(5.082762527590693718096e-04f),x,_mm_set1_ps(6.757896990527504603057e-03f));
 p=_mm_fmadd_ps(p,x,_mm_set1_ps(5.114512081637298353406e-02f));
 p=_mm_fmadd_ps(p,x,_mm_set1_ps(2.473615434895520810817e-01f));
 p=_mm_fmadd_ps(p,x,_mm_set1_ps(7.257664613233124478488e-01f));
 p=_mm_fmadd_ps(p,x,_mm_set1_ps(9.999999999980870924916e-01f));
 auto q=_mm_fmadd_ps(_mm_set1_ps(2.159509375685829852307e-02f),x,_mm_set1_ps(-2.742335390411667452936e-01f));
 q=_mm_fmadd_ps(q,x,_mm_set1_ps(1.f));
 const auto power=_mm_slli_epi32(_mm_add_epi32(_mm_cvttps_epi32(quadrant),_mm_set1_epi32(127)),23);
 _mm_storeu_ps(output,_mm_mul_ps(_mm_div_ps(p,q),_mm_castsi128_ps(power)));
 const auto mask=_mm_movemask_ps(ordinary);
 if(mask!=15)for(int lane=0;lane<4;++lane)if(!(mask&(1<<lane)))output[lane]=detail::frangi_exp(input[lane]);
}
}
