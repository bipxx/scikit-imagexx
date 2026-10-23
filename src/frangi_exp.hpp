#pragma once
#include <cmath>
#include <limits>
namespace skimagexx::detail {
// NumPy's BSD-licensed AVX2/AVX512 float32 exp approximation and evaluation
// order, v2.5.4 loops_exponent_log.dispatch.c.src + npy_simd_data.h.
// See third_party/numpy-LICENSE.txt. Explicit float FMA also works without
// hardware FMA and keeps portable response decisions near exp(x)==1 consistent.
inline float frangi_exp(float value){
 if(std::isnan(value))return value;
 if(value<=-103.97208404541015625f)return 0.f;
 if(value>=88.72283935546875f)return std::numeric_limits<float>::infinity();
 float quadrant=value*1.4426950408889634074f;
 quadrant=float(quadrant+0x1.8p23f);quadrant=float(quadrant-0x1.8p23f);
 auto x=std::fma(quadrant,-6.93145752e-1f,value);x=std::fma(quadrant,-1.42860677e-6f,x);x=std::fma(quadrant,0.f,x);
 auto p=std::fma(5.082762527590693718096e-04f,x,6.757896990527504603057e-03f);
 p=std::fma(p,x,5.114512081637298353406e-02f);p=std::fma(p,x,2.473615434895520810817e-01f);
 p=std::fma(p,x,7.257664613233124478488e-01f);p=std::fma(p,x,9.999999999980870924916e-01f);
 auto q=std::fma(2.159509375685829852307e-02f,x,-2.742335390411667452936e-01f);q=std::fma(q,x,1.f);
 return std::ldexp(float(p/q),int(quadrant));
}
inline double frangi_exp(double value){return std::exp(value);}
}
