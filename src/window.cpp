// Source interpolation weights and ordered accumulation follow SciPy
// ni_splines.c / ni_interpolation.c; BSD-3-Clause notices are retained in
// third_party/scipy-ndimage-LICENSE.txt. Geometry/clip remain explicit providers.
#include "window_internal.hpp"
#include "execution.hpp"
#include <limits>
#include <stdexcept>

namespace skimagexx::detail {
void window_sample(const double* window,std::size_t length,const float* coords,
                   std::size_t count,double* output,int order,int workers) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    if (!length || length>limit || count>limit) throw std::invalid_argument("window sampling extent invalid");
    if (order!=0 && order!=1) throw std::invalid_argument("window sampling supports order0 or1");
    if (workers<1 || workers>1024) throw std::invalid_argument("window worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if (workers!=1) throw std::invalid_argument("this build has no OpenMP support");
#endif
    if (!count) return;
    if (!window || !coords || !output) throw std::invalid_argument("window sampling null buffer");
    const bool vectorized=has_avx2();
    const int team=count>=parallel_min ? workers : 1;
    partition(count,team,[&](std::size_t begin,std::size_t end,std::size_t) {
#ifdef SKIMAGEXX_AVX2
        if (vectorized) { simd::window_sample(window,length,coords,output,begin,end,order);return; }
#else
        (void)vectorized;
#endif
        for(auto i=begin;i<end;++i) output[i]=window_sample_one(window,length,double(coords[i]),order);
    });
}
}
