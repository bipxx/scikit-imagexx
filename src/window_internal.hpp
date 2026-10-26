#pragma once
#include <cmath>
#include <cstddef>

namespace skimagexx::detail {
// Private stage: native aligned, disjoint caller-owned buffers. Coordinates
// are finite source-generated float32 values, window contains length doubles.
// Count0 performs no reads/writes; otherwise all buffers have sufficient size.
// Grid-constant cval0, order0/1, no coordinate generation or clipping here.
void window_sample(const double* window, std::size_t length, const float* coords,
                   std::size_t count, double* output, int order, int workers);
inline double window_tap(const double* window,std::size_t length,std::ptrdiff_t index) {
    return index>=0 && std::size_t(index)<length ? window[index] : 0.;
}
inline double window_sample_one(const double* window,std::size_t length,double x,int order) {
    double total=0.;
    if (order==0) {
        if (!(x>=-.5 && x<double(length))) return total;
        total+=window_tap(window,length,std::ptrdiff_t(std::floor(x+.5)));
    } else {
        if (!(x>=-1. && x<double(length))) return total;
        const auto start=std::floor(x);
        const auto index=std::ptrdiff_t(start);
        // SciPy weights: final weight is1 minus the first, not a shortcut.
        const auto first_weight=1.-(x-start),second_weight=1.-first_weight;
        auto first=window_tap(window,length,index);
        auto second=window_tap(window,length,index+1);
        first*=first_weight;total+=first;
        second*=second_weight;total+=second;
    }
    return total;
}
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
void window_sample(const double*,std::size_t,const float*,double*,
                   std::size_t begin,std::size_t end,int order);
}
#endif
