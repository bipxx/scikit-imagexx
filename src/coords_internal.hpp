#pragma once
#include <cstddef>
namespace skimagexx::detail {
// Private aligned native C powers[N,D,K], D1..3/K>1, disjoint output[K^D].
// Axis products and point accumulation retain source order and typed rounding.
// Status:1divide,2overflow,4underflow,8invalid,16nonfinite,32capture failure.
unsigned moments_coords_reduce(const float*,std::size_t,std::size_t,std::size_t,float*,int);
unsigned moments_coords_reduce(const double*,std::size_t,std::size_t,std::size_t,double*,int);
template<class T> T coords_cell(const T* powers,std::size_t N,std::size_t D,std::size_t K,std::size_t cell) {
    std::size_t exponent[3];
    for(auto axis=D;axis--;){exponent[axis]=cell%K;cell/=K;}
    T sum{};
    for(std::size_t point=0;point<N;++point) {
        T product=T(1);
        for(std::size_t axis=0;axis<D;++axis)product=T(product*powers[(point*D+axis)*K+exponent[axis]]);
        sum=T(sum+product);
    }
    return sum;
}
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
void moments_coords_reduce(const float*,std::size_t,std::size_t,std::size_t,float*,std::size_t,std::size_t);
void moments_coords_reduce(const double*,std::size_t,std::size_t,std::size_t,double*,std::size_t,std::size_t);
}
#endif
