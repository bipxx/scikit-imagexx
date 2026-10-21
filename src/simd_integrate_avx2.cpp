#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t integrate_add(const double* values,double* output,std::size_t count) {
    std::size_t i=0;
    for(;i+16<=count;i+=16) {
        _mm256_storeu_pd(output+i,_mm256_add_pd(_mm256_loadu_pd(output+i),_mm256_loadu_pd(values+i)));
        _mm256_storeu_pd(output+i+4,_mm256_add_pd(_mm256_loadu_pd(output+i+4),_mm256_loadu_pd(values+i+4)));
        _mm256_storeu_pd(output+i+8,_mm256_add_pd(_mm256_loadu_pd(output+i+8),_mm256_loadu_pd(values+i+8)));
        _mm256_storeu_pd(output+i+12,_mm256_add_pd(_mm256_loadu_pd(output+i+12),_mm256_loadu_pd(values+i+12)));
    }
    for(;i+4<=count;i+=4)_mm256_storeu_pd(output+i,_mm256_add_pd(_mm256_loadu_pd(output+i),_mm256_loadu_pd(values+i)));
    return i;
}
}
