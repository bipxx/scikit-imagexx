#include "skimagexx/projective.hpp"
#include <immintrin.h>
#include <cstring>
#include <limits>
namespace skimagexx::simd {
namespace {
double scalar(const double* data,std::ptrdiff_t row,std::ptrdiff_t column,std::size_t i,std::size_t j) {
    double value;std::memcpy(&value,reinterpret_cast<const unsigned char*>(data)+std::ptrdiff_t(i)*row+std::ptrdiff_t(j)*column,8);return value;
}
__m256d coordinates(const double* data,std::ptrdiff_t row,std::ptrdiff_t column,std::size_t i,std::size_t j) {
    const auto* base=reinterpret_cast<const unsigned char*>(data)+std::ptrdiff_t(i)*row+std::ptrdiff_t(j)*column;
    if(row==8)return _mm256_loadu_pd(reinterpret_cast<const double*>(base));
    const auto offsets=_mm256_set_epi64x(3*row,2*row,row,0);
    return _mm256_i64gather_pd(reinterpret_cast<const double*>(base),offsets,1);
}
}
std::size_t projective_apply(const double* points,std::ptrdiff_t point_row,std::ptrdiff_t point_column,
    const double* matrix,std::ptrdiff_t matrix_row,std::ptrdiff_t matrix_column,double* output,
    std::size_t count,std::size_t dimensions) {
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m256d cached[4];
        if(dimensions<=4)for(std::size_t j=0;j<dimensions;++j)cached[j]=coordinates(points,point_row,point_column,i,j);
        const auto coordinate=[&](std::size_t j){return dimensions<=4 ? cached[j]:coordinates(points,point_row,point_column,i,j);};
        auto divisor=_mm256_setzero_pd();
        for(std::size_t j=0;j<dimensions;++j)divisor=_mm256_add_pd(divisor,_mm256_mul_pd(coordinate(j),_mm256_set1_pd(scalar(matrix,matrix_row,matrix_column,dimensions,j))));
        divisor=_mm256_add_pd(divisor,_mm256_set1_pd(scalar(matrix,matrix_row,matrix_column,dimensions,dimensions)));
        divisor=_mm256_blendv_pd(divisor,_mm256_set1_pd(std::numeric_limits<double>::epsilon()),_mm256_cmp_pd(divisor,_mm256_setzero_pd(),_CMP_EQ_OQ));
        for(std::size_t d=0;d<dimensions;++d) {
            auto value=_mm256_setzero_pd();
            for(std::size_t j=0;j<dimensions;++j)value=_mm256_add_pd(value,_mm256_mul_pd(coordinate(j),_mm256_set1_pd(scalar(matrix,matrix_row,matrix_column,d,j))));
            value=_mm256_add_pd(value,_mm256_set1_pd(scalar(matrix,matrix_row,matrix_column,d,dimensions)));
            alignas(32) double lanes[4];_mm256_store_pd(lanes,_mm256_div_pd(value,divisor));
            for(std::size_t lane=0;lane<4;++lane)output[(i+lane)*dimensions+d]=lanes[lane];
        }
    }
    return i;
}
}
