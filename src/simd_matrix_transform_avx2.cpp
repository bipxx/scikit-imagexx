#include "skimagexx/matrix_transform.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd {
std::size_t matrix_transform_fma(const MatrixReadView& points,const double* m,double* out,
    std::size_t begin,std::size_t end,std::size_t d) {
    if (end-begin<4) return 0;
    std::size_t i=begin;
    const auto offsets=_mm256_set_epi64x(3*points.row_stride,2*points.row_stride,points.row_stride,0);
    for(;end-i>=4;i+=4) {
        __m256d x[4];
        for(std::size_t j=0;j<d;++j) {
            const auto* base=static_cast<const unsigned char*>(points.data)+
                std::ptrdiff_t(i)*points.row_stride+std::ptrdiff_t(j)*points.column_stride;
            x[j]=points.row_stride==8 ? _mm256_loadu_pd(reinterpret_cast<const double*>(base)) :
                _mm256_i64gather_pd(reinterpret_cast<const double*>(base),offsets,1);
        }
        auto dot=[&](const double* coefficients) {
            auto v=_mm256_setzero_pd();
            for(std::size_t j=0;j<d;++j)v=_mm256_fmadd_pd(x[j],_mm256_set1_pd(coefficients[j]),v);
            v=_mm256_fmadd_pd(_mm256_set1_pd(1.),_mm256_set1_pd(coefficients[d]),v);
            return _mm256_add_pd(v,_mm256_setzero_pd());
        };
        auto den=dot(m+d*(d+1));
        den=_mm256_blendv_pd(den,_mm256_set1_pd(std::numeric_limits<double>::epsilon()),
            _mm256_cmp_pd(den,_mm256_setzero_pd(),_CMP_EQ_OQ));
        for(std::size_t j=0;j<d;++j) {
            alignas(32) double lanes[4];_mm256_store_pd(lanes,_mm256_div_pd(dot(m+j*(d+1)),den));
            for(std::size_t k=0;k<4;++k)out[(i+k)*d+j]=lanes[k];
        }
    }
    return i-begin;
}
}
