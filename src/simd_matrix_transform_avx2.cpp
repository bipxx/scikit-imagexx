#include "skimagexx/matrix_transform.hpp"
#include <immintrin.h>
#include <limits>
#include <cmath>
#include <cstring>
namespace skimagexx::simd {
void matrix_transform_cooperlake_gemv_n(const MatrixReadView& points,const double* m,
    double* out,std::size_t d) {
    // One validated input row; load each signed/unaligned coordinate once.
    // No row gather or fourth-matrix-row access for K3.
    double x[4];
    for(std::size_t k=0;k<d;++k)std::memcpy(x+k,
        static_cast<const unsigned char*>(points.data)+std::ptrdiff_t(k)*points.column_stride,8);
    const auto k=d+1;
    auto column=[&](std::size_t c) {
        return _mm256_set_pd(d>=3 ? m[3*k+c] : 0.,m[2*k+c],m[k+c],m[c]);
    };
    const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1.);
    __m256d value;
    if (d==2) {
        value=zero;
        for(std::size_t c=0;c<d;++c)value=_mm256_fmadd_pd(column(c),_mm256_set1_pd(x[c]),value);
        value=_mm256_fmadd_pd(column(d),one,value);
        value=_mm256_fmadd_pd(one,value,zero);
    }else {
        value=_mm256_mul_pd(column(1),_mm256_set1_pd(x[1]));
        value=_mm256_fmadd_pd(column(0),_mm256_set1_pd(x[0]),value);
        value=_mm256_fmadd_pd(column(2),_mm256_set1_pd(x[2]),value);
        value=_mm256_fmadd_pd(column(3),d==3 ? one : _mm256_set1_pd(x[3]),value);
        value=_mm256_fmadd_pd(one,value,zero);
        if(d==4)value=_mm256_add_pd(value,_mm256_mul_pd(column(4),one));
    }
    alignas(32) double lanes[4];_mm256_store_pd(lanes,value);
    double den;
    if (d<4)den=lanes[d];
    else {
        den=0.;for(std::size_t c=0;c<d;++c)den=std::fma(m[d*k+c],x[c],den);
        den=std::fma(m[d*k+d],1.,den);den=std::fma(1.,den,0.);
    }
    if(den==0.)den=std::numeric_limits<double>::epsilon();
    _mm256_store_pd(lanes,_mm256_div_pd(value,_mm256_set1_pd(den)));
    for(std::size_t c=0;c<d;++c)out[c]=lanes[c];
}

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
