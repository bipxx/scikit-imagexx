#include "skimagexx/draw_coordinates.hpp"
#include <immintrin.h>
namespace skimagexx::simd {
std::size_t draw_fill(std::intptr_t* output,std::size_t count,std::intptr_t first,bool ramp) {
    std::size_t i=0;
    for(;count-i>=4;i+=4) {
        const auto value=ramp?_mm256_setr_epi64x(first+std::intptr_t(i),first+std::intptr_t(i+1),first+std::intptr_t(i+2),first+std::intptr_t(i+3)):_mm256_set1_epi64x(first);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),value);
    }
    return i;
}
std::size_t draw_ellipse_row(const double* column_sin,const double* column_cos,
    std::size_t count,double row_cos,double row_sin,double row_radius,
    double column_radius,std::uint8_t* mask) {
    const auto rc=_mm256_set1_pd(row_cos),rs=_mm256_set1_pd(row_sin);
    const auto rr=_mm256_set1_pd(row_radius),cr=_mm256_set1_pd(column_radius);
    const auto one=_mm256_set1_pd(1.0);std::size_t hits=0;
    for(std::size_t i=0;i<count;i+=4) {
        const auto a=_mm256_div_pd(_mm256_add_pd(rc,_mm256_loadu_pd(column_sin+i)),rr);
        const auto b=_mm256_div_pd(_mm256_sub_pd(rs,_mm256_loadu_pd(column_cos+i)),cr);
        const auto distance=_mm256_add_pd(_mm256_mul_pd(a,a),_mm256_mul_pd(b,b));
        const unsigned bits=unsigned(_mm256_movemask_pd(_mm256_cmp_pd(distance,one,_CMP_LT_OQ)));
        for(unsigned lane=0;lane<4;++lane){const auto inside=std::uint8_t((bits>>lane)&1u);mask[i+lane]=inside;hits+=inside;}
    }
    return hits;
}
}
