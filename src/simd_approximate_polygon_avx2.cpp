#include "approximate_polygon_simd.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t approximate_polygon_distances(const detail::ApproximatePolygonInput& input,
    std::size_t begin,std::size_t count,double* distances,
    const detail::ApproximatePolygonSegment& s) {
    if(count<4)return 0;
    const auto zero=_mm256_setzero_pd(),sign=_mm256_set1_pd(-0.);
    const auto r0=_mm256_set1_pd(s.r0),c0=_mm256_set1_pd(s.c0);
    const auto r1=_mm256_set1_pd(s.r1),c1=_mm256_set1_pd(s.c1);
    const auto dr=_mm256_set1_pd(s.dr),dc=_mm256_set1_pd(s.dc);
    const auto sine=_mm256_set1_pd(s.sine),cosine=_mm256_set1_pd(s.cosine),offset=_mm256_set1_pd(s.offset);
    const auto stride=input.row_stride;
    const auto indices=_mm256_set_epi64x(static_cast<long long>(stride*3),static_cast<long long>(stride*2),static_cast<long long>(stride),0);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto* point=input.data+std::ptrdiff_t(begin+i)*stride;
        const auto row=_mm256_i64gather_pd(point,indices,8);
        const auto column=_mm256_i64gather_pd(point+input.column_stride,indices,8);
        const auto dr0=_mm256_sub_pd(row,r0),dc0=_mm256_sub_pd(column,c0);
        const auto dr1=_mm256_sub_pd(row,r1),dc1=_mm256_sub_pd(column,c1);
        const auto p0=_mm256_add_pd(_mm256_mul_pd(dr0,dr),_mm256_mul_pd(dc0,dc));
        const auto p1=_mm256_sub_pd(_mm256_mul_pd(_mm256_xor_pd(dr1,sign),dr),_mm256_mul_pd(dc1,dc));
        const auto perp=_mm256_and_pd(_mm256_cmp_pd(p0,zero,_CMP_GT_OQ),_mm256_cmp_pd(p1,zero,_CMP_GT_OQ));
        // Mask unused branches before arithmetic: do not invent overflow or
        // underflow flags from values the source's boolean slices never use.
        const auto pr=_mm256_and_pd(perp,row),pc=_mm256_and_pd(perp,column);
        const auto perpendicular=_mm256_andnot_pd(sign,_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(pr,cosine),_mm256_mul_pd(pc,sine)),_mm256_and_pd(perp,offset)));
        const auto er0=_mm256_andnot_pd(perp,dr0),ec0=_mm256_andnot_pd(perp,dc0);
        const auto er1=_mm256_andnot_pd(perp,dr1),ec1=_mm256_andnot_pd(perp,dc1);
        const auto first=_mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(ec0,ec0),_mm256_mul_pd(er0,er0)));
        const auto last=_mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(ec1,ec1),_mm256_mul_pd(er1,er1)));
        const auto value=_mm256_blendv_pd(_mm256_min_pd(first,last),perpendicular,perp);
        _mm256_storeu_pd(distances+begin+i,value);
    }
    return i;
}
}
