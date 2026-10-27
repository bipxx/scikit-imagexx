#include "polygon.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
namespace {
// Inactive rays use exact zero products and a unit denominator. In particular,
// a lane which has returned VERTEX never evaluates a later nonfinite edge.
__m256d crossing(__m256d x0,__m256d y0,__m256d x1,__m256d y1,__m256d mask) {
    x0=_mm256_and_pd(x0,mask); y0=_mm256_and_pd(y0,mask);
    x1=_mm256_and_pd(x1,mask); y1=_mm256_and_pd(y1,mask);
    const auto numerator=_mm256_sub_pd(_mm256_mul_pd(x0,y1),_mm256_mul_pd(x1,y0));
    const auto denominator=_mm256_blendv_pd(_mm256_set1_pd(1.),_mm256_sub_pd(y1,y0),mask);
    return _mm256_div_pd(numerator,denominator);
}
void labels4(const detail::PolygonView& polygon,__m256d x,__m256d y,std::uint8_t* output) {
    const auto zero=_mm256_setzero_pd(),eps=_mm256_set1_pd(1e-12),minus_eps=_mm256_set1_pd(-1e-12);
    const auto all=_mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    auto active=all,vertices=zero,right=zero,left=zero;
    auto x1=_mm256_sub_pd(_mm256_set1_pd(polygon.x[polygon.vertices-1]),x);
    auto y1=_mm256_sub_pd(_mm256_set1_pd(polygon.y[polygon.vertices-1]),y);
    for (std::size_t i=0; i<polygon.vertices; ++i) {
        const auto x0=_mm256_sub_pd(_mm256_and_pd(_mm256_set1_pd(polygon.x[i]),active),_mm256_and_pd(x,active));
        const auto y0=_mm256_sub_pd(_mm256_and_pd(_mm256_set1_pd(polygon.y[i]),active),_mm256_and_pd(y,active));
        const auto near_x=_mm256_and_pd(_mm256_cmp_pd(x0,minus_eps,_CMP_GT_OQ),_mm256_cmp_pd(x0,eps,_CMP_LT_OQ));
        const auto near_y=_mm256_and_pd(_mm256_cmp_pd(y0,minus_eps,_CMP_GT_OQ),_mm256_cmp_pd(y0,eps,_CMP_LT_OQ));
        const auto vertex=_mm256_and_pd(active,_mm256_and_pd(near_x,near_y));
        vertices=_mm256_or_pd(vertices,vertex);
        active=_mm256_andnot_pd(vertex,active);
        if (!_mm256_movemask_pd(active)) break;
        auto straddles=_mm256_and_pd(active,_mm256_xor_pd(_mm256_cmp_pd(y0,zero,_CMP_GT_OQ),_mm256_cmp_pd(y1,zero,_CMP_GT_OQ)));
        right=_mm256_xor_pd(right,_mm256_and_pd(straddles,_mm256_cmp_pd(crossing(x0,y0,x1,y1,straddles),zero,_CMP_GT_OQ)));
        straddles=_mm256_and_pd(active,_mm256_xor_pd(_mm256_cmp_pd(y0,zero,_CMP_LT_OQ),_mm256_cmp_pd(y1,zero,_CMP_LT_OQ)));
        left=_mm256_xor_pd(left,_mm256_and_pd(straddles,_mm256_cmp_pd(crossing(x0,y0,x1,y1,straddles),zero,_CMP_LT_OQ)));
        x1=x0; y1=y0;
    }
    const unsigned vertex_bits=unsigned(_mm256_movemask_pd(vertices));
    const unsigned right_bits=unsigned(_mm256_movemask_pd(right)),left_bits=unsigned(_mm256_movemask_pd(left));
    for (unsigned lane=0; lane<4; ++lane)
        output[lane]=(vertex_bits&(1u<<lane))?2:(((right_bits^left_bits)&(1u<<lane))?3:((right_bits>>lane)&1u));
}
}

void polygon_points(const detail::PolygonView& polygon,const double* x,const double* y,
                    std::uint8_t* output,std::size_t begin,std::size_t end) {
    auto i=begin;
    for (; end-i>=4; i+=4) labels4(polygon,_mm256_loadu_pd(x+i),_mm256_loadu_pd(y+i),output+i);
    for (; i<end; ++i) output[i]=detail::polygon_label(polygon,x[i],y[i]);
}
void polygon_grid(const detail::PolygonView& polygon,std::size_t columns,
                  std::uint8_t* output,std::size_t begin,std::size_t end) {
    auto i=begin;
    for (; end-i>=4; i+=4) {
        const auto x=_mm256_set_pd(double((i+3)/columns),double((i+2)/columns),double((i+1)/columns),double(i/columns));
        const auto y=_mm256_set_pd(double((i+3)%columns),double((i+2)%columns),double((i+1)%columns),double(i%columns));
        labels4(polygon,x,y,output+i);
    }
    for (; i<end; ++i) output[i]=detail::polygon_label(polygon,double(i/columns),double(i%columns));
}
}
