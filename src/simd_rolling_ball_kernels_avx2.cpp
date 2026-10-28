#include "rolling_ball_kernels_simd.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd { namespace {
using V=__m256d;
V sum(V last,V prefix,V left,V pair45,V lane6,bool spine,bool add,V zero){
    if(!spine)return _mm256_add_pd(zero,_mm256_add_pd(prefix,last));
    const auto lane7=add?_mm256_add_pd(prefix,last):last;
    return _mm256_add_pd(zero,_mm256_add_pd(left,_mm256_add_pd(pair45,_mm256_add_pd(lane6,lane7))));
}
V finish(V value,bool ball,double coefficient,double radius,V zero){
    const auto scaling=_mm256_sub_pd(_mm256_set1_pd(ball?coefficient:1.),value);
    const auto height=_mm256_sqrt_pd(_mm256_max_pd(scaling,zero));
    const auto result=ball?height:_mm256_mul_pd(_mm256_set1_pd(coefficient),height);
    const auto outside=ball?_mm256_cmp_pd(_mm256_sqrt_pd(value),_mm256_set1_pd(radius),_CMP_GT_OQ):_mm256_cmp_pd(scaling,zero,_CMP_LT_OQ);
    return _mm256_blendv_pd(result,_mm256_set1_pd(std::numeric_limits<double>::infinity()),outside);
}
}
std::size_t rolling_kernel_row(const double* last,double* output,std::size_t n,
    const detail::RollingRowPlan& p,bool ball,double coefficient,double radius,double identity){
    const auto zero=_mm256_set1_pd(identity),prefix=_mm256_set1_pd(p.prefix),left=_mm256_set1_pd(p.left),pair45=_mm256_set1_pd(p.pair45),lane6=_mm256_set1_pd(p.lane6);
    std::size_t i=0;for(;n-i>=4;i+=4)_mm256_storeu_pd(output+i,finish(sum(_mm256_loadu_pd(last+i),prefix,left,pair45,lane6,p.spine,p.add_lane7,zero),ball,coefficient,radius,zero));return i;
}
void rolling_kernel_four_rows(const double* last,double* output,std::size_t n,
    const detail::RollingRowPlan* p,bool ball,double coefficient,double radius,double identity){
    const auto prefix=_mm256_setr_pd(p[0].prefix,p[1].prefix,p[2].prefix,p[3].prefix),left=_mm256_setr_pd(p[0].left,p[1].left,p[2].left,p[3].left),pair45=_mm256_setr_pd(p[0].pair45,p[1].pair45,p[2].pair45,p[3].pair45),lane6=_mm256_setr_pd(p[0].lane6,p[1].lane6,p[2].lane6,p[3].lane6),zero=_mm256_set1_pd(identity);
    for(std::size_t i=0;i<n;++i){alignas(32)double values[4];_mm256_store_pd(values,finish(sum(_mm256_set1_pd(last[i]),prefix,left,pair45,lane6,p[0].spine,p[0].add_lane7,zero),ball,coefficient,radius,zero));for(std::size_t lane=0;lane<4;++lane)output[lane*n+i]=values[lane];}
}
}
