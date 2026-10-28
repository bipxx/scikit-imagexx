#include "subdivide_polygon_simd.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t subdivide_polygon_points(const detail::SubdividePolygonInput& input,
    std::size_t begin,std::size_t count,const detail::SubdividePolygonWeights& weights,
    bool circular,double* output) {
    std::size_t i=0;
    for(;i+2<=count;i+=2) {
        auto even=_mm256_setzero_pd(),odd=_mm256_setzero_pd();
        for(unsigned k=0;k<weights.count;++k) {
            const auto first=detail::subdivide_polygon_index(input.count,begin+i,k,weights.count,circular);
            const auto second=detail::subdivide_polygon_index(input.count,begin+i+1,k,weights.count,circular);
            const auto a=std::ptrdiff_t(first)*input.row_stride,b=std::ptrdiff_t(second)*input.row_stride;
            const auto indices=_mm256_set_epi64x(static_cast<long long>(b+input.column_stride),static_cast<long long>(b),static_cast<long long>(a+input.column_stride),static_cast<long long>(a));
            const auto values=_mm256_i64gather_pd(input.data,indices,8);
            even=_mm256_add_pd(even,_mm256_mul_pd(_mm256_set1_pd(weights.even[k]),values));
            odd=_mm256_add_pd(odd,_mm256_mul_pd(_mm256_set1_pd(weights.odd[k]),values));
        }
        _mm256_storeu_pd(output+(begin+i)*4,_mm256_permute2f128_pd(odd,even,0x20));
        _mm256_storeu_pd(output+(begin+i)*4+4,_mm256_permute2f128_pd(odd,even,0x31));
    }
    return i;
}
}
