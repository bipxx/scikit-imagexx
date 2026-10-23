// Independent output lanes preserve ndimage's axis/corner multiplication and
// summation order. BSD reference attribution: scipy-ndimage-LICENSE.txt.
#include "resample_detail.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
std::size_t zoom_sample(const double* input,double* output,const detail::ZoomPlan& plan,
                        std::size_t begin,std::size_t end) {
    // NumPy has at most 64 image axes. This fixed coordinate scratch stays on
    // each worker stack and avoids allocations inside the parallel region.
    if(plan.axes.size()>64)return begin;
    const auto rank=plan.axes.size();
    const detail::ZoomPoint* points[64][4];
    const auto zero=_mm256_setzero_si256();
    for(;end-begin>=4;begin+=4) {
        for(std::size_t axis=0;axis<rank;++axis)for(std::size_t lane=0;lane<4;++lane)
            points[axis][lane]=&plan.axes[axis][((begin+lane)/plan.output_strides[axis])%plan.axes[axis].size()];
        auto total=_mm256_setzero_pd();
        for(std::size_t corner=0;corner<plan.corners;++corner) {
            const auto* taps=plan.footprint.data()+corner*rank;
            auto index=zero,valid=_mm256_cmpeq_epi64(zero,zero);
            for(std::size_t axis=0;axis<rank;++axis) {
                const auto tap=taps[axis];
                const auto offsets=_mm256_set_epi64x(points[axis][3]->offsets[tap],points[axis][2]->offsets[tap],
                    points[axis][1]->offsets[tap],points[axis][0]->offsets[tap]);
                const auto positive=_mm256_cmpgt_epi64(offsets,_mm256_set1_epi64x(-1));
                valid=_mm256_and_si256(valid,positive);
                index=_mm256_add_epi64(index,_mm256_and_si256(offsets,positive));
            }
            auto coefficient=_mm256_mask_i64gather_pd(_mm256_set1_pd(plan.cval),input,index,_mm256_castsi256_pd(valid),8);
            if(plan.order)for(std::size_t axis=0;axis<rank;++axis) {
                const auto tap=taps[axis];
                const auto weights=_mm256_set_pd(points[axis][3]->weights[tap],points[axis][2]->weights[tap],
                    points[axis][1]->weights[tap],points[axis][0]->weights[tap]);
                coefficient=_mm256_mul_pd(coefficient,weights);
            }
            total=_mm256_add_pd(total,coefficient);
        }
        _mm256_storeu_pd(output+begin,total);
    }
    return begin;
}
} // namespace skimagexx::simd
