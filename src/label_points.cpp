#include "label_points.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>

namespace skimagexx::detail {
void label_points_into(const LabelPointsInput& input,std::uint64_t* output,
                       int workers) {
    if(!input.output_count)return;
    const bool vectorized=has_avx2();
    partition(input.output_count,input.output_count>=parallel_min?workers:1,
              [&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(vectorized && end-begin>=16)
            simd::label_points_zero(output+begin,end-begin);
        else
#else
        (void)vectorized;
#endif
            std::fill_n(output+begin,end-begin,std::uint64_t(0));
    });
    // partition has completed before any point store. Repeated destinations
    // must see the same last assignment as the original ordered index array.
    for(std::size_t point=0;point<input.point_count;++point) {
        std::size_t offset=0;
        for(std::size_t axis=0;axis<input.ndim;++axis) {
            std::int64_t index;
            const auto* address=reinterpret_cast<const char*>(input.axes[axis])
                              +std::ptrdiff_t(point)*input.axis_strides[axis];
            std::memcpy(&index,address,sizeof index);
            offset=offset*input.dimensions[axis]+std::size_t(index);
        }
        output[offset]=std::uint64_t(point)+1;
    }
}
}
