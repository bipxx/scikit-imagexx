#include "label_points.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>

namespace skimagexx::detail {
bool label_points_into(const LabelPointsInput& input,std::uint64_t* output,
                       int workers) {
    if(!input.output_count)return label_points_assign(input,output,workers);
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
    return label_points_assign(input,output,workers);
}
bool label_points_assign(const LabelPointsInput& input,std::uint64_t* output,
                         int workers) {
    (void)workers;
    if(!input.output_count)return input.point_count==0;
    // Any constructor clear has completed before these ordered point stores. Repeated destinations
    // must see the same last assignment as the original ordered index array.
    const auto read=[](const std::int64_t* axis,std::ptrdiff_t stride,std::size_t point) {
        std::int64_t value;
        std::memcpy(&value,reinterpret_cast<const char*>(axis)+std::ptrdiff_t(point)*stride,sizeof value);
        return std::uint64_t(value);
    };
    if(input.ndim==2) {
        const auto* first=input.axes[0];const auto* second=input.axes[1];
        const auto fs=input.axis_strides[0],ss=input.axis_strides[1];
        const auto height=input.dimensions[0],width=input.dimensions[1];
        for(std::size_t point=0;point<input.point_count;++point) {
            const auto row=read(first,fs,point),column=read(second,ss,point);
            if(row>=height || column>=width)return false;
            output[row*width+column]=std::uint64_t(point)+1;
        }
        return true;
    }
    if(input.ndim==3) {
        const auto* first=input.axes[0];const auto* second=input.axes[1];const auto* third=input.axes[2];
        const auto fs=input.axis_strides[0],ss=input.axis_strides[1],ts=input.axis_strides[2];
        const auto depth=input.dimensions[0],height=input.dimensions[1],width=input.dimensions[2];
        for(std::size_t point=0;point<input.point_count;++point) {
            const auto plane=read(first,fs,point),row=read(second,ss,point),column=read(third,ts,point);
            if(plane>=depth || row>=height || column>=width)return false;
            output[(plane*height+row)*width+column]=std::uint64_t(point)+1;
        }
        return true;
    }
    for(std::size_t point=0;point<input.point_count;++point) {
        std::size_t offset=0;
        for(std::size_t axis=0;axis<input.ndim;++axis) {
            const auto index=read(input.axes[axis],input.axis_strides[axis],point);
            if(index>=input.dimensions[axis])return false;
            offset=offset*input.dimensions[axis]+index;
        }
        output[offset]=std::uint64_t(point)+1;
    }
    return true;
}
}
