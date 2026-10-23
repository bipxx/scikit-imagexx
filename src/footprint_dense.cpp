#include "skimagexx/footprint_dense.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx {
void footprint_dense_into(void* output, std::size_t count, std::size_t itemsize,
                          const void* zero, const void* one, const double* axis,
                          std::size_t width, DenseFootprint kind, double limit) {
    if(kind < DenseFootprint::rectangle || kind > DenseFootprint::ball)
        throw std::invalid_argument("invalid dense footprint kind");
    const auto max=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(itemsize && count>max/itemsize)throw std::overflow_error("dense footprint storage overflow");
    if(kind!=DenseFootprint::rectangle) {
        std::size_t expected=width;
        const int dimensions=kind==DenseFootprint::ball ? 3:2;
        for(int d=1;d<dimensions;++d) {
            if(width && expected>max/width)throw std::overflow_error("dense footprint shape overflow");
            expected*=width;
        }
        if(count!=expected)throw std::invalid_argument("dense footprint shape mismatch");
    }
    if(!count || !itemsize)return;
    if(!output || !zero || !one || (kind!=DenseFootprint::rectangle && !axis))
        throw std::invalid_argument("nonnull dense footprint buffers required");
    if(output==zero || output==one || output==axis)
        throw std::invalid_argument("disjoint dense footprint buffers required");
    auto* bytes=static_cast<unsigned char*>(output);
    if(kind==DenseFootprint::rectangle) {
        // Seed once per worker, then exponentially copy the completed prefix.
        // This reaches the platform's optimized memcpy for every dtype width.
        detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
            if(begin==end)return;
            auto* target=bytes+begin*itemsize;
            std::memcpy(target,one,itemsize);
            auto filled=std::size_t{1};
            while(filled<end-begin) {
                const auto chunk=std::min(filled,end-begin-filled);
                std::memcpy(target+filled*itemsize,target,chunk*itemsize);filled+=chunk;
            }
        });
        return;
    }
    const auto rows=count/width;
    const bool ball=kind==DenseFootprint::ball;
    detail::partition(rows,detail::active_workers(count),[&](auto begin,auto end,auto) {
        for(auto row=begin;row<end;++row) {
            const auto middle=axis[row%width],outer=ball ? axis[row/width]:0.;
            auto* target=bytes+row*width*itemsize;
            std::size_t column=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())column=simd::footprint_dense_row(target,itemsize,zero,one,axis,width,middle,outer,ball,limit);
#endif
            for(;column<width;++column) {
                auto cost=axis[column]+middle;
                if(ball)cost+=outer;
                std::memcpy(target+column*itemsize,cost<=limit ? one:zero,itemsize);
            }
        }
    });
}
}
