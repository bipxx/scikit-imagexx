#include "unique_rows.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>

namespace skimagexx::detail {
std::size_t unique_rows_indices(const UniqueRowsInput& input,
                               std::int64_t* output,int workers) {
    const auto count=input.row_count,bytes=input.row_bytes;
    if(!count)return 0;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=bytes>=32 && has_avx2();
#endif
    const auto compare=[&](std::int64_t a,std::int64_t b) {
        const auto* first=input.rows+std::size_t(a)*bytes;
        const auto* second=input.rows+std::size_t(b)*bytes;
#ifdef SKIMAGEXX_AVX2
        if(vectorized)return simd::unique_rows_compare(first,second,bytes);
#endif
        return std::memcmp(first,second,bytes);
    };
    const auto less=[&](std::int64_t a,std::int64_t b) {
        const auto order=compare(a,b);return order<0 || (order==0 && a<b);
    };
    const auto chunks=input.row_count*bytes>=parallel_min?
        std::min<std::size_t>(std::size_t(workers),count):std::size_t(1);
    const auto block=count/chunks,extra=count%chunks;
    const auto boundary=[&](std::size_t chunk) {
        return block*chunk+std::min(chunk,extra);
    };
    partition(chunks,int(chunks),[&](auto begin,auto end,auto) {
        for(auto chunk=begin;chunk<end;++chunk) {
            const auto first=boundary(chunk),last=boundary(chunk+1);
            for(auto i=first;i<last;++i)output[i]=std::int64_t(i);
            std::sort(output+first,output+last,less);
        }
    });
    for(std::size_t span=1;span<chunks;span*=2)
        for(std::size_t first=0;first<chunks;first+=span*2) {
            const auto middle=std::min(first+span,chunks);
            const auto last=std::min(first+span*2,chunks);
            if(middle<last)std::inplace_merge(output+boundary(first),
                output+boundary(middle),output+boundary(last),less);
        }
    const auto end=std::unique(output,output+count,[&](auto a,auto b) {return compare(a,b)==0;});
    return std::size_t(end-output);
}
}
