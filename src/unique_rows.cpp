#include "unique_rows.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace skimagexx::detail {
std::size_t unique_rows_indices(const UniqueRowsInput& input,
                               std::int64_t* output,int workers) {
    const auto count=input.row_count,bytes=input.row_bytes;
    if(!count)return 0;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=bytes>=32 && has_avx2();
#endif
    const auto raw_compare=[&](std::int64_t a,std::int64_t b) {
        const auto* first=input.rows+std::size_t(a)*bytes;
        const auto* second=input.rows+std::size_t(b)*bytes;
#ifdef SKIMAGEXX_AVX2
        if(vectorized)return simd::unique_rows_compare(first,second,bytes);
#endif
        return std::memcmp(first,second,bytes);
    };
    // Equal rows need only the first index, not a sort of all index ties.
    std::size_t different=1;
    for(;different<count;++different)
        if(raw_compare(0,std::int64_t(different))!=0)break;
    if(different==count){output[0]=0;return 1;}
    const auto sorted_unique=[&](const auto& compare) {
        const auto less=[&](std::int64_t a,std::int64_t b) {return compare(a,b)<0;};
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
        // Equal rows need no internal index ordering. Keep the first input
        // occurrence by taking the minimum over each complete merged group.
        std::size_t unique_count=0;
        auto minimum=output[0];
        for(std::size_t row=1;row<count;++row) {
            const auto current=output[row];
            if(compare(minimum,current)==0)minimum=std::min(minimum,current);
            else {output[unique_count++]=minimum;minimum=current;}
        }
        output[unique_count++]=minimum;
        return unique_count;
    };
    // Keep wide-row SIMD sorting untouched, without key allocation/loads.
    if(bytes>16)return sorted_unique(raw_compare);
    std::vector<std::uint64_t> keys(count);
    const auto prefix=std::min<std::size_t>(bytes,8);
    for(std::size_t row=0;row<count;++row) {
        std::uint64_t key=0;
        for(std::size_t byte=0;byte<prefix;++byte)
            key=(key<<8)|input.rows[row*bytes+byte];
        keys[row]=key;
    }
    const auto compare=[&](std::int64_t a,std::int64_t b) {
        const auto first=keys[std::size_t(a)],second=keys[std::size_t(b)];
        if(first!=second)return first<second?-1:1;
        if(bytes<=8)return 0;
        return std::memcmp(input.rows+std::size_t(a)*bytes+8,
                           input.rows+std::size_t(b)*bytes+8,bytes-8);
    };
    return sorted_unique(compare);
}
}
