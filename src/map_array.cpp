#include "map_array.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace skimagexx::detail {
void map_array_into(const MapArrayInput& input,MapArrayOutput output,int workers) {
    if(!input.count)return;
    const auto bytes=input.itemsize;
    const auto mask=input.index_bits==64?~std::uint64_t(0):(std::uint64_t(1)<<input.index_bits)-1;
    // The source hash map owns its values, not pointers into the public array.
    // Snapshot before ANY output store, including when output is output_vals.
    std::vector<unsigned char> values(input.key_count*bytes);
    for(std::size_t k=0;k<input.key_count;++k)
        std::memcpy(values.data()+k*bytes,static_cast<const char*>(input.values)+std::ptrdiff_t(k)*input.values_stride,bytes);
    std::uint64_t low=std::numeric_limits<std::uint64_t>::max(),high=0;
    for(std::size_t k=0;k<input.key_count;++k){const auto key=input.keys[k]&mask;low=std::min(low,key);high=std::max(high,key);}
    const auto range=high-low;
    const bool dense=input.key_count && range<65536 && range+1<=std::max<std::size_t>(input.count,256);
    const bool contiguous=output.stride==std::ptrdiff_t(bytes);
    const bool vectorized=has_avx2();
    const auto threads=input.count>=parallel_min?workers:1;
    if(dense) {
        const auto cells=std::size_t(range+1),slot=std::max<std::size_t>(bytes,4);
        std::vector<std::uint64_t> table(((cells+4)*slot+7)/8,0);
        auto* raw=reinterpret_cast<unsigned char*>(table.data());
        for(std::size_t k=0;k<input.key_count;++k)
            std::memcpy(raw+std::size_t((input.keys[k]&mask)-low)*slot,values.data()+k*bytes,bytes);
        partition(input.count,threads,[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
            if(contiguous && vectorized)begin+=simd::map_array_dense(input.indices+begin,end-begin,mask,low,cells,raw,bytes,static_cast<char*>(output.data)+begin*bytes);
#endif
            for(auto i=begin;i<end;++i) {
                const auto index=(input.indices[i]&mask)-low;
                auto* target=static_cast<char*>(output.data)+std::ptrdiff_t(i)*output.stride;
                if(index<cells)std::memcpy(target,raw+std::size_t(index)*slot,bytes);
                else std::memset(target,0,bytes);
            }
        });
        return;
    }
    // Separate from consumer_map_into: that older helper reads live values and
    // uses the global worker getter. This route uses snapshots and this call's
    // explicit worker setting, while retaining last duplicate-key assignment.
    std::unordered_map<std::uint64_t,std::size_t> lookup;lookup.reserve(input.key_count);
    for(std::size_t k=0;k<input.key_count;++k)lookup[input.keys[k]&mask]=k;
    const auto& frozen=lookup;
    partition(input.count,threads,[&](auto begin,auto end,auto) {
        for(auto i=begin;i<end;++i) {
            const auto found=frozen.find(input.indices[i]&mask);
            auto* target=static_cast<char*>(output.data)+std::ptrdiff_t(i)*output.stride;
            if(found==frozen.end())std::memset(target,0,bytes);
            else std::memcpy(target,values.data()+found->second*bytes,bytes);
        }
    });
}
}
