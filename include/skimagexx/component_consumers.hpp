#pragma once
#include "components.hpp"
#include <cstdint>
#include <vector>

namespace skimagexx {
inline constexpr unsigned component_consumers_interface_version=1;
// V1 kernels receive prepared native C keys; public dtype admission, scalar
// casts and observable assignment errors belong to the NumPy C-API binding.
struct ConsumerRaster {
    void* data=nullptr;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::size_t itemsize=1;
};
struct ConsumerCounts {
    std::vector<std::uint64_t> keys;
    std::vector<std::intptr_t> sizes;
    std::uint64_t maximum=0;
    bool empty=true;
};
bool consumer_min_negative(const ComponentsView&);
void consumer_truth_into(const std::intptr_t*,std::uint8_t*,std::size_t,bool invert);
void consumer_nonzero_into(const ComponentsView&,std::uint8_t*);
void consumer_bool_write(const std::uint8_t*,const ConsumerRaster&,bool invert);
ConsumerCounts consumer_count(const std::intptr_t*,std::size_t);
void consumer_filter_into(const std::intptr_t*,const ConsumerCounts&,double threshold,const ConsumerRaster&);
void consumer_border_into(std::uint8_t*,const std::vector<std::size_t>&,
                          const std::vector<std::size_t>& start_stops,const std::vector<std::size_t>& end_starts);
std::vector<std::uint8_t> consumer_border_ids(const std::intptr_t*,const std::uint8_t*,std::size_t,std::size_t);
void consumer_clear_into(const std::intptr_t*,const std::uint8_t*,std::size_t,const void*,const ConsumerRaster&);
std::vector<std::uint64_t> consumer_sorted_unique(const std::uint64_t*,std::size_t);
// Keys and indices use modulo index width, reproducing map_array's key cast.
void consumer_map_into(const std::uint64_t* indices,std::size_t,const std::uint64_t* keys,
                       const void* values,std::size_t key_count,std::size_t itemsize,
                       unsigned index_bits,void* output);
void consumer_dense_into(const std::uint64_t* keys,const void* values,std::size_t key_count,
                         std::size_t itemsize,void* output,std::size_t length);
void consumer_scatter_into(const std::uint64_t* indices,const void* values,std::size_t count,
                           std::size_t itemsize,void* output,std::size_t length);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t consumer_truth(const std::intptr_t*,std::uint8_t*,std::size_t,bool);
std::size_t consumer_bool_write(const std::uint8_t*,std::uint8_t*,std::size_t,bool);
std::size_t consumer_or(const std::uint8_t*,std::uint8_t*,std::size_t);
// Whole32-byte transport blocks only. Callers certify disjoint C-order
// output/read buffers; scalar tails retain generic traversal/error behavior.
std::size_t consumer_filter_store(const std::intptr_t*,std::size_t,
    const std::intptr_t* sizes,std::size_t bins,double threshold,void*,std::size_t);
std::size_t consumer_clear_store(const std::intptr_t*,std::size_t,
    const std::uint8_t* remove,std::size_t number,const void* fill,void*,std::size_t,std::uint8_t& invalid);
}
#endif
