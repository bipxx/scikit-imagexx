#pragma once
#include "convert.hpp"
#include <vector>
namespace skimagexx {
struct MetricImageView {
    const void* data=nullptr;
    PixelType type=PixelType::float64;
    bool half=false,swap=false; // Binary16 storage uses type=uint16, half=true.
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
};
enum class MetricNormalization { none,euclidean,mean,min_max };
struct MetricSums {
    double error=0,energy=0,mean=0,low=0,high=0;
    // Per-operation floating status: overflow=1, underflow=2, invalid=4.
    unsigned subtract_flags=0,square_flags=0,error_reduce_flags=0;
    unsigned energy_flags=0,energy_reduce_flags=0,mean_reduce_flags=0;
    unsigned cast0_flags=0,cast1_flags=0;
    bool nonfinite=false,valid=false,nan=false,range_signaling=false;
};
// Bounds occupy the first storage-width bytes of each word in native byte
// order; bounds are unspecified when has_nan is true.
struct MetricRawRange {std::uint64_t low=0,high=0;bool has_nan=false;};
// Fused raw-value conversion/subtract/square and ordered pairwise reductions.
// Views have equal shapes; either arbitrary byte strides/endianness are allowed.
// single=true requests float32 subtraction/products (half/F32 inputs only).
// Other inputs use float64. Optional workers preserve the fixed reduction tree.
// reduction_block reproduces legacy NumPy iterator-buffer reduction order.
// scalar_power_double reproduces NumPy1.x's 0-D float32 ** Python-int policy.
MetricSums metric_sums(const MetricImageView&,const MetricImageView&,bool single,
                       MetricNormalization,bool capture_underflow=false,bool scalar_power_double=false,
                       std::size_t reduction_block=0);
MetricRawRange metric_raw_range(const MetricImageView&);
namespace detail {
double metric_read(const MetricImageView&,std::ptrdiff_t byte_offset);
}
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t metric_values(const MetricImageView&,const MetricImageView&,
    std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t,std::size_t,bool,
    MetricNormalization,double*,double*,double*);
}
#endif
