#pragma once
#include "convert.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
namespace skimagexx {
enum class RankOperation { minimum=0,maximum=1,mean=2,sum=3,pop=4,gradient=5,median=6,modal=7,entropy=8,autolevel=9,equalize=10,enhance_contrast=11,threshold=12,geometric_mean=13,noise_filter=14,subtract_mean=15,windowed_histogram=16,autolevel_percentile=17,enhance_contrast_percentile=18,gradient_percentile=19,mean_percentile=20,pop_percentile=21,subtract_mean_percentile=22,threshold_percentile=23,sum_percentile=24,percentile=25 };
// Capabilities come from the linked implementation, not just a header version.
std::uint32_t rank_histogram_capabilities() noexcept;
// Prepared C-order image/mask and binary footprint;2D uses a singleton plane.
// Output spatial strides may include a larger destination and pixel vectors.
// The Python bridge validates accessible shapes and retains all owners.
struct RankView {
    const void* image=nullptr;
    const std::uint8_t* mask=nullptr;
    const std::uint8_t* footprint=nullptr;
    void* output=nullptr;
    std::array<std::size_t,3> shape{},footprint_shape{};
    std::array<std::ptrdiff_t,3> center{},output_strides{};
    PixelType output_type=PixelType::uint8;
    std::size_t bins=256;
    bool uint16=false,serial_alias=false;
};
void rank_filter_into(const RankView&,RankOperation);
// rank-statistics-v1: disjoint prepared buffers only. The caller performs the
// noise footprint copy/center-clear stage before constructing this view.
std::uint32_t rank_statistics_capabilities() noexcept;
void rank_statistics_into(const RankView&,RankOperation);
// rank-vector-v1: singleton-plane 2D, contiguous channels of positive depth
// no greater than bins. Spatial strides remain bytes; RankView ABI is unchanged.
std::uint32_t rank_vector_capabilities() noexcept;
void rank_windowed_histogram_into(const RankView&,std::size_t output_depth);
// rank-percentile-v1: separate parameterized 2D/disjoint entry. Finite
// 0<=p0<=p1<=1; percentile/threshold use p0 only (p1 still must be finite and
// in range). Integer CDF/quotient semantics, no interpolated quantiles.
std::uint32_t rank_percentile_capabilities() noexcept;
void rank_percentile_into(const RankView&,RankOperation,double p0,double p1);
// Actual accessed stores, after image/mask normalization. Distinct from the
// conservative legacy serial_alias flag; the RankView ABI is unchanged.
struct RankAliases {bool image_output=false,mask_output=false;};
RankAliases rank_aliases(const RankView&);
void rank_filter_into(const RankView&,RankOperation,RankAliases);
// Operation bits only; each call still certifies its byte/index safety class.
std::uint32_t rank_alias_capabilities() noexcept;
std::uint16_t rank_max_u16(const std::uint16_t*,std::size_t);
// Binary16 multiplication/rint/clip follows img_as_ubyte's half precision.
struct RankHalfConversion {bool nan=false,invalid_multiply=false,overflow=false;};
RankHalfConversion rank_half_u8(const std::uint16_t*,std::uint8_t*,std::size_t);
}
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::uint64_t rank_sum(const void*,std::size_t,bool);
std::int64_t rank_count_prefix(const std::int64_t*,std::size_t);
bool rank_uniform(const void*,bool,std::uint32_t&);
std::uint16_t rank_max_u16(const std::uint16_t*,std::size_t);
std::size_t rank_positive(const void*,std::uint8_t*,std::size_t,PixelType);
// Counts must be nonnegative and <= INT32_MAX; returns the processed prefix.
std::size_t rank_histogram_normalized(const std::int64_t*,void*,std::size_t,double,bool);
}
#endif
