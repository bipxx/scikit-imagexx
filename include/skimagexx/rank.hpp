#pragma once
#include "convert.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
namespace skimagexx {
enum class RankOperation { minimum=0,maximum=1,mean=2,sum=3,pop=4,gradient=5,median=6,modal=7,entropy=8,autolevel=9,equalize=10,enhance_contrast=11,threshold=12 };
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
}
#endif
