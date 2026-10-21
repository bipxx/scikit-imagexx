#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
enum class PixelType { boolean, uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float64 };
struct ConversionInfo {
    bool downcast_without_scaling = false;
    bool nan_to_integer = false;
    bool floating_overflow = false;
    bool floating_underflow = false;
    bool invalid_float_cast = false;
    std::int64_t maximum_signed = 0;
    std::uint64_t maximum_unsigned = 0;
};
// Native scikit-image conversion rules. Contiguous, properly aligned buffers;
// output must have count elements of the target type and must not overlap input.
// Empty buffers are allowed. Public image targets: bool/u8/u16/i16/f32/f64.
// Metadata lets bindings reproduce data-dependent warnings without a second scan.
ConversionInfo convert_into(const void* input, void* output, std::size_t count,
                            PixelType source, PixelType target);
std::size_t pixel_size(PixelType type);
} // namespace skimagexx
