#pragma once
#include "convert.hpp"

namespace skimagexx {
enum class DistanceMetric { cie94, ciede2000, cmc };
struct DistanceParameters {
    double kL=1,kC=1,kH=1,k1=0.045,k2=0.015;
};
// NumPy 1.x promotes scalar float32 powers and Python-scalar arithmetic.
// These one-color paths retain float32 subtraction/hypot before promotion.
// Input pointers address three channels; channel strides may be signed/zero.
double delta_e76_scalar32(const float* first,const float* second,
    std::ptrdiff_t first_channels=1,std::ptrdiff_t second_channels=1);
double delta_e94_scalar32(const float* first,const float* second,
    DistanceParameters parameters={},bool k1_float32=false,bool k2_float32=false,
    std::ptrdiff_t first_channels=1,std::ptrdiff_t second_channels=1);
// Weighted Lab distances with NumPy-compatible intermediate rounding. Float32
// inputs produce float32 unless cmc_scalar_result requests the upstream CMC
// promotion for two scalar colors (the output is then float64).
// wide_trigonometry follows NumPy's radian-constant promotion; true for NumPy 2.
// legacy_ciede2000 reproduces NumPy 1.x float32 array promotion at c7+25**7:
// chroma/hue become double, while lightness and output accumulation stay float.
void delta_e_into(const void* first,const void* second,void* output,std::size_t count,
    PixelType first_type,PixelType second_type,DistanceMetric metric,
    DistanceParameters parameters={},bool wide_trigonometry=true,bool cmc_scalar_result=false,
    std::ptrdiff_t first_pixels=3,std::ptrdiff_t first_channels=1,
    std::ptrdiff_t second_pixels=3,std::ptrdiff_t second_channels=1,bool legacy_ciede2000=false);
// Raw Lab values, without integer intensity normalization. The first three
// channels participate; strides are in input elements and may be negative/zero.
// Output is float32 when both inputs are float32, float64 otherwise. Buffers
// must not overlap. count is the number of broadcasted pixel pairs.
void delta_e76_into(const void* first,const void* second,void* output,std::size_t count,
    PixelType first_type,PixelType second_type,std::ptrdiff_t first_pixels=3,
    std::ptrdiff_t first_channels=1,std::ptrdiff_t second_pixels=3,std::ptrdiff_t second_channels=1);
// Rows may be independently reversed, strided or broadcast. Row offsets and
// pixel/channel strides count input elements. Output has rows*columns values.
// The global worker limit is captured once for the whole image.
void delta_e76_rows_into(const void* first,const void* second,void* output,std::size_t rows,std::size_t columns,
    PixelType first_type,PixelType second_type,const std::ptrdiff_t* first_rows,const std::ptrdiff_t* second_rows,
    std::ptrdiff_t first_pixels=3,std::ptrdiff_t first_channels=1,
    std::ptrdiff_t second_pixels=3,std::ptrdiff_t second_channels=1);
}
