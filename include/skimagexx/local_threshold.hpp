#pragma once
#include "convert.hpp"
#include <cstddef>
#include <vector>
#include <limits>
namespace skimagexx {
enum class LocalThreshold { niblack, sauvola };
struct LocalThresholdParameters {
    double k=0.2, r=std::numeric_limits<double>::quiet_NaN(); // NaN r selects half the dtype range.
    bool divide_double=false, r_double=false, k_double=false;
};
// Contiguous C-order buffers. Float32 inputs use float32 statistics; other
// supported real input types use float64. Strong scalar parameters can promote
// the final output. Padding and cumulative sums follow upstream axis order.
void local_threshold_into(const void* input,void* output,PixelType input_type,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& window,
    LocalThreshold method,const LocalThresholdParameters& parameters={});
}
