#pragma once
#include <cstddef>

namespace skimagexx {
// Scalar planning stays outside the image loop. Squares and normalization are
// explicit so callers can preserve their scalar promotion rules.
struct GaborParameters {
    double cosine, sine, sigma_x_squared, sigma_y_squared;
    double angular_frequency, offset, normalization;
};
// Writes interleaved real/imaginary components, float or double. Dimensions must
// be positive and odd. The global worker cap applies; one is the default.
void gabor_kernel_into(void* output, bool single, std::size_t height,
                       std::size_t width, const GaborParameters& parameters);
}
