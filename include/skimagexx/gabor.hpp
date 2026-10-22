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
// Uses scalar libm transcendental functions with SIMD coordinate arithmetic.
// Useful before integer-valued convolution, where one coefficient ULP can
// change a cast result. No Python or NumPy runtime is required.
void gabor_kernel_accurate_into(void* output, bool single, std::size_t height,
                       std::size_t width, const GaborParameters& parameters);
}
