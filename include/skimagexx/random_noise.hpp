#pragma once

#include <cstddef>

namespace skimagexx {

struct RandomNoiseDraw {
    const double* values;
    std::size_t count;
    double probability;
};

// Combine prepared float64 images and genuine provider noise. Both inputs
// have count elements. Output has the same size and cannot overlap inputs.
// For speckle, compile with contraction disabled to preserve the product
// rounding before addition. Use round-to-nearest.
void random_noise_normal_into(const double* image, const double* noise, double* output,
    std::size_t count, bool speckle);

// Apply two prepared Bernoulli draws. At exact probability 0 or 1, pass a
// null buffer with count zero. Otherwise pass count provider uniforms.
// All spans refer to at most 524288 elements. Invalid input throws before
// writes. Clipping is a separate provider operation.
void random_noise_impulse_into(const double* image, double* output, std::size_t count,
    RandomNoiseDraw flipped, RandomNoiseDraw salted, double low);

}  // namespace skimagexx
