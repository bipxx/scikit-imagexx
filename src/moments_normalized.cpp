#include "moments_normalized.hpp"
#include "skimagexx/image.hpp"

#include <cmath>
#include <limits>

namespace skimagexx::detail {
namespace {

template<class T>
void normalize(const T* input, bool float32_power, T* out) {
    double denominator[7] = {1., 1.};
    for (int degree = 2; degree <= 6; ++degree) {
        const double exponent = degree / 2. + 1.;
        denominator[degree] = float32_power
            ? double(std::pow(float(input[0]), float(exponent)))
            : std::pow(double(input[0]), exponent);
    }
#ifdef SKIMAGEXX_AVX2
    if (has_avx2()) simd::moments_normalize_default(input, denominator, out);
    else
#endif
    for (int row = 0; row != 4; ++row)
        for (int col = 0; col != 4; ++col)
            out[4 * row + col] = T(double(input[4 * row + col]) / denominator[row + col]);
    // The source first division by scale**degree is division by exactly one.
    // The three extra finite divisions above use denominator one and raise no
    // numerical exception; these source-defined low-degree cells are NaN.
    out[0] = out[1] = out[4] = std::numeric_limits<T>::quiet_NaN();
}

}  // namespace

void moments_normalize_default(const float* input, bool float32_power, float* out) {
    normalize(input, float32_power, out);
}

void moments_normalize_default(const double* input, bool float32_power, double* out) {
    normalize(input, float32_power, out);
}

}  // namespace skimagexx::detail
