#include "skimagexx/restoration_denoise.hpp"

#include <array>
#include <cmath>
#include <cstdint>

int main() {
    constexpr std::size_t side = 7;
    constexpr auto size = side * side;
    std::array<double, size> image{};
    std::array<double, size> output{};
    for (std::size_t p = 0; p < size; ++p)
        image[p] = 0.25 + (static_cast<double>(p % side) + static_cast<double>(p / side)) / 25;
    const auto close = [](double lhs, double rhs) { return std::abs(lhs - rhs) < 1e-10; };

    if (!skimagexx::denoise_nl_means(image.data(), output.data(), 1, side, side, 1,
                                    false, 3, 0, 0.1, 0.0)) return 1;
    for (std::size_t p = 0; p < size; ++p)
        if (!close(image[p], output[p])) return 2;

    if (!std::isfinite(skimagexx::estimate_sigma(image.data(), side, side, 1, 0, true))) return 3;
    if (!skimagexx::denoise_wavelet(image.data(), output.data(), side, side, 1,
                                   true, 4, 1, 0.0, false, false, false)) return 4;
    for (std::size_t p = 0; p < size; ++p)
        if (!close(image[p], output[p])) return 5;

    std::array<std::uint8_t, size> mask{};
    mask[size / 2] = 1;
    if (!skimagexx::inpaint_biharmonic(image.data(), mask.data(), output.data(), side, side, 1)) return 6;
    if (!close(output[size / 2], image[size / 2])) return 7;

    constexpr std::array<double, 4> phase{2.7, 3.0, -3.0, -2.7};
    if (!skimagexx::unwrap_phase(phase.data(), output.data(), 1, 1, phase.size(), 1, true)) return 8;
    if (!close(output[2], 3.283185307179586)) return 9;

    constexpr double identity_psf = 1.0;
    if (!skimagexx::richardson_lucy(image.data(), &identity_psf, output.data(),
                                    side, side, 1, 1, 2, false, 0.0)) return 10;
    for (std::size_t p = 0; p < size; ++p)
        if (!close(image[p], output[p])) return 11;
}
