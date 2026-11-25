#include <skimagexx/tomography.hpp>

#include <array>
#include <cmath>

int main() {
    using namespace skimagexx;
    constexpr int side = 8;
    const RadonRotation rotation{1, 0, 0, 0};
    std::array<double, side * side> phantom{};
    phantom[4 * side + 4] = 1;
    if (radon_detector_size(side, side + 1, true) != side ||
        radon_detector_size(side, side + 1, false) != 13 ||
        radon_outside_circle(phantom.data(), side, side)) return 1;
    std::array<double, side> sinogram{};
    radon_into(phantom.data(), side, side, &rotation, 1, true, sinogram.data());
    for (int r = 0; r < side; ++r) {
        if (sinogram[r] != (r == 4 ? 1 : 0)) return 2;
    }
    std::array<double, side * side> inverse{};
    const double cosine = 1, sine = 0;
    iradon_into(sinogram.data(), side, 1, &cosine, &sine, side, TomographyFilter::none,
                TomographyInterpolation::cubic, true, inverse.data());
    if (std::abs(inverse[4 * side + 4] - 1.5707963267948966) > 1e-12 ||
        std::abs(inverse[4 * side + 3]) > 1e-12 || inverse.front() != 0) return 3;
    const double theta = 0, shift = 0;
    std::array<double, side * side> estimate{};
    iradon_sart_into(sinogram.data(), side, 1, &theta, &shift, 0.15, true, estimate.data());
    if (estimate[4 * side + 4] <= 0 || estimate.front() != 0) return 4;
    return 0;
}
