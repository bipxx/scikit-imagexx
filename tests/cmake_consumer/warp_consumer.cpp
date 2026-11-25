#include <skimagexx/warp.hpp>
#include <array>
#include <cmath>

int main() {
    using namespace skimagexx;
    const std::array<double, 6> input{1., 2., 3., 4., 5., 6.};
    std::array<double, 6> output{};
    const WarpShape source{2, 3};
    const WarpMap translation{WarpMapping::matrix, {1., 0., .5, 0., 1., 0., 0., 0., 1.}};
    WarpOptions options;
    options.clip = false;
    if (!warp_into(input.data(), output.data(), PixelType::float64, source, source,
                   translation, options) ||
        output != std::array<double, 6>{1.5, 2.5, 1.5, 4.5, 5.5, 3.}) return 1;

    options.order = 0;
    options.border = GaussianBorder::nearest;
    if (!warp_into(input.data(), output.data(), PixelType::float64, source, source,
                   translation, options) ||
        output != std::array<double, 6>{2., 3., 3., 5., 6., 6.}) return 2;

    std::array<float, 12> coordinates{};
    if (!warp_coords_into(coordinates.data(), source, false, translation) ||
        coordinates != std::array<float, 12>{0, 0, 0, 1, 1, 1, .5f, 1.5f, 2.5f, .5f, 1.5f, 2.5f})
        return 3;

    options.order = 2;
    const WarpMap identity{WarpMapping::matrix, {1., 0., 0., 0., 1., 0., 0., 0., 1.}};
    if (!warp_into(input.data(), output.data(), PixelType::float64, source, source,
                   identity, options) || std::abs(output[0] - input[0]) > 1e-9) return 4;

    const WarpMap polar{WarpMapping::polar_linear, {0., 0., 1., 2.}};
    options.order = 1;
    if (!warp_into(input.data(), output.data(), PixelType::float64, source, source,
                   polar, options) || output[0] != 1. || output[1] != 1.5) return 5;
}
