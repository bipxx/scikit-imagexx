#include "skimagexx/feature_descriptors.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

int main() {
    using namespace skimagexx;
    const std::array<std::uint8_t, 25> pixels = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
    };
    std::array<double, 25> lbp{};
    local_binary_pattern_into(pixels.data(), 5, 5, 4, 1, LbpMethod::standard, lbp.data());
    if (lbp[12] != 11) return 1;

    const std::array<float, 9> integral = {1, 2, 3, 2, 4, 6, 3, 6, 9};
    if (multiblock_lbp(integral.data(), 3, 0, 0, 1, 1) != 255) return 2;

    const HogOptions hog{5, 5, 1, 2, 2, 1, 1, 4, HogNorm::l2};
    std::array<double, 16> gradient_histograms{};
    std::array<double, 25> visualization{};
    hog_into(pixels.data(), hog, gradient_histograms.data(), visualization.data());
    if (gradient_histograms[0] < 0.99 || gradient_histograms[1] != 0 || visualization[6] != 1)
        return 3;

    const DaisyOptions daisy{5, 5, 1, 1, 1, 2, 4, DaisyNorm::l1};
    const double sigmas[] = {0.5, 0.5};
    const double radii[] = {1.0};
    std::array<double, 108> dense{};
    daisy_into(pixels.data(), daisy, sigmas, radii, dense.data());
    double total = 0;
    for (int i = 0; i < 12; ++i) total += dense[i];
    if (std::abs(total - 1) > 1e-12) return 4;

    std::array<double, 25> image{};
    std::copy(pixels.begin(), pixels.end(), image.begin());
    const std::array<double, 4> pattern = {0, 1, 0, 1};
    std::array<double, 16> matched{};
    const TemplateMatchOptions match{5, 5, 2, 2, false, 0};
    if (!match_template_into(image.data(), pattern.data(), match, matched.data()) ||
        std::abs(matched[0] - 1) > 1e-12)
        return 5;

    const BasicFeatureOptions basic{5, 5, 1, true, true, true};
    constexpr double sigma = 0.01;
    std::array<float, 100> features{};
    multiscale_basic_features_into(pixels.data(), basic, &sigma, 1, features.data());
    if (std::abs(features[48] - 2.0f / 255.0f) > 1e-8f ||
        features[49] <= 0 || std::abs(features[50]) > 1e-8f || std::abs(features[51]) > 1e-8f)
        return 6;
    return 0;
}
