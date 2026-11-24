#include "skimagexx/gray_texture.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    using skimagexx::GrayImageView;
    using skimagexx::GrayIntegerType;
    using skimagexx::GrayProbabilityView;
    using skimagexx::GrayProperty;

    const std::array<std::uint8_t, 4> image{0, 0, 1, 1};
    const GrayImageView view{image.data(), 2, 2, 2, 1, GrayIntegerType::u8};

    const auto counts = skimagexx::graycomatrix(view, {1.0}, {0.0}, 2, false, false, 1);
    if (counts.levels != 2 || counts.distances != 1 || counts.angles != 1 ||
        counts.counts != std::vector<std::uint32_t>{1, 0, 0, 1})
        return 1;

    const auto normed = skimagexx::graycomatrix(view, {1.0}, {0.0}, 2, true, true, 0);
    if (normed.probabilities.size() != 4 ||
        normed.probabilities != std::vector<double>{0.5, 0.0, 0.0, 0.5})
        return 2;

    const std::array<std::ptrdiff_t, 4> strides{2 * 8, 8, 8, 8};
    const GrayProbabilityView prob{
        normed.probabilities.data(), 2, 1, 1,
        {strides[0], strides[1], strides[2], strides[3]}};
    for (const auto prop : {
             GrayProperty::contrast,
             GrayProperty::dissimilarity,
             GrayProperty::homogeneity,
             GrayProperty::ASM,
             GrayProperty::energy,
             GrayProperty::mean,
             GrayProperty::variance,
             GrayProperty::stddev,
             GrayProperty::entropy,
             GrayProperty::correlation,
         }) {
        const auto result = skimagexx::graycoprops(prob, prop, 1);
        if (result.distances != 1 || result.angles != 1 || result.values.size() != 1 ||
            !std::isfinite(result.values[0]))
            return 3;
    }

    try {
        skimagexx::graycoprops(prob, GrayProperty::invalid, 1);
        return 4;
    } catch (const std::domain_error&) {
    }
    try {
        skimagexx::graycomatrix(view, {1.0}, {0.0}, 2, false, false, -1);
        return 5;
    } catch (const std::invalid_argument&) {
    }

    std::vector<std::uint8_t> large(257 * 257);
    for (std::size_t k = 0; k < large.size(); ++k)
        large[k] = static_cast<std::uint8_t>(k % 8);
    const GrayImageView big{large.data(), 257, 257, 257, 1, GrayIntegerType::u8};
    const auto p1 = skimagexx::graycomatrix(big, {1.0}, {0.0}, 256, true, true, 1);
    const auto p4 = skimagexx::graycomatrix(big, {1.0}, {0.0}, 256, true, true, 4);
    if (p1.probabilities != p4.probabilities)
        return 6;
    return 0;
}
