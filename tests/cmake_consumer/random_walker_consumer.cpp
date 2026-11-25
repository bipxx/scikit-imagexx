#include <skimagexx/random_walker.hpp>

#include <array>
#include <cmath>
#include <cstdint>

int main() {
    using skimagexx::PixelType;
    using skimagexx::RandomWalkerOptions;
    using skimagexx::random_walker;
    const std::array<double, 6> image{0, 1, 2, 0, 1, 2};
    const std::array<std::int16_t, 6> markers{9, 0, 4, 9, 0, 4};
    RandomWalkerOptions options;
    options.beta = 0;
    options.tolerance = 1e-10;
    options.return_probabilities = true;
    const auto result = random_walker(image.data(), PixelType::float64, markers.data(),
                                      PixelType::int16, {2, 3}, options);
    if (!result || result->classes != 2 || !result->labels.empty() ||
        result->probabilities.size() != 12 || result->convergence_warning ||
        result->probability_warning)
        return 1;
    const std::array<double, 12> expected{0, 0.5, 1, 0, 0.5, 1, 1, 0.5, 0, 1, 0.5, 0};
    for (std::size_t i = 0; i < expected.size(); ++i)
        if (std::abs(result->probabilities[i] - expected[i]) > 1e-12) return 2;

    const std::array<std::uint8_t, 6> separated{0, 2, 200, 1, 3, 201};
    options.beta = 130;
    options.return_probabilities = false;
    const auto classified = random_walker(separated.data(), PixelType::uint8, markers.data(),
                                          PixelType::int16, {2, 3}, options);
    if (!classified || classified->labels != std::vector<std::int64_t>{2, 2, 1, 2, 2, 1} ||
        !classified->probabilities.empty())
        return 3;

    auto masked = markers;
    masked[1] = -1;
    if (random_walker(image.data(), PixelType::float64, masked.data(), PixelType::int16, {2, 3}))
        return 4;
    return 0;
}
