#include <skimagexx/canny.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("canny SDK assertion failed");
}

}  // namespace

int main() {
    using skimagexx::CannyOptions;
    using skimagexx::PixelType;
    using skimagexx::canny_into;
    constexpr std::size_t rows = 7, cols = 9, count = rows * cols;
    std::array<std::uint8_t, count> image{}, mask{}, output{}, expected{};
    for (std::size_t row = 0; row < rows; ++row) {
        std::fill_n(image.begin() + row * cols + 4, cols - 4, std::uint8_t(255));
        if (row > 0 && row + 1 < rows) {
            expected[row * cols + 3] = 1;
            expected[row * cols + 4] = 1;
        }
    }

    require(canny_into(image.data(), PixelType::uint8, nullptr, rows, cols, {}, output.data()));
    require(output == expected);

    CannyOptions options;
    options.quantiles = true;
    options.weights = {0.25, 0.5, 0.25};
    require(canny_into(image.data(), PixelType::uint8, nullptr, rows, cols, options, output.data()));
    // The constant-border bleed division perturbs the interior tie.
    // Reference: separable SciPy correlations followed by stock Canny at sigma=0.
    for (std::size_t row = 2; row + 2 < rows; ++row) expected[row * cols + 4] = 0;
    require(output == expected);

    require(canny_into(image.data(), PixelType::uint8, mask.data(), rows, cols, options, output.data()));
    require(std::none_of(output.begin(), output.end(), [](std::uint8_t value) { return value != 0; }));

    output.fill(9);
    options.low = 0.5;
    require(!canny_into(image.data(), PixelType::uint8, nullptr, rows, cols, options, output.data()));
    require(std::all_of(output.begin(), output.end(), [](std::uint8_t value) { return value == 9; }));
}
