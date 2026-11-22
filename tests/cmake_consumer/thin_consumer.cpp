#include <skimagexx/thin.hpp>

#include <array>
#include <stdexcept>

int main() {
    const auto require = [](bool condition) {
        if (!condition) throw std::runtime_error("thin SDK assertion failed");
    };
    constexpr std::size_t height = 9, width = 11;
    std::array<std::uint8_t, height * width> image{};
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col)
            image[row * width + col] = (row >= 1 && row < 8 && col >= 1 && col < 10) ||
                                       (col == width / 2 && row < height / 3);
    }
    std::array<std::uint8_t, height * width> output{}, expected{};
    for (std::size_t col = 4; col <= 6; ++col) expected[4 * width + col] = 1;
    require(skimagexx::thin_into(image.data(), height, width, true, 0, output.data()));
    require(expected == output);

    expected.fill(0);
    expected[width + width / 2] = 1;
    for (std::size_t row = 2; row <= 6; ++row) {
        for (std::size_t col = 2; col <= 8; ++col) expected[row * width + col] = 1;
    }
    require(skimagexx::thin_into(image.data(), height, width, true, 1, output.data()));
    require(expected == output);
    image[width / 2] = 225;
    require(!skimagexx::thin_into(image.data(), height, width, true, 1, output.data()));
    require(expected == output);
    require(skimagexx::thin_into(image.data(), height, width, false, 1, output.data()));
    require(expected == output);
}
