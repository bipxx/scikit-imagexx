#include <skimagexx/convex_hull_image.hpp>

#include <array>
#include <stdexcept>

int main() {
    constexpr std::size_t extent = 7;
    std::array<std::uint8_t, extent * extent> image{}, output{}, expected{};
    image[8] = image[12] = image[36] = 1;
    const auto require = [](bool condition) {
        if (!condition) throw std::runtime_error("convex_hull_image SDK assertion failed");
    };
    require(skimagexx::convex_hull_image_into(image.data(), extent, extent, false, false, output.data()));
    expected[16] = expected[17] = expected[23] = 1;
    require(output == expected);
    require(skimagexx::convex_hull_image_into(image.data(), extent, extent, false, true, output.data()));
    for (std::size_t r = 1; r <= 5; ++r)
        for (std::size_t c = 1; c <= 6 - r; ++c) expected[r * extent + c] = 1;
    require(output == expected);
    image.fill(0);
    image[6] = 1;
    require(!skimagexx::convex_hull_image_into(image.data(), extent, extent, false, true, output.data()));
    require(output == expected);
    require(skimagexx::convex_hull_image_into(image.data(), extent, extent, true, true, output.data()));
    require(output == image);
}
