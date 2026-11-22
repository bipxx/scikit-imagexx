#include "skimagexx/random_shapes.hpp"

#include <array>
#include <stdexcept>

int main() {
    using skimagexx::random_shape_paint_into;
    std::array<std::uint8_t, 3 * 4 * 3> image;
    image.fill(255);
    std::array<std::uint8_t, 3 * 4> mask{};
    const std::array<std::uint8_t, 3> white{255, 255, 255}, color{7, 13, 19};
    const std::array<std::intptr_t, 4> interleaved{0, 1, 2, 3};
    if (!random_shape_paint_into(image.data(), image.size(), mask.data(), mask.size(), 3, 4, 3,
            {interleaved.data(), 2}, {interleaved.data() + 1, 2}, 2, white.data(), white.size(), false) ||
            mask[1] != 1 || mask[11] != 1 || image[3] != 255)
        return 1;

    const std::array<std::intptr_t, 2> y{1, 2}, x{2, 3};
    const auto before = image;
    const auto old_mask = mask;
    if (random_shape_paint_into(image.data(), image.size(), mask.data(), mask.size(), 3, 4, 3,
            {y.data(), 1}, {x.data(), 1}, 2, color.data(), color.size(), false) || image != before || mask != old_mask)
        return 2;
    if (!random_shape_paint_into(image.data(), image.size(), mask.data(), mask.size(), 3, 4, 3,
            {y.data(), 1}, {x.data(), 1}, 2, color.data(), color.size(), true) ||
            image[6 * 3 + 2] != 19 || image[11 * 3 + 1] != 13 || mask[6] != 1)
        return 3;

    const auto painted = image;
    const auto painted_mask = mask;
    const std::array<std::intptr_t, 2> invalid{2, 4};
    try {
        random_shape_paint_into(image.data(), image.size(), mask.data(), mask.size(), 3, 4, 3,
            {y.data(), 1}, {invalid.data(), 1}, 2, white.data(), white.size(), true);
        return 4;
    } catch (const std::invalid_argument&) {
        if (image != painted || mask != painted_mask)
            return 5;
    }
    try {
        random_shape_paint_into(image.data(), image.size(), mask.data(), mask.size(), 3, 4, 3,
            {y.data(), 1}, {x.data(), 1}, 2, image.data(), color.size(), true);
        return 6;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
