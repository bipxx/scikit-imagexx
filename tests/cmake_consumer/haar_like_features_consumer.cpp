#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include "skimagexx/haar_like_features.hpp"

int main() {
    using namespace skimagexx;
    const auto family = HaarKind::two_x;
    std::array<HaarLikeFeature, 3> boxes{};
    assert(haar_feature_count(2, 2, family) == boxes.size());
    haar_coordinates(2, 2, family, boxes.data(), boxes.size());
    assert(boxes[0].rectangles[0].top == 0 && boxes[0].rectangles[1].left == 1);
    assert(boxes[1].rectangles[0].bottom == 1 && boxes[2].rectangles[0].top == 1);
    const std::array<std::uint8_t, 4> image{3, 8, 9, 22};
    std::array<std::int64_t, 3> values{};
    haar_generated_values(image.data(), PixelType::uint8, 2, 2, family, values.data(), values.size());
    assert((values == std::array<std::int64_t, 3>{2, 4, 2}));
    haar_selected_values(image.data(), PixelType::uint8, 2, 2, boxes.data(), values.data(), values.size());
    assert((values == std::array<std::int64_t, 3>{2, 4, 2}));

    boxes.back().rectangles[1].right = 2;
    values.fill(-9);
    try {
        haar_selected_values(image.data(), PixelType::uint8, 2, 2, boxes.data(), values.data(), values.size());
        return 1;
    } catch (const std::invalid_argument&) {
        assert((values == std::array<std::int64_t, 3>{-9, -9, -9}));
    }
    return 0;
}
