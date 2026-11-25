#include "skimagexx/draw_polygon.hpp"
#include "skimagexx/future_filters.hpp"

#include <array>
#include <vector>

int main() {
    using namespace skimagexx;
    const std::array<std::int16_t, 5> labels{-3, 0, 9, 0, 7};
    const std::array<float, 10> features{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const auto selected = segment_positive_indices(labels.data(), labels.size(), SegmentLabel::i16);
    if (selected != std::vector<std::size_t>{2, 4}) return 1;
    std::array<float, 4> data{};
    std::array<std::int16_t, 2> target{};
    segment_gather_rows(selected, features.data(), 2 * sizeof(float), labels.data(),
                        sizeof(std::int16_t), data.data(), target.data());
    if (data != std::array<float, 4>{4, 5, 8, 9} ||
        target != std::array<std::int16_t, 2>{9, 7}) return 2;

    const std::array<double, 4> y{1, 1, 3, 3};
    const std::array<double, 4> x{1, 3, 3, 1};
    detail::DrawPolygonPlan plan;
    if (!detail::draw_polygon_prepare({y.data(), x.data(), x.size()}, {{1, 1}, {3, 3}},
                                      plan, 1)) return 3;
    std::array<std::intptr_t, 25> output{};
    output[0] = 9;
    segment_stamp_polygon(plan, 5, 5, 3, output.data());
    for (std::size_t row = 0; row < 5; ++row) {
        for (std::size_t col = 0; col < 5; ++col) {
            const auto expected = row == 0 && col == 0 ? 9 :
                (row > 0 && row < 4 && col > 0 && col < 4 ? 3 : 0);
            if (output[row * 5 + col] != expected) return 4;
        }
    }

    const std::array<std::uint8_t, 4> image{7, 9, 7, 255};
    const auto histogram = histogram_u8(image.data(), 2, 2);
    if (histogram[7] != 2 || histogram[9] != 1 || histogram[255] != 1) return 5;
    return 0;
}
