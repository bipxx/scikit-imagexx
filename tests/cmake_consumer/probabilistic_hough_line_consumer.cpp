#include "skimagexx/probabilistic_hough_line.hpp"

#include <array>
#include <limits>
#include <stdexcept>

int main() {
    constexpr int height = 9, width = 12;
    const std::array<double, 1> cosines{0.0}, sines{1.0};
    std::array<std::uint8_t, height * width> mask{};
    for (int x = 2; x < 10; ++x)
        mask[3 * width + x] = x % 2 ? 231 : 3;
    const std::array<std::intptr_t, 8> order{7, 6, 5, 4, 3, 2, 1, 0};
    const auto lines = skimagexx::probabilistic_hough_line_ordered(
        mask.data(), mask.size(), height, width, cosines.data(), sines.data(), 1, order.data(), order.size(), 1, 7, 0);
    if (lines.size() != 1 || lines[0].first.x != 2 || lines[0].first.y != 3 ||
        lines[0].second.x != 9 || lines[0].second.y != 3 || mask[3 * width + 2] != 3)
        return 1;

    if (!skimagexx::probabilistic_hough_line_ordered(
            mask.data(), mask.size(), height, width, nullptr, nullptr, 0, order.data(), order.size(), 1, 7, 0).empty())
        return 2;
    auto duplicate = order;
    duplicate[1] = duplicate[0];
    try {
        skimagexx::probabilistic_hough_line_ordered(mask.data(), mask.size(), height, width, cosines.data(), sines.data(),
            1, duplicate.data(), duplicate.size(), 1, 7, 0);
        return 3;
    } catch (const std::invalid_argument&) {
    }
    const double infinite = std::numeric_limits<double>::infinity();
    try {
        skimagexx::probabilistic_hough_line_ordered(mask.data(), mask.size(), height, width, cosines.data(), &infinite,
            1, order.data(), order.size(), 1, 7, 0);
        return 4;
    } catch (const std::invalid_argument&) {
    }
    try {
        skimagexx::probabilistic_hough_line_ordered(mask.data(), mask.size() - 1, height, width, cosines.data(),
            sines.data(), 1, order.data(), order.size(), 1, 7, 0);
        return 5;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
