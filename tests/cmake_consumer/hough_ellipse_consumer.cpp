#include "skimagexx/hough_ellipse.hpp"

#include <array>
#include <stdexcept>

int main() {
    using skimagexx::hough_ellipse_bounded;

    constexpr std::array indices{
        58, 59, 60, 61, 62, 82, 88, 106, 114, 130, 140, 155, 165, 179, 191, 204, 216,
        229, 241, 254, 266, 279, 291, 304, 316, 329, 341, 355, 365, 380, 390, 406, 414,
        432, 438, 458, 459, 460, 461, 462};
    std::array<std::uint8_t, 25 * 25> image{};
    const auto blank = hough_ellipse_bounded(image.data(), image.size(), 25, 25);
    if (!blank || blank->has_foreground || !blank->detections.empty())
        return 1;
    for (auto index : indices)
        image[index] = 2;
    const auto result = hough_ellipse_bounded(image.data(), image.size(), 25, 25, 8);
    if (!result || !result->has_foreground || result->detections.size() != 1)
        return 2;
    const auto& row = result->detections.front();
    if (row.accumulator != 10 || row.yc != 10 || row.xc != 10 || row.a != 8 || row.b != 6 || row.orientation != 0)
        return 3;
    const auto threshold = hough_ellipse_bounded(image.data(), image.size(), 25, 25, 10);
    if (!threshold || !threshold->has_foreground || !threshold->detections.empty())
        return 4;
    try {
        hough_ellipse_bounded(image.data(), image.size() - 1, 25, 25);
        return 5;
    } catch (const std::invalid_argument&) {
    }
    image.fill(1);
    if (hough_ellipse_bounded(image.data(), image.size(), 25, 25))
        return 6;
    return 0;
}
