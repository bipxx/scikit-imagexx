#include <skimagexx/hough_circle.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("hough_circle SDK output or validation differs");
}

}  // namespace

int main() {
    std::array<std::uint8_t, 6> image{255, 0, 0, 0, 0, 0};
    const std::vector<int> radii{1, 0, 1};
    const auto shape = skimagexx::hough_circle_plane(2, 3, radii, true);
    require(shape && shape->height == 4 && shape->width == 5 && shape->padding == 1);
    std::array<double, 60> result{};
    auto circle = [&](const std::uint8_t* input, std::size_t count, bool normalized) {
        skimagexx::hough_circle_into(input, image.size(), 2, 3, radii, normalized, true, result.data(), count);
    };
    circle(image.data(), result.size(), false);
    require(result[20 + 6] == 8);
    require(std::equal(result.begin(), result.begin() + 20, result.begin() + 40));
    require(result[1] == 2 && result[5] == 2 && result[7] == 2 && result[11] == 2);
    circle(image.data(), result.size(), true);
    require(result[20 + 6] == 1 && result[1] == 0.25 && result[11] == 0.25);
    const auto saved = result;
    bool rejected = false;
    try {
        circle(reinterpret_cast<const std::uint8_t*>(result.data()), result.size(), true);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected && result == saved);
    require(!skimagexx::hough_circle_plane(2, 3, {-1}, false));
}
