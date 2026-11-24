#include <skimagexx/corner_orientations.hpp>

#include <array>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("corner_orientations SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::corner_orientations_f32_into;
    using skimagexx::corner_orientations_f64_into;
    using skimagexx::corner_orientations_u8_into;

    std::array<std::uint8_t, 49> u8_img{};
    u8_img[4 * 7 + 4] = 255;
    const std::array<std::intptr_t, 2> corner{3, 3};
    const std::array<std::uint8_t, 9> mask{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    };
    double angle = 0.0;
    require(corner_orientations_u8_into(u8_img.data(), 7, 7, corner.data(), 1, mask.data(), 3, &angle));
    require(std::abs(angle - std::atan2(1.0, 1.0)) < 1e-12);

    std::array<float, 49> f32_img{};
    f32_img[2 * 7 + 3] = 0.5f;
    float f32_angle = 0.0f;
    require(corner_orientations_f32_into(f32_img.data(), 7, 7, corner.data(), 1, mask.data(), 3, &f32_angle));
    require(std::abs(f32_angle - static_cast<float>(std::atan2(-1.0, 0.0))) < 1e-6f);

    std::array<double, 49> f64_img{};
    f64_img[3 * 7 + 2] = 1.0;
    double f64_angle = 0.0;
    require(corner_orientations_f64_into(f64_img.data(), 7, 7, corner.data(), 1, mask.data(), 3, &f64_angle));
    require(std::abs(f64_angle - std::atan2(0.0, -1.0)) < 1e-12);

    require(corner_orientations_u8_into(u8_img.data(), 7, 7, nullptr, 0, mask.data(), 3, nullptr));

    const double saved = angle;
    const std::array<std::intptr_t, 2> border_corner{0, 3};
    const std::array<std::uint8_t, 4> even_mask{1, 1, 1, 1};
    require(!corner_orientations_u8_into(u8_img.data(), 7, 7, border_corner.data(), 1, mask.data(), 3, &angle));
    require(!corner_orientations_u8_into(u8_img.data(), 7, 7, corner.data(), 1, even_mask.data(), 2, &angle));
    require(!corner_orientations_u8_into(nullptr, 7, 7, corner.data(), 1, mask.data(), 3, &angle));
    require(angle == saved);

    const int nearest = std::fegetround();
    require(nearest == FE_TONEAREST && std::fesetround(FE_DOWNWARD) == 0);
    const bool refused = !corner_orientations_u8_into(
        u8_img.data(), 7, 7, corner.data(), 1, mask.data(), 3, &angle);
    require(std::fesetround(nearest) == 0 && refused && angle == saved);
}
