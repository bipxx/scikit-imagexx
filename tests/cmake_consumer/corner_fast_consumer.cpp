#include <skimagexx/corner_fast.hpp>

#include <array>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("corner_fast SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::corner_fast_f32_into;
    using skimagexx::corner_fast_f64_into;
    using skimagexx::corner_fast_u8_into;

    std::array<std::uint8_t, 49> u8_img{};
    u8_img[3 * 7 + 3] = 255;
    std::array<double, 49> u8_out{};
    require(corner_fast_u8_into(u8_img.data(), 7, 7, 12, 0.15, u8_out.data()));
    require(u8_out[3 * 7 + 3] == 16.0);
    for (std::size_t r = 0; r < 7; ++r) {
        for (std::size_t c = 0; c < 7; ++c) {
            if (r != 3 || c != 3) {
                require(u8_out[r * 7 + c] == 0.0);
            }
        }
    }

    std::array<float, 49> f32_img{};
    for (std::size_t i = 0; i < 49; ++i) {
        f32_img[i] = 0.5f;
    }
    constexpr int ring_dy[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    constexpr int ring_dx[16] = {3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3};
    for (int k = 0; k < 12; ++k) {
        f32_img[(3 + ring_dy[(12 + k) & 15]) * 7 + (3 + ring_dx[(12 + k) & 15])] = 0.875f;
    }
    std::array<float, 49> f32_out{};
    require(corner_fast_f32_into(f32_img.data(), 7, 7, 12, 0.15f, f32_out.data()));
    require(std::abs(f32_out[3 * 7 + 3] - 4.5f) < 1e-6f);

    std::array<double, 49> f64_img{};
    for (std::size_t i = 0; i < 49; ++i) {
        f64_img[i] = 0.5;
    }
    for (int k = 0; k < 9; ++k) {
        f64_img[(3 + ring_dy[k]) * 7 + (3 + ring_dx[k])] = 0.1;
    }
    std::array<double, 49> f64_out{};
    require(corner_fast_f64_into(f64_img.data(), 7, 7, 9, 0.15, f64_out.data()));
    require(std::abs(f64_out[3 * 7 + 3] - 3.6) < 1e-12);

    std::array<double, 12> small_out{};
    require(corner_fast_u8_into(u8_img.data(), 2, 6, 9, 0.15, small_out.data()));
    for (double v : small_out) {
        require(v == 0.0);
    }

    const auto prior = u8_out;
    require(!corner_fast_u8_into(u8_img.data(), 1, 7, 9, 0.15, u8_out.data()));
    require(!corner_fast_u8_into(u8_img.data(), 7, 7, 0, 0.15, u8_out.data()));
    require(!corner_fast_u8_into(u8_img.data(), 7, 7, 17, 0.15, u8_out.data()));
    require(!corner_fast_u8_into(u8_img.data(), 7, 7, 12, -0.1, u8_out.data()));
    require(!corner_fast_u8_into(nullptr, 7, 7, 12, 0.15, u8_out.data()));
    f64_img[0] = 300.0;
    require(!corner_fast_f64_into(f64_img.data(), 7, 7, 9, 0.15, f64_out.data()));
    require(u8_out == prior);

    const int nearest = std::fegetround();
    require(nearest == FE_TONEAREST && std::fesetround(FE_DOWNWARD) == 0);
    const bool refused = !corner_fast_u8_into(u8_img.data(), 7, 7, 12, 0.15, u8_out.data());
    require(std::fesetround(nearest) == 0 && refused && u8_out == prior);
}
