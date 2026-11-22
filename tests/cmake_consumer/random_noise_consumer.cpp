#include "skimagexx/random_noise.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

int main() {
    using skimagexx::random_noise_impulse_into;
    using skimagexx::random_noise_normal_into;

    const std::array<double, 4> image{-0.0, 0x1.0000000000001p-1, -0.25, 1.0};
    const std::array<double, 4> samples{0.25, -0x1.fffffffffffffp-1, 0.5, 0.0};
    std::array<double, 4> out;
    random_noise_normal_into(image.data(), samples.data(), out.data(), image.size(), false);
    if (out[0] != 0.25 || out[2] != 0.25 || out[3] != 1.0)
        return 1;
    const volatile double rounded_product = image[1] * samples[1];
    random_noise_normal_into(image.data(), samples.data(), out.data(), image.size(), true);
    if (out[1] != image[1] + rounded_product || out[2] != -0.375)
        return 2;

    random_noise_impulse_into(image.data(), out.data(), image.size(), {nullptr, 0, 0.0}, {nullptr, 0, 0.0}, -1.0);
    if (!std::signbit(out[0]) || out != image)
        return 3;
    const std::array<double, 4> draw{0, 0.25, 0.5, 1};
    random_noise_impulse_into(image.data(), out.data(), image.size(), {draw.data(), 4, 0.25}, {nullptr, 0, 0.0}, -1.0);
    if (out != std::array<double, 4>{-1.0, -1.0, -0.25, 1.0})
        return 4;
    const auto before = out;
    try {
        random_noise_impulse_into(image.data(), out.data(), image.size(), {draw.data(), 3, 0.25}, {nullptr, 0, 1.0}, 0.0);
        return 5;
    } catch (const std::invalid_argument&) {
        if (out != before)
            return 6;
    }
    try {
        random_noise_normal_into(out.data(), samples.data(), out.data(), out.size(), false);
        return 7;
    } catch (const std::invalid_argument&) {
        if (out != before)
            return 8;
    }
    return 0;
}
