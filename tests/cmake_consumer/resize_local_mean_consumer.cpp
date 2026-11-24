#include "skimagexx/resize_local_mean.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

int main() {
    using skimagexx::PixelType;
    using skimagexx::resize_local_mean_into;

    std::array<std::uint8_t, 16 * 16> normalized{};
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 16; ++x)
            normalized[y * 16 + x] = std::uint8_t(257 * y + 17 * x + 13 * y * x);
    std::array<double, 8 * 8> output{};
    if (!resize_local_mean_into(normalized.data(), PixelType::uint8, normalized.size(), 16, 16,
            output.data(), output.size(), 8, 8, false) || output[5] != 0x1.dededededededp-2)
        return 1;

    std::array<std::uint16_t, 24 * 32> raw;
    raw.fill(65535);
    std::array<double, 3 * 8> raw_result;
    raw_result.fill(-1);
    if (resize_local_mean_into(raw.data(), PixelType::uint16, raw.size(), 24, 32,
            raw_result.data(), raw_result.size(), 3, 8, false) ||
            !std::all_of(raw_result.begin(), raw_result.end(), [](double x) { return x == -1; }))
        return 2;
    if (!resize_local_mean_into(raw.data(), PixelType::uint16, raw.size(), 24, 32,
            raw_result.data(), raw_result.size(), 3, 8, true) ||
            !std::all_of(raw_result.begin(), raw_result.end(), [](double x) { return x == 65535; }))
        return 3;

    try {
        resize_local_mean_into(output.data(), PixelType::uint8, normalized.size(), 16, 16,
            output.data(), output.size(), 8, 8, false);
        return 4;
    } catch (const std::invalid_argument&) {
    }
    try {
        resize_local_mean_into(normalized.data(), PixelType::uint8, normalized.size() - 1, 16, 16,
            output.data(), output.size(), 8, 8, false);
        return 5;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
