#include <skimagexx/data_loaders_a.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>

int main() {
    const std::array<std::uint16_t, 5> original{0, 1, 256, 32768, 65535};
    std::array<std::uint16_t, 5> first{};
    std::array<std::uint16_t, 5> second{};
    const auto bytes = original.size() * sizeof(original[0]);
    skimagexx::copy_sample_data_into(original.data(), bytes, first.data(), bytes);
    skimagexx::copy_sample_data_into(original.data(), bytes, second.data(), bytes);
    if (first != original || second != original) {
        return 1;
    }
    first[0] = 99;
    if (second != original) {
        return 2;
    }
    skimagexx::copy_sample_data_into(nullptr, 0, nullptr, 0);
    try {
        skimagexx::copy_sample_data_into(original.data(), bytes, first.data(), bytes - 1);
        return 3;
    } catch (const std::invalid_argument&) {
        if (first[0] != 99) {
            return 4;
        }
    }
    try {
        skimagexx::copy_sample_data_into(nullptr, bytes, second.data(), bytes);
        return 5;
    } catch (const std::invalid_argument&) {
        return second == original ? 0 : 6;
    }
}
