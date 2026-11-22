#include "skimagexx/thin.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace skimagexx {
namespace {

using Table = std::array<bool, 256>;

constexpr std::array<Table, 2> thinning_tables() {
    std::array<Table, 2> tables{};
    for (unsigned code = 0; code < 256; ++code) {
        const auto bit = [code](unsigned k) { return ((code >> (k % 8)) & 1u) != 0; };
        unsigned transitions = 0, clockwise = 0, counterclockwise = 0;
        for (unsigned k = 0; k < 8; k += 2) {
            transitions += !bit(k) && (bit(k + 1) || bit(k + 2));
            clockwise += bit(k + 1) || bit(k + 2);
            counterclockwise += bit(k + 1) || bit(k);
        }
        const auto pairs = std::min(clockwise, counterclockwise);
        const bool common = transitions == 1 && (pairs == 2 || pairs == 3);
        tables[0][code] = common && !((bit(1) || bit(2) || !bit(7)) && bit(0));
        tables[1][code] = common && !((bit(5) || bit(6) || !bit(3)) && bit(4));
    }
    return tables;
}

constexpr auto tables = thinning_tables();

} // namespace

bool thin_into(const std::uint8_t* input, std::size_t height, std::size_t width,
               bool canonical_boolean, std::size_t max_iterations, std::uint8_t* output) {
    if (!input || !output || height == 0 || width == 0 ||
        height > thin_max_extent || width > thin_max_extent || max_iterations > thin_max_iterations)
        return false;

    const auto stride = width + 2;
    std::vector<std::uint8_t> padded((height + 2) * stride);
    auto* const first = padded.data() + stride + 1;
    for (std::size_t row = 0; row < height; ++row) {
        auto* state = first + row * stride;
        for (std::size_t col = 0; col < width; ++col) {
            const auto value = *input++;
            if (canonical_boolean && value > 1) return false;
            state[col] = value != 0;
        }
    }

    std::vector<std::uint8_t*> erase;
    erase.reserve(width * 2);
    for (std::size_t iteration = 0; max_iterations == 0 || iteration < max_iterations; ++iteration) {
        bool changed = false;
        for (const auto& table : tables) {
            erase.clear();
            for (std::size_t row = 0; row < height; ++row) {
                auto* middle = first + row * stride;
                const auto* above = middle - stride;
                const auto* below = middle + stride;
                for (std::size_t col = 0; col < width; ++col) {
                    if (middle[col] == 0) continue;
                    const auto* a = above + col;
                    const auto* m = middle + col;
                    const auto* b = below + col;
                    const unsigned code = m[1] | (a[1] << 1) | (a[0] << 2) | (a[-1] << 3) |
                                          (m[-1] << 4) | (b[-1] << 5) | (b[0] << 6) | (b[1] << 7);
                    if (table[code]) erase.push_back(middle + col);
                }
            }
            // Commit all removals together before the next neighborhood scan.
            for (auto* pixel : erase) *pixel = 0;
            changed |= !erase.empty();
        }
        if (!changed) break;
    }

    for (std::size_t row = 0; row < height; ++row)
        std::copy_n(first + row * stride, width, output + row * width);
    return true;
}

} // namespace skimagexx
