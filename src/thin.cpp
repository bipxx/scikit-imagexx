#include "skimagexx/thin.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace skimagexx {
namespace {

constexpr auto deletion_table() {
    std::array<std::uint8_t, 256> table{};
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
        if (common) {
            table[code] = unsigned(!((bit(1) || bit(2) || !bit(7)) && bit(0)))
                | (unsigned(!((bit(5) || bit(6) || !bit(3)) && bit(4))) << 1);
        }
    }
    return table;
}

constexpr auto deletion = deletion_table();

unsigned neighborhood(const std::uint8_t* pixel, std::ptrdiff_t stride) {
    const auto* above = pixel - stride;
    const auto* below = pixel + stride;
    return pixel[1] | (above[1] << 1) | (above[0] << 2) | (above[-1] << 3)
        | (pixel[-1] << 4) | (below[-1] << 5) | (below[0] << 6) | (below[1] << 7);
}

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

    const auto pitch = static_cast<std::ptrdiff_t>(stride);
    const std::array<std::ptrdiff_t, 8> neighbors{
        1, 1 - pitch, -pitch, -1 - pitch, -1, pitch - 1, pitch, pitch + 1
    };
    std::array<std::vector<std::uint32_t>, 2> frontier;
    std::vector<std::uint8_t> queued(padded.size());
    for (std::size_t row = 0; row < height; ++row) {
        const auto begin = (row + 1) * stride + 1;
        for (auto i = begin; i < begin + width; ++i) {
            const auto* pixel = padded.data() + i;
            // Four occupied cardinal neighbors cannot satisfy the deletion rule.
            if (!*pixel || (pixel[-1] & pixel[1] & pixel[-pitch] & pixel[pitch])) continue;
            const auto phases = deletion[neighborhood(pixel, pitch)];
            queued[i] = phases;
            if (phases & 1) frontier[0].push_back(static_cast<std::uint32_t>(i));
            if (phases & 2) frontier[1].push_back(static_cast<std::uint32_t>(i));
        }
    }

    std::vector<std::uint32_t> erase;
    for (std::size_t iteration = 0; max_iterations == 0 || iteration < max_iterations; ++iteration) {
        bool changed = false;
        const bool last_iteration = iteration + 1 == max_iterations;
        for (unsigned phase = 0; phase < 2; ++phase) {
            const auto phase_bit = 1u << phase;
            erase.clear();
            for (const auto i : frontier[phase]) {
                queued[i] &= static_cast<std::uint8_t>(~phase_bit);
                const auto* pixel = padded.data() + i;
                if (*pixel && (deletion[neighborhood(pixel, pitch)] & phase_bit)) erase.push_back(i);
            }
            frontier[phase].clear();

            // Decisions are simultaneous within a phase. Only surviving neighbors
            // can have a different decision in either of the upcoming phases.
            for (const auto i : erase) padded[i] = 0;
            changed |= !erase.empty();
            if (last_iteration && phase == 1) break;
            const auto upcoming = last_iteration ? 2u : 3u;
            for (const auto i : erase) {
                for (const auto delta : neighbors) {
                    const auto j = static_cast<std::uint32_t>(i + delta);
                    if (!padded[j]) continue;
                    const auto pending = upcoming & ~queued[j];
                    if (pending & 1) frontier[0].push_back(j);
                    if (pending & 2) frontier[1].push_back(j);
                    queued[j] |= static_cast<std::uint8_t>(pending);
                }
            }
        }
        if (!changed) break;
    }

    for (std::size_t row = 0; row < height; ++row)
        std::copy_n(first + row * stride, width, output + row * width);
    return true;
}

} // namespace skimagexx
