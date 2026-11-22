#include "skimagexx/medial_axis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace skimagexx {
namespace {

// Keep a foreground center if its removal changes 8-connectivity, or the
// three-by-three neighborhood has fewer than three foreground pixels.
constexpr std::array<std::uint8_t, 512> keep{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

constexpr auto corner_scores() {
    std::array<unsigned, 512> result{};
    for (unsigned code = 0; code < result.size(); ++code) {
        result[code] = 9;
        for (unsigned bits = code; bits; bits >>= 1) {
            result[code] -= bits & 1;
        }
    }
    return result;
}

constexpr auto corner = corner_scores();

unsigned neighborhood(const std::uint8_t* center, std::ptrdiff_t pitch) {
    const auto* top = center - pitch;
    const auto* bottom = center + pitch;
    return top[-1] | (top[0] << 1) | (top[1] << 2)
        | (center[-1] << 3) | (center[0] << 4) | (center[1] << 5)
        | (bottom[-1] << 6) | (bottom[0] << 7) | (bottom[1] << 8);
}

struct Site {
    std::size_t offset;
    double distance;
    unsigned corner_score;
    std::int64_t priority;
};

}  // namespace

bool medial_axis_ordered(
    const std::uint8_t* foreground, const double* distances, std::size_t distance_count,
    const std::int64_t* priorities, std::size_t priority_count,
    std::size_t rows, std::size_t columns, std::vector<std::uint8_t>& output) {
    if (!foreground || !distances || rows < 1 || columns < 1
        || rows > medial_axis_max_extent || columns > medial_axis_max_extent
        || distance_count != rows * columns || priority_count > distance_count
        || (priority_count && !priorities)) {
        return false;
    }
    const auto pitch = columns + 2;
    const auto signed_pitch = static_cast<std::ptrdiff_t>(pitch);
    std::vector<std::uint8_t> state((rows + 2) * pitch);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* first = foreground + row * columns;
        if (std::any_of(first, first + columns, [](auto value) { return value > 1; })) {
            return false;
        }
        std::copy_n(first, columns, state.data() + (row + 1) * pitch + 1);
    }

    std::vector<Site> sites;
    sites.reserve(priority_count);
    std::vector<std::uint8_t> seen(priority_count);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < columns; ++col) {
            const auto offset = (row + 1) * pitch + col + 1;
            if (!state[offset]) {
                continue;
            }
            const auto distance = distances[row * columns + col];
            if (sites.size() == priority_count || !std::isfinite(distance) || distance < 0) {
                return false;
            }
            const auto priority = priorities[sites.size()];
            if (priority < 0 || std::size_t(priority) >= priority_count || seen[std::size_t(priority)]) {
                return false;
            }
            seen[std::size_t(priority)] = 1;
            sites.push_back({offset, distance, corner[neighborhood(state.data() + offset, signed_pitch)], priority});
        }
    }
    if (sites.size() != priority_count) {
        return false;
    }
    std::sort(sites.begin(), sites.end(), [](const Site& a, const Site& b) {
        return std::tie(a.distance, a.corner_score, a.priority) < std::tie(b.distance, b.corner_score, b.priority);
    });
    // Scores and ranks are fixed before deletion. Each table lookup sees the
    // updates from all earlier sites, exactly once in that ordered pass.
    for (const auto& site : sites) {
        if (!keep[neighborhood(state.data() + site.offset, signed_pitch)]) {
            state[site.offset] = 0;
        }
    }
    std::vector<std::uint8_t> result(distance_count);
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(state.data() + (row + 1) * pitch + 1, columns, result.data() + row * columns);
    }
    output = std::move(result);
    return true;
}

}  // namespace skimagexx
