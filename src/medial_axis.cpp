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
constexpr std::array<std::uint8_t, 256> keep{
    1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

constexpr auto corner_scores() {
    std::array<std::uint8_t, 256> result{};
    for (unsigned code = 0; code < result.size(); ++code) {
        result[code] = 8;
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
        | (center[-1] << 3) | (center[1] << 4)
        | (bottom[-1] << 5) | (bottom[0] << 6) | (bottom[1] << 7);
}

struct Site {
    double distance;
    std::uint32_t offset;
    unsigned score;
};

template <class Visit>
bool sweep_buckets(const std::vector<Site>& ranked, Visit visit) {
    // The EDT returns square roots of integer squared distances. Bound the
    // bucket storage by the number of sites, and keep the general SDK path
    // for arbitrary distances or sparse fields with a large distance range.
    constexpr unsigned scores = 9;
    const auto limit = static_cast<unsigned>((4 * ranked.size() + 32) / scores);
    double farthest = 0;
    for (const auto& site : ranked) {
        farthest = std::max(farthest, site.distance);
    }
    if (farthest > limit || farthest * farthest >= limit + 0.5) return false;
    const auto largest = static_cast<unsigned>(farthest * farthest + 0.5);
    std::vector<double> distances(largest + 1, -1.0);
    std::vector<int> heads((largest + 1) * scores, -1);
    std::vector<int> next(ranked.size());
    for (auto i = ranked.size(); i-- > 0;) {
        const auto& site = ranked[i];
        const auto squared = static_cast<unsigned>(site.distance * site.distance + 0.5);
        auto& distance = distances[squared];
        if (distance >= 0 && distance != site.distance) return false;
        distance = site.distance;
        auto& head = heads[squared * scores + site.score];
        next[i] = head;
        head = static_cast<int>(i);
    }
    for (auto i : heads) {
        for (; i != -1; i = next[i]) visit(ranked[i]);
    }
    return true;
}

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

    std::vector<Site> sites(priority_count);
    std::size_t count = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < columns; ++col) {
            const auto offset = (row + 1) * pitch + col + 1;
            if (!state[offset]) {
                continue;
            }
            const auto distance = distances[row * columns + col];
            if (count == priority_count || !std::isfinite(distance) || distance < 0) {
                return false;
            }
            const auto priority = priorities[count++];
            if (priority < 0 || std::size_t(priority) >= priority_count || sites[std::size_t(priority)].offset) {
                return false;
            }
            sites[std::size_t(priority)] = {
                distance, static_cast<std::uint32_t>(offset),
                corner[neighborhood(state.data() + offset, signed_pitch)]
            };
        }
    }
    if (count != priority_count) {
        return false;
    }
    // Scores and ranks are fixed before deletion. Each table lookup sees the
    // updates from all earlier sites, exactly once in that ordered pass.
    const auto visit = [&](const Site& site) {
        if (!keep[neighborhood(state.data() + site.offset, signed_pitch)]) {
            state[site.offset] = 0;
        }
    };
    if (!sweep_buckets(sites, visit)) {
        std::stable_sort(sites.begin(), sites.end(), [](const Site& a, const Site& b) {
            return std::tie(a.distance, a.score) < std::tie(b.distance, b.score);
        });
        for (const auto& site : sites) visit(site);
    }
    std::vector<std::uint8_t> result(distance_count);
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(state.data() + (row + 1) * pitch + 1, columns, result.data() + row * columns);
    }
    output = std::move(result);
    return true;
}

}  // namespace skimagexx
