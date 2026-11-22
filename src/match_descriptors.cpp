#include "skimagexx/match_descriptors.hpp"

#include <cfenv>
#include <utility>

namespace skimagexx {
namespace {

using PackedRow = std::array<std::uint64_t, match_descriptors_max_width / 64>;

bool pack(const std::uint8_t* source, std::size_t width, std::vector<PackedRow>& rows) {
    for (auto& row : rows) {
        for (std::size_t k = 0; k < width; ++k) {
            const auto value = *source++;
            if (value > 1) return false;
            row[k / 64] |= std::uint64_t(value) << (k % 64);
        }
    }
    return true;
}

unsigned popcount(std::uint64_t word) {
    word -= (word >> 1) & UINT64_C(0x5555555555555555);
    word = (word & UINT64_C(0x3333333333333333)) + ((word >> 2) & UINT64_C(0x3333333333333333));
    word = (word + (word >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return static_cast<unsigned>((word * UINT64_C(0x0101010101010101)) >> 56);
}

struct Nearest {
    unsigned count;
    std::size_t index = 0;
};

struct RowMatches {
    Nearest best;
    unsigned second;
};

} // namespace

bool match_descriptors_bool(const std::uint8_t* left, std::size_t left_rows,
                            const std::uint8_t* right, std::size_t right_rows,
                            std::size_t width, const DescriptorMatchOptions& options,
                            std::vector<DescriptorMatch>& matches) {
    const auto inf = std::numeric_limits<double>::infinity();
    const auto distance = options.max_distance;
    const auto ratio = options.max_ratio;
    if (!left || !right || left_rows == 0 || left_rows > match_descriptors_max_rows ||
        right_rows == 0 || right_rows > match_descriptors_max_rows ||
        width == 0 || width > match_descriptors_max_width ||
        !(distance == inf || (distance >= 0.0 && distance <= 1.0)) ||
        !(ratio >= 0.0 && ratio <= 1.0))
        return false;

    const bool filter_distance = distance < inf;
    const bool filter_ratio = ratio < 1.0;
    if ((filter_distance || filter_ratio) &&
        ((width & (width - 1)) != 0 || std::fegetround() != FE_TONEAREST))
        return false;

    std::vector<PackedRow> packed_left(left_rows);
    std::vector<PackedRow> packed_right(right_rows);
    if (!pack(left, width, packed_left) || !pack(right, width, packed_right))
        return false;

    const auto no_count = static_cast<unsigned>(width + 1);
    const auto words = (width + 63) / 64;
    std::vector<Nearest> reverse(options.cross_check ? right_rows : 0, Nearest{no_count});
    std::vector<RowMatches> rows;
    rows.reserve(left_rows);
    for (std::size_t i = 0; i < left_rows; ++i) {
        Nearest best{no_count};
        auto second = no_count;
        for (std::size_t j = 0; j < right_rows; ++j) {
            unsigned count = 0;
            for (std::size_t k = 0; k < words; ++k)
                count += popcount(packed_left[i][k] ^ packed_right[j][k]);
            if (count < best.count) {
                second = best.count;
                best = {count, j};
            } else if (count < second) {
                second = count;
            }
            if (options.cross_check && count < reverse[j].count)
                reverse[j] = {count, i};
        }
        rows.push_back({best, second});
    }

    const double normalize =
        filter_distance || filter_ratio ? 1.0 / static_cast<double>(width) : 0.0;
    std::vector<DescriptorMatch> selected;
    selected.reserve(left_rows);
    for (std::size_t i = 0; i < left_rows; ++i) {
        const auto [best, second] = rows[i];
        if (options.cross_check && reverse[best.index].index != i) continue;
        if (filter_distance || filter_ratio) {
            const double best_distance = best.count * normalize;
            if (filter_distance && !(best_distance < distance)) continue;
            if (filter_ratio) {
                double alternative = second == no_count ? inf : second * normalize;
                if (alternative == 0.0) alternative = std::numeric_limits<double>::epsilon();
                if (!(best_distance / alternative < ratio)) continue;
            }
        }
        selected.push_back({static_cast<std::intptr_t>(i), static_cast<std::intptr_t>(best.index)});
    }

    matches = std::move(selected);
    return true;
}

} // namespace skimagexx
