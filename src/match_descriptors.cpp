#include "skimagexx/match_descriptors.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <cfenv>
#include <utility>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace skimagexx {
namespace {

using PackedRow = std::array<std::uint64_t, match_descriptors_max_width / 64>;

bool pack(const std::uint8_t* source, std::size_t width, std::vector<PackedRow>& rows) {
    for (auto& row : rows) {
        for (std::size_t start = 0; start < width; start += 64) {
            const auto bits = std::min<std::size_t>(64, width - start);
            std::uint64_t word = 0;
            for (std::size_t k = 0; k < bits; ++k) {
                const auto value = *source++;
                if (value > 1) return false;
                word |= std::uint64_t(value) << k;
            }
            row[start / 64] = word;
        }
    }
    return true;
}

unsigned software_popcount(std::uint64_t word) {
    word -= (word >> 1) & UINT64_C(0x5555555555555555);
    word = (word & UINT64_C(0x3333333333333333)) + ((word >> 2) & UINT64_C(0x3333333333333333));
    word = (word + (word >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return static_cast<unsigned>((word * UINT64_C(0x0101010101010101)) >> 56);
}

unsigned software_distance(const PackedRow& left, const PackedRow& right, std::size_t words) {
    unsigned count = 0;
    for (std::size_t k = 0; k < words; ++k)
        count += software_popcount(left[k] ^ right[k]);
    return count;
}

#if (defined(_MSC_VER) && defined(_M_X64)) || \
    (!defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__))
#ifndef _MSC_VER
__attribute__((target("popcnt")))
#endif
unsigned hardware_distance(const PackedRow& left, const PackedRow& right, std::size_t words) {
    unsigned count = 0;
    for (std::size_t k = 0; k < words; ++k) {
#ifdef _MSC_VER
        count += static_cast<unsigned>(__popcnt64(left[k] ^ right[k]));
#else
        count += static_cast<unsigned>(__builtin_popcountll(left[k] ^ right[k]));
#endif
    }
    return count;
}

bool has_hardware_distance() {
    if (!has_avx2()) return false;
#ifdef _MSC_VER
    static const bool supported = [] {
        int registers[4];
        __cpuid(registers, 1);
        return (registers[2] & (1 << 23)) != 0;
    }();
    return supported;
#else
    return __builtin_cpu_supports("popcnt") != 0;
#endif
}
#endif

struct Nearest {
    unsigned count;
    std::size_t index = 0;
};

struct RowMatches {
    Nearest best;
    unsigned second;
};

template<auto distance>
void find_matches(const std::vector<PackedRow>& left, const std::vector<PackedRow>& right,
                  std::size_t words, unsigned no_count, std::vector<Nearest>& reverse,
                  std::vector<RowMatches>& rows) {
    const bool cross_check = !reverse.empty();
    for (std::size_t i = 0; i < left.size(); ++i) {
        const auto& left_row = left[i];
        Nearest best{no_count};
        auto second = no_count;
        for (std::size_t j = 0; j < right.size(); ++j) {
            const auto count = distance(left_row, right[j], words);
            if (count < best.count) {
                second = best.count;
                best = {count, j};
            } else if (count < second) {
                second = count;
            }
            if (cross_check && count < reverse[j].count)
                reverse[j] = {count, i};
        }
        rows.push_back({best, second});
    }
}

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
#if (defined(_MSC_VER) && defined(_M_X64)) || \
    (!defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__))
    if (has_hardware_distance())
        find_matches<hardware_distance>(packed_left, packed_right, words, no_count, reverse, rows);
    else
#endif
        find_matches<software_distance>(packed_left, packed_right, words, no_count, reverse, rows);

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
