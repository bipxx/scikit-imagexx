#include "skimagexx/local_extrema.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

struct Position {
    std::size_t row;
    std::size_t column;
};

constexpr std::uint8_t unseen = 0;
constexpr std::uint8_t extremum = 1;
constexpr std::uint8_t rejected = 2;
constexpr std::uint8_t visiting = 3;

template<bool Corners, class Visit>
bool any_neighbor(Position p, std::size_t rows, std::size_t columns, const Visit& visit) {
    const auto [y, x] = p;
    const bool up = y > 0;
    const bool down = y + 1 < rows;
    const bool left = x > 0;
    const bool right = x + 1 < columns;
    if ((left && visit(y, x - 1)) || (right && visit(y, x + 1)) ||
        (up && visit(y - 1, x)) || (down && visit(y + 1, x)))
        return true;
    if constexpr (Corners) {
        return (up && ((left && visit(y - 1, x - 1)) || (right && visit(y - 1, x + 1)))) ||
               (down && ((left && visit(y + 1, x - 1)) || (right && visit(y + 1, x + 1))));
    } else {
        return false;
    }
}

template<class T, bool Minima, bool Corners>
void mark_extrema(const T* image, std::size_t rows, std::size_t columns,
                  bool allow_borders, std::uint8_t* mask, std::size_t mask_stride) {
    const auto better = [](T value, T level) {
        if constexpr (Minima)
            return value < level;
        else
            return value > level;
    };
    const auto disallowed_border = [&](Position p) {
        return !allow_borders && (p.row == 0 || p.row + 1 == rows ||
                                  p.column == 0 || p.column + 1 == columns);
    };

    std::vector<Position> plateau;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            auto& initial = mask[row * mask_stride + column];
            if (initial != unseen)
                continue;

            const Position start{row, column};
            const T level = image[row * columns + column];
            bool tied = false;
            const bool inadmissible = disallowed_border(start) ||
                any_neighbor<Corners>(start, rows, columns, [&](auto y, auto x) {
                    const T value = image[y * columns + x];
                    if (better(value, level))
                        return true;
                    if (value != level)
                        return false;
                    tied = true;
                    return mask[y * mask_stride + x] == rejected;
                });
            if (inadmissible || !tied) {
                initial = inadmissible ? rejected : extremum;
                continue;
            }

            // A better neighbor or a previously rejected equal neighbor
            // disqualifies the entire plateau. Mark the visited subset and
            // let remaining equal pixels find that rejection in later scans.
            plateau.clear();
            plateau.push_back(start);
            initial = visiting;
            bool reject = false;
            bool has_strict_neighbor = false;
            for (std::size_t i = 0; i < plateau.size() && !reject; ++i) {
                const Position p = plateau[i];
                reject = disallowed_border(p) ||
                    any_neighbor<Corners>(p, rows, columns, [&](auto y, auto x) {
                        const T value = image[y * columns + x];
                        if (value != level) {
                            has_strict_neighbor = true;
                            return better(value, level);
                        }
                        auto& flag = mask[y * mask_stride + x];
                        if (flag == unseen) {
                            plateau.push_back({y, x});
                            flag = visiting;
                        }
                        return flag == rejected;
                    });
            }
            // No unequal neighbor can occur only for a constant image.
            const auto state = !reject && has_strict_neighbor ? extremum : rejected;
            for (const auto p : plateau)
                mask[p.row * mask_stride + p.column] = state;
        }
    }
}

}  // namespace

template<class T>
void local_extrema_2d_into(const T* image, std::size_t image_count,
                           std::size_t rows, std::size_t columns,
                           LocalExtremaKind kind, LocalExtremaNeighbors neighbors,
                           bool allow_borders, std::uint8_t* mask,
                           std::size_t mask_count, std::size_t mask_stride) {
    const auto max = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (!image || !mask || rows < 3 || columns < 3 || mask_stride < columns ||
        rows > max / columns || image_count < rows * columns ||
        mask_stride > (max - columns) / (rows - 1) ||
        mask_count < (rows - 1) * mask_stride + columns ||
        (kind != LocalExtremaKind::maxima && kind != LocalExtremaKind::minima) ||
        (neighbors != LocalExtremaNeighbors::four && neighbors != LocalExtremaNeighbors::eight)) {
        throw std::invalid_argument("Invalid local extrema image, mask or options");
    }

    const bool corners = neighbors == LocalExtremaNeighbors::eight;
    for (std::size_t row = 0; row < rows; ++row)
        std::fill_n(mask + row * mask_stride, columns, unseen);

    if (kind == LocalExtremaKind::minima) {
        if (corners)
            mark_extrema<T, true, true>(image, rows, columns, allow_borders, mask, mask_stride);
        else
            mark_extrema<T, true, false>(image, rows, columns, allow_borders, mask, mask_stride);
    } else if (corners) {
        mark_extrema<T, false, true>(image, rows, columns, allow_borders, mask, mask_stride);
    } else {
        mark_extrema<T, false, false>(image, rows, columns, allow_borders, mask, mask_stride);
    }
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
            auto& flag = mask[row * mask_stride + column];
            flag = std::uint8_t(flag == extremum);
        }
}

#define SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(T) \
    template void local_extrema_2d_into<T>(const T*, std::size_t, std::size_t, std::size_t, \
                                           LocalExtremaKind, LocalExtremaNeighbors, bool, \
                                           std::uint8_t*, std::size_t, std::size_t)
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::uint8_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::int8_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::uint16_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::int16_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::uint32_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::int32_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::uint64_t);
SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE(std::int64_t);
#undef SKIMAGEXX_LOCAL_EXTREMA_INSTANTIATE

}  // namespace skimagexx
