#include "skimagexx/local_extrema.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx {

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

    const bool minima = kind == LocalExtremaKind::minima;
    const bool corners = neighbors == LocalExtremaNeighbors::eight;
    for (std::size_t row = 0; row < rows; ++row)
        std::fill_n(mask + row * mask_stride, columns, std::uint8_t{0});

    // Zero means unseen, two means discovered, and one means accepted.
    // Reject a plateau only after checking every connected equal pixel.
    std::vector<std::size_t> plateau;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            auto& initial = mask[row * mask_stride + column];
            if (initial != 0)
                continue;

            const auto first = row * columns + column;
            const T level = image[first];
            bool admissible = true;
            bool has_strict_neighbor = false;
            plateau.clear();
            plateau.push_back(first);
            initial = 2;
            for (std::size_t i = 0; i < plateau.size(); ++i) {
                const auto p = plateau[i] / columns;
                const auto q = plateau[i] % columns;
                if (!allow_borders && (p == 0 || p == rows - 1 || q == 0 || q == columns - 1))
                    admissible = false;

                for (auto y = p == 0 ? p : p - 1; y <= std::min(p + 1, rows - 1); ++y) {
                    for (auto x = q == 0 ? q : q - 1; x <= std::min(q + 1, columns - 1); ++x) {
                        if ((y == p && x == q) || (!corners && y != p && x != q))
                            continue;
                        const auto pos = y * columns + x;
                        const T value = image[pos];
                        if (value != level) {
                            has_strict_neighbor = true;
                            if (minima ? value < level : value > level)
                                admissible = false;
                        } else {
                            auto& flag = mask[y * mask_stride + x];
                            if (flag == 0) {
                                flag = 2;
                                plateau.push_back(pos);
                            }
                        }
                    }
                }
            }
            // With four or eight neighbors, no unequal neighbor means the
            // entire image is constant. Original scikit-image emits no extrema.
            if (admissible && has_strict_neighbor)
                for (const auto p : plateau)
                    mask[p / columns * mask_stride + p % columns] = 1;
        }
    }
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column) {
            auto& flag = mask[row * mask_stride + column];
            flag = std::uint8_t(flag == 1);
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
