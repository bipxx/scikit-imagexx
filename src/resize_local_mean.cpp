#include "skimagexx/resize_local_mean.hpp"
#include "skimagexx/downscale.hpp"

#include <cfenv>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

struct Span {
    std::uintptr_t begin, end;
};

Span checked_span(const void* data, std::size_t count, std::size_t item_size, std::size_t alignment) {
    const auto start = reinterpret_cast<std::uintptr_t>(data);
    if (!data || start % alignment || count > (std::numeric_limits<std::uintptr_t>::max() - start) / item_size)
        throw std::invalid_argument("invalid resize_local_mean buffer");
    return {start, start + count * item_size};
}

template<class Word>
void normalized_two_by_two(const Word* source, int width, double* output, int rows, int columns) {
    constexpr double scale = 1.0 / double(std::numeric_limits<Word>::max());
    for (int y = 0; y < rows; ++y) {
        const auto* first = source + 2 * y * width;
        const auto* second = first + width;
        for (int x = 0; x < columns; ++x) {
            const int left = 2 * x, right = left + 1;
            const double top_left = double(first[left]) * scale;
            const double bottom_left = double(second[left]) * scale;
            const double top_right = double(first[right]) * scale;
            const double bottom_right = double(second[right]) * scale;
            // Stock contracts y before x. Halving commutes with each rounded
            // sum here because all nonzero normalized words are normal and
            // all intermediate values are in [0, 4].
            const double column_left = top_left + bottom_left;
            const double column_right = top_right + bottom_right;
            output[y * columns + x] = (column_left + column_right) * 0.25;
        }
    }
}

bool supported_factor(int factor) {
    return factor == 2 || factor == 4 || factor == 8;
}

}  // namespace

bool resize_local_mean_into(const void* source, PixelType type, std::size_t source_count,
    int height, int width, double* output, std::size_t output_count, int rows, int columns, bool preserve_range) {
    if ((type != PixelType::uint8 && type != PixelType::uint16) || height < 16 || height > 2048 ||
        width < 16 || width > 2048 || rows < 1 || rows > height || columns < 1 || columns > width ||
        height % rows || width % columns)
        return false;
    const int dy = height / rows, dx = width / columns;
    if (!supported_factor(dy) || !supported_factor(dx) || std::fegetround() != FE_TONEAREST ||
        (!preserve_range && (dy != 2 || dx != 2)))
        return false;
    const std::size_t item_size = type == PixelType::uint8 ? sizeof(std::uint8_t) : sizeof(std::uint16_t);
    if (source_count != std::size_t(height * width) || output_count != std::size_t(rows * columns))
        throw std::invalid_argument("resize_local_mean element count does not match shape");
    const auto input_span = checked_span(source, source_count, item_size, item_size);
    const auto output_span = checked_span(output, output_count, sizeof(double), alignof(double));
    if (input_span.begin < output_span.end && output_span.begin < input_span.end)
        throw std::invalid_argument("resize_local_mean input and output overlap");

    if (preserve_range) {
        const LocalMeanView view{source, type, false, false, {std::size_t(height), std::size_t(width)},
                                {std::ptrdiff_t(width * item_size), std::ptrdiff_t(item_size)}};
        return downscale_local_mean_into(view, {std::size_t(dy), std::size_t(dx)}, nullptr, output,
                                         {std::size_t(columns) * sizeof(double), sizeof(double)});
    }
    if (type == PixelType::uint8)
        normalized_two_by_two(static_cast<const std::uint8_t*>(source), width, output, rows, columns);
    else
        normalized_two_by_two(static_cast<const std::uint16_t*>(source), width, output, rows, columns);
    return true;
}

}  // namespace skimagexx
