#include "skimagexx/random_shapes.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

struct Span {
    std::uintptr_t begin;
    std::uintptr_t end;
};

template<class T>
Span extent(const T* data, std::size_t count) {
    if (count == 0)
        return {0, 0};
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if (!data || address % alignof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(T))
        throw std::invalid_argument("invalid random_shapes buffer span");
    return {address, address + count * sizeof(T)};
}

bool overlaps(Span a, Span b) {
    return a.begin < b.end && b.begin < a.end;
}

}  // namespace

bool random_shape_paint_into(std::uint8_t* image, std::size_t image_count,
    std::uint8_t* occupied, std::size_t occupied_count, int height, int width, int channels,
    RandomShapeIndexView rows, RandomShapeIndexView columns, std::size_t count,
    const std::uint8_t* color, std::size_t color_count, bool allow_overlap) {
    if (height < 1 || height > 256 || width < 1 || width > 256 || channels < 1 || channels > 8 ||
        image_count != std::size_t(height * width * channels) || occupied_count != std::size_t(height * width) ||
        color_count != std::size_t(channels) || count > occupied_count ||
        rows.stride < 1 || rows.stride > 2 || columns.stride < 1 || columns.stride > 2)
        throw std::invalid_argument("random_shapes storage and shape differ");

    const auto out = extent(image, image_count);
    const auto mask = extent(occupied, occupied_count);
    const std::array read_spans{
        extent(rows.data, count ? (count - 1) * rows.stride + 1 : 0),
        extent(columns.data, count ? (count - 1) * columns.stride + 1 : 0),
        extent(color, color_count)};
    if (overlaps(out, mask) || std::any_of(read_spans.begin(), read_spans.end(), [&](Span input) {
            return overlaps(out, input) || overlaps(mask, input);
        }))
        throw std::invalid_argument("random_shapes writable storage overlaps an argument");

    bool conflict = false;
    for (std::size_t i = 0; i < count; ++i) {
        const auto y = rows.data[i * rows.stride], x = columns.data[i * columns.stride];
        if (y < 0 || y >= height || x < 0 || x >= width)
            throw std::invalid_argument("random_shapes coordinate is outside the image");
        if (!allow_overlap && occupied[y * width + x])
            conflict = true;
    }
    if (conflict)
        return false;
    for (std::size_t i = 0; i < count; ++i) {
        const auto index = rows.data[i * rows.stride] * width + columns.data[i * columns.stride];
        std::copy_n(color, channels, image + index * channels);
        occupied[index] = 1;
    }
    return true;
}

}  // namespace skimagexx
