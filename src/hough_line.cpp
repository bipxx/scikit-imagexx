#include "skimagexx/hough_line.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

struct ByteSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

template<class T>
ByteSpan byte_span(const T* data, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || begin % alignof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        throw std::invalid_argument("invalid hough_line buffer");
    return {begin, begin + count * sizeof(T)};
}

bool overlaps(ByteSpan left, ByteSpan right) {
    return left.begin < right.end && right.begin < left.end;
}

}  // namespace

int hough_line_offset(std::size_t height, std::size_t width, std::size_t angles) {
    if (height == 0 || width == 0 || height > 4096 || width > 4096 ||
        angles == 0 || angles > 1024 || height * width > 1048576 ||
        height * width > 134217728 / angles)
        throw std::invalid_argument("hough_line shape is outside the bounded domain");
    return static_cast<int>(std::ceil(std::sqrt(double(height * height + width * width))));
}

void hough_line_into(const std::uint8_t* image, std::size_t image_count,
                     std::size_t height, std::size_t width,
                     const double* cosines, const double* sines, std::size_t angles,
                     std::uint64_t* accumulator, std::size_t accumulator_count) {
    if (std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("hough_line requires round-to-nearest");
    const auto offset = hough_line_offset(height, width, angles);
    if (image_count != height * width || accumulator_count != std::size_t(2 * offset + 1) * angles)
        throw std::invalid_argument("hough_line buffer counts must match the image and accumulator");
    const auto output = byte_span(accumulator, accumulator_count);
    if (overlaps(output, byte_span(image, image_count)) ||
        overlaps(output, byte_span(cosines, angles)) || overlaps(output, byte_span(sines, angles)))
        throw std::invalid_argument("hough_line output overlaps input storage");
    for (std::size_t angle = 0; angle < angles; ++angle) {
        const double cosine = cosines[angle];
        const double sine = sines[angle];
        if (!std::isfinite(cosine) || !std::isfinite(sine) ||
            std::abs(cosine) * double(width - 1) + std::abs(sine) * double(height - 1) > offset)
            throw std::invalid_argument("hough_line coefficients exceed the distance grid");
    }

    std::fill_n(accumulator, accumulator_count, std::uint64_t{0});
    for (std::size_t row = 0; row < height; ++row) {
        const auto* pixels = image + row * width;
        for (std::size_t column = 0; column < width; ++column) {
            if (pixels[column] == 0)
                continue;
            for (std::size_t angle = 0; angle < angles; ++angle) {
                // The pinned rounding helper separately adjusts by half before truncating.
                const double projection = cosines[angle] * double(column) + sines[angle] * double(row);
                const double adjusted = projection > 0.0 ? projection + 0.5 : projection - 0.5;
                const auto distance = std::size_t(offset + static_cast<int>(adjusted));
                ++accumulator[distance * angles + angle];
            }
        }
    }
}

}  // namespace skimagexx
