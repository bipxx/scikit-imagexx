#include "skimagexx/hough_circle.hpp"
#include "skimagexx/circle_perimeter.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

struct Pixel {
    int row;
    int column;
};

template<class T>
std::uintptr_t span_end(const T* data, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (!data || begin % alignof(T) || count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        throw std::invalid_argument("invalid hough_circle buffer span");
    return begin + count * sizeof(T);
}

}  // namespace

std::optional<HoughCirclePlane> hough_circle_plane(int height, int width,
    const std::vector<int>& radii, bool full_output) {
    if (height < 1 || height > 512 || width < 1 || width > 512 || radii.empty() || radii.size() > 16)
        return std::nullopt;
    int max_radius = 0;
    std::size_t visits = 0;
    for (int radius : radii) {
        if (radius < 0 || radius > 64)
            return std::nullopt;
        max_radius = std::max(max_radius, radius);
        visits += 8 * std::size_t(radius + 1);
    }
    if (std::size_t(height) * width > 67108864 / visits)
        return std::nullopt;
    const int pad = full_output ? max_radius : 0;
    HoughCirclePlane plane{height + 2 * pad, width + 2 * pad, pad};
    if (std::size_t(plane.height) * plane.width * radii.size() > 4194304)
        return std::nullopt;
    return plane;
}

void hough_circle_into(const std::uint8_t* image, std::size_t image_count, int height, int width,
    const std::vector<int>& radii, bool normalize, bool full_output, double* output, std::size_t output_count) {
    const auto plane = hough_circle_plane(height, width, radii, full_output);
    if (!plane || image_count != std::size_t(height) * width ||
        output_count != std::size_t(plane->height) * plane->width * radii.size() || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("hough_circle shape, counts, controls or round mode are outside the bounded domain");
    const auto image_end = span_end(image, image_count);
    const auto radii_end = span_end(radii.data(), radii.size());
    const auto output_end = span_end(output, output_count);
    const auto output_begin = reinterpret_cast<std::uintptr_t>(output);
    if ((output_begin < image_end && reinterpret_cast<std::uintptr_t>(image) < output_end) ||
        (output_begin < radii_end && reinterpret_cast<std::uintptr_t>(radii.data()) < output_end))
        throw std::invalid_argument("hough_circle output overlaps an input");

    std::vector<Pixel> nonzero;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (image[row * width + col])
                nonzero.push_back({row, col});
        }
    }
    std::fill_n(output, output_count, 0.0);
    if (nonzero.empty())
        return;

    const int pitch = plane->width;
    const int area = plane->height * pitch;
    std::array<std::intptr_t, 520> rows, columns;
    std::array<int, 520> shifts;
    for (std::size_t i = 0; i < radii.size(); ++i) {
        const int radius = radii[i];
        detail::CirclePerimeter circle;
        if (!detail::prepare_circle_perimeter({0, 0, radius}, circle))
            throw std::logic_error("bounded hough_circle perimeter was refused");
        const auto taps = detail::circle_perimeter_count(circle, 1);
        detail::circle_perimeter_fill(circle, {rows.data(), columns.data(), 1, 1}, 1);
        for (std::size_t k = 0; k < taps; ++k)
            shifts[k] = int(rows[k]) * pitch + int(columns[k]);
        const double increment = normalize ? 1.0 / double(taps) : 1.0;
        auto* layer = output + i * area;
        for (const auto& pixel : nonzero) {
            if (plane->padding || (pixel.row >= radius && pixel.row < height - radius &&
                                   pixel.column >= radius && pixel.column < width - radius)) {
                auto* center = layer + (pixel.row + plane->padding) * pitch + pixel.column + plane->padding;
                for (std::size_t k = 0; k < taps; ++k)
                    center[shifts[k]] += increment;
            } else {
                for (std::size_t k = 0; k < taps; ++k) {
                    const int row = pixel.row + int(rows[k]);
                    const int col = pixel.column + int(columns[k]);
                    if (row >= 0 && row < height && col >= 0 && col < width)
                        layer[row * pitch + col] += increment;
                }
            }
        }
    }
}

}  // namespace skimagexx
