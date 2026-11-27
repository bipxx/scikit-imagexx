#include "skimagexx/hough_circle.hpp"
#include "skimagexx/circle_perimeter.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

constexpr int max_circle_radius = 64;
constexpr std::size_t max_circle_taps = 8 * (max_circle_radius + 1);

#ifdef _MSC_VER
#define HOUGH_CIRCLE_RESTRICT __restrict
#else
#define HOUGH_CIRCLE_RESTRICT __restrict__
#endif

void add_mask(std::uint16_t* HOUGH_CIRCLE_RESTRICT counts,
              const std::uint16_t* HOUGH_CIRCLE_RESTRICT mask, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i)
        counts[i] += mask[i];
}

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
        if (radius < 0 || radius > max_circle_radius)
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

    // A halo also handles clipped votes without a bounds check on each tap.
    const int pad = *std::max_element(radii.begin(), radii.end());
    const int pitch = width + 2 * pad;
    const int area = plane->height * plane->width;
    std::size_t foreground = 0;
    for (std::size_t p = 0; p < image_count; ++p)
        foreground += image[p] != 0;
    if (!foreground) {
        std::fill_n(output, output_count, 0.0);
        return;
    }

    const int first_center = pad * pitch + pad;
    const auto mask_span = std::size_t(height - 1) * pitch + width;
    // For dense masks, each shift can add a contiguous span to the halo.
    // Compare against the padded span because its row gaps are also processed.
    const bool dense = foreground > mask_span / 2;
    std::vector<std::uint16_t> mask(dense ? mask_span : 0);
    std::vector<int> nonzero;
    if (!dense)
        nonzero.reserve(foreground);
    for (int row = 0; row < height; ++row) {
        const auto* image_row = image + row * width;
        if (dense) {
            auto* mask_row = mask.data() + row * pitch;
            for (int col = 0; col < width; ++col)
                mask_row[col] = image_row[col] != 0;
        } else {
            const int origin = first_center + row * pitch;
            for (int col = 0; col < width; ++col) {
                if (image_row[col])
                    nonzero.push_back(origin + col);
            }
        }
    }

    std::vector<std::uint16_t> counts;
    std::array<const double*, max_circle_radius + 1> completed{};
    std::array<std::intptr_t, max_circle_taps> rows, columns;
    std::array<int, max_circle_taps> shifts;
    std::array<double, max_circle_taps + 1> values;
    for (std::size_t i = 0; i < radii.size(); ++i) {
        const int radius = radii[i];
        auto* layer = output + i * area;
        if (completed[radius]) {
            std::copy_n(completed[radius], area, layer);
            continue;
        }
        completed[radius] = layer;

        detail::CirclePerimeter circle;
        if (!detail::prepare_circle_perimeter({0, 0, radius}, circle))
            throw std::logic_error("bounded hough_circle perimeter was refused");
        const auto taps = detail::circle_perimeter_count(circle, 1);
        detail::circle_perimeter_fill(circle, {rows.data(), columns.data(), 1, 1}, 1);
        for (std::size_t k = 0; k < taps; ++k)
            shifts[k] = int(rows[k]) * pitch + int(columns[k]);
        const double increment = normalize ? 1.0 / double(taps) : 1.0;
        // Each tap contributes at most once to any cell. Repeated addition
        // gives each count the same rounding as the floating-point vote loop.
        values[0] = 0.0;
        for (std::size_t k = 0; k < taps; ++k)
            values[k + 1] = values[k] + increment;
        if (radius == 0) {
            if (full_output && !dense) {
                std::fill_n(layer, area, 0.0);
                for (int center : nonzero) layer[center] = values[taps];
            } else if (full_output) {
                std::fill_n(layer, area, 0.0);
                auto* dest = layer + first_center;
                for (std::size_t p = 0; p < mask_span; ++p)
                    dest[p] = mask[p] ? values[taps] : 0.0;
            } else {
                for (std::size_t p = 0; p < image_count; ++p)
                    layer[p] = image[p] ? values[taps] : 0.0;
            }
            continue;
        }
        counts.assign(std::size_t(height + 2 * pad) * pitch, 0);
        if (dense) {
            for (std::size_t k = 0; k < taps; ++k)
                add_mask(counts.data() + first_center + shifts[k], mask.data(), mask_span);
        } else {
            for (int index : nonzero) {
                auto* center = counts.data() + index;
                for (std::size_t k = 0; k < taps; ++k)
                    ++center[shifts[k]];
            }
        }
        const int crop = full_output ? 0 : pad;
        for (int row = 0; row < plane->height; ++row) {
            const auto* source = counts.data() + (row + crop) * pitch + crop;
            auto* dest = layer + row * plane->width;
            for (int col = 0; col < plane->width; ++col)
                dest[col] = values[source[col]];
        }
    }
}

}  // namespace skimagexx
