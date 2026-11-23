#include "skimagexx/hough_circle_peaks.hpp"
#include "skimagexx/components.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

// For nonnegative votes, constant-zero padding cannot increase the maximum.
void maximum_line(const double* source, double* target, int length, int stride, int radius) {
    std::array<int, 256> deque;
    int head = 0, tail = 0, next = 0;
    for (int position = 0; position < length; ++position) {
        while (head < tail && deque[head] < position - radius)
            ++head;
        for (; next < length && next <= position + radius; ++next) {
            while (head < tail && source[deque[tail - 1] * stride] <= source[next * stride])
                --tail;
            deque[tail++] = next;
        }
        target[position * stride] = source[deque[head] * stride];
    }
}

struct Component {
    double maximum = 0;
    int samples = 0;
    int row_sum = 0;
    int column_sum = 0;
    int first_id = 0;
};

int rounded_centroid(int sum, int samples) {
    const int whole = sum / samples;
    const int twice_remainder = 2 * (sum % samples);
    return whole + (twice_remainder > samples || (twice_remainder == samples && whole % 2));
}

void suppress(double* filtered, int height, int width, int row, int column, int radius_x, int radius_y) {
    const int left = column - radius_x;
    const int right = column + radius_x + 1;
    // The original shared local peak helper excludes row zero, including before wrapping x.
    for (int y = std::max(1, row - radius_y); y < std::min(height, row + radius_y + 1); ++y) {
        auto* direct = filtered + y * width;
        std::fill(direct + std::max(0, left), direct + std::min(width, right), 0.0);
        auto* mirror = filtered + (height - y) * width;
        if (left < 0)
            std::fill(mirror + width + left, mirror + width, 0.0);
        if (right > width)
            std::fill(mirror, mirror + right - width, 0.0);
    }
}

}  // namespace

std::optional<std::vector<HoughCirclePeak>> hough_circle_peak_indices(const double* plane,
    std::size_t count, int height, int width, int min_xdistance, int min_ydistance,
    std::optional<double> threshold) {
    if (height < 1 || height > 256 || width < 1 || width > 256 ||
        count != std::size_t(height * width) || min_xdistance < 0 || min_xdistance > std::min(32, width) ||
        min_ydistance < 0 || min_ydistance > 32 || (threshold && (!std::isfinite(*threshold) || *threshold < 0)))
        throw std::invalid_argument("hough_circle_peaks plane or controls are outside the bounded domain");
    const auto address = reinterpret_cast<std::uintptr_t>(plane);
    if (!plane || address % alignof(double) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(double))
        throw std::invalid_argument("invalid hough_circle_peaks plane span");
    if (std::fegetround() != FE_TONEAREST)
        return std::nullopt;

    double maximum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const double vote = plane[i];
        // Keep provider floating-point warning behavior for tiny subnormal arithmetic.
        if (!std::isfinite(vote) || vote < 0 || (vote > 0 && vote < 2 * std::numeric_limits<double>::min()))
            return std::nullopt;
        maximum = std::max(maximum, vote);
    }
    const double level = threshold.value_or(0.5 * maximum);
    std::vector<HoughCirclePeak> peaks;
    if (maximum <= level)
        return peaks;

    std::vector<double> vertical(count), filtered(count);
    for (int x = 0; x < width; ++x)
        maximum_line(plane + x, vertical.data() + x, height, width, min_ydistance);
    for (int row = 0; row < height; ++row)
        maximum_line(vertical.data() + row * width, filtered.data() + row * width, width, 1, min_xdistance);

    std::vector<std::uint8_t> mask(count);
    for (std::size_t i = 0; i < count; ++i)
        mask[i] = plane[i] == filtered[i] && plane[i] > level;
    std::vector<std::int32_t> labels(count);
    const auto component_count = components_label_bool_into(mask.data(), labels.data(), false,
        {std::size_t(height), std::size_t(width)}, 0, 2);
    std::vector<Component> components(component_count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int position = y * width + x;
            const auto id = labels[position];
            if (!id)
                continue;
            auto& c = components[id - 1];
            c.maximum = std::max(c.maximum, filtered[position]);
            ++c.samples;
            c.row_sum += y;
            c.column_sum += x;
            c.first_id = id;
        }
    }
    std::sort(components.begin(), components.end(), [](const Component& a, const Component& b) {
        return a.maximum == b.maximum ? a.first_id > b.first_id : a.maximum > b.maximum;
    });

    peaks.reserve(components.size());
    for (const auto& c : components) {
        const int row = rounded_centroid(c.row_sum, c.samples);
        const int column = rounded_centroid(c.column_sum, c.samples);
        const double vote = filtered[row * width + column];
        if (vote > level) {
            suppress(filtered.data(), height, width, row, column, min_xdistance, min_ydistance);
            peaks.push_back({vote, row, column});
        }
    }
    return peaks;
}

}  // namespace skimagexx
