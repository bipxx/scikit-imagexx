#include "skimagexx/hough_line_peaks.hpp"
#include "skimagexx/components.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

// Constant-zero padding does not increase maxima of a nonnegative accumulator.
void maximum_line(const std::uint64_t* input, std::uint64_t* output,
                  int length, int stride, int radius) {
    std::array<int, 1024> deque;
    int head = 0, tail = 0, next = 0;
    for (int point = 0; point < length; ++point) {
        while (head < tail && deque[head] < point - radius)
            ++head;
        for (; next < length && next <= point + radius; ++next) {
            while (head < tail && input[deque[tail - 1] * stride] <= input[next * stride])
                --tail;
            deque[tail++] = next;
        }
        output[point * stride] = input[deque[head] * stride];
    }
}

struct Component {
    std::uint64_t maximum = 0;
    std::size_t samples = 0;
    std::size_t row_sum = 0;
    std::size_t column_sum = 0;
    std::size_t first_id = 0;
};

int rounded_centroid(std::size_t sum, std::size_t samples) {
    const auto whole = sum / samples;
    const auto twice_remainder = 2 * (sum % samples);
    return int(whole + (twice_remainder > samples || (twice_remainder == samples && whole % 2)));
}

void suppress(std::uint64_t* filtered, int height, int width, int row, int column,
              int min_distance, int min_angle) {
    const int left = column - min_angle;
    const int right = column + min_angle + 1;
    // Stock excludes distance row zero before it wraps angles and mirrors rows.
    for (int y = std::max(1, row - min_distance); y < std::min(height, row + min_distance + 1); ++y) {
        auto* direct = filtered + y * width;
        std::fill(direct + std::max(0, left), direct + std::min(width, right), std::uint64_t{0});
        auto* mirror = filtered + (height - y) * width;
        if (left < 0)
            std::fill(mirror + width + left, mirror + width, std::uint64_t{0});
        if (right > width)
            std::fill(mirror, mirror + right - width, std::uint64_t{0});
    }
}

}  // namespace

std::vector<HoughLinePeak> hough_line_peak_indices(
    const std::uint64_t* accumulator, std::size_t count, int height, int width,
    int min_distance, int min_angle, std::optional<double> threshold) {
    if (height < 1 || height > 1024 || width < 1 || width > 512 ||
        count != std::size_t(height) * std::size_t(width) || count >= 262144 ||
        min_distance < 0 || min_distance > 32 || min_angle < 0 || std::min(min_angle, width) > 32 ||
        (threshold && (!std::isfinite(*threshold) || *threshold < 0)))
        throw std::invalid_argument("hough_line_peaks shape or controls are outside the bounded domain");
    const auto address = reinterpret_cast<std::uintptr_t>(accumulator);
    if (!accumulator || address % alignof(std::uint64_t) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(std::uint64_t))
        throw std::invalid_argument("invalid hough_line_peaks accumulator span");
    min_angle = std::min(min_angle, width);
    const auto maximum = *std::max_element(accumulator, accumulator + count);
    if (maximum > 1048576)
        throw std::overflow_error("hough_line_peaks accumulator votes are outside the bounded domain");
    const double level = threshold.value_or(0.5 * double(maximum));
    if (double(maximum) <= level)
        return {};

    std::vector<std::uint64_t> vertical(count);
    std::vector<std::uint64_t> filtered(count);
    for (int x = 0; x < width; ++x)
        maximum_line(accumulator + x, vertical.data() + x, height, width, min_distance);
    for (int row = 0; row < height; ++row)
        maximum_line(vertical.data() + row * width, filtered.data() + row * width, width, 1, min_angle);

    std::vector<std::uint8_t> mask(count);
    for (std::size_t i = 0; i < count; ++i)
        mask[i] = accumulator[i] == filtered[i] && double(accumulator[i]) > level;
    std::vector<std::int32_t> labels(count);
    const auto label_count = components_label_bool_into(mask.data(), labels.data(), false,
        {std::size_t(height), std::size_t(width)}, 0, 2);
    std::vector<Component> components(label_count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto position = y * width + x;
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

    std::vector<HoughLinePeak> peaks;
    peaks.reserve(components.size());
    for (const auto& c : components) {
        const auto row = rounded_centroid(c.row_sum, c.samples);
        const auto column = rounded_centroid(c.column_sum, c.samples);
        const auto value = filtered[row * width + column];
        if (double(value) > level) {
            suppress(filtered.data(), height, width, row, column, min_distance, min_angle);
            peaks.push_back({value, row, column});
        }
    }
    return peaks;
}

}  // namespace skimagexx
