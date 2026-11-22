#include "skimagexx/hough_ellipse.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace skimagexx {
namespace {

constexpr int point_limit = 128;
constexpr std::size_t edge_limit = 1823;
constexpr double pi = 0x1.921fb54442d18p+1;

struct Point {
    double row;
    double column;
};

// An ascending exact-dyadic grid with NumPy's exclusive arange stop and
// right-inclusive last histogram bin. Return the first maximum bin.
std::optional<std::pair<int, double>> histogram(const double* values, std::size_t size, double step,
    std::array<double, edge_limit>& edges, std::array<int, edge_limit>& counts) {
    const auto end = *std::max_element(values, values + size);
    const auto bins = std::size_t(std::ceil((end + step) / step));
    if (bins < 2 || bins > edges.size())
        return std::nullopt;
    for (std::size_t i = 0; i < bins; ++i)
        edges[i] = double(i) * step;
    std::fill_n(counts.begin(), bins - 1, 0);
    for (std::size_t i = 0; i < size; ++i) {
        const double value = values[i];
        if (value >= 0.0 && value <= edges[bins - 1]) {
            const auto top = std::upper_bound(edges.begin(), edges.begin() + bins, value);
            const auto index = std::size_t(top - edges.begin() - 1);
            ++counts[std::min(index, bins - 2)];
        }
    }
    const auto peak = std::max_element(counts.begin(), counts.begin() + bins - 1);
    return std::pair<int, double>{*peak, edges[std::size_t(peak - counts.begin())]};
}

}  // namespace

std::optional<HoughEllipseResult> hough_ellipse_bounded(const std::uint8_t* image, std::size_t image_count,
    int height, int width, int threshold, double accuracy, int min_size, std::optional<int> max_size) {
    if (height < 0 || height > 64 || width < 0 || width > 64 || threshold < 0 || threshold > 128 ||
        min_size < 1 || min_size > 64 || (max_size && (*max_size < 1 || *max_size > 32)) ||
        (accuracy != 0.75 && accuracy != 1.0 && accuracy != 1.5 && accuracy != 2.0) ||
        std::fegetround() != FE_TONEAREST)
        return std::nullopt;
    if (image_count != std::size_t(height) * width ||
        (image_count && (!image || image_count > std::numeric_limits<std::uintptr_t>::max() -
                                                    reinterpret_cast<std::uintptr_t>(image))))
        throw std::invalid_argument("invalid hough_ellipse image span");

    HoughEllipseResult output;
    std::array<Point, point_limit> points;
    std::size_t foreground = 0;
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            if (image[row * width + column]) {
                if (foreground == points.size())
                    return std::nullopt;
                points[foreground++] = {double(row), double(column)};
            }
        }
    }
    if (!foreground)
        return output;
    output.has_foreground = true;

    const int extent = std::min(height, width);
    const int half = extent / 2;
    const double minor_limit = max_size.value_or(half + (extent % 2 && half % 2));
    const double max_minor_squared = minor_limit * minor_limit;
    const double step = accuracy * accuracy;
    std::array<double, point_limit> minor_squares;
    std::array<double, edge_limit> edges;
    std::array<int, edge_limit> counts;
    for (std::size_t i = 0; i < foreground; ++i) {
        const auto p1 = points[i];
        for (std::size_t j = 0; j < i; ++j) {
            const auto p2 = points[j];
            const double pair_dx = p1.column - p2.column;
            const double pair_dy = p1.row - p2.row;
            double a = 0.5 * std::sqrt(pair_dx * pair_dx + pair_dy * pair_dy);
            if (a <= 0.5 * min_size)
                continue;
            const double xc = 0.5 * (p1.column + p2.column);
            const double yc = 0.5 * (p1.row + p2.row);
            std::size_t accepted = 0;
            for (std::size_t k = 0; k < foreground; ++k) {
                const auto p3 = points[k];
                const double center_dx = p3.column - xc;
                const double center_dy = p3.row - yc;
                const double d = std::sqrt(center_dx * center_dx + center_dy * center_dy);
                if (d <= min_size)
                    continue;
                const double dx = p3.column - p1.column;
                const double dy = p3.row - p1.row;
                double cosine_squared = (a * a + d * d - dx * dx - dy * dy) / (2 * a * d);
                cosine_squared *= cosine_squared;
                const double divisor = a * a - d * d * cosine_squared;
                if (divisor > 0 && cosine_squared < 1) {
                    const double b_squared = a * a * d * d * (1 - cosine_squared) / divisor;
                    if (b_squared <= max_minor_squared)
                        minor_squares[accepted++] = b_squared;
                }
            }
            if (!accepted)
                continue;
            const auto peak = histogram(minor_squares.data(), accepted, step, edges, counts);
            if (!peak)
                return std::nullopt;
            if (peak->first <= threshold)
                continue;
            double b = std::sqrt(peak->second);
            double orientation = std::atan2(pair_dx, pair_dy);
            if (orientation != 0.0) {
                orientation = pi - orientation;
                if (orientation > pi) {
                    orientation -= pi / 2.0;
                    std::swap(a, b);
                }
            }
            output.detections.push_back({peak->first, yc, xc, a, b, orientation});
        }
    }
    return output;
}

}  // namespace skimagexx
