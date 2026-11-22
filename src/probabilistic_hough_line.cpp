#include "skimagexx/probabilistic_hough_line.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

constexpr std::size_t foreground_limit = 4096;
constexpr int fixed_unit = 1 << 16;

template<class T>
bool invalid_span(const T* data, std::size_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return count && (!data || address % alignof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(T));
}

int round_half_away(double value) {
    return static_cast<int>(value > 0 ? value + 0.5 : value - 0.5);
}

int fixed_floor(int value) {
    return (value < 0 ? value - (fixed_unit - 1) : value) / fixed_unit;
}

using Point = ProbabilisticHoughPoint;

struct Walk {
    Point start;
    Point step;
    bool x_major;

    Walk(Point origin, double cosine, double sine)
        : start(origin), x_major(std::abs(sine) > std::abs(cosine)) {
        const double a = -sine;
        const double b = cosine;
        if (x_major) {
            step = {a > 0 ? 1 : -1, round_half_away(b * fixed_unit / std::abs(a))};
            start.y = start.y * fixed_unit + fixed_unit / 2;
        } else {
            step = {round_half_away(a * fixed_unit / std::abs(b)), b > 0 ? 1 : -1};
            start.x = start.x * fixed_unit + fixed_unit / 2;
        }
    }

    Point position(Point fixed) const {
        return x_major ? Point{fixed.x, fixed_floor(fixed.y)} : Point{fixed_floor(fixed.x), fixed.y};
    }
};

template<class Visit>
void follow(const Walk& walk, int direction, Visit visit) {
    Point fixed = walk.start;
    while (visit(walk.position(fixed))) {
        fixed.x += direction * walk.step.x;
        fixed.y += direction * walk.step.y;
    }
}

}  // namespace

std::vector<ProbabilisticHoughSegment> probabilistic_hough_line_ordered(
    const std::uint8_t* image, std::size_t image_count, int height, int width,
    const double* cosines, const double* sines, std::size_t angle_count,
    const std::intptr_t* permutation, std::size_t count, int threshold, int line_length, int line_gap) {
    if (height < 1 || height > 256 || width < 1 || width > 256 || image_count != std::size_t(height * width) ||
        count > foreground_limit || angle_count > 256 || threshold < 1 || threshold > 4096 ||
        line_length < 0 || line_length > 256 || line_gap < 0 || line_gap > 256 ||
        invalid_span(image, image_count) || invalid_span(permutation, count) ||
        invalid_span(cosines, angle_count) || invalid_span(sines, angle_count) || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("probabilistic_hough_line input is outside the bounded domain");

    const int offset = int(std::ceil(std::sqrt(double(height * height + width * width))));
    for (std::size_t k = 0; k < angle_count; ++k) {
        const double x = std::abs(cosines[k]);
        const double y = std::abs(sines[k]);
        if (!std::isfinite(x) || !std::isfinite(y) || x > 1.0 || y > 1.0 || std::max(x, y) < 0.5 ||
            x * (width - 1) + y * (height - 1) >= offset - 0.5)
            throw std::invalid_argument("probabilistic_hough_line direction exceeds the distance grid");
    }

    std::array<bool, foreground_limit> seen{};
    for (std::size_t k = 0; k < count; ++k) {
        const auto index = std::size_t(permutation[k]);
        if (index >= count || seen[index])
            throw std::invalid_argument("probabilistic_hough_line requires a foreground permutation");
        seen[index] = true;
    }
    std::vector<Point> points;
    points.reserve(count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image[y * width + x]) {
                if (points.size() == count)
                    throw std::invalid_argument("probabilistic_hough_line foreground count differs");
                points.push_back({x, y});
            }
        }
    }
    if (points.size() != count)
        throw std::invalid_argument("probabilistic_hough_line foreground count differs");

    std::vector<ProbabilisticHoughSegment> lines;
    if (!angle_count || !count)
        return lines;
    std::vector<std::uint8_t> mask(image, image + image_count);
    std::vector<int> accumulator(std::size_t(2 * offset) * angle_count);
    const auto vote = [&](Point p, std::size_t angle) -> int& {
        const double projection = cosines[angle] * p.x + sines[angle] * p.y;
        return accumulator[std::size_t(offset + round_half_away(projection)) * angle_count + angle];
    };
    while (count > 0) {
        const Point start = points[std::size_t(permutation[--count])];
        if (!mask[start.y * width + start.x])
            continue;
        int strongest = threshold - 1;
        std::size_t winner = 0;
        for (std::size_t k = 0; k < angle_count; ++k) {
            const int value = ++vote(start, k);
            if (value > strongest) {
                strongest = value;
                winner = k;
            }
        }
        if (strongest < threshold)
            continue;

        const Walk walk(start, cosines[winner], sines[winner]);
        const auto endpoint = [&](int direction) {
            Point last = start;
            int gap = 0;
            follow(walk, direction, [&](Point point) {
                if (point.x < 0 || point.x >= width || point.y < 0 || point.y >= height)
                    return false;
                if (mask[point.y * width + point.x]) {
                    last = point;
                    gap = 0;
                } else if (++gap > line_gap) {
                    return false;
                }
                return true;
            });
            return last;
        };
        const Point first = endpoint(1);
        const Point second = endpoint(-1);
        if (std::abs(first.x - second.x) < line_length && std::abs(first.y - second.y) < line_length)
            continue;

        const auto remove_votes = [&](int direction, Point end) {
            follow(walk, direction, [&](Point point) {
                auto& marked = mask[point.y * width + point.x];
                if (marked) {
                    marked = 0;
                    // All marked pixels are removed, including pixels still waiting to vote.
                    for (std::size_t k = 0; k < angle_count; ++k)
                        --vote(point, k);
                }
                return point.x != end.x || point.y != end.y;
            });
        };
        remove_votes(1, first);
        remove_votes(-1, second);
        lines.push_back({first, second});
        // The original cap exceeds the maximum foreground count in this domain.
    }
    return lines;
}

}  // namespace skimagexx
