#include "skimagexx/convex_hull_image.hpp"
#include "polygon.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace skimagexx {
namespace {

using Point = std::pair<int, int>;

int turn(const Point& a, const Point& b, const Point& c) {
    return (b.first - a.first) * (c.second - a.second) -
           (b.second - a.second) * (c.first - a.first);
}

// Points are doubled pixel coordinates. The four offsets represent the
// middle of each pixel edge, and can be applied to the row endpoints.
void add_point(std::vector<Point>& points, int row, int col, bool offset) {
    if (offset) {
        points.emplace_back(row - 1, col);
        points.emplace_back(row + 1, col);
        points.emplace_back(row, col - 1);
        points.emplace_back(row, col + 1);
    } else {
        points.emplace_back(row, col);
    }
}

template<class Iterator>
void append_chain(std::vector<Point>& hull, Iterator begin, Iterator end) {
    const auto base = hull.size();
    for (; begin != end; ++begin) {
        while (hull.size() >= base + 2 && turn(hull[hull.size() - 2], hull.back(), *begin) <= 0)
            hull.pop_back();
        hull.push_back(*begin);
    }
    hull.pop_back();
}

} // namespace

bool convex_hull_image_into(const std::uint8_t* image, std::size_t height, std::size_t width,
                            bool offset_coordinates, bool include_borders, std::uint8_t* output) {
    if (!image || !output || height == 0 || width == 0 || height > convex_hull_image_max_extent ||
        width > convex_hull_image_max_extent)
        return false;

    std::vector<Point> points;
    points.reserve(height * (offset_coordinates ? 8 : 2));
    auto first_row = height, last_row = std::size_t{0};
    auto first_col = width, last_col = std::size_t{0};
    for (std::size_t row = 0; row < height; ++row) {
        auto start = width, end = std::size_t{0};
        const auto* line = image + row * width;
        for (std::size_t col = 0; col < width; ++col) {
            const auto value = line[col];
            if (value > 1) return false;
            if (value) {
                if (start == width) start = col;
                end = col;
            }
        }
        if (start == width) continue;
        if (first_row == height) first_row = row;
        last_row = row;
        first_col = std::min(first_col, start);
        last_col = std::max(last_col, end);
        add_point(points, static_cast<int>(2 * row), static_cast<int>(2 * start), offset_coordinates);
        if (start != end)
            add_point(points, static_cast<int>(2 * row), static_cast<int>(2 * end), offset_coordinates);
    }
    if (points.size() < 3) return false;
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() < 3) return false;
    std::vector<Point> hull;
    hull.reserve(points.size());
    append_chain(hull, points.begin(), points.end());
    append_chain(hull, points.rbegin(), points.rend());
    if (hull.size() < 3) return false;

    // A half-pixel offset cannot reach the next integer coordinate outside
    // the bounding box of the foreground centers.
    std::vector<double> x, y;
    x.reserve(hull.size());
    y.reserve(hull.size());
    for (const auto& vertex : hull) {
        x.push_back(0.5 * vertex.first - static_cast<double>(first_row));
        y.push_back(0.5 * vertex.second - static_cast<double>(first_col));
    }
    const auto rows = last_row - first_row + 1;
    const auto cols = last_col - first_col + 1;
    std::vector<std::uint8_t> labels(rows * cols);
    detail::grid_points_in_poly({x.data(), y.data(), hull.size()}, rows, cols, labels.data(), 1);
    std::fill_n(output, height * width, std::uint8_t{0});
    for (std::size_t row = 0; row < rows; ++row) {
        auto* target = output + (first_row + row) * width + first_col;
        const auto* source = labels.data() + row * cols;
        for (std::size_t col = 0; col < cols; ++col)
            target[col] = include_borders ? source[col] != 0 : source[col] == 1;
    }
    return true;
}

} // namespace skimagexx
