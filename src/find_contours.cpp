#include "skimagexx/find_contours.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <limits>
#include <list>
#include <type_traits>
#include <unordered_map>

namespace skimagexx {
namespace {

constexpr std::size_t max_pixels = 131072;
constexpr double magnitude = 0x1p40;
constexpr auto no_path = std::numeric_limits<std::size_t>::max();

struct EqualPoint {
    bool operator()(ContourPoint a, ContourPoint b) const noexcept {
        return a.row == b.row && a.column == b.column;
    }
};

struct HashPoint {
    std::size_t operator()(ContourPoint point) const noexcept {
        const auto row = std::hash<double>{}(point.row);
        const auto column = std::hash<double>{}(point.column);
        return row ^ (column + 0x9e3779b9u + (row << 6) + (row >> 2));
    }
};

class OrderedContours {
public:
    void add(ContourPoint from, ContourPoint to) {
        if (EqualPoint{}(from, to)) {
            return;
        }
        const auto tail = take(starts_, to);
        const auto head = take(ends_, from);
        if (tail == no_path && head == no_path) {
            const auto index = paths_.size();
            paths_.emplace_back(std::initializer_list<ContourPoint>{from, to});
            starts_[from] = index;
            ends_[to] = index;
        } else if (head == no_path) {
            paths_[tail].push_front(from);
            starts_[from] = tail;
        } else if (tail == no_path) {
            paths_[head].push_back(to);
            ends_[to] = head;
        } else if (head == tail) {
            paths_[head].push_back(to);
        } else {
            join(head, tail);
        }
    }

    Contours finish() const {
        Contours result;
        for (const auto& path : paths_) {
            if (!path.empty()) {
                result.emplace_back(path.begin(), path.end());
            }
        }
        return result;
    }

private:
    using Endpoints = std::unordered_map<ContourPoint, std::size_t, HashPoint, EqualPoint>;
    Endpoints starts_;
    Endpoints ends_;
    std::vector<std::list<ContourPoint>> paths_;

    static std::size_t take(Endpoints& endpoints, ContourPoint point) {
        const auto item = endpoints.find(point);
        if (item == endpoints.end()) {
            return no_path;
        }
        const auto index = item->second;
        endpoints.erase(item);
        return index;
    }

    void join(std::size_t head, std::size_t tail) {
        // Keep the earlier creation index regardless of geometric direction.
        auto& left = paths_[head];
        auto& right = paths_[tail];
        if (head < tail) {
            left.splice(left.end(), right);
            starts_[left.front()] = head;
            ends_[left.back()] = head;
        } else {
            starts_.erase(left.front());
            right.splice(right.begin(), left);
            starts_[right.front()] = tail;
            ends_[right.back()] = tail;
        }
    }
};

enum Edge : unsigned { top, right, bottom, left };
using Segment = std::array<Edge, 2>;
constexpr std::array<Segment, 16> segments{{
    {top, top}, {top, left}, {right, top}, {right, left},
    {left, bottom}, {top, bottom}, {top, top}, {right, bottom},
    {bottom, right}, {top, top}, {bottom, top}, {bottom, left},
    {left, right}, {top, right}, {left, top}, {top, top}
}};

double fraction(double from, double to, double level) {
    return from == to ? 0.0 : (level - from) / (to - from);
}

void connect_cell(OrderedContours& paths, double row, double column, double level,
                  bool connect_high, double ul, double ur, double ll, double lr) {
    const auto code = unsigned(ul > level) | (unsigned(ur > level) << 1)
        | (unsigned(ll > level) << 2) | (unsigned(lr > level) << 3);
    if (code == 0 || code == 15) {
        return;
    }
    const std::array<ContourPoint, 4> points{{
        {row, column + fraction(ul, ur, level)},
        {row + fraction(ur, lr, level), column + 1},
        {row + 1, column + fraction(ll, lr, level)},
        {row + fraction(ul, ll, level), column}
    }};
    auto add = [&](Edge from, Edge to) { paths.add(points[from], points[to]); };
    if (code == 6) {
        if (connect_high) {
            add(left, top);
            add(right, bottom);
        } else {
            add(right, top);
            add(left, bottom);
        }
    } else if (code == 9) {
        if (connect_high) {
            add(top, right);
            add(bottom, left);
        } else {
            add(top, left);
            add(bottom, right);
        }
    } else {
        add(segments[code][0], segments[code][1]);
    }
}

bool in_numeric_range(double value) {
    return value >= -magnitude && value <= magnitude;
}

template <class Pixel>
bool contours(const Pixel* image, const std::uint8_t* mask, std::size_t rows,
              std::size_t columns, ContourOptions options, Contours& output) {
    if (!image || !find_contours_shape_supported(rows, columns)
        || std::fegetround() != FE_TONEAREST
        || (options.default_level && !std::is_same_v<Pixel, double>)
        || (!options.default_level && !in_numeric_range(options.level))) {
        return false;
    }
    double level = options.level;
    if constexpr (!std::is_same_v<Pixel, std::uint8_t>) {
        double low = image[0];
        double high = low;
        for (std::size_t i = 0; i < rows * columns; ++i) {
            const double sample = image[i];
            if (!in_numeric_range(sample)) {
                return false;
            }
            if (options.default_level) {
                low = std::min(low, sample);
                high = std::max(high, sample);
            }
        }
        if (options.default_level) {
            level = (low + high) / 2.0;
        }
    }

    OrderedContours paths;
    for (std::size_t row = 0; row + 1 < rows; ++row) {
        for (std::size_t column = 0; column + 1 < columns; ++column) {
            const auto a = row * columns + column;
            const auto b = a + columns;
            if (!mask || (mask[a] && mask[a + 1] && mask[b] && mask[b + 1])) {
                connect_cell(paths, double(row), double(column), level, options.connect_high,
                             image[a], image[a + 1], image[b], image[b + 1]);
            }
        }
    }
    output = paths.finish();
    return true;
}

}  // namespace

bool find_contours_shape_supported(std::size_t rows, std::size_t columns) {
    return rows >= 2 && columns >= 2 && rows <= max_pixels / columns;
}

bool find_contours(const double* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions options, Contours& output) {
    return contours(image, mask, rows, columns, options, output);
}

bool find_contours(const float* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions options, Contours& output) {
    return contours(image, mask, rows, columns, options, output);
}

bool find_contours(const std::uint8_t* image, const std::uint8_t* mask, std::size_t rows,
                   std::size_t columns, ContourOptions options, Contours& output) {
    return contours(image, mask, rows, columns, options, output);
}

}  // namespace skimagexx
