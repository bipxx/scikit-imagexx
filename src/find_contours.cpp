#include "skimagexx/find_contours.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <type_traits>

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
    static std::uint64_t bits(double value) noexcept {
        std::uint64_t result;
        std::memcpy(&result, &value, sizeof(result));
        return value == 0.0 ? 0 : result;
    }

    std::size_t operator()(ContourPoint point) const noexcept {
        const auto row = bits(point.row);
        const auto column = bits(point.column);
        auto hash = row ^ ((column << 29) | (column >> 35));
        // Integer coordinates concentrate their entropy in the high bits.
        hash = (hash ^ (hash >> 33)) * 0xff51afd7ed558ccdULL;
        return static_cast<std::size_t>(hash ^ (hash >> 33));
    }
};

class Endpoints {
    struct Entry {
        ContourPoint point{};
        std::size_t path = no_path;
    };
    std::vector<Entry> entries_;
    std::size_t size_ = 0;

    std::size_t find(ContourPoint point) const {
        const auto mask = entries_.size() - 1;
        auto slot = HashPoint{}(point) & mask;
        while (entries_[slot].path != no_path && !EqualPoint{}(entries_[slot].point, point)) {
            slot = (slot + 1) & mask;
        }
        return slot;
    }

public:
    void set(ContourPoint point, std::size_t path) {
        if (size_ >= entries_.size() / 2) {
            std::vector<Entry> larger(entries_.empty() ? 16 : entries_.size() * 2);
            entries_.swap(larger);
            for (const auto& entry : larger) {
                if (entry.path != no_path) entries_[find(entry.point)] = entry;
            }
        }
        auto& entry = entries_[find(point)];
        size_ += entry.path == no_path;
        entry = {point, path};
    }

    std::size_t take(ContourPoint point) {
        if (!size_) return no_path;
        auto hole = find(point);
        const auto path = entries_[hole].path;
        if (path == no_path) return no_path;
        --size_;

        // Close the gap for entries whose probe sequence crosses it.
        const auto mask = entries_.size() - 1;
        for (auto slot = (hole + 1) & mask; entries_[slot].path != no_path;
             slot = (slot + 1) & mask) {
            const auto home = HashPoint{}(entries_[slot].point) & mask;
            if (((hole - home) & mask) < ((slot - home) & mask)) {
                entries_[hole] = entries_[slot];
                hole = slot;
            }
        }
        entries_[hole].path = no_path;
        return path;
    }
};

class OrderedContours {
public:
    void add(ContourPoint from, ContourPoint to) {
        if (EqualPoint{}(from, to)) {
            return;
        }
        const auto tail = starts_.take(to);
        const auto head = ends_.take(from);
        if (tail == no_path && head == no_path) {
            const auto index = paths_.size();
            const auto first = add_vertex(from);
            const auto last = add_vertex(to);
            vertices_[first].next = last;
            paths_.push_back({first, last, 2});
            starts_.set(from, index);
            ends_.set(to, index);
        } else if (head == no_path) {
            auto& path = paths_[tail];
            path.first = add_vertex(from, path.first);
            ++path.size;
            starts_.set(from, tail);
        } else if (tail == no_path || head == tail) {
            auto& path = paths_[head];
            const auto last = add_vertex(to);
            vertices_[path.last].next = last;
            path.last = last;
            ++path.size;
            if (tail == no_path) ends_.set(to, head);
        } else {
            join(head, tail);
        }
    }

    Contours finish() const {
        Contours result;
        for (const auto& path : paths_) {
            if (path.size) {
                auto& contour = result.emplace_back();
                contour.reserve(path.size);
                for (auto i = path.first; i != no_path; i = vertices_[i].next) {
                    contour.push_back(vertices_[i].point);
                }
            }
        }
        return result;
    }

private:
    struct Vertex { ContourPoint point; std::size_t next; };
    struct Path { std::size_t first, last, size; };
    Endpoints starts_;
    Endpoints ends_;
    std::vector<Vertex> vertices_;
    std::vector<Path> paths_;

    std::size_t add_vertex(ContourPoint point, std::size_t next = no_path) {
        const auto index = vertices_.size();
        vertices_.push_back({point, next});
        return index;
    }

    void join(std::size_t head, std::size_t tail) {
        // Keep the earlier creation index regardless of geometric direction.
        auto& left = paths_[head];
        auto& right = paths_[tail];
        const auto first = vertices_[left.first].point;
        if (head > tail) starts_.take(first);
        vertices_[left.last].next = right.first;
        const Path merged{left.first, right.last, left.size + right.size};
        const auto keep = std::min(head, tail);
        paths_[std::max(head, tail)].size = 0;
        paths_[keep] = merged;
        starts_.set(first, keep);
        ends_.set(vertices_[merged.last].point, keep);
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
                  bool connect_high, unsigned code, double ul, double ur, double ll, double lr) {
    auto point = [&](Edge edge) -> ContourPoint {
        switch (edge) {
        case top: return {row, column + fraction(ul, ur, level)};
        case right: return {row + fraction(ur, lr, level), column + 1};
        case bottom: return {row + 1, column + fraction(ll, lr, level)};
        case left: return {row + fraction(ul, ll, level), column};
        }
        return {};
    };
    auto add = [&](Edge from, Edge to) { paths.add(point(from), point(to)); };
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
        const auto row_start = row * columns;
        auto left_code = unsigned(image[row_start] > level)
            | (unsigned(image[row_start + columns] > level) << 2);
        for (std::size_t column = 0; column + 1 < columns; ++column) {
            const auto a = row_start + column;
            const auto b = a + columns;
            const auto right_code = unsigned(image[a + 1] > level)
                | (unsigned(image[b + 1] > level) << 2);
            const auto code = left_code | (right_code << 1);
            left_code = right_code;
            if (code != 0 && code != 15
                && (!mask || (mask[a] && mask[a + 1] && mask[b] && mask[b + 1]))) {
                connect_cell(paths, double(row), double(column), level, options.connect_high,
                             code, image[a], image[a + 1], image[b], image[b + 1]);
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
