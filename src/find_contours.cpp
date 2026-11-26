#include "skimagexx/find_contours.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

namespace skimagexx {
namespace {

constexpr std::size_t max_pixels = 131072;
constexpr double magnitude = 0x1p40;
// A cell emits at most two segments and a segment adds at most two vertices.
using Index = std::uint32_t;
constexpr auto no_path = std::numeric_limits<Index>::max();
static_assert(4 * max_pixels < no_path);

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

    std::uint32_t operator()(ContourPoint point) const noexcept {
        const auto row = bits(point.row);
        const auto column = bits(point.column);
        auto hash = row ^ ((column << 29) | (column >> 35));
        // Integer coordinates concentrate their entropy in the high bits.
        hash = (hash ^ (hash >> 33)) * 0xff51afd7ed558ccdULL;
        return static_cast<std::uint32_t>(hash ^ (hash >> 33));
    }
};

class Endpoints {
    struct Entry {
        ContourPoint point{};
        Index path = no_path;
        std::uint32_t hash = 0;
    };
    std::vector<Entry> entries_;
    std::size_t size_ = 0;

    std::size_t find(ContourPoint point, std::uint32_t hash) const {
        const auto mask = entries_.size() - 1;
        auto slot = hash & mask;
        while (entries_[slot].path != no_path
               && (entries_[slot].hash != hash || !EqualPoint{}(entries_[slot].point, point))) {
            slot = (slot + 1) & mask;
        }
        return slot;
    }

public:
    void set(ContourPoint point, Index path) {
        if (entries_.empty()) entries_.resize(16);
        const auto hash = HashPoint{}(point);
        auto slot = find(point, hash);
        if (entries_[slot].path != no_path) {
            entries_[slot].path = path;
            return;
        }
        if (size_ >= entries_.size() / 2) {
            std::vector<Entry> larger(entries_.size() * 2);
            entries_.swap(larger);
            for (const auto& entry : larger) {
                if (entry.path != no_path) entries_[find(entry.point, entry.hash)] = entry;
            }
            slot = find(point, hash);
        }
        ++size_;
        entries_[slot] = {point, path, hash};
    }

    Index take(ContourPoint point) {
        if (!size_) return no_path;
        auto hole = find(point, HashPoint{}(point));
        const auto path = entries_[hole].path;
        if (path == no_path) return no_path;
        --size_;

        // Close the gap for entries whose probe sequence crosses it.
        const auto mask = entries_.size() - 1;
        for (auto slot = (hole + 1) & mask; entries_[slot].path != no_path;
             slot = (slot + 1) & mask) {
            const auto home = entries_[slot].hash & mask;
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
            const auto index = static_cast<Index>(paths_.size());
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
    struct Vertex { ContourPoint point; Index next; };
    struct Path { Index first, last, size; };
    Endpoints starts_;
    Endpoints ends_;
    std::vector<Vertex> vertices_;
    std::vector<Path> paths_;

    Index add_vertex(ContourPoint point, Index next = no_path) {
        const auto index = static_cast<Index>(vertices_.size());
        vertices_.push_back({point, next});
        return index;
    }

    void join(Index head, Index tail) {
        // Keep the earlier creation index regardless of geometric direction.
        auto& left = paths_[head];
        auto& right = paths_[tail];
        const auto first = vertices_[left.first].point;
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

constexpr unsigned block_width = 16;
constexpr unsigned block_mask = (1u << block_width) - 1;

template <class Pixel, class Threshold, std::size_t... Bit>
unsigned classify_block(const Pixel* pixels, Threshold level, std::index_sequence<Bit...>) {
    return ((unsigned(pixels[Bit] > level) << Bit) | ...);
}

template <class Pixel, class Threshold>
void classify_row(const Pixel* pixels, std::size_t columns, Threshold level,
                  std::vector<unsigned>& blocks) {
    std::size_t column = 0;
    for (; column + block_width <= columns; column += block_width) {
        blocks[column / block_width] =
            classify_block(pixels + column, level, std::make_index_sequence<block_width>{});
    }
    if (column < columns) {
        unsigned tail = 0;
        for (std::size_t i = 0; i < columns - column; ++i) {
            tail |= unsigned(pixels[column + i] > level) << i;
        }
        blocks[column / block_width] = tail;
    }
}

constexpr unsigned crossing_index(std::uint32_t bit) {
    // The shifted De Bruijn product gives a distinct index for each set bit.
    return std::uint32_t(bit * 0x077cb531u) >> 27;
}

constexpr auto crossing_positions() {
    std::array<unsigned, 32> positions{};
    for (unsigned bit = 0; bit < block_width; ++bit) {
        positions[crossing_index(1u << bit)] = bit;
    }
    return positions;
}

constexpr auto positions = crossing_positions();

template <class Pixel, class Threshold>
Contours trace(const Pixel* image, const std::uint8_t* mask, std::size_t rows,
               std::size_t columns, double level, Threshold threshold, bool connect_high) {
    OrderedContours paths;
    // The trailing zero supplies the lookahead bit for the last block.
    const auto blocks = (columns + block_width - 1) / block_width + 1;
    std::vector<unsigned> above(blocks), below(blocks);
    classify_row(image, columns, threshold, above);
    for (std::size_t row = 0; row + 1 < rows; ++row) {
        const auto row_start = row * columns;
        classify_row(image + row_start + columns, columns, threshold, below);
        for (std::size_t base = 0, block = 0; base < columns - 1; base += block_width, ++block) {
            const auto top_bits = above[block] | ((above[block + 1] & 1u) << block_width);
            const auto bottom_bits = below[block] | ((below[block + 1] & 1u) << block_width);
            // A cell crosses the level if its upper-left sample disagrees
            // with any other corner. Skip uniform blocks in one bit test.
            const auto valid = block_mask >> (block_width - std::min<std::size_t>(block_width, columns - 1 - base));
            auto crossings = ((top_bits ^ (top_bits >> 1)) | (top_bits ^ bottom_bits)
                              | (top_bits ^ (bottom_bits >> 1))) & valid;
            while (crossings) {
                const auto bit = positions[crossing_index(crossings & (0u - crossings))];
                crossings &= crossings - 1;
                const auto column = base + bit;
                const auto a = row_start + column;
                const auto b = a + columns;
                if (mask && !(mask[a] && mask[a + 1] && mask[b] && mask[b + 1])) continue;
                const auto code = ((top_bits >> bit) & 3u) | (((bottom_bits >> bit) & 3u) << 2);
                connect_cell(paths, double(row), double(column), level, connect_high,
                             code, image[a], image[a + 1], image[b], image[b + 1]);
            }
        }
        above.swap(below);
    }
    return paths.finish();
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
            if (low == high) {
                output.clear();
                return true;
            }
            level = (low + high) / 2.0;
        }
    }

    if constexpr (std::is_same_v<Pixel, std::uint8_t>) {
        if (level < 0 || level >= 255) {
            output.clear();
            return true;
        }
        // Comparing an integer sample to a fractional threshold gives the
        // same classification as comparing it to the floored threshold.
        output = trace(image, mask, rows, columns, level, static_cast<std::uint8_t>(level), options.connect_high);
    } else {
        output = trace(image, mask, rows, columns, level, level, options.connect_high);
    }
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
