#include "skimagexx/felzenszwalb.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
extern std::ptrdiff_t felzenszwalb_rgb_cost_row(const double* a, const double* b,
                                              double* out, std::ptrdiff_t cols);
extern std::ptrdiff_t felzenszwalb_gray_cost_row(const double* a, const double* b,
                                               double* out, std::ptrdiff_t cols, bool paired);
}
#endif
namespace {

using Index = std::ptrdiff_t;

struct GridEdges {
    Index width;
    Index right;
    Index down_end;
    Index diagonal_end;
    Index count;

    GridEdges(Index height, Index w)
        : width(w), right(height * (w - 1)), down_end(right + (height - 1) * w),
          diagonal_end(down_end + (height - 1) * (w - 1)),
          count(diagonal_end + (height - 1) * (w - 1)) {}

    template<class Position>
    std::pair<Position, Position> endpoints(Index edge, const Position* origins) const {
        if (edge < right) {
            const Index left = origins[edge];
            return {Position(left), Position(left + 1)};
        }
        if (edge < down_end) {
            const auto top = edge - right;
            return {Position(top), Position(top + width)};
        }
        if (edge < diagonal_end) {
            const Index top = origins[edge - down_end];
            return {Position(top), Position(top + width + 1)};
        }
        const Index top = origins[edge - diagonal_end];
        return {Position(top + 1), Position(top + width)};
    }
};

template<Index fixed_channels>
void fill_costs(const double* image, double* out, Index height, Index width, Index channels) {
    const auto count = fixed_channels ? fixed_channels : channels;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized = fixed_channels && has_avx2();
#endif
    const auto fill = [&](Index rows, Index cols, Index first, Index delta) {
        if (cols == 0) return;
        for (Index row = 0; row < rows; ++row) {
            const auto* a = image + (row * width + first) * channels;
            const auto* b = a + delta * channels;
            Index col = 0;
#ifdef SKIMAGEXX_AVX2
            if constexpr (fixed_channels != 0) {
                if (vectorized && cols >= 4) {
                    if constexpr (fixed_channels == 3)
                        col = simd::felzenszwalb_rgb_cost_row(a, b, out, cols);
                    else
                        col = simd::felzenszwalb_gray_cost_row(a, b, out, cols, fixed_channels == 2);
                    a += col * channels;
                    b += col * channels;
                    out += col;
                }
            }
#endif
            for (; col < cols; ++col, a += channels, b += channels) {
                double squared = 0;
                for (Index channel = 0; channel < count; ++channel) {
                    const double difference = a[channel] - b[channel];
                    squared += difference * difference;
                }
                *out++ = std::sqrt(squared);
            }
        }
    };
    fill(height, width - 1, 0, 1);
    // Vertical edges form one contiguous span in both image streams.
    if (height > 1)
        fill(1, (height - 1) * width, 0, width);
    fill(height - 1, width - 1, 0, width + 1);
    fill(height - 1, width - 1, 1, width - 1);
}

template<class Position>
Position find_root(Position* parent, Position pixel) {
    auto next = parent[pixel];
    while (next != pixel) {
        const auto grandparent = parent[next];
        if (next == grandparent)
            return next;
        parent[pixel] = grandparent;
        pixel = grandparent;
        next = parent[pixel];
    }
    return pixel;
}

template<class Position>
void label_components(const double* costs, const Index* order, Index* labels,
                      Index height, Index width, double scale, Index min_size, Position* parent) {
    const GridEdges edges(height, width);
    const auto pixels = height * width;
    // Right edges and both diagonal blocks share the same row-major origin
    // map. Vertical edges already use their top endpoint as the local index.
    std::unique_ptr<Position[]> origins(new Position[edges.right]);
    Index origin = 0;
    for (Index row = 0; row < height && width > 1; ++row)
        for (Index col = 0; col < width - 1; ++col)
            origins[origin++] = static_cast<Position>(row * width + col);

    std::iota(parent, parent + pixels, Position{0});
    std::vector<Position> sizes(static_cast<std::size_t>(pixels), 1);
    scale /= 255;
    std::vector<float> thresholds(static_cast<std::size_t>(pixels), static_cast<float>(0.0 + scale));
    const auto join = [&](Position a, Position b) {
        const auto root = std::min(a, b);
        parent[std::max(a, b)] = root;
        sizes[root] = sizes[a] + sizes[b];
        return root;
    };

    for (Index i = 0; i < edges.count; ++i) {
        const auto edge = order[i];
        const auto [first, second] = edges.endpoints(edge, origins.get());
        const auto a = find_root(parent, first);
        const auto b = find_root(parent, second);
        if (a == b) continue;
        const double cost = costs[edge];
        if (cost < std::min(thresholds[a], thresholds[b])) {
            const auto root = join(a, b);
            thresholds[root] = static_cast<float>(cost + scale / static_cast<double>(sizes[root]));
        }
    }
    if (min_size > 1) {
        for (Index i = 0; i < edges.count; ++i) {
            const auto [first, second] = edges.endpoints(order[i], origins.get());
            const auto a = find_root(parent, first);
            const auto b = find_root(parent, second);
            if (a != b && (sizes[a] < min_size || sizes[b] < min_size)) join(a, b);
        }
    }

    // Each non-root parent has a smaller index, so its final label is ready.
    Index next = 0;
    for (Index pixel = 0; pixel < pixels; ++pixel) {
        const auto root = parent[pixel];
        labels[pixel] = root == pixel ? next++ : labels[root];
    }
}

} // namespace

Index felzenszwalb_edge_count(Index height, Index width) {
    return GridEdges(height, width).count;
}

void felzenszwalb_costs_into(const double* image, double* costs, Index height,
                             Index width, Index channels) {
    switch (channels) {
    case 1:
        fill_costs<1>(image, costs, height, width, channels);
        break;
    case 2:
        fill_costs<2>(image, costs, height, width, channels);
        break;
    case 3:
        fill_costs<3>(image, costs, height, width, channels);
        break;
    default:
        fill_costs<0>(image, costs, height, width, channels);
    }
}

void felzenszwalb_labels_into(const double* costs, const Index* order, Index* labels,
                              Index height, Index width, double scale, Index min_size) {
    const auto pixels = height * width;
    if (pixels <= std::numeric_limits<std::int32_t>::max()) {
        std::unique_ptr<std::int32_t[]> parent(new std::int32_t[pixels]);
        label_components(costs, order, labels, height, width, scale, min_size, parent.get());
    } else {
        label_components(costs, order, labels, height, width, scale, min_size, labels);
    }
}

} // namespace skimagexx
