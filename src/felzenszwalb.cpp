#include "skimagexx/felzenszwalb.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

namespace skimagexx {
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

    std::pair<Index, Index> endpoints(Index edge) const {
        if (edge < right) {
            const auto left = edge + edge / (width - 1);
            return {left, left + 1};
        }
        if (edge < down_end) {
            const auto top = edge - right;
            return {top, top + width};
        }
        if (edge < diagonal_end) {
            edge -= down_end;
            const auto top = edge + edge / (width - 1);
            return {top, top + width + 1};
        }
        edge -= diagonal_end;
        const auto top = edge + edge / (width - 1);
        return {top + 1, top + width};
    }
};

template<Index fixed_channels>
void fill_costs(const double* image, double* out, Index height, Index width, Index channels) {
    const auto count = fixed_channels ? fixed_channels : channels;
    const auto fill = [&](Index rows, Index cols, Index first, Index delta) {
        if (cols == 0) return;
        for (Index row = 0; row < rows; ++row) {
            const auto* a = image + (row * width + first) * channels;
            const auto* b = a + delta * channels;
            for (Index col = 0; col < cols; ++col, a += channels, b += channels) {
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
    fill(height - 1, width, 0, width);
    fill(height - 1, width - 1, 0, width + 1);
    fill(height - 1, width - 1, 1, width - 1);
}

Index find_root(Index* parent, Index pixel) {
    while (parent[pixel] != pixel) {
        parent[pixel] = parent[parent[pixel]];
        pixel = parent[pixel];
    }
    return pixel;
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
    case 3:
        fill_costs<3>(image, costs, height, width, channels);
        break;
    default:
        fill_costs<0>(image, costs, height, width, channels);
    }
}

void felzenszwalb_labels_into(const double* costs, const Index* order, Index* labels,
                              Index height, Index width, double scale, Index min_size) {
    const GridEdges edges(height, width);
    const auto pixels = height * width;
    std::iota(labels, labels + pixels, Index{0});
    std::vector<Index> sizes(static_cast<std::size_t>(pixels), 1);
    std::vector<double> internal_cost(static_cast<std::size_t>(pixels), 0);
    scale /= 255;
    const auto join = [&](Index a, Index b) {
        const auto root = std::min(a, b);
        labels[std::max(a, b)] = root;
        sizes[root] = sizes[a] + sizes[b];
        return root;
    };

    for (Index i = 0; i < edges.count; ++i) {
        const auto edge = order[i];
        const auto [first, second] = edges.endpoints(edge);
        const auto a = find_root(labels, first);
        const auto b = find_root(labels, second);
        if (a == b) continue;
        const auto threshold_a = static_cast<float>(internal_cost[a] + scale / static_cast<double>(sizes[a]));
        const auto threshold_b = static_cast<float>(internal_cost[b] + scale / static_cast<double>(sizes[b]));
        if (costs[edge] < std::min(threshold_a, threshold_b)) internal_cost[join(a, b)] = costs[edge];
    }
    if (min_size > 1) {
        for (Index i = 0; i < edges.count; ++i) {
            const auto [first, second] = edges.endpoints(order[i]);
            const auto a = find_root(labels, first);
            const auto b = find_root(labels, second);
            if (a != b && (sizes[a] < min_size || sizes[b] < min_size)) join(a, b);
        }
    }

    // Roots are always the smallest pixel in their respective component.
    // Earlier indices therefore hold their final labels when visited here.
    Index next = 0;
    for (Index pixel = 0; pixel < pixels; ++pixel) labels[pixel] = find_root(labels, pixel);
    for (Index pixel = 0; pixel < pixels; ++pixel) {
        labels[pixel] = labels[pixel] == pixel ? next++ : labels[labels[pixel]];
    }
}

} // namespace skimagexx
