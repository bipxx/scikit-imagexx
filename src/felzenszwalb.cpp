#include "skimagexx/felzenszwalb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
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

template<Index FixedChannels>
void fill_costs(const double* image, double* out, Index height, Index width, Index channels) {
    const GridEdges edges(height, width);
    const auto count = FixedChannels ? FixedChannels : channels;
    const auto cost = [count](const double* a, const double* b) {
        double squared = 0;
        for (Index c = 0; c < count; ++c) {
            const double difference = a[c] - b[c];
            squared += difference * difference;
        }
        return std::sqrt(squared);
    };
    auto* right = out;
    auto* down = out + edges.right;
    auto* diagonal = out + edges.down_end;
    auto* other_diagonal = out + edges.diagonal_end;
    const Index row_stride = width * channels;
    for (Index y = 1; y < height; ++y) {
        const auto* top = image;
        const auto* bottom = top + row_stride;
        for (Index x = 1; x < width; ++x, top += channels, bottom += channels) {
            *right++ = cost(top, top + channels);
            *down++ = cost(top, bottom);
            *diagonal++ = cost(top, bottom + channels);
            *other_diagonal++ = cost(top + channels, bottom);
        }
        *down++ = cost(top, bottom);
        image += row_stride;
    }
    for (Index x = 1; x < width; ++x, image += channels)
        *right++ = cost(image, image + channels);
}

template<class Position>
Position find_root(Position* parent, Position pixel) {
    auto next = parent[pixel];
    while (next >= 0) {
        const auto grandparent = parent[next];
        if (grandparent < 0)
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

    // Roots store their negated component size in the parent array.
    std::fill_n(parent, pixels, Position{-1});
    scale /= 255;
    std::vector<float> thresholds(static_cast<std::size_t>(pixels), static_cast<float>(0.0 + scale));
    const auto join = [&](Position a, Position b) {
        const auto root = std::min(a, b);
        const auto size = parent[a] + parent[b];
        parent[std::max(a, b)] = root;
        parent[root] = size;
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
            thresholds[root] = static_cast<float>(cost + scale / static_cast<double>(-parent[root]));
        }
    }
    if (min_size > 1) {
        for (Index i = 0; i < edges.count; ++i) {
            const auto [first, second] = edges.endpoints(order[i], origins.get());
            const auto a = find_root(parent, first);
            const auto b = find_root(parent, second);
            if (a != b && (-parent[a] < min_size || -parent[b] < min_size)) join(a, b);
        }
    }

    // Each non-root parent has a smaller index, so its final label is ready.
    Index next = 0;
    for (Index pixel = 0; pixel < pixels; ++pixel) {
        const auto root = parent[pixel];
        labels[pixel] = root < 0 ? next++ : labels[root];
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
