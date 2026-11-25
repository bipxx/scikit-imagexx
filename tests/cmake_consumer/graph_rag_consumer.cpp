#include <skimagexx/graph_rag.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace skimagexx::graph_rag;
    const Label labels[]{2, 2, 0, 1, 1, 0};
    const auto edges = adjacency(labels, {2, 3}, 1);
    if (edges.size() != 3 || edges[0].first != 2 || edges[0].second != 1 ||
        edges[1].first != 2 || edges[1].second != 0 ||
        edges[2].first != 1 || edges[2].second != 0) {
        return 1;
    }
    const double rgb[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1, 2, 2, 2, 3, 4, 5};
    const auto regions = region_colors(labels, rgb, 6, 3);
    if (regions[0].count != 2 || regions[0].total != std::array<double, 3>{10, 12, 14} ||
        regions[2].total != std::array<double, 3>{5, 7, 9}) {
        return 2;
    }
    const double weights[]{1, 2, 3, 4, 5, 6};
    const auto border = boundary(labels, weights, {2, 3}, 1, false);
    if (border.size() != 3 || border[0].edge.first != 0 || border[0].edge.second != 1 ||
        border[0].count != 2 || border[0].total != 11) {
        return 3;
    }
    const auto centers = centroids(labels, 2, 3, 3);
    if (centers != std::vector<std::array<Label, 2>>{{1, 0}, {0, 0}} ||
        components(5, {{2, 3}, {1, 2}}) != std::vector<Label>{0, 1, 1, 1, 2}) {
        return 4;
    }
    const std::int32_t offsets[]{0, 1, 1, 2};
    const std::int32_t indices[]{1, 1};
    const double costs[]{2, 3};
    if (central_distances(3, offsets, indices, costs, 2) != std::vector<double>{7, 5, 8}) {
        return 5;
    }
    const double eig[]{-1, 0, 1};
    const double degrees[]{2, 3, 2};
    const std::int32_t columns[]{0, 2, 5, 7};
    const std::int32_t rows[]{0, 1, 0, 1, 2, 1, 2};
    const double similarity[]{1, 1, 1, 1, 1, 1, 1};
    const auto cut = spectral_cut(eig, degrees, 3, columns, rows, similarity, 2);
    if (cut.mask != std::vector<std::uint8_t>{0, 1, 1} || cut.cost != 0.7) {
        return 6;
    }
    std::cout << "graph_rag consumer passed\n";
}
