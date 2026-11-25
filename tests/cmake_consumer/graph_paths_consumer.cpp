#include <skimagexx/graph_paths.hpp>

#include <array>
#include <cmath>
#include <limits>

int main() {
    namespace graph = skimagexx::graph_paths;
    const std::array<double, 6> image{1, 100, 2, 100, 3, 4};
    const auto offsets = graph::lattice_offsets(2, 1);
    const graph::Grid grid({2, 3}, offsets, {1, 2, 2, 1});
    std::array<double, 6> distance{};
    std::array<graph::Offset, 6> predecessor{};
    const graph::Search geometric{graph::Cost::geometric, true, true};
    auto end = graph::find_costs(grid, image.data(), {0}, {5}, geometric, distance.data(),
                                 predecessor.data());
    if (end != 5 || distance[5] != 11.5 ||
        graph::traceback(grid, distance.data(), predecessor.data(), end) != graph::Shape{0, 2, 4, 5}) {
        return 1;
    }
    graph::Hooks hooks;
    int meetings = 0;
    hooks.connect = [&](graph::Index left, graph::Index right, graph::Index,
                        graph::Index, double, double) {
        if (left != right) ++meetings;
    };
    const graph::Search connect{graph::Cost::connect};
    graph::find_costs(grid, image.data(), {0, 5}, {}, connect, distance.data(),
                      predecessor.data(), hooks);
    if (meetings == 0 || distance[0] != 1 || distance[5] != 4) return 2;

    const graph::Grid directed({2, 3}, {{0, 1}}, {1});
    const auto path = graph::shortest_path(directed, image.data(), 1);
    if (path.first != graph::Shape{0, 2, 4} || path.second != 2) return 3;

    const graph::Grid pixels({2, 3}, offsets, {1, 2, 2, 1}, true);
    const std::uint8_t mask[]{1, 1, 0, 1, 0, 2};
    const auto edges = graph::pixel_edges(pixels, mask);
    if (edges.nodes != graph::Shape{0, 1, 3, 5} ||
        edges.indptr != graph::Shape{0, 2, 3, 4, 4} ||
        edges.indices != graph::Shape{1, 2, 0, 0} ||
        edges.distances != std::vector<double>{2, 1, 2, 1}) {
        return 4;
    }
}
