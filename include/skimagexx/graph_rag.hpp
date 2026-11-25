#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::graph_rag {

using Label = std::int64_t;

struct Edge {
    Label first;
    Label second;
};

struct RegionColor {
    Label label;
    std::size_t count = 0;
    std::array<double, 3> total{};
};

struct BoundaryEdge {
    Edge edge;
    std::size_t count = 0;
    double total = 0.0;
};

struct BoundarySample {
    Edge edge;
    std::size_t pixel;
};

// Images are contiguous in row-major order. Shapes have one to three axes.
// The first encounter determines the ordering and orientation of each edge.
std::vector<Edge> adjacency(const Label* labels, const std::vector<std::size_t>& shape,
                            int connectivity);
std::vector<RegionColor> region_colors(const Label* labels, const double* rgb,
                                       std::size_t size, std::size_t label_count);
std::vector<BoundaryEdge> boundary(const Label* labels, const double* values,
                                  const std::vector<std::size_t>& shape,
                                  int connectivity, bool single_precision);
std::vector<BoundarySample> boundary_samples(const Label* labels,
                                            const std::vector<std::size_t>& shape,
                                            int connectivity);
std::vector<std::array<Label, 2>> centroids(const Label* labels, std::size_t height,
                                           std::size_t width, std::size_t label_count);

// Edges use positions in graph node order. Components are numbered by their
// first node, independently of union order.
std::vector<Label> components(std::size_t node_count, const std::vector<Edge>& edges);

// Exact totals for a connected, undirected interpretation of square CSR with
// positive integral weights representable within the distance-sum bound.
// Returns an empty vector for a disconnected or unsupported graph.
std::vector<double> central_distances(std::size_t nodes, const std::int32_t* rows,
                                      const std::int32_t* columns,
                                      const double* weights, std::size_t entries);

struct SpectralCut {
    std::vector<std::uint8_t> mask;
    double cost;
};

// Score evenly spaced cuts on an eigenvector of the normalized Laplacian.
// Weights are an undirected CSC matrix and degrees follow its column order.
SpectralCut spectral_cut(const double* eigenvector, const double* degrees,
                         std::size_t nodes, const std::int32_t* columns,
                         const std::int32_t* rows, const double* weights,
                         std::size_t cuts);

} // namespace skimagexx::graph_rag
