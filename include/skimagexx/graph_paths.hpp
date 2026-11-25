#pragma once

#include "skimagexx/image.hpp"

#include <functional>
#include <limits>

namespace skimagexx::graph_paths {

using Index = std::int32_t;
using Offset = std::int16_t;
using Shape = std::vector<Index>;
using Offsets = std::vector<Shape>;

// The grid uses Fortran-order indices by default, including in callbacks.
class Grid {
public:
    Grid(Shape shape, Offsets offsets, std::vector<double> lengths, bool row_major = false);

    const Shape& shape() const { return shape_; }
    const Shape& strides() const { return strides_; }
    const Offsets& offsets() const { return offsets_; }
    Index size() const { return size_; }
    Shape coordinates(Index index) const;
    void coordinates(Index index, Index* point) const;
    Index delta(std::size_t move) const { return deltas_[move]; }
    double length(std::size_t move) const { return lengths_[move]; }
    std::uint64_t moves_2d(Index x, Index y) const { return x_moves_[x] & y_moves_[y]; }
    bool has_move_masks() const { return !x_moves_.empty(); }
    bool contains_move(const Index* point, std::size_t move) const;

private:
    Shape shape_, strides_, deltas_;
    Offsets offsets_;
    std::vector<double> lengths_;
    Index size_ = 1;
    std::vector<std::uint64_t> x_moves_, y_moves_;
};

Offsets lattice_offsets(std::size_t dimensions, int connectivity);

enum class Cost { destination, geometric, connect, difference };

struct Hooks {
    std::function<int(Index, double)> goal;
    std::function<double(double, double, double)> travel;
    std::function<void(Index, Index, double)> examine;
    std::function<void(Index, Index, double)> update;
    std::function<void(Index, Index, Index, Index, double, double)> connect;
};

struct Search {
    Cost cost = Cost::destination;
    bool all_ends = true;
    bool use_ends = false;
    std::ptrdiff_t max_visits = std::numeric_limits<std::ptrdiff_t>::max();
    double step_cost = std::numeric_limits<double>::infinity();
};

// Buffers have grid.size() elements and must not overlap. Returns the last
// matched end index, or -1. The caller supplies initialized grid coordinates
// as bounded flat indices.
Index find_costs(const Grid& grid, const double* costs, const Shape& starts,
                 const Shape& ends, const Search& options, double* distance,
                 Offset* predecessor, const Hooks& hooks = {}, Shape* pending = nullptr);
Shape traceback(const Grid& grid, const double* distance, const Offset* predecessor, Index end);
std::pair<Shape, double> shortest_path(const Grid& grid, const double* costs, int axis);

struct PixelEdges {
    Shape nodes, indptr, indices;
    std::vector<double> distances;
};

// The grid and byte mask use C-order indices. Pass null for a full lattice.
PixelEdges pixel_edges(const Grid& grid, const std::uint8_t* mask);

} // namespace skimagexx::graph_paths
