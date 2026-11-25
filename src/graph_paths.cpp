#include "skimagexx/graph_paths.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace skimagexx::graph_paths {
namespace {

constexpr auto infinity = std::numeric_limits<double>::infinity();
constexpr auto max_size = std::numeric_limits<Index>::max() / 2;

// A binary tournament over dense slots. Equal priorities choose the leftmost
// slot and extraction moves the last slot into the vacancy. These two rules
// are observable in scikit-image's traceback and callback order.
class Frontier {
public:
    explicit Frontier(Index size) : positions_(size, -1), tree_(256, infinity) {}

    bool empty() const { return nodes_.empty(); }
    void reset_slots(Shape slots) {
        while (slots.size() > capacity_) grow();
        nodes_ = std::move(slots);
    }
    Shape remaining_slots() { return std::move(nodes_); }

    bool push(double value, Index node, bool replace = false) {
        auto slot = positions_[node];
        if (slot < 0) {
            if (nodes_.size() == capacity_) grow();
            slot = static_cast<Index>(nodes_.size());
            positions_[node] = slot;
            nodes_.push_back(node);
        } else if (!replace && !(value < tree_[capacity_ + slot])) {
            return false;
        }
        const auto leaf = capacity_ + slot;
        tree_[leaf] = value;
        propagate(leaf);
        return true;
    }

    std::pair<Index, double> pop() {
        std::size_t leaf = 2;
        while (leaf < capacity_) {
            leaf = (leaf + (tree_[leaf] <= tree_[leaf + 1] ? 0 : 1)) * 2;
        }
        leaf += tree_[leaf] <= tree_[leaf + 1] ? 0 : 1;
        const auto value = tree_[leaf];
        // An all-infinite front still has a valid leftmost slot. NaN inputs
        // are screened at the Python boundary.
        const auto slot = leaf - capacity_;
        const auto node = nodes_[slot];
        const auto last = capacity_ + nodes_.size() - 1;
        const auto moved = nodes_.back();
        positions_[moved] = static_cast<Index>(slot);
        nodes_[slot] = moved;
        positions_[node] = -1;
        nodes_.pop_back();
        tree_[leaf] = tree_[last];
        tree_[last] = infinity;
        propagate(leaf);
        if (leaf != last) propagate(last);
        return {node, value};
    }

private:
    void propagate(std::size_t leaf) {
        for (auto parent = leaf / 2; parent > 1; parent /= 2) {
            const auto value = std::min(tree_[2 * parent], tree_[2 * parent + 1]);
            if (value == tree_[parent]) break;
            tree_[parent] = value;
        }
    }

    void grow() {
        const auto old_capacity = capacity_;
        capacity_ *= 2;
        tree_.resize(capacity_ * 2, infinity);
        std::copy_n(tree_.data() + old_capacity, old_capacity, tree_.data() + capacity_);
        for (auto parent = capacity_ - 1; parent > 1; --parent) {
            tree_[parent] = std::min(tree_[2 * parent], tree_[2 * parent + 1]);
        }
    }

    std::vector<Index> positions_, nodes_;
    std::vector<double> tree_;
    std::size_t capacity_ = 128;
};

template<class Visit>
void visit_moves(const Grid& grid, Index node, Visit visit) {
    const auto count = grid.offsets().size();
    if (grid.has_move_masks()) {
        const auto stride = std::max(grid.strides()[0], grid.strides()[1]);
        const auto slow = node / stride;
        const auto fast = node - slow * stride;
        const auto mask = grid.strides()[0] == 1 ? grid.moves_2d(fast, slow)
                                                 : grid.moves_2d(slow, fast);
        const auto all = std::numeric_limits<std::uint64_t>::max() >> (64 - count);
        if (mask == all) {
            for (std::size_t move = 0; move < count; ++move) visit(move);
        } else {
            for (std::size_t move = 0; move < count; ++move) {
                if ((mask >> move) & 1) visit(move);
            }
        }
    } else {
        std::array<Index, 9> point;
        grid.coordinates(node, point.data());
        for (std::size_t move = 0; move < count; ++move) {
            if (grid.contains_move(point.data(), move)) visit(move);
        }
    }
}

template<Cost cost_mode>
Index search(const Grid& grid, const double* costs, const Shape& starts,
             const Shape& ends, const Search& options, double* distance,
             Offset* predecessor, const Hooks& hooks, Shape* pending) {
    std::fill_n(distance, grid.size(), infinity);
    std::fill_n(predecessor, grid.size(), Offset{-2});
    Frontier queue(grid.size());
    // The upstream reset clears priorities and cross references while keeping
    // the dense slots occupied by a prior early-stop search. Reused slots
    // can change the order in which equal priorities are extracted.
    if (pending) queue.reset_slots(std::move(*pending));
    Shape origins;
    if (hooks.connect) origins.resize(grid.size(), -1);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const auto node = starts[i];
        predecessor[node] = -1;
        constexpr bool start_cost = cost_mode == Cost::destination || cost_mode == Cost::connect;
        queue.push(start_cost ? costs[node] : 0.0, node, true);
        if (hooks.connect) origins[node] = static_cast<Index>(i);
    }
    std::vector<std::uint8_t> targets;
    if (ends.size() > 8) {
        targets.resize(grid.size());
        for (const auto end : ends) targets[end] = 1;
    }
    std::size_t found = 0;
    Index last_end = -1;
    for (std::ptrdiff_t iteration = 0; iteration < options.max_visits && !queue.empty(); ++iteration) {
        const auto entry = queue.pop();
        const auto node = entry.first;
        const auto cumulative = entry.second;
        distance[node] = cumulative;
        if (hooks.goal) {
            const auto goal = hooks.goal(node, cumulative);
            if (goal == 1) continue;
            if (goal > 1) break;
        }
        if (options.use_ends) {
            const bool reached = targets.empty()
                ? std::find(ends.begin(), ends.end(), node) != ends.end()
                : targets[node] != 0;
            if (reached) {
                ++found;
                last_end = node;
            }
            if ((found && !options.all_ends) || found == ends.size()) break;
        }
        visit_moves(grid, node, [&](std::size_t move) {
            const auto next = node + grid.delta(move);
            const auto length = grid.length(move);
            if (distance[next] != infinity) {
                if (hooks.examine) hooks.examine(node, next, length);
                if (hooks.connect && origins[node] >= 0 && origins[next] >= 0
                    && origins[node] != origins[next]) {
                    hooks.connect(origins[node], origins[next], node, next, cumulative, distance[next]);
                }
                return;
            }
            const auto destination = costs[next];
            if (destination < 0 || destination >= infinity || destination > options.step_cost) return;
            auto step = destination;
            if constexpr (cost_mode == Cost::geometric || cost_mode == Cost::connect) {
                step = (length * 0.5) * (costs[node] + destination);
            } else if constexpr (cost_mode == Cost::difference) {
                step = std::abs(costs[node] - destination);
            } else if (hooks.travel) {
                step = hooks.travel(costs[node], destination, length);
            }
            const auto candidate = cumulative + step;
            if (!std::isnan(candidate) && candidate != infinity && queue.push(candidate, next)) {
                predecessor[next] = static_cast<Offset>(move);
                if (hooks.update) hooks.update(node, next, length);
                if (hooks.connect) origins[next] = origins[node];
            }
        });
    }
    if (pending) *pending = queue.remaining_slots();
    return last_end;
}

} // namespace

Grid::Grid(Shape shape, Offsets offsets, std::vector<double> lengths, bool row_major)
    : shape_(std::move(shape)), strides_(shape_.size()), offsets_(std::move(offsets)),
      lengths_(std::move(lengths)) {
    if (shape_.empty() || shape_.size() > 9 || offsets_.empty() || offsets_.size() != lengths_.size()
        || offsets_.size() > static_cast<std::size_t>(std::numeric_limits<Offset>::max())) {
        throw std::invalid_argument("invalid grid dimensionality or offsets");
    }
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        const auto axis = row_major ? shape_.size() - 1 - i : i;
        const auto extent = shape_[axis];
        if (extent <= 0 || size_ > max_size / extent) throw std::invalid_argument("invalid grid extent");
        strides_[axis] = size_;
        size_ *= extent;
    }
    for (const auto& offset : offsets_) {
        if (offset.size() != shape_.size()) throw std::invalid_argument("invalid offset dimensionality");
        std::int64_t delta = 0;
        bool usable = true;
        for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
            const auto component = offset[axis];
            if (component <= -shape_[axis] || component >= shape_[axis]) usable = false;
            delta += static_cast<std::int64_t>(component) * strides_[axis];
        }
        deltas_.push_back(usable ? static_cast<Index>(delta) : 0);
    }
    if (shape_.size() == 2 && offsets_.size() <= 64 && shape_[0] > 1 && shape_[1] > 1
        && shape_[0] + shape_[1] <= 65536) {
        auto fill_masks = [&](std::vector<std::uint64_t>& masks, std::size_t axis) {
            masks.resize(shape_[axis]);
            for (std::size_t move = 0; move < offsets_.size(); ++move) {
                const auto component = offsets_[move][axis];
                if (component >= shape_[axis] || component <= -shape_[axis]) continue;
                const auto begin = std::max(Index{0}, -component);
                const auto end = shape_[axis] - std::max(Index{0}, component);
                const auto bit = std::uint64_t{1} << move;
                for (auto i = begin; i < end; ++i) masks[i] |= bit;
            }
        };
        fill_masks(x_moves_, 0);
        fill_masks(y_moves_, 1);
    }
}

Shape Grid::coordinates(Index index) const {
    Shape point(shape_.size());
    coordinates(index, point.data());
    return point;
}

void Grid::coordinates(Index index, Index* point) const {
    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
        point[axis] = (index / strides_[axis]) % shape_[axis];
    }
}

bool Grid::contains_move(const Index* point, std::size_t move) const {
    for (std::size_t axis = 0; axis < shape_.size(); ++axis) {
        const auto coordinate = static_cast<std::int64_t>(point[axis]) + offsets_[move][axis];
        if (coordinate < 0 || coordinate >= shape_[axis]) return false;
    }
    return true;
}

Offsets lattice_offsets(std::size_t dimensions, int connectivity) {
    if (dimensions == 0 || dimensions > 9) throw std::invalid_argument("unsupported lattice dimensionality");
    Offsets offsets;
    Shape component(dimensions, -1);
    for (;;) {
        const auto steps = std::count_if(component.begin(), component.end(), [](Index i) { return i != 0; });
        if (steps > 0 && steps <= connectivity) offsets.push_back(component);
        auto axis = dimensions;
        while (axis != 0 && component[axis - 1] == 1) component[--axis] = -1;
        if (axis == 0) break;
        ++component[axis - 1];
    }
    return offsets;
}

Index find_costs(const Grid& grid, const double* costs, const Shape& starts,
                 const Shape& ends, const Search& options, double* distance,
                 Offset* predecessor, const Hooks& hooks, Shape* pending) {
    switch (options.cost) {
    case Cost::geometric:
        return search<Cost::geometric>(grid, costs, starts, ends, options, distance, predecessor, hooks, pending);
    case Cost::connect:
        return search<Cost::connect>(grid, costs, starts, ends, options, distance, predecessor, hooks, pending);
    case Cost::difference:
        return search<Cost::difference>(grid, costs, starts, ends, options, distance, predecessor, hooks, pending);
    default:
        return search<Cost::destination>(grid, costs, starts, ends, options, distance, predecessor, hooks, pending);
    }
}

Shape traceback(const Grid& grid, const double* distance, const Offset* predecessor, Index end) {
    if (end < 0 || end >= grid.size()) throw std::invalid_argument("graph index out of bounds");
    if (distance[end] == infinity) {
        throw std::invalid_argument("no minimum-cost path was found to the specified end point");
    }
    Shape path;
    for (auto node = end; node >= 0 && node < grid.size();) {
        path.push_back(node);
        const auto move = predecessor[node];
        if (move == -1) {
            std::reverse(path.begin(), path.end());
            return path;
        }
        if (move < 0 || static_cast<std::size_t>(move) >= grid.offsets().size()
            || path.size() >= static_cast<std::size_t>(grid.size())) break;
        node -= grid.delta(static_cast<std::size_t>(move));
    }
    throw std::invalid_argument("invalid traceback offsets");
}

std::pair<Shape, double> shortest_path(const Grid& grid, const double* costs, int axis) {
    if (axis < 0 || static_cast<std::size_t>(axis) >= grid.shape().size()) {
        throw std::invalid_argument("graph axis out of bounds");
    }
    const auto stride = grid.strides()[axis];
    const auto extent = grid.shape()[axis];
    const auto slab = stride * extent;
    Shape starts, ends;
    starts.reserve(grid.size() / extent);
    ends.reserve(grid.size() / extent);
    for (Index slab_start = 0; slab_start < grid.size(); slab_start += slab) {
        for (Index offset = 0; offset < stride; ++offset) {
            starts.push_back(slab_start + offset);
            ends.push_back(slab_start + offset + slab - stride);
        }
    }
    std::vector<double> distance(grid.size());
    std::vector<Offset> predecessor(grid.size());
    const Search options{Cost::difference, false, true};
    const auto end = find_costs(grid, costs, starts, ends, options, distance.data(), predecessor.data());
    if (end < 0) throw std::invalid_argument("no minimum-cost path was found to the specified end point");
    return {traceback(grid, distance.data(), predecessor.data(), end), distance[end]};
}

PixelEdges pixel_edges(const Grid& grid, const std::uint8_t* mask) {
    PixelEdges result;
    Shape ids;
    if (mask) {
        ids.resize(grid.size(), -1);
        for (Index i = 0; i < grid.size(); ++i) {
            if (mask[i]) {
                ids[i] = static_cast<Index>(result.nodes.size());
                result.nodes.push_back(i);
            }
        }
    } else {
        result.nodes.resize(grid.size());
        std::iota(result.nodes.begin(), result.nodes.end(), Index{0});
    }
    result.indptr.reserve(result.nodes.size() + 1);
    const auto upper = result.nodes.size() * grid.offsets().size();
    if (upper > static_cast<std::size_t>(max_size)) throw std::invalid_argument("pixel graph is too large");
    result.indices.reserve(upper);
    result.distances.reserve(upper);
    for (const auto node : result.nodes) {
        result.indptr.push_back(static_cast<Index>(result.indices.size()));
        visit_moves(grid, node, [&](std::size_t move) {
            const auto next = node + grid.delta(move);
            const auto id = mask ? ids[next] : next;
            if (id >= 0) {
                result.indices.push_back(id);
                result.distances.push_back(grid.length(move));
            }
        });
    }
    result.indptr.push_back(static_cast<Index>(result.indices.size()));
    return result;
}

} // namespace skimagexx::graph_paths
