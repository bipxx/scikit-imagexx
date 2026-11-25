#include "skimagexx/graph_rag.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace skimagexx::graph_rag {
namespace {

struct EdgeEqual {
    bool operator()(const Edge& a, const Edge& b) const {
        return a.first == b.first && a.second == b.second;
    }
};

struct EdgeHash {
    std::size_t operator()(const Edge& edge) const {
        const auto a = std::hash<Label>{}(edge.first);
        const auto b = std::hash<Label>{}(edge.second);
        return a ^ (b + std::size_t{0x9e3779b9} + (a << 6) + (a >> 2));
    }
};

Edge ordered(Label a, Label b) {
    return {std::min(a, b), std::max(a, b)};
}

// A radius-one reflect or nearest footprint has identical boundary samples.
// The footprint order matches C-order ndimage traversal.
template<class Visit>
void neighborhoods(const Label* labels, const std::vector<std::size_t>& shape,
                   int connectivity, Visit visit) {
    if (shape.empty() || shape.size() > 3) {
        throw std::invalid_argument("RAG shape must have one to three axes");
    }
    std::array<std::size_t, 3> extents{1, 1, 1};
    std::copy(shape.rbegin(), shape.rend(), extents.rbegin());
    const auto [planes, height, width] = extents;
    if (!planes || !height || !width) {
        return;
    }
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        planes > std::numeric_limits<std::size_t>::max() / (height * width)) {
        throw std::invalid_argument("RAG shape overflows");
    }
    const auto plane_size = height * width;
    connectivity = std::max(1, connectivity);
    std::vector<std::array<int, 3>> offsets;
    const int dz_max = shape.size() == 3 ? 1 : 0;
    const int dy_max = shape.size() >= 2 ? 1 : 0;
    for (int dz = -dz_max; dz <= dz_max; ++dz) {
        for (int dy = -dy_max; dy <= dy_max; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int distance = std::abs(dz) + std::abs(dy) + std::abs(dx);
                if (distance && distance <= connectivity) {
                    offsets.push_back({dz, dy, dx});
                }
            }
        }
    }
    for (std::size_t z = 0, i = 0; z < planes; ++z) {
        const std::array<std::size_t, 3> base{
            (z - (z != 0)) * plane_size, z * plane_size,
            (z + (z + 1 < planes)) * plane_size};
        for (std::size_t y = 0; y < height; ++y) {
            const std::array<std::size_t, 3> row{
                (y - (y != 0)) * width, y * width, (y + (y + 1 < height)) * width};
            std::array<std::size_t, 26> starts{};
            std::size_t n = 0;
            for (const auto& off : offsets) {
                starts[n++] = base[static_cast<std::size_t>(1 + off[0])] +
                              row[static_cast<std::size_t>(1 + off[1])];
            }
            for (std::size_t x = 0; x < width; ++x, ++i) {
                const std::array<std::size_t, 3> cols{
                    x - (x != 0), x, x + (x + 1 < width)};
                visit(i, labels[i], [&](auto neighbor) {
                    for (std::size_t j = 0; j < n; ++j) {
                        neighbor(labels[starts[j] + cols[static_cast<std::size_t>(1 + offsets[j][2])]]);
                    }
                });
            }
        }
    }
}

std::size_t root(std::vector<std::size_t>& parents, std::size_t node) {
    while (parents[node] != node) {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    return node;
}

template<class Lower, class Upper>
void visit_boundary(const Label* labels, const std::vector<std::size_t>& shape,
                    int connectivity, Lower lower, Upper upper) {
    auto emit = [&](std::size_t i, Label center, Label low, Label high) {
        if (low != center) {
            lower(BoundarySample{{low, center}, i});
        }
        if (high != center) {
            upper(BoundarySample{{center, high}, i});
        }
    };
    if (shape.size() == 2 && shape[0] && shape[1] &&
        shape[0] <= std::numeric_limits<std::size_t>::max() / shape[1]) {
        const auto height = shape[0];
        const auto width = shape[1];
        for (std::size_t y = 0; y < height; ++y) {
            const auto* current = labels + y * width;
            const auto* above = y > 0 ? current - width : current;
            const auto* below = y + 1 < height ? current + width : current;
            auto vertical = [&](std::size_t x) {
                return std::minmax(current[x], above[x]);
            };
            auto extrema = [&](std::size_t x) {
                const auto bounds = vertical(x);
                return std::make_pair(std::min(bounds.first, below[x]),
                                      std::max(bounds.second, below[x]));
            };
            auto middle = extrema(0);
            auto previous = middle;
            for (std::size_t x = 0; x < width; ++x) {
                const auto right = x + (x + 1 < width);
                const auto next = extrema(right);
                const auto left = x - (x > 0);
                auto low = middle.first;
                auto high = middle.second;
                if (connectivity >= 2) {
                    low = std::min(low, std::min(previous.first, next.first));
                    high = std::max(high, std::max(previous.second, next.second));
                } else {
                    low = std::min(low, std::min(current[left], current[right]));
                    high = std::max(high, std::max(current[left], current[right]));
                }
                emit(y * width + x, current[x], low, high);
                previous = middle;
                middle = next;
            }
        }
        return;
    }
    neighborhoods(labels, shape, connectivity, [&](std::size_t i, Label center, auto visit) {
        Label low = center;
        Label high = center;
        visit([&](Label neighbor) {
            low = std::min(low, neighbor);
            high = std::max(high, neighbor);
        });
        emit(i, center, low, high);
    });
}

// Match NumPy's contiguous double reduction, including its eight lanes.
double sum(const double* values, std::size_t n) {
    if (n < 8) {
        double total = -0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total += values[i];
        }
        return total;
    }
    if (n <= 128) {
        std::array<double, 8> partial{};
        std::copy_n(values, 8, partial.begin());
        const auto end = n - n % 8;
        for (std::size_t i = 8; i < end; i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                partial[j] += values[i + j];
            }
        }
        double total = ((partial[0] + partial[1]) + (partial[2] + partial[3])) +
                       ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for (std::size_t i = end; i < n; ++i) {
            total += values[i];
        }
        return total;
    }
    const auto halfway = n / 2 - (n / 2) % 8;
    return sum(values, halfway) + sum(values + halfway, n - halfway);
}

} // namespace

std::vector<Edge> adjacency(const Label* labels, const std::vector<std::size_t>& shape,
                            int connectivity) {
    std::vector<Edge> edges;
    std::unordered_set<Edge, EdgeHash, EdgeEqual> seen;
    neighborhoods(labels, shape, connectivity, [&](std::size_t, Label center, auto visit) {
        visit([&](Label neighbor) {
            if (neighbor != center && seen.insert(ordered(center, neighbor)).second) {
                edges.push_back({center, neighbor});
            }
        });
    });
    return edges;
}

std::vector<RegionColor> region_colors(const Label* labels, const double* rgb,
                                       std::size_t size, std::size_t label_count) {
    std::vector<RegionColor> regions(label_count);
    for (std::size_t i = 0; i < label_count; ++i) {
        regions[i].label = static_cast<Label>(i);
    }
    for (std::size_t i = 0; i < size; ++i) {
        const auto label = static_cast<std::size_t>(labels[i]);
        if (label >= label_count) {
            throw std::invalid_argument("RAG label out of range");
        }
        auto& region = regions[label];
        ++region.count;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            region.total[channel] += rgb[3 * i + channel];
        }
    }
    return regions;
}

std::vector<BoundaryEdge> boundary(const Label* labels, const double* values,
                                  const std::vector<std::size_t>& shape,
                                  int connectivity, bool single_precision) {
    std::unordered_map<Edge, BoundaryEdge, EdgeHash, EdgeEqual> edges;
    std::vector<BoundarySample> upper;
    auto accumulate = [&](const BoundarySample& sample) {
        auto& weight = edges.try_emplace(sample.edge, BoundaryEdge{sample.edge}).first->second;
        const double value = values[sample.pixel];
        const double total = weight.count ? weight.total + value : value;
        ++weight.count;
        weight.total = single_precision ? static_cast<double>(static_cast<float>(total)) : total;
    };
    visit_boundary(labels, shape, connectivity, accumulate, [&](const auto& sample) {
        upper.push_back(sample);
    });
    for (const auto& sample : upper) {
        accumulate(sample);
    }
    std::vector<BoundaryEdge> result;
    result.reserve(edges.size());
    for (const auto& item : edges) {
        result.push_back(item.second);
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.edge.first < b.edge.first ||
               (a.edge.first == b.edge.first && a.edge.second < b.edge.second);
    });
    return result;
}

std::vector<BoundarySample> boundary_samples(const Label* labels,
                                            const std::vector<std::size_t>& shape,
                                            int connectivity) {
    std::vector<BoundarySample> lower, upper;
    visit_boundary(labels, shape, connectivity,
                   [&](const auto& sample) { lower.push_back(sample); },
                   [&](const auto& sample) { upper.push_back(sample); });
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

std::vector<std::array<Label, 2>> centroids(const Label* labels, std::size_t height,
                                           std::size_t width, std::size_t label_count) {
    struct Moments {
        std::size_t count = 0;
        Label row = 0;
        Label column = 0;
    };
    std::vector<Moments> regions(label_count);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const auto label = static_cast<std::size_t>(*labels++);
            if (label >= label_count) {
                throw std::invalid_argument("RAG centroid label out of range");
            }
            auto& region = regions[label];
            ++region.count;
            region.row += static_cast<Label>(y);
            region.column += static_cast<Label>(x);
        }
    }
    std::vector<std::array<Label, 2>> results;
    for (std::size_t i = 1; i < regions.size(); ++i) {
        const auto& region = regions[i];
        if (region.count) {
            const auto count = static_cast<Label>(region.count);
            results.push_back({region.row / count, region.column / count});
        }
    }
    return results;
}

std::vector<Label> components(std::size_t node_count, const std::vector<Edge>& edges) {
    std::vector<std::size_t> parents(node_count);
    std::iota(parents.begin(), parents.end(), std::size_t{0});
    for (const auto& edge : edges) {
        const auto a = static_cast<std::size_t>(edge.first);
        const auto b = static_cast<std::size_t>(edge.second);
        if (a >= node_count || b >= node_count) {
            throw std::invalid_argument("RAG endpoint out of range");
        }
        const auto first = root(parents, a);
        const auto second = root(parents, b);
        parents[std::max(first, second)] = std::min(first, second);
    }
    std::vector<Label> numbered(node_count);
    Label next = 0;
    for (std::size_t i = 0; i < node_count; ++i) {
        const auto representative = root(parents, i);
        numbered[i] = representative == i ? next++ : numbered[representative];
    }
    return numbered;
}

std::vector<double> central_distances(std::size_t nodes, const std::int32_t* rows,
                                      const std::int32_t* columns,
                                      const double* weights, std::size_t entries) {
    if (nodes < 2 || nodes > 10000 || rows[0] != 0 ||
        static_cast<std::size_t>(rows[nodes]) != entries) {
        return {};
    }
    using Arc = std::pair<std::size_t, double>;
    std::vector<std::vector<Arc>> adj(nodes);
    bool uniform = true;
    const double unit = entries ? weights[0] : 1.0;
    const double max_weight = 0x1p53 / (static_cast<double>(nodes) * static_cast<double>(nodes));
    for (std::size_t i = 0; i < nodes; ++i) {
        if (rows[i] < 0 || rows[i] > rows[i + 1] ||
            static_cast<std::size_t>(rows[i + 1]) > entries) {
            return {};
        }
        for (auto k = rows[i]; k < rows[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(columns[k]);
            const auto weight = weights[k];
            if (j >= nodes || !(weight > 0 && weight <= max_weight) ||
                weight != std::floor(weight) || (k > rows[i] && columns[k - 1] >= columns[k])) {
                return {};
            }
            uniform = uniform && weight == unit;
            if (i != j) {
                adj[i].emplace_back(j, weight);
                adj[j].emplace_back(i, weight);
            }
        }
    }
    for (auto& neighbors : adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end(), [](Arc a, Arc b) {
            return a.first == b.first;
        }), neighbors.end());
    }

    // Undirected trees need just two traversals for all distance sums.
    std::vector<std::size_t> order{0};
    std::vector<std::size_t> parent(nodes, nodes);
    std::vector<double> parent_weight(nodes);
    parent[0] = 0;
    std::size_t arcs = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
        const auto vertex = order[i];
        arcs += adj[vertex].size();
        for (const auto& arc : adj[vertex]) {
            if (parent[arc.first] == nodes) {
                parent[arc.first] = vertex;
                parent_weight[arc.first] = arc.second;
                order.push_back(arc.first);
            }
        }
    }
    if (order.size() != nodes) {
        return {};
    }
    std::vector<double> totals(nodes);
    if (arcs == 2 * (nodes - 1)) {
        std::vector<std::size_t> subtree(nodes, 1);
        for (std::size_t i = nodes - 1; i > 0; --i) {
            const auto vertex = order[i];
            subtree[parent[vertex]] += subtree[vertex];
            totals[0] += static_cast<double>(subtree[vertex]) * parent_weight[vertex];
        }
        for (std::size_t i = 1; i < nodes; ++i) {
            const auto vertex = order[i];
            totals[vertex] = totals[parent[vertex]] +
                (static_cast<double>(nodes) - 2.0 * static_cast<double>(subtree[vertex])) *
                parent_weight[vertex];
        }
        return totals;
    }

    const int workers = detail::active_workers(nodes * arcs);
    if (uniform) {
        detail::partition(nodes, workers, [&](auto begin, auto end, auto) {
            std::vector<int> steps(nodes);
            std::vector<std::size_t> queue;
            queue.reserve(nodes);
            for (auto source = begin; source < end; ++source) {
                std::fill(steps.begin(), steps.end(), -1);
                steps[source] = 0;
                queue.assign(1, source);
                int total_steps = 0;
                for (std::size_t i = 0; i < queue.size(); ++i) {
                    const auto vertex = queue[i];
                    const int next_step = steps[vertex] + 1;
                    for (const auto& arc : adj[vertex]) {
                        if (steps[arc.first] == -1) {
                            steps[arc.first] = next_step;
                            total_steps += next_step;
                            queue.push_back(arc.first);
                        }
                    }
                }
                // The admission bound keeps all weighted totals exact in double.
                totals[source] = total_steps * unit;
            }
        });
        return totals;
    }

    const auto infinity = std::numeric_limits<double>::infinity();
    detail::partition(nodes, workers, [&](auto begin, auto end, auto) {
        std::vector<double> distances(nodes);
        using Candidate = std::pair<double, std::size_t>;
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
        for (auto source = begin; source < end; ++source) {
            std::fill(distances.begin(), distances.end(), infinity);
            distances[source] = 0.0;
            queue.emplace(0.0, source);
            while (!queue.empty()) {
                const auto [distance, vertex] = queue.top();
                queue.pop();
                if (distance != distances[vertex]) {
                    continue;
                }
                totals[source] += distance;
                for (const auto& arc : adj[vertex]) {
                    const auto candidate = distance + arc.second;
                    if (candidate < distances[arc.first]) {
                        distances[arc.first] = candidate;
                        queue.emplace(candidate, arc.first);
                    }
                }
            }
        }
    });
    return totals;
}

SpectralCut spectral_cut(const double* eigenvector, const double* degrees,
                         std::size_t nodes, const std::int32_t* columns,
                         const std::int32_t* rows, const double* weights,
                         std::size_t cuts) {
    SpectralCut best{std::vector<std::uint8_t>(nodes), std::numeric_limits<double>::infinity()};
    if (!nodes || !cuts) {
        return best;
    }
    const auto extrema = std::minmax_element(eigenvector, eigenvector + nodes);
    const auto low = *extrema.first;
    const auto high = *extrema.second;
    if (high - low <= 1e-8 + 1e-5 * std::abs(high)) {
        return best;
    }
    const double step = (high - low) / static_cast<double>(cuts);
    const auto threshold_at = [&](std::size_t candidate) {
        return low + (step == 0.0 ?
            static_cast<double>(candidate) / static_cast<double>(cuts) * (high - low) :
            static_cast<double>(candidate) * step);
    };
    std::vector<std::uint8_t> mask(nodes);
    std::vector<double> a, b;
    a.reserve(nodes);
    b.reserve(nodes);
    const auto exact_cost = [&]() {
        a.clear();
        b.clear();
        for (std::size_t i = 0; i < nodes; ++i) {
            (mask[i] ? a : b).push_back(degrees[i]);
        }
        if (a.empty() || b.empty()) {
            return std::numeric_limits<double>::infinity();
        }
        double crossing = 0.0;
        for (std::size_t i = 0; i < nodes; ++i) {
            for (auto k = columns[i]; k < columns[i + 1]; ++k) {
                if (mask[rows[k]] != mask[i]) {
                    crossing += weights[k];
                }
            }
        }
        crossing *= 0.5;
        return crossing / sum(a.data(), a.size()) + crossing / sum(b.data(), b.size());
    };

    double degree_total = 0.0;
    double weight_total = 0.0;
    bool sweepable = cuts > 2 && std::isfinite(step);
    for (std::size_t i = 0; sweepable && i < nodes; ++i) {
        sweepable = std::isfinite(eigenvector[i]) && degrees[i] > 0;
        degree_total += degrees[i];
        for (auto k = columns[i]; k < columns[i + 1]; ++k) {
            sweepable = sweepable && weights[k] >= 0;
            weight_total += weights[k];
        }
    }
    if (!sweepable || !std::isfinite(degree_total) || !std::isfinite(weight_total)) {
        for (std::size_t candidate = 0; candidate < cuts; ++candidate) {
            const auto threshold = threshold_at(candidate);
            for (std::size_t i = 0; i < nodes; ++i) {
                mask[i] = eigenvector[i] > threshold;
            }
            const auto cost = exact_cost();
            if (cost < best.cost) {
                best = {mask, cost};
            }
        }
        return best;
    }

    std::vector<std::size_t> order(nodes);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](auto first, auto second) {
        return eigenvector[first] < eigenvector[second];
    });
    // Retain both incidences so each directed entry contributes exactly once
    // when an endpoint moves, even for asymmetric matrices.
    using Arc = std::pair<std::size_t, double>;
    std::vector<std::vector<Arc>> incoming(nodes);
    std::size_t entries = 0;
    for (std::size_t i = 0; i < nodes; ++i) {
        for (auto k = columns[i]; k < columns[i + 1]; ++k) {
            const auto j = static_cast<std::size_t>(rows[k]);
            if (i != j) {
                incoming[j].emplace_back(i, weights[k]);
                ++entries;
            }
        }
    }

    // Bound the difference between incremental sums and the ordered cost.
    // Each edge is added once and subtracted once during the sweep.
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    const double degree_error = 8 * epsilon * (nodes + 1) * degree_total;
    const double crossing_error = 8 * epsilon * (entries + 1) * weight_total;
    struct Candidate {
        std::size_t moved;
        double lower;
    };
    std::vector<Candidate> candidates;
    double best_upper = std::numeric_limits<double>::infinity();
    std::fill(mask.begin(), mask.end(), std::uint8_t{1});
    std::size_t moved = 0;
    double crossing = 0.0, sum_b = 0.0;
    for (std::size_t candidate = 0; candidate < cuts && moved < nodes; ++candidate) {
        const auto threshold = threshold_at(candidate);
        const auto previous = moved;
        while (moved < nodes && eigenvector[order[moved]] <= threshold) {
            const auto i = order[moved++];
            mask[i] = 0;
            sum_b += degrees[i];
            for (auto k = columns[i]; k < columns[i + 1]; ++k) {
                const auto j = static_cast<std::size_t>(rows[k]);
                if (i != j) {
                    crossing += mask[j] ? weights[k] : -weights[k];
                }
            }
            for (const auto& [j, weight] : incoming[i]) {
                crossing += mask[j] ? weight : -weight;
            }
        }
        if (moved == previous || moved == nodes) {
            continue;
        }
        const double sum_a = degree_total - sum_b;
        const double low_crossing = 0.5 * std::max(0.0, crossing - crossing_error);
        const double high_crossing = 0.5 * (crossing + crossing_error);
        const double lower = (low_crossing / (sum_a + degree_error) +
                              low_crossing / (sum_b + degree_error)) * (1 - 8 * epsilon);
        if (lower <= best_upper) {
            candidates.push_back({moved, lower});
            if (std::min(sum_a, sum_b) > degree_error) {
                const auto upper = (high_crossing / (sum_a - degree_error) +
                                    high_crossing / (sum_b - degree_error)) * (1 + 8 * epsilon);
                best_upper = std::min(best_upper, upper);
            }
        }
    }

    // Re-evaluate contenders in threshold order to preserve strict ties and
    // the exact cost used by the Python caller to decide whether to recurse.
    std::fill(mask.begin(), mask.end(), std::uint8_t{1});
    moved = 0;
    for (const auto& candidate : candidates) {
        if (candidate.lower > best_upper || candidate.lower >= best.cost) {
            continue;
        }
        while (moved < candidate.moved) {
            mask[order[moved++]] = 0;
        }
        const auto cost = exact_cost();
        if (cost < best.cost) {
            best = {mask, cost};
        }
        if (best.cost == 0.0) {
            break;
        }
    }
    return best;
}

} // namespace skimagexx::graph_rag
