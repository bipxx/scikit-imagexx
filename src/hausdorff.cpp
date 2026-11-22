#include "skimagexx/hausdorff.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <limits>
#include <utility>

namespace skimagexx {
namespace {

constexpr std::size_t max_points = 256;
constexpr std::size_t max_bytes = 262144;
constexpr std::size_t max_extent = 16384;
using Point = std::array<int, 4>;
using Points = std::array<Point, max_points>;

template<std::size_t Rank>
bool enumerate(const HausdorffMask& input, Points& points, std::size_t& count) {
    count = 0;
    const auto end = input.shape.begin() + Rank;
    if (std::any_of(input.shape.begin(), end, [](auto extent) { return extent > max_extent; }))
        return false;
    if (std::find(input.shape.begin(), end, 0) != end) return true;
    std::size_t size = 1;
    for (auto it = input.shape.begin(); it != end; ++it) {
        if (size > max_bytes / *it) return false;
        size *= *it;
    }
    if (!input.bytes) return false;
    for (std::size_t offset = 0; offset < size; ++offset) {
        if (!input.bytes[offset]) continue;
        if (count == max_points) return false;
        auto index = offset;
        auto& point = points[count++];
        for (std::size_t axis = Rank - 1; axis > 0; --axis) {
            point[axis] = static_cast<int>(index % input.shape[axis]);
            index /= input.shape[axis];
        }
        point[0] = static_cast<int>(index);
    }
    return true;
}

struct Neighbor {
    unsigned squared = std::numeric_limits<unsigned>::max();
    std::size_t index = 0;
    bool tied = false;
};

void observe(Neighbor& nearest, unsigned squared, std::size_t index) {
    if (squared < nearest.squared)
        nearest = {squared, index, false};
    else if (squared == nearest.squared)
        nearest.tied = true;
}

class ScopedEnvironment {
public:
    ScopedEnvironment() : saved_(std::feholdexcept(&state_) == 0) {}
    ~ScopedEnvironment() { if (saved_) std::fesetenv(&state_); }
    bool ready() const { return saved_ && std::fegetround() == FE_TONEAREST; }

private:
    std::fenv_t state_;
    bool saved_;
};

template<std::size_t Rank>
bool compute(const HausdorffMask& a, const HausdorffMask& b,
             HausdorffOperation operation, HausdorffResult& result) {
    Points left{}, right{};
    HausdorffResult output;
    auto& nl = output.count[0];
    auto& nr = output.count[1];
    if (!enumerate<Rank>(a, left, nl) || !enumerate<Rank>(b, right, nr)) return false;
    if (!nl || !nr) {
        output.distance = nl || nr ? std::numeric_limits<double>::infinity() : 0.0;
        result = std::move(output);
        return true;
    }

    std::array<Neighbor, max_points> from_left, from_right;
    for (std::size_t i = 0; i < nl; ++i) {
        for (std::size_t j = 0; j < nr; ++j) {
            unsigned squared = 0;
            for (std::size_t axis = 0; axis < Rank; ++axis) {
                const int delta = left[i][axis] - right[j][axis];
                squared += static_cast<unsigned>(delta * delta);
            }
            observe(from_left[i], squared, j);
            observe(from_right[j], squared, i);
        }
    }

    if (operation == HausdorffOperation::modified) {
        const auto mean = [](const auto& nearest, std::size_t count) {
            double sum = 0;
            for (std::size_t i = 0; i < count; ++i)
                sum += std::sqrt(static_cast<double>(nearest[i].squared));
            return sum / static_cast<double>(count);
        };
        output.distance = std::max(mean(from_right, nr), mean(from_left, nl));
    } else {
        const auto less = [](const Neighbor& x, const Neighbor& y) { return x.squared < y.squared; };
        const auto l = std::max_element(from_left.begin(), from_left.begin() + nl, less);
        const auto r = std::max_element(from_right.begin(), from_right.begin() + nr, less);
        const bool left_query = l->squared > r->squared;
        const auto& winner = left_query ? *l : *r;
        output.distance = std::sqrt(static_cast<double>(winner.squared));
        if (operation == HausdorffOperation::pair) {
            // A single SciPy leaf visits targets in input order and replaces
            // a neighbor only for a strictly smaller distance.
            if (winner.tied && (left_query ? nr : nl) > 16) return false;
            if (left_query)
                output.attaining = {static_cast<std::size_t>(l - from_left.begin()), l->index};
            else
                output.attaining = {r->index, static_cast<std::size_t>(r - from_right.begin())};
            output.points[0].assign(left.begin(), left.begin() + nl);
            output.points[1].assign(right.begin(), right.begin() + nr);
        }
    }
    result = std::move(output);
    return true;
}

} // namespace

bool hausdorff(const HausdorffMask& a, const HausdorffMask& b, std::size_t rank,
               HausdorffOperation operation, HausdorffResult& result) {
    const ScopedEnvironment environment;
    if (!environment.ready()) return false;
    switch (rank) {
    case 1: return compute<1>(a, b, operation, result);
    case 2: return compute<2>(a, b, operation, result);
    case 3: return compute<3>(a, b, operation, result);
    case 4: return compute<4>(a, b, operation, result);
    default: return false;
    }
}

} // namespace skimagexx
