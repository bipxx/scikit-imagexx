#include "skimagexx/expand_labels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

// Axis-ordered separable Euclidean feature transform. Retain feature
// coordinates to preserve floating operation order in the pinned SciPy
// feature transform and in the final NumPy distance reduction.
// https://github.com/scipy/scipy/blob/v1.18.1/scipy/ndimage/src/ni_morphology.c
using Coordinate = std::int32_t;
template<std::size_t Rank> using Point = std::array<Coordinate, Rank>;

template<std::size_t Begin, std::size_t End, std::size_t Rank>
void advance_coordinates(Point<Rank>& point, const std::size_t* shape) {
    for (std::size_t axis = End; axis-- > Begin;) {
        if (++point[axis] < static_cast<Coordinate>(shape[axis]))
            return;
        point[axis] = 0;
    }
}

inline double squared_offset(Coordinate site, Coordinate position, double spacing) {
    const double difference = double(site - position) * spacing;
    return difference * difference;
}

template<class T, std::size_t Rank>
bool first_axis(const T* labels, std::size_t count, const std::size_t* shape,
                std::vector<Point<Rank>>& features) {
    const auto stride = count / shape[0];
    const auto length = static_cast<Coordinate>(shape[0]);
    bool any_seed = false;
    Point<Rank> site{};
    for (std::size_t offset = 0; offset < stride; ++offset, advance_coordinates<1, Rank>(site, shape)) {
        site[0] = -1;
        auto index = offset;
        for (Coordinate i = 0; i < length; ++i, index += stride) {
            if (labels[index] != 0) {
                any_seed = true;
                site[0] = i;
            }
            features[index] = site;
        }
        if (site[0] < 0)
            continue;

        index = offset + (shape[0] - 1) * stride;
        for (Coordinate i = length - 1; i >= 0; --i) {
            auto& previous = features[index];
            if (previous[0] == i)
                site[0] = i;
            else if (previous[0] < 0 || site[0] - i < i - previous[0])
                previous = site;
            if (i != 0)
                index -= stride;
        }
    }
    return any_seed;
}

template<std::size_t Rank>
struct Site {
    Point<Rank> feature;
    double orthogonal;
};

template<std::size_t Rank>
bool hidden(const Site<Rank>& a, const Site<Rank>& b, const Site<Rank>& c,
            std::size_t axis, double spacing) {
    const double left = double(b.feature[axis] - a.feature[axis]) * spacing;
    const double right = double(c.feature[axis] - b.feature[axis]) * spacing;
    const double length = left + right;
    const double gap = length * b.orthogonal - right * a.orthogonal - left * c.orthogonal;
    return gap - left * right * length > 0.0;
}

template<class T, std::size_t Rank>
void assign_label(const T* labels, const Point<Rank>& feature, const std::size_t* shape,
                  double squared_distance, double distance, T& result) {
    if (std::sqrt(squared_distance) > distance) {
        result = 0;
        return;
    }
    auto index = static_cast<std::size_t>(feature[0]);
    for (std::size_t axis = 1; axis < Rank; ++axis)
        index = index * shape[axis] + static_cast<std::size_t>(feature[axis]);
    result = labels[index];
}

template<std::size_t Axis, class T, std::size_t Rank>
void other_axis(const T* labels, std::size_t count, const std::size_t* shape,
                const double* sampling, double distance, std::vector<Point<Rank>>& features,
                std::vector<Site<Rank>>& envelope, T* result) {
    const auto length = static_cast<Coordinate>(shape[Axis]);
    const double spacing = sampling[Axis];
    std::size_t stride = 1;
    for (std::size_t later = Axis + 1; later < Rank; ++later)
        stride *= shape[later];
    const auto span = shape[Axis] * stride;
    Point<Rank> fixed{};
    for (std::size_t block = 0; block < count; block += span, advance_coordinates<0, Axis>(fixed, shape)) {
        for (std::size_t offset = 0; offset < stride; ++offset) {
            const auto start = block + offset;
            const auto stop = start + span;
            envelope.clear();
            for (auto index = start; index < stop; index += stride) {
                const auto feature = features[index];
                if (feature[0] < 0)
                    continue;
                double orthogonal = 0.0;
                // Only earlier axes contribute, and their fixed coordinates
                // are shared by every line in the block.
                for (std::size_t earlier = 0; earlier < Axis; ++earlier)
                    orthogonal += squared_offset(feature[earlier], fixed[earlier], sampling[earlier]);
                const Site<Rank> next{feature, orthogonal};
                while (envelope.size() >= 2 &&
                       hidden(envelope[envelope.size() - 2], envelope.back(), next, Axis, spacing))
                    envelope.pop_back();
                envelope.push_back(next);
            }
            if (envelope.empty()) {
                // No intermediate candidates means this line keeps its
                // absent features. The final sweep writes all outcomes.
                if constexpr (Axis + 1 == Rank)
                    std::fill_n(result + start, length, T(0));
                continue;
            }
            std::size_t site = 0;
            auto index = start;
            for (Coordinate pos = 0; pos < length; ++pos, index += stride) {
                const auto squared_distance = [&](const Site<Rank>& candidate) {
                    return candidate.orthogonal + squared_offset(candidate.feature[Axis], pos, spacing);
                };
                auto best = squared_distance(envelope[site]);
                while (site + 1 < envelope.size()) {
                    const auto next = squared_distance(envelope[site + 1]);
                    if (best <= next)
                        break;
                    best = next;
                    ++site;
                }
                const auto& feature = envelope[site].feature;
                if constexpr (Axis + 1 == Rank)
                    assign_label(labels, feature, shape, best, distance, result[index]);
                else
                    features[index] = feature;
            }
        }
    }
}

template<class T>
void expand_line(const T* labels, std::size_t count, double spacing, double distance, T* result) {
    const auto* first = std::find_if(labels, labels + count, [](T label) { return label != 0; });
    if (first == labels + count) {
        std::fill_n(result, count, T(0));
        return;
    }

    auto nearest = static_cast<Coordinate>(first - labels);
    Coordinate position = 0;
    const auto write_through = [&](Coordinate last) {
        const T label = labels[nearest];
        for (; position <= last; ++position) {
            const double squared = squared_offset(nearest, position, spacing);
            result[position] = std::sqrt(squared) > distance ? T(0) : label;
        }
    };
    for (auto next = nearest + 1; next < static_cast<Coordinate>(count); ++next) {
        if (labels[next] != 0) {
            // The midpoint belongs to the lower feature coordinate on a tie.
            write_through(nearest + (next - nearest) / 2);
            nearest = next;
        }
    }
    write_through(static_cast<Coordinate>(count - 1));
}

template<class T, std::size_t Rank>
void expand(const T* labels, std::size_t count, const std::size_t* shape,
            const double* sampling, double distance, T* result) {
    std::vector<Point<Rank>> features(count);
    if (!first_axis(labels, count, shape, features)) {
        std::fill_n(result, count, T(0));
        return;
    }
    std::vector<Site<Rank>> envelope;
    envelope.reserve(*std::max_element(shape + 1, shape + Rank));
    other_axis<1>(labels, count, shape, sampling, distance, features, envelope, result);
    if constexpr (Rank == 3)
        other_axis<2>(labels, count, shape, sampling, distance, features, envelope, result);
}

}  // namespace

template<class T>
void expand_labels_into(const T* labels, std::size_t label_count,
                        std::size_t rank, const std::size_t* shape,
                        const double* sampling, double distance,
                        T* result, std::size_t result_count) {
    if (!labels || !result || !shape || !sampling || rank < 1 || rank > 3)
        throw std::invalid_argument("expand_labels requires valid storage and rank 1 to 3");
    constexpr std::size_t max_elements = 64 * 1024 * 1024;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto length = shape[axis];
        if (length == 0 || length > 65536 || count > max_elements / length ||
            !(sampling[axis] >= 1.0 / 65536.0 && sampling[axis] <= 65536.0))
            throw std::invalid_argument("expand_labels shape or spacing is outside the native domain");
        count *= length;
    }
    if (label_count < count || result_count < count)
        throw std::invalid_argument("expand_labels buffers have insufficient extent");
    if (!(distance >= 0)) {
        std::fill_n(result, count, T(0));
    } else if (distance == 0) {
        std::copy_n(labels, count, result);
    } else if (rank == 1) {
        expand_line(labels, count, sampling[0], distance, result);
    } else if (rank == 2) {
        expand<T, 2>(labels, count, shape, sampling, distance, result);
    } else {
        expand<T, 3>(labels, count, shape, sampling, distance, result);
    }
}

#define SKIMAGEXX_INSTANTIATE_EXPANSION(T) \
    template void expand_labels_into<T>(const T*, std::size_t, std::size_t, const std::size_t*, \
                                        const double*, double, T*, std::size_t)
SKIMAGEXX_INSTANTIATE_EXPANSION(std::int8_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::uint8_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::int16_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::uint16_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::int32_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::uint32_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::int64_t);
SKIMAGEXX_INSTANTIATE_EXPANSION(std::uint64_t);
#undef SKIMAGEXX_INSTANTIATE_EXPANSION

}  // namespace skimagexx
