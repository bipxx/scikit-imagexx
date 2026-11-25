#include "skimagexx/remove_objects_by_distance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {

using Coordinates = std::array<std::size_t, 3>;

struct Layout {
    Coordinates shape{1, 1, 1};
    Coordinates stride{1, 1, 1};

    Coordinates coordinates(std::size_t index) const {
        return {index / stride[0], index / stride[1] % shape[1], index % shape[2]};
    }
};

struct Object {
    std::size_t count = 0;
    std::size_t first = 0;
    bool removed = false;
};

struct Point {
    std::array<double, 3> coordinates{};
    std::size_t next = 0;
    std::uint32_t label = 0;
};

template<class T>
bool is_boundary(const T* labels, std::size_t index, const Coordinates& where,
                 const Layout& layout, std::size_t rank, bool full_connectivity) {
    const T label = labels[index];
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if ((where[axis] && labels[index - layout.stride[axis]] != label) ||
            (where[axis] + 1 < layout.shape[axis] && labels[index + layout.stride[axis]] != label))
            return true;
    }
    if (!full_connectivity || rank == 1)
        return false;

    Coordinates low{}, high{};
    for (std::size_t axis = 0; axis < rank; ++axis) {
        low[axis] = where[axis] ? where[axis] - 1 : 0;
        high[axis] = std::min(where[axis] + 1, layout.shape[axis] - 1);
    }
    for (std::size_t x = low[0]; x <= high[0]; ++x)
        for (std::size_t y = low[1]; y <= high[1]; ++y)
            for (std::size_t z = low[2]; z <= high[2]; ++z)
                if (labels[x * layout.stride[0] + y * layout.stride[1] + z] != label)
                    return true;
    return false;
}

// Median partitioning stores the spatial tree in the point vector itself.
// Each leaf is a small contiguous neighborhood.
constexpr std::ptrdiff_t leaf_size = 16;

using Points = std::vector<Point>;
using PointIterator = Points::iterator;

void partition_points(PointIterator first, PointIterator last, std::size_t axis,
                      std::size_t rank) {
    const auto size = last - first;
    if (size <= leaf_size)
        return;
    const auto middle = first + size / 2;
    std::nth_element(first, middle, last, [axis](const Point& a, const Point& b) {
        return a.coordinates[axis] < b.coordinates[axis];
    });
    const auto next_axis = (axis + 1) % rank;
    partition_points(first, middle, next_axis, rank);
    partition_points(middle + 1, last, next_axis, rank);
}

struct Neighborhood {
    std::size_t rank;
    ObjectDistanceNorm norm;
    double radius;
    double cutoff;
    std::vector<Object>& objects;
    std::size_t remaining;

    bool contains(const Point& center, const Point& neighbor) const {
        double distance = 0;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            const double delta = std::abs(center.coordinates[axis] - neighbor.coordinates[axis]);
            switch (norm) {
            case ObjectDistanceNorm::manhattan:
                distance += delta;
                break;
            case ObjectDistanceNorm::euclidean:
                distance += delta * delta;
                break;
            case ObjectDistanceNorm::chebyshev:
                distance = std::max(distance, delta);
                break;
            }
        }
        return distance <= cutoff;
    }

    void consider(const Point& center, const Point& neighbor) {
        auto& candidate = objects[neighbor.label];
        if (neighbor.label != center.label && !candidate.removed && contains(center, neighbor)) {
            candidate.removed = true;
            --remaining;
        }
    }

    void remove_neighbors(const Point& center, Points::const_iterator first,
                          Points::const_iterator last, std::size_t axis) {
        const auto size = last - first;
        if (remaining <= 1)
            return;
        if (size <= leaf_size) {
            for (; first != last && remaining > 1; ++first)
                consider(center, *first);
            return;
        }

        const auto middle = first + size / 2;
        const double delta = center.coordinates[axis] - middle->coordinates[axis];
        const auto next_axis = (axis + 1) % rank;
        if (std::abs(delta) <= radius)
            consider(center, *middle);
        if (delta <= radius)
            remove_neighbors(center, first, middle, next_axis);
        if (delta >= -radius)
            remove_neighbors(center, middle + 1, last, next_axis);
    }
};

}  // namespace

template<class T>
bool remove_objects_by_distance_into(
    const T* labels, std::size_t label_count, std::size_t rank,
    const std::size_t* shape, const double* spacing, double min_distance,
    ObjectDistanceNorm norm, const double* priority, std::size_t priority_count,
    T* output, std::size_t output_count) {
    if (!labels || !shape || !spacing || !output || rank == 0 || rank > 3 ||
        label_count != output_count || label_count > object_distance_max_elements ||
        !(0 <= min_distance && min_distance <= 1048576) ||
        (!priority && priority_count != 0) ||
        (norm != ObjectDistanceNorm::manhattan && norm != ObjectDistanceNorm::euclidean &&
         norm != ObjectDistanceNorm::chebyshev))
        throw std::invalid_argument("invalid object distance descriptor");

    Layout layout;
    std::size_t size = 1;
    bool full_connectivity = norm == ObjectDistanceNorm::chebyshev;
    for (std::size_t axis = rank; axis-- > 0;) {
        if (shape[axis] == 0 || shape[axis] > object_distance_max_extent ||
            size > object_distance_max_elements / shape[axis])
            throw std::invalid_argument("unsupported object distance shape");
        const double factor = spacing[axis];
        if (!(1.0 / 16 <= factor && factor <= 16) || std::trunc(factor * 16) != factor * 16)
            throw std::invalid_argument("unsupported object distance spacing");
        layout.shape[axis] = shape[axis];
        layout.stride[axis] = size;
        size *= shape[axis];
        full_connectivity = full_connectivity || factor != spacing[0];
    }
    if (size != label_count)
        throw std::invalid_argument("object distance span does not match shape");

    std::size_t max_label = 0;
    for (std::size_t index = 0; index < size; ++index) {
        const T label = labels[index];
        if constexpr (std::is_signed_v<T>) {
            if (label < 0)
                return false;
        }
        if (static_cast<std::uint64_t>(label) > object_distance_max_label)
            return false;
        max_label = std::max(max_label, static_cast<std::size_t>(label));
    }
    if (max_label <= 1 && !priority) {
        std::copy_n(labels, size, output);
        return true;
    }

    std::vector<Object> objects(max_label + 1);
    Points points;
    for (std::size_t index = 0; index < size; ++index) {
        const auto label = static_cast<std::uint32_t>(labels[index]);
        if (!label)
            continue;
        ++objects[label].count;
        const auto where = layout.coordinates(index);
        if (!is_boundary(labels, index, where, layout, rank, full_connectivity))
            continue;
        Point point;
        point.label = label;
        for (std::size_t axis = 0; axis < rank; ++axis)
            point.coordinates[axis] = static_cast<double>(where[axis]) * spacing[axis];
        points.push_back(point);
    }
    if (points.empty()) {
        std::copy_n(labels, size, output);
        return true;
    }

    partition_points(points.begin(), points.end(), 0, rank);
    std::vector<std::uint32_t> order;
    for (std::size_t i = 0; i < points.size(); ++i) {
        auto& point = points[i];
        auto& object = objects[point.label];
        if (object.first == 0) {
            if (priority && (point.label >= priority_count || !std::isfinite(priority[point.label])))
                return false;
            order.push_back(point.label);
        }
        point.next = object.first;
        object.first = i + 1;
    }
    const auto score = [&](std::uint32_t id) {
        return priority ? priority[id] : static_cast<double>(objects[id].count);
    };
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        const double left = score(a), right = score(b);
        return left == right ? a > b : left > right;
    });

    const double cutoff = norm == ObjectDistanceNorm::euclidean
        ? min_distance * min_distance : min_distance;
    Neighborhood neighborhood{rank, norm, min_distance, cutoff, objects, order.size()};
    for (const auto label : order) {
        if (neighborhood.remaining <= 1)
            break;
        const auto& object = objects[label];
        if (!object.removed) {
            for (auto next = object.first; next && neighborhood.remaining > 1;) {
                const auto& point = points[next - 1];
                neighborhood.remove_neighbors(point, points.cbegin(), points.cend(), 0);
                next = point.next;
            }
        }
    }
    for (std::size_t index = 0; index < size; ++index) {
        const T label = labels[index];
        output[index] = objects[static_cast<std::size_t>(label)].removed ? T(0) : label;
    }
    return true;
}

#define SKIMAGEXX_DISTANCE_INSTANTIATE(T)                                                    \
    template bool remove_objects_by_distance_into(                                          \
        const T*, std::size_t, std::size_t, const std::size_t*, const double*, double,        \
        ObjectDistanceNorm, const double*, std::size_t, T*, std::size_t)
SKIMAGEXX_DISTANCE_INSTANTIATE(std::int8_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::uint8_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::int16_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::uint16_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::int32_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::uint32_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::int64_t);
SKIMAGEXX_DISTANCE_INSTANTIATE(std::uint64_t);
#undef SKIMAGEXX_DISTANCE_INSTANTIATE

}  // namespace skimagexx
