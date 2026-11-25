// Flood and heap ordering from scikit-image 0.26.0 _watershed_cy.pyx,
// heap_watershed.pxi and heap_general.pxi, BSD-3-Clause. See
// third_party/scikit-image-LICENSE.txt.
#include <skimagexx/watershed.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;

#ifdef _MSC_VER
#define WATERSHED_INLINE __forceinline
#else
#define WATERSHED_INLINE inline
#endif

template<class Position, bool TrackSource>
struct Item {
    double value;
    std::uint32_t age;
    Position index;
    Position source;
};

template<class Position>
struct Item<Position, false> {
    double value;
    std::uint32_t age;
    Position index;
};

template<class Entry>
WATERSHED_INLINE bool smaller(const Entry& a, const Entry& b) {
    return a.value < b.value || (a.value == b.value && a.age < b.age);
}

template<class Position, bool TrackSource>
class FloodHeap {
public:
    using Entry = Item<Position, TrackSource>;
    bool empty() const { return size_ == 0; }

    WATERSHED_INLINE void push(double value, std::uint32_t age, Position index, Position source) {
        if (size_ == capacity_)
            grow();
        const auto item = [&] {
            if constexpr (TrackSource)
                return Entry{value, age, index, source};
            else
                return Entry{value, age, index};
        }();
        auto child = size_++;
        auto* items = items_.get();
        while (child) {
            const auto parent = (child - 1) / 2;
            if (!smaller(item, items[parent]))
                break;
            items[child] = items[parent];
            child = parent;
        }
        items[child] = item;
    }

    WATERSHED_INLINE Entry pop() {
        auto* items = items_.get();
        const Entry result = items[0];
        if (--size_ == 0)
            return result;
        const Entry last = items[size_];

        std::size_t parent = 0;
        for (auto left = 1u + 2 * parent; left < size_; left = 1u + 2 * parent) {
            // Keep the comparison with the displaced item first. A NaN is
            // unordered, so comparing only the two children changes ties.
            auto next = smaller(items[left], last) ? left : parent;
            const auto right = left + 1;
            if (right < size_ && smaller(items[right], next == parent ? last : items[left]))
                next = right;
            if (next == parent)
                break;
            items[parent] = items[next];
            parent = next;
        }
        items[parent] = last;
        return result;
    }

private:
    void grow() {
        const auto capacity = capacity_ ? 2 * capacity_ : 1024;
        std::unique_ptr<Entry[]> next(new Entry[capacity]);
        if (size_)
            std::copy_n(items_.get(), size_, next.get());
        items_ = std::move(next);
        capacity_ = capacity;
    }

    std::unique_ptr<Entry[]> items_;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

template<class Position>
double distance(Position a, Position b, const std::vector<Index>& strides) {
    double squared = 0;
    for (const auto wide_stride : strides) {
        const auto stride = static_cast<Position>(wide_stride);
        const double delta = static_cast<double>(a / stride - b / stride);
        squared += delta * delta;
        a %= stride;
        b %= stride;
    }
    return std::sqrt(squared);
}

// Bias the signed 32-bit age into unsigned comparison order. Unsigned wrap
// preserves the stock conversion when the insertion counter crosses INT_MAX.
constexpr std::uint32_t seed_age = UINT32_C(0x80000000);

template<class T, class Position, bool Compact, bool Line>
void flood(const std::vector<double>& image, std::vector<T>& labels,
           std::vector<std::uint8_t>& mask, const std::vector<Index>& neighbors,
           const std::vector<Index>& strides, double compactness,
           Position axis2_width, const std::vector<Position>& row_indices,
           FloodHeap<Position, Compact || Line>& heap) {
    std::uint32_t age = seed_age + 1;
    while (!heap.empty()) {
        const auto item = heap.pop();
        auto source = item.index;
        if constexpr (Compact || Line)
            source = item.source;
        if constexpr (Compact || Line) {
            const bool visited = Compact ? !mask[item.index] : labels[item.index] != 0;
            if (visited && item.index != source)
                continue;
        }
        const auto label = labels[source];
        if constexpr (Compact || Line) {
            if constexpr (Compact) {
                labels[item.index] = label;
                mask[item.index] = 0;
            } else if (mask[item.index]) {
                for (const auto offset : neighbors) {
                    const auto next = item.index + offset;
                    if (mask[next] && labels[next] && labels[next] != label) {
                        mask[item.index] = 0;
                        break;
                    }
                }
                if (mask[item.index])
                    labels[item.index] = label;
            }
        }
        Position row_delta{};
        Position column_delta{};
        if constexpr (Compact) {
            if (axis2_width) {
                row_delta = row_indices[item.index] - row_indices[source];
                column_delta = item.index - source - row_delta * axis2_width;
            }
        }
        const auto visit = [&](Position next, int row_step, int column_step) {
            if (!mask[next] || (Line && labels[next]))
                return;
            double value = image[next];
            if constexpr (Compact) {
                double separation;
                if (axis2_width) {
                    const double dy = static_cast<double>(row_delta + row_step);
                    const double dx = static_cast<double>(column_delta + column_step);
                    separation = std::sqrt(dy * dy + dx * dx);
                } else {
                    separation = distance(next, source, strides);
                }
                value += compactness * separation;
            } else if constexpr (!Line) {
                labels[next] = label;
                mask[next] = 0;
            }
            if (value < item.value)
                value = item.value;
            heap.push(value, ++age, next, source);
        };
        if (axis2_width) {
            visit(item.index - axis2_width, -1, 0);
            visit(item.index - 1, 0, -1);
            visit(item.index + 1, 0, 1);
            visit(item.index + axis2_width, 1, 0);
        } else {
            for (const auto offset : neighbors)
                visit(static_cast<Position>(item.index + offset), 0, 0);
        }
    }
}

#undef WATERSHED_INLINE

}  // namespace

template<class T>
void watershed_into(const double* image, const T* markers, const std::uint8_t* mask,
                    std::size_t count, const std::size_t* shape, std::size_t rank,
                    double compactness, bool watershed_line, T* output,
                    const WatershedNeighborhood* neighborhood, bool boolean_markers) {
    if (!rank || !shape || (neighborhood && neighborhood->padding.size() != rank) ||
        (boolean_markers && !std::is_same_v<T, std::uint8_t>))
        throw std::invalid_argument("invalid watershed dimensions or marker type");
    if (std::find(shape, shape + rank, 0) != shape + rank) {
        if (count)
            throw std::invalid_argument("watershed count disagrees with empty shape");
        return;
    }

    const auto max = static_cast<std::size_t>(std::numeric_limits<Index>::max());
    std::vector<Index> strides(rank);
    std::vector<Index> row_advances(rank);
    std::size_t padded_count = 1;
    std::size_t original_count = 1;
    Index first = 0;
    for (auto axis = rank; axis-- > 0;) {
        const auto padding = neighborhood ? neighborhood->padding[axis] : 1;
        if (shape[axis] > max || padding > (max - shape[axis]) / 2 ||
            shape[axis] + 2 * padding > max / padded_count)
            throw std::invalid_argument("watershed padded dimensions are too large");
        strides[axis] = static_cast<Index>(padded_count);
        first += static_cast<Index>(padding * padded_count);
        row_advances[axis] = static_cast<Index>(2 * padding * padded_count);
        original_count *= shape[axis];
        padded_count *= shape[axis] + 2 * padding;
    }
    if (count != original_count || !image || !markers || !output)
        throw std::invalid_argument("watershed buffers do not match shape");

    std::vector<Index> axis_offsets;
    if (!neighborhood) {
        for (const auto stride : strides)
            axis_offsets.push_back(-stride);
        axis_offsets.insert(axis_offsets.end(), strides.rbegin(), strides.rend());
    }
    const auto& offsets = neighborhood ? neighborhood->offsets : axis_offsets;
    for (const auto offset : offsets) {
        if (offset < -first || offset > first)
            throw std::invalid_argument("watershed offsets exceed padded storage");
    }

    std::vector<double> elevations(padded_count);
    std::vector<T> labels(padded_count);
    std::vector<std::uint8_t> active(padded_count);
    // Walk row starts with an odometer, skipping padding rather than dividing
    // each pixel index to recover coordinates.
    std::vector<std::size_t> coordinates(rank, 0);
    const auto width = shape[rank - 1];
    auto visit_rows = [&](const auto& visit) {
        Index row = first;
        for (std::size_t start = 0; start < count; start += width) {
            visit(start, row);
            if (start + width == count)
                break;
            row += static_cast<Index>(width) + row_advances.back();
            for (auto axis = rank - 1; axis-- > 0;) {
                if (++coordinates[axis] < shape[axis])
                    break;
                coordinates[axis] = 0;
                row += row_advances[axis];
            }
        }
    };

    const auto run = [&](auto position, auto compact, auto line) {
        using Position = decltype(position);
        constexpr bool Compact = decltype(compact)::value;
        constexpr bool Line = decltype(line)::value;
        FloodHeap<Position, Compact || Line> heap;
        const auto axis2_width = static_cast<Position>(!neighborhood && rank == 2 ? strides[0] : 0);
        std::vector<Position> row_indices;
        Position row_index = 0;
        if constexpr (Compact) {
            if (axis2_width)
                row_indices.resize(padded_count);
        }
        visit_rows([&](std::size_t start, Index row) {
            std::copy_n(image + start, width, elevations.data() + row);
            if (!row_indices.empty())
                std::fill_n(row_indices.data() + row, width, ++row_index);
            for (std::size_t column = 0; column < width; ++column) {
                const auto i = start + column;
                const auto p = static_cast<Position>(row + static_cast<Index>(column));
                const bool included = !mask || mask[i];
                const T marker = markers[i];
                // Only the line path examines the labels of active neighbors.
                active[p] = included && (Line || !marker);
                if (included && marker) {
                    labels[p] = boolean_markers ? T(1) : marker;
                    heap.push(elevations[p], seed_age, p, p);
                }
            }
        });
        flood<T, Position, Compact, Line>(elevations, labels, active, offsets, strides,
                                         compactness, axis2_width, row_indices, heap);
    };
    const auto dispatch = [&](auto position) {
        if (compactness > 0)
            run(position, std::true_type{}, std::false_type{});
        else if (watershed_line)
            run(position, std::false_type{}, std::true_type{});
        else
            run(position, std::false_type{}, std::false_type{});
    };
    if (padded_count <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        dispatch(std::int32_t{});
    else
        dispatch(Index{});

    std::fill(coordinates.begin(), coordinates.end(), 0);
    visit_rows([&](std::size_t start, Index row) {
        std::copy_n(labels.data() + row, width, output + start);
    });
}

#define INSTANTIATE(T) \
    template void watershed_into<T>(const double*, const T*, const std::uint8_t*, \
                                    std::size_t, const std::size_t*, std::size_t, \
                                    double, bool, T*, const WatershedNeighborhood*, bool)
INSTANTIATE(std::int8_t);
INSTANTIATE(std::uint8_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::int64_t);
INSTANTIATE(std::uint64_t);
#undef INSTANTIATE

}  // namespace skimagexx
