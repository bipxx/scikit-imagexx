// Flood and heap ordering from scikit-image 0.26.0 _watershed_cy.pyx,
// heap_watershed.pxi and heap_general.pxi, BSD-3-Clause. See
// third_party/scikit-image-LICENSE.txt.
#include <skimagexx/watershed.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;

#ifdef _MSC_VER
#define WATERSHED_INLINE __forceinline
#else
#define WATERSHED_INLINE inline
#endif

template<class Position, bool TrackSource, class Priority = double>
struct Item {
    Priority priority;
    std::uint32_t age;
    Position index;
    Position source;
};

template<class Position, class Priority>
struct Item<Position, false, Priority> {
    Priority priority;
    std::uint32_t age;
    Position index;
};

constexpr auto sign_bit = UINT64_C(1) << 63;
static_assert(sizeof(double) == sizeof(std::uint64_t));

std::uint64_t order_key(double value) {
    std::uint64_t bits = 0;
    if (value != 0)
        std::memcpy(&bits, &value, sizeof(bits));
    return bits ^ ((UINT64_C(0) - (bits >> 63)) | sign_bit);
}

template<class Position, bool TrackSource, bool Ordered>
class FloodHeap {
public:
    using Entry = Item<Position, TrackSource, std::conditional_t<Ordered, std::uint64_t, double>>;

    explicit FloodHeap(std::size_t capacity)
        : items_(new Entry[capacity]), capacity_(capacity) {}

    bool empty() const { return size_ == 0; }

    static double value(const Entry& item) {
        if constexpr (Ordered) {
            const auto key = item.priority;
            const auto bits = key ^ (((key >> 63) - 1) | sign_bit);
            double decoded;
            std::memcpy(&decoded, &bits, sizeof(decoded));
            return decoded;
        } else {
            return item.priority;
        }
    }

    WATERSHED_INLINE void push(double value, std::uint32_t age, Position index, Position source) {
        if (size_ == capacity_)
            grow();
        const auto priority = [&] {
            if constexpr (Ordered)
                return order_key(value);
            else
                return value;
        }();
        const auto item = [&] {
            if constexpr (TrackSource)
                return Entry{priority, age, index, source};
            else
                return Entry{priority, age, index};
        }();
        auto child = size_++;
        auto* items = items_.get();
        while (child) {
            const auto parent = (child - 1) / arity;
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
        for (auto left = 1u + arity * parent; left < size_; left = 1u + arity * parent) {
            // Start with the displaced entry to preserve the binary heap's
            // behavior when floating-point comparisons are unordered.
            const Entry* next = &last;
            for (auto child = left, end = left + std::min(arity, size_ - left); child < end; ++child)
                if (smaller(items[child], *next))
                    next = items + child;
            if (next == &last)
                break;
            items[parent] = *next;
            parent = static_cast<std::size_t>(next - items);
        }
        items[parent] = last;
        return result;
    }

private:
    static constexpr std::size_t arity = Ordered ? 4 : 2;

    static WATERSHED_INLINE bool smaller(const Entry& a, const Entry& b) {
        if constexpr (Ordered)
            return (a.priority < b.priority) | ((a.priority == b.priority) & (a.age < b.age));
        else
            return a.priority < b.priority || (a.priority == b.priority && a.age < b.age);
    }

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

template<class Position, bool TrackSource>
struct BucketEntry {
    unsigned priority;
    Position index;
    Position source;
};

template<class Position>
struct BucketEntry<Position, false> {
    unsigned priority;
    Position index;
};

template<class Position, bool TrackSource>
class FloodBuckets {
public:
    using Entry = BucketEntry<Position, TrackSource>;

    bool empty() {
        while (priority_ < buckets_.size() && head_ == buckets_[priority_].size()) {
            buckets_[priority_].clear();
            ++priority_;
            head_ = 0;
        }
        return priority_ == buckets_.size();
    }

    static double value(const Entry& item) { return item.priority; }

    void push(double value, std::uint32_t, Position index, Position source) {
        const auto priority = static_cast<unsigned>(value);
        priority_ = std::min(priority_, priority);
        if constexpr (TrackSource)
            buckets_[priority].push_back({priority, index, source});
        else
            buckets_[priority].push_back({priority, index});
    }

    Entry pop() { return buckets_[priority_][head_++]; }

private:
    std::array<std::vector<Entry>, 256> buckets_;
    unsigned priority_ = 256;
    std::size_t head_ = 0;
};

template<class Position, bool TrackSource>
class FloodRadixQueue {
public:
    using Entry = BucketEntry<Position, TrackSource>;

    bool empty() {
        if (head_ < buckets_[0].size())
            return false;
        buckets_[0].clear();
        head_ = 0;
        for (std::size_t i = 1; i < buckets_.size(); ++i) {
            auto& bucket = buckets_[i];
            if (bucket.empty())
                continue;
            priority_ = std::min_element(bucket.begin(), bucket.end(),
                [](const Entry& a, const Entry& b) { return a.priority < b.priority; })->priority;
            // Redistribute in insertion order so equal priorities remain FIFO.
            for (const auto& entry : bucket)
                insert(entry);
            bucket.clear();
            return false;
        }
        return true;
    }

    static double value(const Entry& item) { return item.priority; }

    void push(double value, std::uint32_t, Position index, Position source) {
        const auto priority = static_cast<unsigned>(value);
        if constexpr (TrackSource)
            insert({priority, index, source});
        else
            insert({priority, index});
    }

    Entry pop() { return buckets_[0][head_++]; }

private:
    void insert(const Entry& entry) {
        auto difference = entry.priority ^ priority_;
        unsigned bucket = 0;
        if (difference >= 256) { difference >>= 8; bucket += 8; }
        if (difference >= 16) { difference >>= 4; bucket += 4; }
        if (difference >= 4) { difference >>= 2; bucket += 2; }
        if (difference >= 2) { difference >>= 1; ++bucket; }
        buckets_[bucket + difference].push_back(entry);
    }

    // The flood never inserts a priority below the one just popped.
    // Entries sharing the highest differing bit move to lower buckets.
    std::array<std::vector<Entry>, 17> buckets_;
    unsigned priority_ = 0;
    std::size_t head_ = 0;
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

template<class T, class Position, bool Compact, bool Line, class Queue>
void flood(const double* image, T* labels,
           std::vector<std::uint8_t>& mask, const std::vector<Index>& neighbors,
           const std::vector<Index>& strides, double compactness,
           Position axis2_width, const std::vector<Position>& row_indices,
           Queue& heap) {
    std::uint32_t age = seed_age + 1;
    while (!heap.empty()) {
        const auto item = heap.pop();
        const auto current_value = heap.value(item);
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
            if (value < current_value)
                value = current_value;
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

template<class T, class Position, bool Compact, bool Line, class Queue>
void fill_and_flood_queue(const double* elevations, T* labels,
                          std::vector<std::uint8_t>& active, const std::vector<Index>& offsets,
                          const std::vector<Index>& strides, double compactness,
                          Position axis2_width, const std::vector<Position>& row_indices,
                          const std::vector<Index>& seeds, Queue& queue) {
    for (const auto p : seeds) {
        const auto seed = static_cast<Position>(p);
        queue.push(elevations[p], seed_age, seed, seed);
    }
    flood<T, Position, Compact, Line>(elevations, labels, active, offsets, strides,
                                      compactness, axis2_width, row_indices, queue);
}

template<class T, class Position, bool Compact, bool Line, class VisitRows>
void run_watershed_mode(const double* elevations, T* labels,
                        std::vector<std::uint8_t>& active, const std::vector<Index>& offsets,
                        const std::vector<Index>& strides, double compactness,
                        const WatershedNeighborhood* neighborhood, std::size_t rank,
                        std::size_t width, std::size_t count, std::size_t padded_count,
                        std::vector<std::size_t>& coordinates, const std::vector<Index>& seeds,
                        bool ordered, unsigned levels, VisitRows&& visit_rows) {
    const auto axis2_width = static_cast<Position>(!neighborhood && rank == 2 ? strides[0] : 0);
    std::vector<Position> row_indices;
    if constexpr (Compact) {
        if (axis2_width) {
            row_indices.resize(padded_count);
            Position row_index = 0;
            std::fill(coordinates.begin(), coordinates.end(), 0);
            visit_rows([&](std::size_t, Index row) {
                std::fill_n(row_indices.data() + row, width, ++row_index);
            });
        }
    }
    const auto capacity = std::max(seeds.size(), std::min(count, std::size_t(1024)));
    if constexpr (!Compact) {
        if (ordered && levels) {
            if (levels <= 256) {
                FloodBuckets<Position, Line> buckets;
                fill_and_flood_queue<T, Position, Compact, Line>(
                    elevations, labels, active, offsets, strides, compactness,
                    axis2_width, row_indices, seeds, buckets);
            } else {
                FloodRadixQueue<Position, Line> buckets;
                fill_and_flood_queue<T, Position, Compact, Line>(
                    elevations, labels, active, offsets, strides, compactness,
                    axis2_width, row_indices, seeds, buckets);
            }
            return;
        }
    }
    if (ordered) {
        FloodHeap<Position, Compact || Line, true> heap(capacity);
        fill_and_flood_queue<T, Position, Compact, Line>(
            elevations, labels, active, offsets, strides, compactness,
            axis2_width, row_indices, seeds, heap);
    } else {
        FloodHeap<Position, Compact || Line, false> heap(capacity);
        fill_and_flood_queue<T, Position, Compact, Line>(
            elevations, labels, active, offsets, strides, compactness,
            axis2_width, row_indices, seeds, heap);
    }
}

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

    std::unique_ptr<double[]> elevations(new double[padded_count]);
    std::unique_ptr<T[]> labels(new T[padded_count]);
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

    bool finite = true;
    unsigned levels = compactness == 0 ? 1 : 0;
    std::vector<Index> seeds;
    visit_rows([&](std::size_t start, Index row) {
        for (std::size_t column = 0; column < width; ++column) {
            const auto i = start + column;
            const auto p = row + static_cast<Index>(column);
            const double value = image[i];
            elevations[p] = value;
            finite &= std::isfinite(value);
            if (levels) {
                if (value >= 0 && value <= 65535 && value == static_cast<unsigned>(value))
                    levels = std::max(levels, static_cast<unsigned>(value) + 1);
                else
                    levels = 0;
            }
            const bool included = !mask || mask[i];
            const T marker = markers[i];
            active[p] = included && ((watershed_line && !(compactness > 0)) || !marker);
            labels[p] = included ? (boolean_markers ? T(marker != 0) : marker) : T(0);
            if (included && marker)
                seeds.push_back(p);
        }
    });
    // All seeds have the same age. The stock binary heap's layout breaks
    // equal seed priorities, and NaNs are unordered under its comparator.
    // Later entries have a total order until the signed insertion age wraps.
    const auto max_pushes_per_pixel = (compactness > 0 || watershed_line)
        ? std::max(std::size_t(1), offsets.size()) : 1;
    bool ordered = finite && std::isfinite(compactness) &&
        count <= (seed_age - 2) / max_pushes_per_pixel;
    if (ordered && seeds.size() > 1) {
        const auto unique_levels = [&](auto& seen) {
            for (const auto p : seeds) {
                const auto key = static_cast<unsigned>(elevations[p]);
                if (seen.test(key))
                    return false;
                seen.set(key);
            }
            return true;
        };
        if (levels && levels <= 256) {
            std::bitset<256> seen;
            ordered = unique_levels(seen);
        } else if (levels) {
            std::bitset<65536> seen;
            ordered = unique_levels(seen);
        } else {
            std::unordered_set<std::uint64_t> seen;
            for (const auto p : seeds) {
                if (!seen.insert(order_key(elevations[p])).second) {
                    ordered = false;
                    break;
                }
            }
        }
    }

    const auto dispatch = [&](auto position) {
        using Position = decltype(position);
        if (compactness > 0)
            run_watershed_mode<T, Position, true, false>(
                elevations.get(), labels.get(), active, offsets, strides, compactness,
                neighborhood, rank, width, count, padded_count, coordinates, seeds,
                ordered, levels, visit_rows);
        else if (watershed_line)
            run_watershed_mode<T, Position, false, true>(
                elevations.get(), labels.get(), active, offsets, strides, compactness,
                neighborhood, rank, width, count, padded_count, coordinates, seeds,
                ordered, levels, visit_rows);
        else
            run_watershed_mode<T, Position, false, false>(
                elevations.get(), labels.get(), active, offsets, strides, compactness,
                neighborhood, rank, width, count, padded_count, coordinates, seeds,
                ordered, levels, visit_rows);
    };
    if (padded_count <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        dispatch(std::int32_t{});
    else
        dispatch(Index{});

    std::fill(coordinates.begin(), coordinates.end(), 0);
    visit_rows([&](std::size_t start, Index row) {
        std::copy_n(labels.get() + row, width, output + start);
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
