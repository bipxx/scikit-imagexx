#include "skimagexx/normalized_mutual_information.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {
constexpr std::size_t max_samples = 2147483647;
constexpr std::size_t max_cells = 1048576;
constexpr std::size_t scratch_bytes = 64 * 1024 * 1024;

std::size_t view_size(const MetricImageView& image) {
    (void)pixel_size(image.type);
    if (image.half && image.type != PixelType::uint16)
        throw std::invalid_argument("NMI binary16 storage must use uint16");
    if (image.shape.size() != image.strides.size())
        throw std::invalid_argument("NMI requires one byte stride per axis");
    const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t span = 0;
    for (std::size_t axis = 0; axis < image.shape.size(); ++axis) {
        const auto step = image.strides[axis];
        const auto magnitude = step < 0 ? std::size_t(-(step + 1)) + 1 : std::size_t(step);
        const auto length = image.shape[axis] ? image.shape[axis] - 1 : 0;
        if (length && magnitude > (limit - span) / length)
            throw std::overflow_error("NMI byte strides overflow");
        span += length * magnitude;
    }
    if (std::find(image.shape.begin(), image.shape.end(), 0) != image.shape.end()) return 0;
    std::size_t count = 1;
    for (auto length : image.shape) {
        if (length > max_samples / count) return max_samples + 1;
        count *= length;
    }
    if (count && !image.data) throw std::invalid_argument("NMI requires nonnull image data");
    return count;
}

std::vector<std::size_t> common_shape(const MetricImageView& a, const MetricImageView& b) {
    if (a.shape.size() != b.shape.size()) throw std::invalid_argument("NMI inputs must have equal rank");
    auto shape = a.shape;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) shape[axis] = std::max(shape[axis], b.shape[axis]);
    return shape;
}

std::size_t sample_count(const std::vector<std::size_t>& shape) {
    if (std::find(shape.begin(), shape.end(), 0) != shape.end()) return 0;
    std::size_t count = 1;
    for (auto length : shape) {
        if (length > max_samples / count) return max_samples + 1;
        count *= length;
    }
    return count;
}

bool range(const MetricImageView& image, std::size_t size, std::size_t padded,
           double& low, double& high) {
    low = high = 0;
    if (!size) return true;
    // Reuses the existing portable/runtime-dispatched strided range primitive.
    const auto raw = metric_raw_range(image);
    if (raw.has_nan) return false;
    if (image.type == PixelType::uint64 && raw.high > (std::uint64_t{1} << 53)) return false;
    if (image.type == PixelType::int64) {
        std::int64_t lo, hi;
        std::memcpy(&lo, &raw.low, sizeof(lo)); std::memcpy(&hi, &raw.high, sizeof(hi));
        if (lo < -(std::int64_t{1} << 53) || hi > (std::int64_t{1} << 53)) return false;
    }
    auto scalar = image; scalar.swap = false;
    scalar.data = &raw.low; low = detail::metric_read(scalar, 0);
    scalar.data = &raw.high; high = detail::metric_read(scalar, 0);
    if (!std::isfinite(low) || !std::isfinite(high) || std::abs(low) > 1e100 || std::abs(high) > 1e100) return false;
    // Keep the frozen unsafe-edge frontier consistent across NumPy profiles:
    // 1.x widens these bounds, whereas 2.x may overflow in their input dtype.
    const double scalar_max = image.half ? 65504. : image.type == PixelType::float32 ? double(std::numeric_limits<float>::max()) : std::numeric_limits<double>::max();
    if (high - low > scalar_max) return false;
    const double normal = image.half ? 0x1p-14 : image.type == PixelType::float32 ? double(std::numeric_limits<float>::min()) : std::numeric_limits<double>::min();
    if ((low != 0 && std::abs(low) < normal) || (high != 0 && std::abs(high) < normal)) return false;
    if (size < padded) { low = std::min(low, 0.); high = std::max(high, 0.); }
    return true;
}

struct Cursor {
    const MetricImageView& a;
    const MetricImageView& b;
    const std::vector<std::size_t>& shape;
    std::vector<std::size_t> coordinate;
    std::ptrdiff_t offset0 = 0, offset1 = 0;
    std::size_t outside0 = 0, outside1 = 0;
    Cursor(const MetricImageView& first, const MetricImageView& second,
           const std::vector<std::size_t>& dimensions)
        : a(first), b(second), shape(dimensions), coordinate(shape.size()) {}
    void reset(std::size_t index) {
        offset0 = offset1 = 0; outside0 = outside1 = 0;
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            coordinate[axis] = index % shape[axis]; index /= shape[axis];
            const auto position = coordinate[axis];
            outside0 += position >= a.shape[axis]; outside1 += position >= b.shape[axis];
            if (a.shape[axis]) offset0 += std::ptrdiff_t(std::min(position, a.shape[axis] - 1)) * a.strides[axis];
            if (b.shape[axis]) offset1 += std::ptrdiff_t(std::min(position, b.shape[axis] - 1)) * b.strides[axis];
        }
    }
    void advance() {
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            const auto old = coordinate[axis];
            const auto next = old + 1 == shape[axis] ? 0 : old + 1;
            coordinate[axis] = next;
            const auto update = [&](const MetricImageView& image, std::ptrdiff_t& offset, std::size_t& outside) {
                outside -= old >= image.shape[axis]; outside += next >= image.shape[axis];
                if (image.shape[axis]) {
                    const auto before = std::min(old, image.shape[axis] - 1), after = std::min(next, image.shape[axis] - 1);
                    offset += (std::ptrdiff_t(after) - std::ptrdiff_t(before)) * image.strides[axis];
                }
            };
            update(a, offset0, outside0); update(b, offset1, outside1);
            if (next) break;
        }
    }
};

std::size_t bin(double value, const std::vector<double>& edges) {
    if (value < edges.front() || value > edges.back()) return edges.size();
    if (value == edges.back()) return edges.size() - 2;
    return std::size_t(std::upper_bound(edges.begin(), edges.end(), value) - edges.begin()) - 1;
}

struct Sum {
    double total = 0, correction = 0;
    void add(double value) {
        const double next = total + value;
        correction += std::abs(total) >= std::abs(value) ? (total - next) + value : (value - next) + total;
        total = next;
    }
    double value() const { return total + correction; }
};

double entropy(const std::vector<double>& values) {
    Sum normalizer;
    for (auto value : values) normalizer.add(value);
    const auto total = normalizer.value();
    Sum result;
    for (auto value : values) if (value > 0) {
        const double probability = value / total;
        // Like entr(0), an underflowed zero probability contributes exactly0.
        if (probability > 0) result.add(-probability * std::log(probability));
    }
    return result.value();
}

void check_edges(const std::vector<double>& edges, const std::vector<double>& widths) {
    if (widths.empty() || widths.size() > 4096 || edges.size() != widths.size() + 1)
        throw std::invalid_argument("NMI edge and width dimensions differ");
    for (auto edge : edges) if (!std::isfinite(edge)) throw std::invalid_argument("NMI requires finite edges");
    for (std::size_t i = 0; i < widths.size(); ++i)
        if (!(edges[i] < edges[i + 1]) || !std::isfinite(widths[i]) || widths[i] < 1e-100 || widths[i] > 1e100)
            throw std::invalid_argument("NMI requires positive finite bounded widths");
}
} // namespace

NmiPreparation nmi_prepare(const MetricImageView& a, const MetricImageView& b) {
    const auto shape = common_shape(a, b);
    const auto na = view_size(a), nb = view_size(b), count = sample_count(shape);
    NmiPreparation result; result.samples = count;
    if (!count || count > max_samples || na > max_samples || nb > max_samples) return result;
    result.admitted = range(a, na, count, result.bounds[0], result.bounds[1]) &&
                      range(b, nb, count, result.bounds[2], result.bounds[3]);
    return result;
}

NmiStatistics nmi_statistics(const MetricImageView& a, const MetricImageView& b,
    const std::vector<double>& edges0, const std::vector<double>& edges1,
    const std::vector<double>& widths0, const std::vector<double>& widths1) {
    const auto shape = common_shape(a, b);
    const auto na = view_size(a), nb = view_size(b), count = sample_count(shape);
    check_edges(edges0, widths0); check_edges(edges1, widths1);
    const auto rows = widths0.size(), columns = widths1.size();
    if (rows > max_cells / columns) throw std::invalid_argument("NMI histogram exceeds cell bound");
    NmiStatistics result; result.bins = {rows, columns};
    if (!count || count > max_samples || na > max_samples || nb > max_samples) return result;
    const auto cells = rows * columns;
    const int workers = std::min(detail::active_workers(count), int(scratch_bytes / sizeof(std::uint64_t) / cells));
    auto histograms = std::vector<std::vector<std::uint64_t>>(std::size_t(workers), std::vector<std::uint64_t>(cells));
    std::vector<unsigned char> valid(std::size_t(workers), 1);
    std::vector<Cursor> cursors; cursors.reserve(std::size_t(workers));
    for (int worker = 0; worker < workers; ++worker) cursors.emplace_back(a, b, shape);
    detail::partition(count, workers, [&](std::size_t begin, std::size_t end, std::size_t worker) {
        auto& cursor = cursors[worker]; cursor.reset(begin);
        auto& histogram = histograms[worker];
        for (auto index = begin; index < end; ++index) {
            const double x = cursor.outside0 ? 0 : detail::metric_read(a, cursor.offset0);
            const double y = cursor.outside1 ? 0 : detail::metric_read(b, cursor.offset1);
            if (std::isfinite(x) && std::isfinite(y)) {
                const auto row = bin(x, edges0), column = bin(y, edges1);
                if (row < rows && column < columns) ++histogram[row * columns + column];
            } else valid[worker] = 0;
            cursor.advance();
        }
    });
    if (std::find(valid.begin(), valid.end(), 0) != valid.end()) return result;
    result.counts = std::move(histograms[0]);
    for (int worker = 1; worker < workers; ++worker)
        for (std::size_t i = 0; i < cells; ++i) result.counts[i] += histograms[std::size_t(worker)][i];
    std::uint64_t included = 0; std::size_t occupied = 0;
    for (auto value : result.counts) { included += value; occupied += value != 0; }
    if (!included) return result;
    if (occupied == 1) { result.admitted = true; return result; }
    std::vector<double> density(cells), row_values(rows), column_values(columns);
    std::vector<Sum> column_sums(columns);
    Sum total; double largest = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        Sum row_sum;
        for (std::size_t column = 0; column < columns; ++column) {
            const auto index = row * columns + column;
            double value = double(result.counts[index]) / widths0[row];
            value = value / widths1[column]; value = value / double(included);
            density[index] = value; total.add(value); largest = std::max(largest, value);
            row_sum.add(value); column_sums[column].add(value);
        }
        row_values[row] = row_sum.value();
    }
    if (largest / total.value() > 0.99) return result;
    for (std::size_t column = 0; column < columns; ++column) column_values[column] = column_sums[column].value();
    result.entropy = {entropy(column_values), entropy(row_values), entropy(density)};
    result.admitted = true;
    return result;
}
} // namespace skimagexx
