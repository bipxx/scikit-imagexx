#include "skimagexx/peak_local_max.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

struct Box {
    int top, bottom, left, right;
    std::size_t area() const {
        return std::size_t(bottom - top) * std::size_t(right - left);
    }
};

struct Cell { int row, column; };

std::size_t offset(int row, int column, int width) {
    return std::size_t(row) * std::size_t(width) + std::size_t(column);
}

int distance_power(int row, int column, PeakNorm norm) {
    row = std::abs(row);
    column = std::abs(column);
    switch (norm) {
        case PeakNorm::manhattan: return row + column;
        case PeakNorm::euclidean: return row * row + column * column;
        default: return (std::max)(row, column);
    }
}

std::vector<Cell> suppress(const std::vector<Cell>& ordered, const Box& box,
                          int radius, PeakNorm norm, double p, bool inclusive, std::size_t count) {
    if (ordered.empty() || !count)
        return {};
    if (!inclusive && radius == 1) {
        const auto size = static_cast<std::ptrdiff_t>((std::min)(count, ordered.size()));
        return {ordered.begin(), ordered.begin() + size};
    }

    const int limit = distance_power(radius, 0, norm) + (inclusive ? 1 : 0);
    const double powered_radius = norm == PeakNorm::minkowski ? std::pow(radius, p) : 0;
    const auto rejects = [=](int row, int column) {
        if (norm != PeakNorm::minkowski)
            return distance_power(row, column, norm) < limit;
        const double distance = std::pow(std::abs(row), p) + std::pow(column, p);
        return distance <= powered_radius && (inclusive || std::pow(distance, 1 / p) < radius);
    };
    std::vector<int> reach(std::size_t(1 + 2 * radius));
    for (int row = -radius; row <= radius; ++row) {
        int column = radius;
        while (column >= 0 && !rejects(row, column))
            --column;
        reach[std::size_t(row + radius)] = column;
    }
    const int width = box.right - box.left;
    std::vector<std::uint8_t> blocked(box.area());
    std::vector<Cell> result;
    result.reserve((std::min)(count, ordered.size()));
    for (const auto cell : ordered) {
        if (blocked[offset(cell.row - box.top, cell.column - box.left, width)])
            continue;
        result.push_back(cell);
        if (result.size() == count)
            break;
        for (int row = (std::max)(box.top, cell.row - radius);
             row < (std::min)(box.bottom, cell.row + radius + 1); ++row) {
            const int extent = reach[std::size_t(row - cell.row + radius)];
            if (extent < 0)
                continue;
            const auto first = offset(row - box.top, (std::max)(0, cell.column - box.left - extent), width);
            const auto last = offset(row - box.top, (std::min)(width, cell.column - box.left + extent + 1), width);
            std::fill(blocked.data() + first, blocked.data() + last, std::uint8_t{1});
        }
    }
    return result;
}

template<class T>
auto sort_intensity(T value) {
    if constexpr (std::is_same_v<T, std::uint8_t>)
        return static_cast<std::uint8_t>(-int(value));
    else
        return -value;
}

template<class Get>
void sort_cells(std::vector<Cell>& cells, Get value_at) {
    std::sort(cells.begin(), cells.end(), [value_at](Cell a, Cell b) {
        const auto x = sort_intensity(value_at(a.row, a.column));
        const auto y = sort_intensity(value_at(b.row, b.column));
        if (x != y)
            return x < y;
        return a.row == b.row ? a.column < b.column : a.row < b.row;
    });
}

template<class Get, class T>
bool is_maximum(Get at, const Box& box, int radius, int row, int column, T value) {
    const int left = (std::max)(box.left, column - radius);
    const int right = (std::min)(box.right, column + radius + 1);
    for (int c = left; c < right; ++c)
        if (at(row, c) > value)
            return false;
    const int top = (std::max)(box.top, row - radius);
    const int bottom = (std::min)(box.bottom, row + radius + 1);
    for (int r = top; r < bottom; ++r) {
        if (r == row)
            continue;
        for (int c = left; c < right; ++c)
            if (at(r, c) > value)
                return false;
    }
    return true;
}

template<class T, class Label>
std::vector<Cell> detect_labeled(const T* image, const Label* labels, int width, const Box& inside,
                                double threshold, const LocalPeakOptions& options) {
    constexpr auto unset = (std::numeric_limits<int>::max)();
    std::vector<Box> boxes(1);
    for (int r = inside.top; r < inside.bottom; ++r) {
        for (int c = inside.left; c < inside.right; ++c) {
            const auto label = labels[offset(r, c, width)];
            if (label < 0 || label > 65535)
                throw std::overflow_error("unsupported peak label value");
            if (!label)
                continue;
            const auto index = static_cast<std::size_t>(label);
            if (index >= boxes.size())
                boxes.resize(index + 1, {unset, 0, unset, 0});
            auto& box = boxes[index];
            box = {(std::min)(box.top, r), (std::max)(box.bottom, r + 1),
                   (std::min)(box.left, c), (std::max)(box.right, c + 1)};
        }
    }
    std::size_t work_left = 8 * inside.area() + 65536;
    const auto norm = options.suppress_connected ? PeakNorm::chebyshev : options.norm;
    std::vector<Cell> merged;
    for (std::size_t id = 1; id < boxes.size(); ++id) {
        const auto& box = boxes[id];
        if (box.top == unset)
            continue;
        if (box.area() > work_left)
            throw std::overflow_error("overlapping peak label bounds exceed native work budget");
        work_left -= box.area();
        const auto member = [=](int r, int c) {
            return labels[offset(r, c, width)] == static_cast<Label>(id);
        };
        const auto at = [=](int r, int c) {
            return member(r, c) ? image[offset(r, c, width)] : (std::numeric_limits<T>::lowest)();
        };
        bool trivial = true;
        std::vector<Cell> cells;
        for (int r = box.top; r < box.bottom; ++r) {
            for (int c = box.left; c < box.right; ++c) {
                const auto value = at(r, c);
                const bool above = value > threshold;
                if (!above && (!trivial || !member(r, c)))
                    continue;
                if (box.area() == 1 || is_maximum(at, box, options.min_distance, r, c, value)) {
                    if (above)
                        cells.push_back({r, c});
                } else if (trivial && member(r, c)) {
                    trivial = false;
                }
            }
        }
        if (trivial && box.area() > 1) {
            // binary_opening with SciPy's default four-connected cross.
            const auto eroded = [&](int r, int c) {
                return r > box.top && r + 1 < box.bottom && c > box.left && c + 1 < box.right &&
                    member(r, c) && member(r - 1, c) && member(r + 1, c) &&
                    member(r, c - 1) && member(r, c + 1);
            };
            cells.erase(std::remove_if(cells.begin(), cells.end(), [&](Cell cell) {
                const auto [r, c] = cell;
                return !member(r, c) || eroded(r, c) || eroded(r - 1, c) ||
                    eroded(r + 1, c) || eroded(r, c - 1) || eroded(r, c + 1);
            }), cells.end());
        }
        sort_cells(cells, at);
        const auto kept = suppress(cells, box, options.min_distance, norm, options.minkowski_p, false,
                                   options.max_peaks_per_label);
        merged.insert(merged.end(), kept.begin(), kept.end());
    }
    return merged;
}

template<class T>
std::pair<T, T> intensity_range(const T* image, std::size_t size) {
    T low = image[0], high = image[0];
    for (std::size_t i = 0; i < size; ++i) {
        const T value = image[i];
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(value))
                throw std::overflow_error("nonfinite peak image");
        }
        low = (std::min)(value, low);
        high = (std::max)(value, high);
    }
    return {low, high};
}

template<class T>
double peak_threshold(T low, T high, const LocalPeakOptions& options) {
    const double absolute = options.threshold_abs.value_or(low);
    if (!options.threshold_rel)
        return std::is_same_v<T, float> ? double(static_cast<float>(absolute)) : absolute;
    const double relative = *options.threshold_rel;
    double product;
    if constexpr (std::is_same_v<T, float>) {
        if (options.wide_relative_product)
            product = relative * high;
        else
            product = static_cast<float>(relative) * high;
        if (std::isfinite(product))
            return static_cast<float>((std::max)(absolute, product));
    } else {
        product = relative * high;
        if (std::isfinite(product))
            return (std::max)(absolute, product);
    }
    throw std::overflow_error("nonfinite relative peak threshold");
}

void check_extent(int height, int width) {
    if (height <= 0 || width <= 0 || height > 65536 || width > 65536 ||
        std::size_t(height) * std::size_t(width) > 16777216)
        throw std::invalid_argument("unsupported peak image extent");
}

template<class T, class Value>
T pairwise_sum(std::size_t begin, std::size_t end, Value value) {
    const auto count = end - begin;
    if (count < 8) {
        T result = -T(0);
        for (auto i = begin; i < end; ++i)
            result += value(i);
        return result;
    }
    if (count > 128) {
        const auto middle = begin + (count / 16) * 8;
        return pairwise_sum<T>(begin, middle, value) + pairwise_sum<T>(middle, end, value);
    }
    T lanes[8];
    for (std::size_t lane = 0; lane < 8; ++lane)
        lanes[lane] = value(begin + lane);
    const auto tail = end - count % 8;
    for (auto i = begin + 8; i < tail; i += 8)
        for (std::size_t lane = 0; lane < 8; ++lane)
            lanes[lane] += value(i + lane);
    T result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
        ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (auto i = tail; i < end; ++i)
        result += value(i);
    return result;
}

// Two-by-two partial-pivot LU, with the single-precision normal equations
// promoted to double as in np.linalg.solve.
template<class T>
bool solve(T aa, T bb, T cc, T yy, T xx, std::array<T, 2>& estimate) {
    double a = aa, b = bb, c = bb, d = cc, y = yy, x = xx;
    if (std::abs(c) > std::abs(a)) {
        std::swap(a, c);
        std::swap(b, d);
        std::swap(y, x);
    }
    if (a == 0)
        return false;
    c /= a;
    d -= c * b;
    if (d == 0)
        return false;
    x = (x - c * y) / d;
    y = (y - b * x) / a;
    estimate = {static_cast<T>(y), static_cast<T>(x)};
    return true;
}

template<class T>
struct CornerSample {
    T xx, xy, yy;
    int row, column;
};

template<class T>
std::array<T, 2> refine(const std::vector<CornerSample<T>>& samples, double critical_dot,
                         double critical_edge) {
    using Sample = CornerSample<T>;
    auto total = [&](T Sample::* member) {
        return pairwise_sum<T>(0, samples.size(), [&](std::size_t i) { return samples[i].*member; });
    };
    auto moment = [&](T Sample::* member, int Sample::* coordinate) {
        return pairwise_sum<double>(0, samples.size(), [&](std::size_t i) {
            return double(samples[i].*member) * (samples[i].*coordinate);
        });
    };
    const auto xx = total(&Sample::xx), xy = total(&Sample::xy), yy = total(&Sample::yy);
    const auto xx_y = moment(&Sample::xx, &Sample::row), xx_x = moment(&Sample::xx, &Sample::column);
    const auto xy_y = moment(&Sample::xy, &Sample::row), xy_x = moment(&Sample::xy, &Sample::column);
    const auto yy_y = moment(&Sample::yy, &Sample::row), yy_x = moment(&Sample::yy, &Sample::column);
    const auto nan = (std::numeric_limits<T>::quiet_NaN)();
    std::array<T, 2> dot{}, edge{};
    if (!solve(xx, -xy, yy, T(xx_y - xy_x), T(yy_x - xy_y), dot) ||
        !solve(yy, xy, xx, T(yy_y + xy_x), T(xx_x + xy_y), edge))
        return {nan, nan};

    const double var_dot = pairwise_sum<double>(0, samples.size(), [&](std::size_t i) {
        const auto& [gx, gxy, gy, y, x] = samples[i];
        const double ry = y - double(dot[0]), rx = x - double(dot[1]);
        return gx * (ry * ry) - (T(2) * gxy) * (rx * ry) + gy * (rx * rx);
    });
    const double var_edge = pairwise_sum<double>(0, samples.size(), [&](std::size_t i) {
        const auto& [gx, gxy, gy, y, x] = samples[i];
        const double ry = y - double(edge[0]), rx = x - double(edge[1]);
        return gy * (ry * ry) + (T(2) * gxy) * (rx * ry) + gx * (rx * rx);
    });
    constexpr auto epsilon = (std::numeric_limits<double>::epsilon)();
    if (var_dot < epsilon && var_edge < epsilon)
        return {nan, nan};
    const double ratio = var_dot == 0 ? (std::numeric_limits<double>::infinity)() : var_edge / var_dot;
    return ratio > critical_dot ? dot : ratio < critical_edge ? edge : std::array<T, 2>{nan, nan};
}

}  // namespace

template<class T>
std::vector<PeakCoordinate> peak_local_max(const T* image, int height, int width,
                                         const LocalPeakOptions& options, PeakLabels labels) {
    check_extent(height, width);
    const int radius = options.min_distance;
    if (!image || radius < 1 || radius > 8 || options.border[0] < 0 || options.border[1] < 0 ||
        (options.norm == PeakNorm::minkowski && !(options.minkowski_p >= 1 && options.minkowski_p <= 32)) ||
        (options.threshold_abs && !std::isfinite(*options.threshold_abs)) ||
        (options.threshold_rel && !std::isfinite(*options.threshold_rel)))
        throw std::invalid_argument("invalid local peak parameters");
    const Box full{0, height, 0, width};
    const auto [low, high] = intensity_range(image, full.area());
    const double threshold = peak_threshold(low, high, options);
    const Box inside{options.border[0], height - options.border[0], options.border[1], width - options.border[1]};
    if (inside.top >= inside.bottom || inside.left >= inside.right || !options.max_peaks)
        return {};
    const auto at = [=](int r, int c) { return image[offset(r, c, width)]; };
    std::vector<Cell> cells;
    if (std::holds_alternative<std::monostate>(labels)) {
        if ((low == high && full.area() > 1) || high <= threshold)
            return {};
        for (int r = inside.top; r < inside.bottom; ++r) {
            for (int c = inside.left; c < inside.right; ++c) {
                const auto value = at(r, c);
                if (value > threshold && is_maximum(at, full, radius, r, c, value))
                    cells.push_back({r, c});
            }
        }
        sort_cells(cells, at);
        cells = suppress(cells, inside, radius, options.suppress_connected ? PeakNorm::chebyshev : options.norm,
                         options.minkowski_p, false, options.suppress_connected ? cells.size() : options.max_peaks);
    } else {
        cells = std::visit([&](auto view) -> std::vector<Cell> {
            if constexpr (std::is_same_v<decltype(view), std::monostate>) {
                return {};
            } else {
                if (!view)
                    throw std::invalid_argument("null peak labels");
                return detect_labeled(image, view, width, inside, threshold, options);
            }
        }, labels);
        if (!options.suppress_connected && cells.size() > options.max_peaks) {
            sort_cells(cells, at);
            cells.erase(std::unique(cells.begin(), cells.end(), [](Cell a, Cell b) {
                return a.row == b.row && a.column == b.column;
            }), cells.end());
            cells = suppress(cells, inside, radius, options.norm, options.minkowski_p, false, options.max_peaks);
        }
    }
    if (options.suppress_connected) {
        // The original tree rejects subsequent indices including duplicates.
        cells = suppress(cells, inside, radius, options.norm, options.minkowski_p, true, options.max_peaks);
    }
    std::vector<PeakCoordinate> result;
    result.reserve(cells.size());
    for (auto cell : cells)
        result.push_back({cell.row, cell.column});
    return result;
}

template<class Input, class T>
void corner_subpix_into(const Input* image, int height, int width, const PeakCoordinate* corners,
    std::size_t count, int window_size, double critical_dot, double critical_edge, T* output) {
    check_extent(height, width);
    if (!image || (count && (!corners || !output)) || window_size < 3 || window_size > 31 ||
        !std::isfinite(critical_dot) || !std::isfinite(critical_edge) || critical_dot < critical_edge)
        throw std::invalid_argument("invalid corner refinement parameters");
    if (!count)
        return;
    const auto [low, high] = intensity_range(image, std::size_t(height) * std::size_t(width));
    if (low < -1.0e10 || high > 1.0e10)
        throw std::overflow_error("corner image intensity outside native range");
    const int extent = (window_size - 1) / 2;
    const auto at = [=](int r, int c) {
        return r < 0 || r >= height || c < 0 || c >= width ? T(0) : T(image[offset(r, c, width)]);
    };
    std::vector<CornerSample<T>> samples;
    const auto side = std::size_t(2 * extent + 1);
    samples.reserve(side * side);
    for (std::size_t i = 0; i < count; ++i) {
        const auto [row, column] = corners[i];
        if (row < 1 || row >= height - 1 || column < 1 || column >= width - 1)
            throw std::overflow_error("non-interior corner requires original slicing semantics");
        samples.clear();
        for (int y = -extent; y <= extent; ++y) {
            for (int x = -extent; x <= extent; ++x) {
                const int r = int(row) + y, c = int(column) + x;
                const auto dx = [&](int q) { return T(double(at(q, c + 1)) - at(q, c - 1)); };
                const auto dy = [&](int q) { return T(double(at(r + 1, q)) - at(r - 1, q)); };
                const T gx = T(2.0 * dx(r) + (double(dx(r - 1)) + dx(r + 1)));
                const T gy = T(2.0 * dy(c) + (double(dy(c - 1)) + dy(c + 1)));
                samples.push_back({gx * gx, gx * gy, gy * gy, y, x});
            }
        }
        const auto result = refine(samples, critical_dot, critical_edge);
        output[2 * i] = T(double(row + extent) + double(result[0])) - T(extent);
        output[2 * i + 1] = T(double(column + extent) + double(result[1])) - T(extent);
    }
}

template std::vector<PeakCoordinate> peak_local_max(const float*, int, int, const LocalPeakOptions&, PeakLabels);
template std::vector<PeakCoordinate> peak_local_max(const double*, int, int, const LocalPeakOptions&, PeakLabels);
template std::vector<PeakCoordinate> peak_local_max(const std::uint8_t*, int, int, const LocalPeakOptions&, PeakLabels);

template void corner_subpix_into(const float*, int, int, const PeakCoordinate*, std::size_t, int, double, double, float*);
template void corner_subpix_into(const double*, int, int, const PeakCoordinate*, std::size_t, int, double, double, double*);
template void corner_subpix_into(const std::uint8_t*, int, int, const PeakCoordinate*, std::size_t, int, double, double, double*);

}  // namespace skimagexx
