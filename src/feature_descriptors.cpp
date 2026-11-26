#include "skimagexx/feature_descriptors.hpp"

#include "execution.hpp"
#include "fft_internal.hpp"
#include "simd.hpp"
#include "skimagexx/draw_coordinates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
constexpr double pi = 3.14159265358979323846;

struct LinearSample {
    int low, high;
    double weight;
};

std::vector<LinearSample> samples(int count, const std::vector<double>& offsets) {
    std::vector<LinearSample> result;
    result.reserve(count * offsets.size());
    for (int i = 0; i < count; ++i) {
        for (const double offset : offsets) {
            const double position = i + offset;
            const auto low = static_cast<int>(std::floor(position));
            result.push_back({low, static_cast<int>(std::ceil(position)), position - low});
        }
    }
    return result;
}

template<LbpMethod Method>
void lbp_rows(const std::uint8_t* image, int rows, int cols, int points, int margin,
              const std::vector<LinearSample>& y, const std::vector<LinearSample>& x,
              double* output) {
    detail::partition(static_cast<std::size_t>(rows),
                      detail::active_workers(static_cast<std::size_t>(rows) * cols),
                      [&](auto begin, auto end, auto) {
        for (auto yi = begin; yi < end; ++yi) {
            const auto* vertical = y.data() + yi * points;
            // Resolve row pointers once per row and point, and use null for padding.
            std::array<const std::uint8_t*, 64> top{}, bottom{};
            for (int p = 0; p < points; ++p) {
                if (vertical[p].low >= 0 && vertical[p].low < rows)
                    top[p] = image + static_cast<std::size_t>(vertical[p].low) * cols;
                if (vertical[p].high >= 0 && vertical[p].high < rows)
                    bottom[p] = image + static_cast<std::size_t>(vertical[p].high) * cols;
            }
            const auto* current = image + yi * cols;
            auto* dest = output + yi * cols;
            auto row_pixels = [&](auto interior, int first, int last) {
                for (int xi = first; xi < last; ++xi) {
                    const auto* horizontal = x.data() + std::size_t(xi) * points;
                    double sum = 0, squares = 0;
                    unsigned code = 0;
                    int ones = 0, changes = 0, first_one = -1, first_zero = -1;
                    bool previous = false;
                    for (int p = 0; p < points; ++p) {
                        const auto& col = horizontal[p];
                        const auto get = [&](const std::uint8_t* row, int index) -> double {
                            if constexpr (!decltype(interior)::value) {
                                if (!row || index < 0 || index >= cols) return 0;
                            }
                            return row[index];
                        };
                        const double upper = (1 - col.weight) * get(top[p], col.low) +
                                             col.weight * get(top[p], col.high);
                        const double lower = (1 - col.weight) * get(bottom[p], col.low) +
                                             col.weight * get(bottom[p], col.high);
                        const double v = (1 - vertical[p].weight) * upper + vertical[p].weight * lower;
                        if constexpr (Method == LbpMethod::variance) {
                            sum += v;
                            squares += v * v;
                        } else {
                            const bool bit = v >= current[xi];
                            if constexpr (Method == LbpMethod::standard || Method == LbpMethod::rotation) {
                                code |= static_cast<unsigned>(bit) << p;
                            } else {
                                ones += bit;
                                changes += (p != 0 && bit != previous);
                                previous = bit;
                                if constexpr (Method == LbpMethod::oriented_uniform) {
                                    if (bit && first_one < 0) first_one = p;
                                    if (!bit && first_zero < 0) first_zero = p;
                                }
                            }
                        }
                    }
                    if constexpr (Method == LbpMethod::variance) {
                        const double variance = (squares - sum * sum / points) / points;
                        dest[xi] = variance != 0 ? variance : std::numeric_limits<double>::quiet_NaN();
                    } else if constexpr (Method == LbpMethod::uniform) {
                        dest[xi] = changes <= 2 ? ones : points + 1;
                    } else if constexpr (Method == LbpMethod::oriented_uniform) {
                        const int fixed = points * (points - 1);
                        if (changes > 2) dest[xi] = fixed + 2;
                        else if (!ones) dest[xi] = 0;
                        else if (ones == points) dest[xi] = fixed + 1;
                        else dest[xi] = 1 + (ones - 1) * points +
                                        (first_one == 0 ? ones - first_zero : points - first_one);
                    } else {
                        unsigned minimum = code;
                        if constexpr (Method == LbpMethod::rotation) {
                            for (int p = 1; p < points; ++p) {
                                code = (code >> 1) | ((code & 1U) << (points - 1));
                                minimum = std::min(minimum, code);
                            }
                        }
                        dest[xi] = minimum;
                    }
                }
            };
            if (yi >= static_cast<std::size_t>(margin) &&
                yi < static_cast<std::size_t>(rows - margin) && margin < cols - margin) {
                row_pixels(std::false_type{}, 0, margin);
                row_pixels(std::true_type{}, margin, cols - margin);
                row_pixels(std::false_type{}, cols - margin, cols);
            } else {
                row_pixels(std::false_type{}, 0, cols);
            }
        }
    });
}

int fast_extent(int extent) {
    for (;; ++extent) {
        int remaining = extent;
        for (const int prime : {2, 3, 5}) {
            while (remaining % prime == 0) remaining /= prime;
        }
        if (remaining == 1) return extent;
    }
}

template<class T>
void correlate(const T* image, const T* pattern, int rows, int cols, int ph, int pw,
               int oh, int ow, T* output) {
    // The wrapped tail of circular convolution only affects the leading
    // ph-1 rows or pw-1 columns, which the valid crop discards.
    const std::vector<std::size_t> dims{std::size_t(fast_extent(rows)),
                                        std::size_t(fast_extent(cols))};
    const std::vector<std::size_t> axes{0, 1};
    const auto stride = dims[1];
    std::vector<T> space(dims[0] * stride);
    const auto frequency_count = dims[0] * (stride / 2 + 1);
    std::vector<std::complex<T>> spectrum(frequency_count), kernel(frequency_count);
    const int workers = detail::active_workers(std::size_t(rows) * cols);
    for (int r = 0; r < rows; ++r)
        std::copy_n(image + std::size_t(r) * cols, cols, space.data() + r * stride);
    detail::fft_r2c(dims, axes, space.data(), spectrum.data(), workers);
    std::fill(space.begin(), space.end(), T{0});
    for (int r = 0; r < ph; ++r) {
        for (int c = 0; c < pw; ++c)
            space[r * stride + c] = pattern[std::size_t(ph - 1 - r) * pw + (pw - 1 - c)];
    }
    detail::fft_r2c(dims, axes, space.data(), kernel.data(), workers);
    for (std::size_t i = 0; i < frequency_count; ++i) spectrum[i] *= kernel[i];
    detail::fft_c2r(dims, axes, spectrum.data(), space.data(), workers);
    for (int r = 0; r < oh; ++r)
        std::copy_n(space.data() + (r + ph - 1) * stride + pw - 1, ow, output + std::size_t(r) * ow);
}

void gradient_axis(const float* image, int rows, int cols, bool vertical, float* result) {
    if (cols <= 0) return;
    for (int r = 0; r < rows; ++r) {
        const auto offset = std::size_t(r) * cols;
        const auto* row = image + offset;
        auto* dest = result + offset;
        if (vertical) {
            const auto* prev = r > 0 ? row - cols : row;
            const auto* next = r + 1 < rows ? row + cols : row;
            const float weight = r > 0 && r + 1 < rows ? 0.5f : 1.0f;
            for (int c = 0; c < cols; ++c) dest[c] = (next[c] - prev[c]) * weight;
        } else {
            dest[0] = (row[cols > 1 ? 1 : 0] - row[0]) * 1.0f;
            for (int c = 1; c + 1 < cols; ++c) dest[c] = (row[c + 1] - row[c - 1]) * 0.5f;
            if (cols > 1) dest[cols - 1] = (row[cols - 1] - row[cols - 2]) * 1.0f;
        }
    }
}
} // namespace

void local_binary_pattern_into(const std::uint8_t* image, int rows, int cols,
                               int points, double radius, LbpMethod method, double* output) {
    std::vector<double> dy(points), dx(points);
    for (int p = 0; p < points; ++p) {
        const double angle = 2 * pi * p / points;
        dy[p] = std::nearbyint(-radius * std::sin(angle) * 1e5) / 1e5;
        dx[p] = std::nearbyint(radius * std::cos(angle) * 1e5) / 1e5;
    }
    const auto y = samples(rows, dy);
    const auto x = samples(cols, dx);
    const auto margin = static_cast<int>(std::ceil(radius));
    switch (method) {
        case LbpMethod::standard: lbp_rows<LbpMethod::standard>(image, rows, cols, points, margin, y, x, output); break;
        case LbpMethod::rotation: lbp_rows<LbpMethod::rotation>(image, rows, cols, points, margin, y, x, output); break;
        case LbpMethod::uniform: lbp_rows<LbpMethod::uniform>(image, rows, cols, points, margin, y, x, output); break;
        case LbpMethod::oriented_uniform: lbp_rows<LbpMethod::oriented_uniform>(image, rows, cols, points, margin, y, x, output); break;
        case LbpMethod::variance: lbp_rows<LbpMethod::variance>(image, rows, cols, points, margin, y, x, output); break;
    }
}

int multiblock_lbp(const float* integral, int cols, int row, int col, int width, int height) {
    auto sum = [&](int r, int c) {
        const auto* bottom = integral + std::size_t(r + height - 1) * cols;
        float value = bottom[c + width - 1];
        const auto* top = r > 0 ? integral + std::size_t(r - 1) * cols : nullptr;
        if (top && c > 0) value += top[c - 1];
        if (top) value -= top[c + width - 1];
        if (c > 0) value -= bottom[c - 1];
        return value;
    };
    const float center = sum(row + height, col + width);
    constexpr int dr[] = {0, 0, 0, 1, 2, 2, 2, 1};
    constexpr int dc[] = {0, 1, 2, 2, 2, 1, 0, 0};
    int code = 0;
    for (int p = 0; p < 8; ++p)
        code = (code << 1) | (sum(row + dr[p] * height, col + dc[p] * width) >= center);
    return code;
}

template<class Input, class Output>
void hog_into(const Input* image, const HogOptions& o, Output* features, Output* visualization) {
    const int cell_rows = o.rows / o.cell_rows, cell_cols = o.cols / o.cell_cols;
    const int block_rows = cell_rows - o.block_rows + 1, block_cols = cell_cols - o.block_cols + 1;
    const std::size_t block_size = std::size_t(o.block_rows) * o.block_cols * o.orientations;
    const double interval = 180.0 / o.orientations;
    const float cell_area = static_cast<float>(o.cell_rows * o.cell_cols);
    const int workers = detail::active_workers(std::size_t(o.rows) * o.cols);
    std::vector<double> histogram(std::size_t(cell_rows) * cell_cols * o.orientations);
    detail::partition(std::size_t(cell_rows), workers, [&](auto begin, auto end, auto) {
        std::vector<float> totals(o.orientations);
        const int row_stride = o.cols * o.channels;
        for (auto r = begin; r < end; ++r) {
            for (int c = 0; c < cell_cols; ++c) {
                std::fill(totals.begin(), totals.end(), 0.0f);
                for (int ry = 0; ry < o.cell_rows; ++ry) {
                    const auto y = r * o.cell_rows + ry;
                    const auto* row = image + y * row_stride;
                    const bool vertical_interior = y > 0 && y + 1 < std::size_t(o.rows);
                    const auto* p = row + c * o.cell_cols * o.channels;
                    for (int cx = 0; cx < o.cell_cols; ++cx) {
                        const int x = c * o.cell_cols + cx;
                        const bool horizontal_interior = x > 0 && x + 1 < o.cols;
                        double best_x = 0, best_y = 0, best = -1;
                        for (int ch = 0; ch < o.channels; ++ch) {
                            const Output dy = vertical_interior
                                ? Output(p[ch + row_stride]) - Output(p[ch - row_stride]) : Output{0};
                            const Output dx = horizontal_interior
                                ? Output(p[ch + o.channels]) - Output(p[ch - o.channels]) : Output{0};
                            const auto magnitude = double(dx) * dx + double(dy) * dy;
                            const double selection = std::is_same_v<Output, float> && o.channels > 1
                                ? double(static_cast<float>(std::sqrt(magnitude))) : magnitude;
                            if (selection > best) {
                                best = selection;
                                best_x = dx;
                                best_y = dy;
                            }
                        }
                        double degrees = std::atan2(best_y, best_x) * (180.0 / pi);
                        if (degrees < 0) degrees += 180;
                        else if (degrees >= 180) degrees -= 180;
                        auto bin = std::min(o.orientations - 1, static_cast<int>(degrees / interval));
                        if (bin > 0 && degrees < bin * interval) --bin;
                        if (bin + 1 < o.orientations && degrees >= (bin + 1) * interval) ++bin;
                        if (degrees < (bin + 1) * interval)
                            totals[bin] = static_cast<float>(totals[bin] + std::hypot(best_x, best_y));
                        p += o.channels;
                    }
                }
                auto* dest = histogram.data() + (r * cell_cols + c) * o.orientations;
                for (int i = 0; i < o.orientations; ++i)
                    dest[i] = totals[i] / cell_area;
            }
        }
    });

    if (visualization) {
        std::fill_n(visualization, std::size_t(o.rows) * o.cols, Output{0});
        const auto radius = std::min(o.cell_rows, o.cell_cols) / 2 - 1;
        std::vector<double> dy(o.orientations), dx(o.orientations);
        for (int i = 0; i < o.orientations; ++i) {
            const double angle = pi * (i + 0.5) / o.orientations;
            dy[i] = radius * std::cos(angle);
            dx[i] = radius * std::sin(angle);
        }
        std::vector<std::intptr_t> yr(2 * radius + 1), xr(2 * radius + 1);
        for (int r = 0; r < cell_rows; ++r) {
            for (int c = 0; c < cell_cols; ++c) {
                const double cy = r * o.cell_rows + o.cell_rows / 2;
                const double cx = c * o.cell_cols + o.cell_cols / 2;
                for (int i = 0; i < o.orientations; ++i) {
                    const auto y0 = static_cast<std::intptr_t>(cy - dy[i]), y1 = static_cast<std::intptr_t>(cy + dy[i]);
                    const auto x0 = static_cast<std::intptr_t>(cx + dx[i]), x1 = static_cast<std::intptr_t>(cx - dx[i]);
                    const auto count = std::max(std::abs(y0 - y1), std::abs(x0 - x1)) + 1;
                    detail::draw_line_into(y0, x0, y1, x1, {yr.data(), xr.data(), 1, 1}, 1);
                    const auto value = histogram[(std::size_t(r) * cell_cols + c) * o.orientations + i];
                    for (std::intptr_t p = 0; p < count; ++p) {
                        auto& pixel = visualization[yr[p] * o.cols + xr[p]];
                        pixel = static_cast<Output>(pixel + value);
                    }
                }
            }
        }
    }

    const bool l1 = o.norm == HogNorm::l1 || o.norm == HogNorm::l1_sqrt;
    detail::partition(std::size_t(block_rows), workers, [&](auto begin, auto end, auto) {
        std::vector<double> values(block_size);
        for (auto r = begin; r < end; ++r) {
            for (int c = 0; c < block_cols; ++c) {
                auto* dest = features + (r * block_cols + c) * block_size;
                for (int br = 0; br < o.block_rows; ++br) {
                    const auto* source = histogram.data() + ((r + br) * cell_cols + c) * o.orientations;
                    std::copy_n(source, o.block_cols * o.orientations,
                                values.data() + std::size_t(br) * o.block_cols * o.orientations);
                }
                double total = 0;
                for (const auto v : values) total += l1 ? v : v * v;
                double divisor = l1 ? total + 1e-5 : std::sqrt(total + 1e-10);
                if (o.norm == HogNorm::l2_hys) {
                    total = 0;
                    for (auto& v : values) {
                        v = std::min(v / divisor, 0.2);
                        total += v * v;
                    }
                    divisor = std::sqrt(total + 1e-10);
                }
                for (std::size_t i = 0; i < block_size; ++i) {
                    const double v = values[i] / divisor;
                    dest[i] = static_cast<Output>(o.norm == HogNorm::l1_sqrt ? std::sqrt(v) : v);
                }
            }
        }
    });
}

namespace {

struct DaisySample {
    std::ptrdiff_t image_offset;
    int descriptor_offset;
    int row_offset;
};

struct DaisyScale {
    double sigma;
    std::vector<DaisySample> samples;
    std::vector<double> weights;
    std::vector<int> sampled_rows;
    std::vector<int> top, bottom, left, right;

    DaisyScale(double sigma, int rows, int cols) : sigma(sigma) {
        if (!std::isfinite(sigma) || sigma < 0)
            throw std::invalid_argument("sigma and truncate must be finite and nonnegative");
        const auto reach = std::floor(4 * sigma + 0.5);
        if (!std::isfinite(reach) || reach > 100000)
            throw std::invalid_argument("Gaussian radius exceeds the supported limit (100000)");
        const auto radius = static_cast<int>(reach);
        if (radius == 0) return;
        weights.reserve(2 * radius + 1);
        double sum = 0;
        for (int i = -radius; i <= radius; ++i) {
            const double distance = double(i) / sigma;
            const double weight = std::exp(-0.5 * distance * distance);
            weights.push_back(weight);
            sum += weight;
        }
        for (auto& weight : weights) weight /= sum;

        const auto reflect = [](std::ptrdiff_t index, std::ptrdiff_t extent) {
            const auto period = 2 * extent;
            index %= period;
            if (index < 0) index += period;
            return static_cast<int>(index < extent ? index : period - 1 - index);
        };
        for (int i = 0; i < radius; ++i) {
            top.push_back(reflect(i - radius, rows));
            bottom.push_back(reflect(std::ptrdiff_t(rows) + i, rows));
            left.push_back(reflect(i - radius, cols));
            right.push_back(reflect(std::ptrdiff_t(cols) + i, cols));
        }
    }
};

template<class T>
void daisy_convolve(const T* const* sources, T* output, int cols, const std::vector<double>& weights,
                    bool vectorized) {
#ifdef SKIMAGEXX_AVX2
    if (vectorized) {
        simd::convolve(sources, output, cols, weights.data(), weights.size(), 1.0);
        return;
    }
#else
    (void)vectorized;
#endif
    // Match the SIMD primitive's symmetric accumulation for full eight-pixel
    // blocks and its ordered accumulation for the tail.
    const auto middle = weights.size() / 2;
    const int full = cols - cols % 8;
    for (int c = 0; c < full; ++c) {
        double sum = 0;
        sum += double(sources[middle][c]) * weights[middle];
        for (std::size_t k = 0; k < middle; ++k) {
            const double pair = double(sources[k][c]) + double(sources[2 * middle - k][c]);
            sum += pair * weights[k];
        }
        output[c] = static_cast<T>(sum);
    }
    for (int c = full; c < cols; ++c) {
        double sum = 0;
        for (std::size_t k = 0; k < weights.size(); ++k) sum += sources[k][c] * weights[k];
        output[c] = static_cast<T>(sum);
    }
}

template<class T>
void daisy_smooth(const T* histogram, T* vertical, T* filtered, const DaisyOptions& o,
                  const DaisyScale& scale, const T** row_sources, T* padded_rows,
                  std::size_t padded_stride, int threads, bool vectorized) {
    const auto taps = scale.weights.size();
    const auto radius = static_cast<std::ptrdiff_t>(taps / 2);
    detail::partition(scale.sampled_rows.size(), threads, [&](auto begin, auto end, auto worker) {
        auto* sources = row_sources + worker * taps;
        for (auto i = begin; i < end; ++i) {
            const auto r = scale.sampled_rows[i];
            const auto first = static_cast<std::ptrdiff_t>(r) - radius;
            const auto last = first + static_cast<std::ptrdiff_t>(taps);
            for (auto y = first; y < last; ++y) {
                const auto reflected = y < 0 ? scale.top[y + radius] :
                    y >= o.rows ? scale.bottom[y - o.rows] : y;
                sources[y - first] = histogram + std::size_t(reflected) * o.cols;
            }
            daisy_convolve(sources, vertical + std::size_t(r) * o.cols, o.cols, scale.weights, vectorized);
        }
    });
    detail::partition(scale.sampled_rows.size(), threads, [&](auto begin, auto end, auto worker) {
        auto* sources = row_sources + worker * taps;
        auto* padded = padded_rows + worker * padded_stride;
        for (std::size_t k = 0; k < taps; ++k) sources[k] = padded + k;
        for (auto i = begin; i < end; ++i) {
            const auto offset = std::size_t(scale.sampled_rows[i]) * o.cols;
            const auto* row = vertical + offset;
            std::copy_n(row, o.cols, padded + radius);
            for (std::ptrdiff_t k = 0; k < radius; ++k) {
                padded[k] = row[scale.left[k]];
                padded[radius + o.cols + k] = row[scale.right[k]];
            }
            daisy_convolve(sources, filtered + offset, o.cols, scale.weights, vectorized);
        }
    });
}

template<class T> struct DaisyGradient { T magnitude, angle; };

template<class Input, class Output>
void daisy_gradients(const Input* row, const Input* below, int first, int last, int cols,
                     DaisyGradient<Output>* gradients) {
    constexpr auto scale = std::is_same_v<Input, std::uint8_t> ? Output(1.0 / 255.0) : Output{1};
    for (int c = first; c < last; ++c) {
        const auto center = Output(row[c]) * scale;
        const auto dx = c + 1 < cols ? Output(row[c + 1]) * scale - center : Output{0};
        const auto dy = below ? Output(below[c]) * scale - center : Output{0};
        gradients[c - first] = {std::sqrt(dx * dx + dy * dy), std::atan2(dy, dx)};
    }
}

} // namespace

template<class Input, class Output>
void daisy_into(const Input* image, const DaisyOptions& o, const double* sigmas,
                const double* radii, Output* output) {
    if (o.orientations <= 0) return;
    if (!image || o.rows <= 0 || o.cols <= 0)
        throw std::invalid_argument("image must be nonempty and nonnull");
    constexpr auto size_limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double);
    if (std::size_t(o.rows) > size_limit / o.cols)
        throw std::overflow_error("image dimensions overflow");
    const auto size = std::size_t(o.rows) * o.cols;
    const int oh = 1 + (o.rows - 2 * o.radius - 1) / o.step;
    const int ow = 1 + (o.cols - 2 * o.radius - 1) / o.step;
    const int width = (1 + o.rings * o.histograms) * o.orientations;
    std::vector<DaisyScale> scales;
    std::size_t max_taps = 0;
    for (int ring = 0; ring <= o.rings; ++ring) {
        const auto sigma = sigmas[ring];
        auto it = std::find_if(scales.begin(), scales.end(), [sigma](const auto& scale) {
            return scale.sigma == sigma;
        });
        auto& scale = it == scales.end() ? scales.emplace_back(sigma, o.rows, o.cols) : *it;
        max_taps = std::max(max_taps, scale.weights.size());
        if (ring == 0) {
            scale.samples.push_back({0, 0, 0});
            continue;
        }
        for (int j = 0; j < o.histograms; ++j) {
            const double theta = 2 * pi * j / o.histograms;
            const auto y = static_cast<int>(std::nearbyint(radii[ring - 1] * std::sin(theta)));
            const auto x = static_cast<int>(std::nearbyint(radii[ring - 1] * std::cos(theta)));
            const int group = 1 + (ring - 1) * o.histograms + j;
            scale.samples.push_back({std::ptrdiff_t(y) * o.cols + x, group * o.orientations, y});
        }
    }
    for (auto& scale : scales) {
        if (scale.weights.empty()) continue;
        // Both passes produce the same set of output rows because the second
        // pass is horizontal. Rows outside the ring grid are never sampled.
        std::vector<bool> sampled(o.rows);
        for (const auto& sample : scale.samples) {
            const auto first = o.radius + sample.row_offset;
            for (int r = 0; r < oh; ++r) sampled[first + r * o.step] = true;
        }
        for (int r = 0; r < o.rows; ++r) {
            if (sampled[r]) scale.sampled_rows.push_back(r);
        }
    }

    const auto kappa = Output(o.orientations / pi);
    // Keep adjacent planes contiguous while bounding scratch for large inputs.
    constexpr std::size_t plane_budget = 16 * 1024 * 1024;
    const auto batch = static_cast<int>(std::clamp(plane_budget / sizeof(Output) / size,
                                                  std::size_t{1}, std::size_t(o.orientations)));
    const int threads = std::min(o.rows, detail::active_workers(size));
    std::unique_ptr<Output[]> planes(new Output[size * batch]);
    std::unique_ptr<DaisyGradient<Output>[]> gradients;
    if (batch < o.orientations) {
        gradients.reset(new DaisyGradient<Output>[size]);
        detail::partition(o.rows, threads, [&](auto begin, auto end, auto) {
            for (auto r = begin; r < end; ++r) {
                const auto* row = image + r * o.cols;
                const auto* below = r + 1 < std::size_t(o.rows) ? row + o.cols : nullptr;
                daisy_gradients(row, below, 0, o.cols, o.cols, gradients.get() + r * o.cols);
            }
        });
    }
    const auto padded_stride = o.cols + (max_taps ? max_taps - 1 : 0);
    std::vector<const Output*> row_sources(max_taps * threads);
    std::unique_ptr<Output[]> vertical, filtered, padded_rows;
    if (max_taps) {
        vertical.reset(new Output[size]);
        filtered.reset(new Output[size]);
        padded_rows.reset(new Output[padded_stride * threads]);
    }
    const bool vectorized = max_taps && has_avx2();
    for (int first_bin = 0; first_bin < o.orientations; first_bin += batch) {
        const int count = std::min(batch, o.orientations - first_bin);
        std::vector<Output> orientations(count);
        for (int b = 0; b < count; ++b)
            orientations[b] = Output(2 * (first_bin + b) * pi / o.orientations - pi);
        detail::partition(o.rows, threads, [&](auto begin, auto end, auto) {
            std::array<DaisyGradient<Output>, 128> tile;
            for (auto r = begin; r < end; ++r) {
                const auto offset = r * o.cols;
                const auto* row = image + offset;
                const auto* below = r + 1 < std::size_t(o.rows) ? row + o.cols : nullptr;
                for (int c = 0; c < o.cols;) {
                    const int last = c + std::min(o.cols - c, static_cast<int>(tile.size()));
                    const auto* local = gradients ? gradients.get() + offset + c : tile.data();
                    if (!gradients) daisy_gradients(row, below, c, last, o.cols, tile.data());
                    for (int b = 0; b < count; ++b) {
                        auto* dest = planes.get() + b * size + offset + c;
                        const auto angle = orientations[b];
                        for (int i = 0; i < last - c; ++i)
                            dest[i] = std::exp(kappa * std::cos(local[i].angle - angle)) * local[i].magnitude;
                    }
                    c = last;
                }
            }
        });
        for (int b = 0; b < count; ++b) {
            const auto* histogram = planes.get() + b * size;
            for (const auto& scale : scales) {
                const auto* smoothed = histogram;
                if (!scale.weights.empty()) {
                    daisy_smooth(histogram, vertical.get(), filtered.get(), o, scale, row_sources.data(),
                                 padded_rows.get(), padded_stride, threads, vectorized);
                    smoothed = filtered.get();
                }
                for (int r = 0; r < oh; ++r) {
                    const auto* row = smoothed + std::size_t(r * o.step + o.radius) * o.cols + o.radius;
                    auto* dest = output + std::size_t(r) * ow * width;
                    for (int c = 0; c < ow; ++c) {
                        const auto* center = row + c * o.step;
                        for (const auto& sample : scale.samples)
                            dest[sample.descriptor_offset + first_bin + b] = center[sample.image_offset];
                        dest += width;
                    }
                }
            }
        }
    }
    if (o.norm == DaisyNorm::off) return;
    const auto chunk = o.norm == DaisyNorm::per_histogram ? o.orientations : width;
    for (std::size_t i = 0; i < std::size_t(oh) * ow * width; i += chunk) {
        auto* dest = output + i;
        Output total = 0;
        for (int k = 0; k < chunk; ++k) {
            dest[k] += Output(1e-10);
            total += o.norm == DaisyNorm::l1 ? dest[k] : dest[k] * dest[k];
        }
        if (o.norm != DaisyNorm::l1) total = std::sqrt(total);
        for (int k = 0; k < chunk; ++k) dest[k] /= total;
    }
}

template<class T>
bool match_template_into(const T* image, const T* pattern, const TemplateMatchOptions& o, T* output) {
    const int ph = o.template_rows, pw = o.template_cols;
    const auto volume = std::size_t(ph) * pw;
    double mean = 0, ssd = 0;
    for (std::size_t i = 0; i < volume; ++i) {
        if (!std::isfinite(pattern[i])) return false;
        mean += pattern[i];
    }
    mean /= static_cast<double>(volume);
    for (std::size_t i = 0; i < volume; ++i) {
        const double delta = pattern[i] - mean;
        ssd += delta * delta;
    }
    if (!std::isfinite(ssd)) return false;
    // Stock NCC subtracts uncentered moments in the input precision. Leave
    // near-constant templates with a large DC component to its FFT path.
    const double contrast = std::is_same_v<T, float> ? 4 : 16;
    if (ssd > 0 && std::abs(mean) > contrast * std::sqrt(ssd / static_cast<double>(volume)))
        return false;
    const int top = o.pad ? ph / 2 : 0, left = o.pad ? pw / 2 : 0;
    const int rows = o.rows + (o.pad ? ph - 1 : 0), cols = o.cols + (o.pad ? pw - 1 : 0);
    const int oh = rows - ph + 1, ow = cols - pw + 1;
    std::vector<T> padded;
    if (o.pad) {
        padded.assign(std::size_t(rows) * cols, static_cast<T>(o.constant));
        for (int r = 0; r < o.rows; ++r)
            std::copy_n(image + std::size_t(r) * o.cols, o.cols, padded.data() + std::size_t(r + top) * cols + left);
        image = padded.data();
    }

    struct Moments {
        T sum, squares;
        Moments operator+(const Moments& right) const {
            return {sum + right.sum, squares + right.squares};
        }
        Moments operator-(const Moments& right) const {
            return {sum - right.sum, squares - right.squares};
        }
    };
    const auto constant = static_cast<T>(o.constant);
    const Moments background{constant, constant * constant};
    // Match the two ordered prefixes used by the upstream window sums. The
    // omitted top/left padding still contributes to the prefix roundoff.
    Moments initial{0, 0};
    for (int i = 0; i < ph - top; ++i) initial = initial + background;
    const auto stride = std::size_t(cols);
    std::unique_ptr<Moments[]> integral(new Moments[(std::size_t(rows) + 1) * stride]);
    std::fill_n(integral.get(), stride, initial);
    std::vector<Moments> outside(std::size_t(rows) + 1);
    outside[0] = initial;
    T minimum = std::numeric_limits<T>::infinity(), maximum = -minimum;
    for (int r = 0; r < rows; ++r) {
        auto* dest = integral.get() + (std::size_t(r) + 1) * stride;
        const auto* above = dest - stride;
        const auto* source = image + std::size_t(r) * cols;
        outside[r + 1] = outside[r] + background;
        for (int c = 0; c < cols; ++c) {
            const T v = source[c];
            if (!std::isfinite(v)) return false;
            minimum = std::min(minimum, v);
            maximum = std::max(maximum, v);
            dest[c] = above[c] + Moments{v, v * v};
            if (!std::isfinite(dest[c].squares)) return false;
        }
    }
    if (ssd == 0) {
        std::fill_n(output, std::size_t(oh) * ow, T{0});
        return true;
    }
    if (minimum < maximum && std::max(double(minimum), -double(maximum)) >
        contrast * (double(maximum) - double(minimum)))
        return false;

    const bool use_fft = volume > 96;
    if (use_fft) correlate(image, pattern, rows, cols, ph, pw, oh, ow, output);
    const T template_mean = static_cast<T>(mean);
    const T template_ssd = static_cast<T>(ssd);
    const T template_volume = static_cast<T>(volume);
    std::vector<Moments> across(std::size_t(cols) + 1);
    for (int r = 0; r < oh; ++r) {
        const auto* a = integral.get() + std::size_t(r) * stride;
        const auto* b = a + ph * stride;
        across[0] = {0, 0};
        const auto outside_window = outside[r + ph] - outside[r];
        for (int i = 0; i < pw - left; ++i) across[0] = across[0] + outside_window;
        for (int c = 0; c < cols; ++c) across[c + 1] = across[c] + (b[c] - a[c]);
        for (int c = 0; c < ow; ++c) {
            auto& dest = output[std::size_t(r) * ow + c];
            const auto window = across[c + pw] - across[c];
            const T scaled_square = (window.sum * window.sum) / template_volume;
            const T variance = window.squares - scaled_square;
            const T denom = std::sqrt(std::max(T{0}, variance * template_ssd));
            if (denom <= std::numeric_limits<T>::epsilon()) {
                dest = 0;
                continue;
            }
            double dot = 0;
            if (use_fft) {
                dot = dest;
            } else {
                for (int pr = 0; pr < ph; ++pr) {
                    const auto* values = image + std::size_t(r + pr) * cols + c;
                    const auto* weights = pattern + std::size_t(pr) * pw;
                    for (int pc = 0; pc < pw; ++pc) dot += double(values[pc]) * weights[pc];
                }
            }
            const T numerator = static_cast<T>(dot) - window.sum * template_mean;
            dest = numerator / denom;
        }
    }
    return true;
}

template<class T>
void multiscale_basic_features_into(const T* image, const BasicFeatureOptions& o,
                                    const double* sigmas, int scales, float* output) {
    const auto count = std::size_t(o.rows) * o.cols;
    const int per_scale = int(o.intensity) + int(o.edges) + 2 * int(o.texture);
    const auto output_stride = std::size_t(o.channels) * scales * per_scale;
    const float scale = std::is_same_v<T, std::uint8_t> ? 1.0f / 255.0f : 1.0f;
    std::unique_ptr<float[]> channel(new float[count]), blurred(new float[count]);
    std::unique_ptr<float[]> edge(o.edges ? new float[count] : nullptr);
    std::unique_ptr<float[]> derivative(o.texture ? new float[count] : nullptr);
    std::unique_ptr<float[]> rr(o.texture ? new float[count] : nullptr);
    std::unique_ptr<float[]> rc(o.texture ? new float[count] : nullptr);
    std::unique_ptr<float[]> cc(o.texture ? new float[count] : nullptr);
    for (int ch = 0; ch < o.channels; ++ch) {
        const float* input = channel.get();
        if constexpr (std::is_same_v<T, float>) {
            if (o.channels == 1) input = image;
        }
        if (input == channel.get()) {
            for (std::size_t i = 0; i < count; ++i)
                channel[i] = static_cast<float>(image[i * o.channels + ch]) * scale;
        }
        for (int s = 0; s < scales; ++s) {
            gaussian_into(input, blurred.get(), o.rows, o.cols, sigmas[s]);
            if (o.edges) sobel_into(blurred.get(), edge.get(), o.rows, o.cols);
            if (o.texture) {
                gradient_axis(blurred.get(), o.rows, o.cols, true, derivative.get());
                gradient_axis(derivative.get(), o.rows, o.cols, true, rr.get());
                gradient_axis(derivative.get(), o.rows, o.cols, false, rc.get());
                gradient_axis(blurred.get(), o.rows, o.cols, false, derivative.get());
                gradient_axis(derivative.get(), o.rows, o.cols, false, cc.get());
            }
            const auto offset = (std::size_t(ch) * scales + s) * per_scale;
            for (std::size_t i = 0; i < count; ++i) {
                auto* dest = output + i * output_stride + offset;
                int feature = 0;
                if (o.intensity) dest[feature++] = blurred[i];
                if (o.edges) dest[feature++] = edge[i];
                if (o.texture) {
                    const float trace = (rr[i] + cc[i]) * 0.5f, diff = (rr[i] - cc[i]) * 0.5f;
                    const float term = std::sqrt(rc[i] * rc[i] + diff * diff);
                    dest[feature] = trace + term;
                    dest[feature + 1] = trace - term;
                }
            }
        }
    }
}

template void hog_into(const std::uint8_t*, const HogOptions&, double*, double*);
template void hog_into(const float*, const HogOptions&, float*, float*);
template void hog_into(const double*, const HogOptions&, double*, double*);
template void daisy_into(const std::uint8_t*, const DaisyOptions&, const double*, const double*, double*);
template void daisy_into(const float*, const DaisyOptions&, const double*, const double*, float*);
template void daisy_into(const double*, const DaisyOptions&, const double*, const double*, double*);
template bool match_template_into(const float*, const float*, const TemplateMatchOptions&, float*);
template bool match_template_into(const double*, const double*, const TemplateMatchOptions&, double*);
template void multiscale_basic_features_into(const std::uint8_t*, const BasicFeatureOptions&, const double*, int, float*);
template void multiscale_basic_features_into(const float*, const BasicFeatureOptions&, const double*, int, float*);
template void multiscale_basic_features_into(const double*, const BasicFeatureOptions&, const double*, int, float*);

} // namespace skimagexx
