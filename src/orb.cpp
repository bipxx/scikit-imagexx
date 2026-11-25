// The fixed rBRIEF sample positions come from scikit-image 0.26.0's
// orb_descriptor_positions.txt (BSD-3-Clause; third_party/scikit-image-LICENSE.txt).
#include "skimagexx/orb.hpp"
#include "skimagexx/gaussian_nd.hpp"
#include "skimagexx/structure_tensor.hpp"
#include "execution.hpp"
#include "resample_detail.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;

struct Offset {
    int row;
    int col;
};

constexpr Offset ring[] = {
    {0, 3}, {1, 3}, {2, 2}, {3, 1}, {3, 0}, {3, -1}, {2, -2}, {1, -3},
    {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3},
};
constexpr int umax[] = {15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3};

struct Pair {
    Offset first;
    Offset second;
};
constexpr Pair pattern[] = {
    {{8, -3}, {9, 5}},
    {{4, 2}, {7, -12}},
    {{-11, 9}, {-8, 2}},
    {{7, -12}, {12, -13}},
    {{2, -13}, {2, 12}},
    {{1, -7}, {1, 6}},
    {{-2, -10}, {-2, -4}},
    {{-13, -13}, {-11, -8}},
    {{-13, -3}, {-12, -9}},
    {{10, 4}, {11, 9}},
    {{-13, -8}, {-8, -9}},
    {{-11, 7}, {-9, 12}},
    {{7, 7}, {12, 6}},
    {{-4, -5}, {-3, 0}},
    {{-13, 2}, {-12, -3}},
    {{-9, 0}, {-7, 5}},
    {{12, -6}, {12, -1}},
    {{-3, 6}, {-2, 12}},
    {{-6, -13}, {-4, -8}},
    {{11, -13}, {12, -8}},
    {{4, 7}, {5, 1}},
    {{5, -3}, {10, -3}},
    {{3, -7}, {6, 12}},
    {{-8, -7}, {-6, -2}},
    {{-2, 11}, {-1, -10}},
    {{-13, 12}, {-8, 10}},
    {{-7, 3}, {-5, -3}},
    {{-4, 2}, {-3, 7}},
    {{-10, -12}, {-6, 11}},
    {{5, -12}, {6, -7}},
    {{5, -6}, {7, -1}},
    {{1, 0}, {4, -5}},
    {{9, 11}, {11, -13}},
    {{4, 7}, {4, 12}},
    {{2, -1}, {4, 4}},
    {{-4, -12}, {-2, 7}},
    {{-8, -5}, {-7, -10}},
    {{4, 11}, {9, 12}},
    {{0, -8}, {1, -13}},
    {{-13, -2}, {-8, 2}},
    {{-3, -2}, {-2, 3}},
    {{-6, 9}, {-4, -9}},
    {{8, 12}, {10, 7}},
    {{0, 9}, {1, 3}},
    {{7, -5}, {11, -10}},
    {{-13, -6}, {-11, 0}},
    {{10, 7}, {12, 1}},
    {{-6, -3}, {-6, 12}},
    {{10, -9}, {12, -4}},
    {{-13, 8}, {-8, -12}},
    {{-13, 0}, {-8, -4}},
    {{3, 3}, {7, 8}},
    {{5, 7}, {10, -7}},
    {{-1, 7}, {1, -12}},
    {{3, -10}, {5, 6}},
    {{2, -4}, {3, -10}},
    {{-13, 0}, {-13, 5}},
    {{-13, -7}, {-12, 12}},
    {{-13, 3}, {-11, 8}},
    {{-7, 12}, {-4, 7}},
    {{6, -10}, {12, 8}},
    {{-9, -1}, {-7, -6}},
    {{-2, -5}, {0, 12}},
    {{-12, 5}, {-7, 5}},
    {{3, -10}, {8, -13}},
    {{-7, -7}, {-4, 5}},
    {{-3, -2}, {-1, -7}},
    {{2, 9}, {5, -11}},
    {{-11, -13}, {-5, -13}},
    {{-1, 6}, {0, -1}},
    {{5, -3}, {5, 2}},
    {{-4, -13}, {-4, 12}},
    {{-9, -6}, {-9, 6}},
    {{-12, -10}, {-8, -4}},
    {{10, 2}, {12, -3}},
    {{7, 12}, {12, 12}},
    {{-7, -13}, {-6, 5}},
    {{-4, 9}, {-3, 4}},
    {{7, -1}, {12, 2}},
    {{-7, 6}, {-5, 1}},
    {{-13, 11}, {-12, 5}},
    {{-3, 7}, {-2, -6}},
    {{7, -8}, {12, -7}},
    {{-13, -7}, {-11, -12}},
    {{1, -3}, {12, 12}},
    {{2, -6}, {3, 0}},
    {{-4, 3}, {-2, -13}},
    {{-1, -13}, {1, 9}},
    {{7, 1}, {8, -6}},
    {{1, -1}, {3, 12}},
    {{9, 1}, {12, 6}},
    {{-1, -9}, {-1, 3}},
    {{-13, -13}, {-10, 5}},
    {{7, 7}, {10, 12}},
    {{12, -5}, {12, 9}},
    {{6, 3}, {7, 11}},
    {{5, -13}, {6, 10}},
    {{2, -12}, {2, 3}},
    {{3, 8}, {4, -6}},
    {{2, 6}, {12, -13}},
    {{9, -12}, {10, 3}},
    {{-8, 4}, {-7, 9}},
    {{-11, 12}, {-4, -6}},
    {{1, 12}, {2, -8}},
    {{6, -9}, {7, -4}},
    {{2, 3}, {3, -2}},
    {{6, 3}, {11, 0}},
    {{3, -3}, {8, -8}},
    {{7, 8}, {9, 3}},
    {{-11, -5}, {-6, -4}},
    {{-10, 11}, {-5, 10}},
    {{-5, -8}, {-3, 12}},
    {{-10, 5}, {-9, 0}},
    {{8, -1}, {12, -6}},
    {{4, -6}, {6, -11}},
    {{-10, 12}, {-8, 7}},
    {{4, -2}, {6, 7}},
    {{-2, 0}, {-2, 12}},
    {{-5, -8}, {-5, 2}},
    {{7, -6}, {10, 12}},
    {{-9, -13}, {-8, -8}},
    {{-5, -13}, {-5, -2}},
    {{8, -8}, {9, -13}},
    {{-9, -11}, {-9, 0}},
    {{1, -8}, {1, -2}},
    {{7, -4}, {9, 1}},
    {{-2, 1}, {-1, -4}},
    {{11, -6}, {12, -11}},
    {{-12, -9}, {-6, 4}},
    {{3, 7}, {7, 12}},
    {{5, 5}, {10, 8}},
    {{0, -4}, {2, 8}},
    {{-9, 12}, {-5, -13}},
    {{0, 7}, {2, 12}},
    {{-1, 2}, {1, 7}},
    {{5, 11}, {7, -9}},
    {{3, 5}, {6, -8}},
    {{-13, -4}, {-8, 9}},
    {{-5, 9}, {-3, -3}},
    {{-4, -7}, {-3, -12}},
    {{6, 5}, {8, 0}},
    {{-7, 6}, {-6, 12}},
    {{-13, 6}, {-5, -2}},
    {{1, -10}, {3, 10}},
    {{4, 1}, {8, -4}},
    {{-2, -2}, {2, -13}},
    {{2, -12}, {12, 12}},
    {{-2, -13}, {0, -6}},
    {{4, 1}, {9, 3}},
    {{-6, -10}, {-3, -5}},
    {{-3, -13}, {-1, 1}},
    {{7, 5}, {12, -11}},
    {{4, -2}, {5, -7}},
    {{-13, 9}, {-9, -5}},
    {{7, 1}, {8, 6}},
    {{7, -8}, {7, 6}},
    {{-7, -4}, {-7, 1}},
    {{-8, 11}, {-7, -8}},
    {{-13, 6}, {-12, -8}},
    {{2, 4}, {3, 9}},
    {{10, -5}, {12, 3}},
    {{-6, -5}, {-6, 7}},
    {{8, -3}, {9, -8}},
    {{2, -12}, {2, 8}},
    {{-11, -2}, {-10, 3}},
    {{-12, -13}, {-7, -9}},
    {{-11, 0}, {-10, -5}},
    {{5, -3}, {11, 8}},
    {{-2, -13}, {-1, 12}},
    {{-1, -8}, {0, 9}},
    {{-13, -11}, {-12, -5}},
    {{-10, -2}, {-10, 11}},
    {{-3, 9}, {-2, -13}},
    {{2, -3}, {3, 2}},
    {{-9, -13}, {-4, 0}},
    {{-4, 6}, {-3, -10}},
    {{-4, 12}, {-2, -7}},
    {{-6, -11}, {-4, 9}},
    {{6, -3}, {6, 11}},
    {{-13, 11}, {-5, 5}},
    {{11, 11}, {12, 6}},
    {{7, -5}, {12, -2}},
    {{-1, 12}, {0, 7}},
    {{-4, -8}, {-3, -2}},
    {{-7, 1}, {-6, 7}},
    {{-13, -12}, {-8, -13}},
    {{-7, -2}, {-6, -8}},
    {{-8, 5}, {-6, -9}},
    {{-5, -1}, {-4, 5}},
    {{-13, 7}, {-8, 10}},
    {{1, 5}, {5, -13}},
    {{1, 0}, {10, -13}},
    {{9, 12}, {10, -1}},
    {{5, -8}, {10, -9}},
    {{-1, 11}, {1, -13}},
    {{-9, -3}, {-6, 2}},
    {{-1, -10}, {1, 12}},
    {{-13, 1}, {-8, -10}},
    {{8, -11}, {10, -6}},
    {{2, -13}, {3, -6}},
    {{7, -13}, {12, -9}},
    {{-10, -10}, {-5, -7}},
    {{-10, -8}, {-8, -13}},
    {{4, -6}, {8, 5}},
    {{3, 12}, {8, -13}},
    {{-4, 2}, {-3, -3}},
    {{5, -13}, {10, -12}},
    {{4, -13}, {5, -1}},
    {{-9, 9}, {-4, 3}},
    {{0, 3}, {3, -9}},
    {{-12, 1}, {-6, 1}},
    {{3, 2}, {4, -8}},
    {{-10, -10}, {-10, 9}},
    {{8, -13}, {12, 12}},
    {{-8, -12}, {-6, -5}},
    {{2, 2}, {3, 7}},
    {{10, 6}, {11, -8}},
    {{6, 8}, {8, -12}},
    {{-7, 10}, {-6, 5}},
    {{-3, -9}, {-3, 9}},
    {{-1, -13}, {-1, 5}},
    {{-3, -7}, {-3, 4}},
    {{-8, -2}, {-8, 3}},
    {{4, 2}, {12, 12}},
    {{2, -5}, {3, 11}},
    {{6, -9}, {11, -13}},
    {{3, -1}, {7, 12}},
    {{11, -1}, {12, 4}},
    {{-3, 0}, {-3, 6}},
    {{4, -11}, {4, 12}},
    {{2, -4}, {2, 1}},
    {{-10, -6}, {-8, 1}},
    {{-13, 7}, {-11, 1}},
    {{-13, 12}, {-11, -13}},
    {{6, 0}, {11, -13}},
    {{0, -1}, {1, 4}},
    {{-13, 3}, {-9, -2}},
    {{-9, 8}, {-6, -3}},
    {{-13, -6}, {-8, -2}},
    {{5, -9}, {8, 10}},
    {{2, 7}, {3, -9}},
    {{-1, -6}, {-1, -1}},
    {{9, 5}, {11, -2}},
    {{11, -3}, {12, -8}},
    {{3, 0}, {3, 5}},
    {{-1, 4}, {0, 10}},
    {{3, -6}, {4, 5}},
    {{-13, 0}, {-10, 5}},
    {{5, 8}, {12, 11}},
    {{8, 9}, {9, -6}},
    {{7, -4}, {8, -12}},
    {{-10, 4}, {-10, 9}},
    {{7, 3}, {12, 4}},
    {{9, -7}, {10, -2}},
    {{7, 0}, {12, -2}},
    {{-1, -6}, {0, -11}},
};
static_assert(std::size(pattern) == orb_descriptor_size);

std::size_t checked_count(const void* input, std::size_t rows, std::size_t cols) {
    if (!input || !rows || !cols)
        throw std::invalid_argument("ORB requires a nonempty image");
    const auto limit = std::size_t(std::numeric_limits<Index>::max()) / sizeof(double);
    if (rows > limit / cols) throw std::overflow_error("ORB dimensions overflow");
    return rows * cols;
}

template<class T>
constexpr PixelType pixel_type = std::is_same_v<T, float> ? PixelType::float32 : PixelType::float64;

bool in_bounds(double row, double col, Index rows, Index cols, int border) {
    return row > border - 1 && row < double(rows - border + 1)
        && col > border - 1 && col < double(cols - border + 1);
}

bool has_arc(unsigned bits, int length) {
    if (!bits) return false;
    bits |= bits << 16;
    // A bit survives an intersection with its displaced copy if the
    // corresponding arc is present in both of the overlapping blocks.
    auto runs = bits;
    for (int block = 1; block < length;) {
        const int step = std::min(block, length - block);
        runs &= runs >> step;
        block += step;
    }
    return (runs & 0xffffu) != 0;
}

template<class T>
T fast_score(const T* center, const std::array<Index, 16>& offsets,
             int length, T threshold) {
    const T upper = *center + threshold;
    const T lower = *center - threshold;
    T samples[16];
    unsigned bright = 0, dark = 0;
    for (unsigned i = 0; i < 16; ++i) {
        const T value = center[offsets[i]];
        samples[i] = value;
        bright |= unsigned(value > upper) << i;
        dark |= unsigned(value <= upper && value < lower) << i;
    }
    if (length >= 12) {
        const auto cardinal_count = [](unsigned bits) {
            return (bits & 1u) + ((bits >> 4) & 1u)
                 + ((bits >> 8) & 1u) + ((bits >> 12) & 1u);
        };
        if (cardinal_count(bright) < 3 && cardinal_count(dark) < 3) return 0;
    }
    if (!has_arc(bright, length) && !has_arc(dark, length)) return 0;

    T score = 0;
    for (const T value : samples)
        score = T(double(score) + std::abs(double(value - *center)));
    return score;
}

template<class T>
T orientation_at(const T* center, Index cols) {
    T horizontal = 0, vertical = 0;
    for (int row = -15; row <= 15; ++row) {
        T row_sum = 0;
        for (int col = -umax[std::abs(row)]; col <= umax[std::abs(row)]; ++col) {
            const T value = center[row * cols + col];
            horizontal += value * T(col);
            row_sum += value;
        }
        vertical += row_sum * T(row);
    }
    return T(std::atan2(double(vertical), double(horizontal)));
}

template<class T>
OrbOctave detect(const T* image, Index rows, Index cols, int fast_n,
                 double threshold, double k, const std::array<double, 9>& weights) {
    if (rows < 7 || cols < 7) return {};
    const auto count = std::size_t(rows * cols);
    std::vector<T> score(count);
    std::array<Index, 16> offsets;
    for (std::size_t i = 0; i < offsets.size(); ++i)
        offsets[i] = ring[i].row * cols + ring[i].col;

    detail::partition(std::size_t(rows - 6), detail::active_workers(count),
        [&](auto begin, auto end, auto) {
            for (Index row = Index(begin) + 3; row < Index(end) + 3; ++row)
                for (Index col = 3; col < cols - 3; ++col) {
                    const Index at = row * cols + col;
                    score[at] = fast_score(image + at, offsets, fast_n, T(threshold));
                }
        });

    OrbOctave result{};
    std::vector<std::uint8_t> selected(count);
    for (Index row = 3; row < rows - 3; ++row)
        for (Index col = 3; col < cols - 3; ++col) {
            const Index at = row * cols + col;
            const T value = score[at];
            if (value <= 0 || selected[at - cols - 1] || selected[at - cols]
                    || selected[at - cols + 1] || selected[at - 1])
                continue;
            if (value < score[at - cols - 1] || value < score[at - cols]
                    || value < score[at - cols + 1] || value < score[at - 1]
                    || value < score[at + 1] || value < score[at + cols - 1]
                    || value < score[at + cols] || value < score[at + cols + 1])
                continue;
            // Adjacent local maxima necessarily tie. Visit ties in the
            // reference's row-major order before filtering the 16px border.
            result.had_fast_peaks = true;
            selected[at] = 1;
            if (in_bounds(double(row), double(col), rows, cols, 16))
                result.points.push_back({row, col, 0, 0});
        }

    if (result.points.empty()) return result;
    std::stable_sort(result.points.begin(), result.points.end(), [&](const auto& a, const auto& b) {
        return score[a.row * cols + a.col] > score[b.row * cols + b.col];
    });

    const std::vector<double> coefficients(weights.begin(), weights.end());
    const auto border = GaussianBorder::constant;
    const std::vector<GaussianPass> passes{{0, coefficients, border}, {1, coefficients, border}};
    std::array<std::vector<T>, 3> fields;
    for (auto& field : fields) field.resize(count);
    void* outputs[] = {fields[0].data(), fields[1].data(), fields[2].data()};
    if (!structure_tensor_into(image, pixel_type<T>, outputs,
                               {std::size_t(rows), std::size_t(cols)}, passes, {border, border}))
        throw std::invalid_argument("ORB input exceeded its numerical range");

    detail::partition(result.points.size(), detail::active_workers(count),
        [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) {
                auto& point = result.points[i];
                const Index at = point.row * cols + point.col;
                point.orientation = double(orientation_at(image + at, cols));
                const T rr = fields[0][at], rc = fields[1][at], cc = fields[2][at];
                const T determinant = rr * cc - rc * rc;
                const T trace = rr + cc;
                point.response = double(determinant - T(k) * (trace * trace));
            }
        });
    return result;
}

template<class T>
void reduce(const T* input, T* output, Index rows, Index cols, std::size_t target_rows,
            std::size_t target_cols, const std::vector<double>& weights, bool legacy_zoom) {
    const std::vector<std::size_t> shape{std::size_t(rows), std::size_t(cols)};
    const auto border = GaussianBorder::reflect;
    std::vector<T> smooth(std::size_t(rows * cols));
    gaussian_nd_into(input, smooth.data(), pixel_type<T>, pixel_type<T>, shape,
                     {{0, weights, border}, {1, weights, border}}, true);

    detail::ZoomOptions options;
    options.legacy_reflect = legacy_zoom;
    options.identity_copy = target_rows == shape[0] && target_cols == shape[1];
    detail::zoom_typed_into(smooth.data(), output, pixel_type<T>, shape,
                            {target_rows, target_cols}, options);
    const auto bounds = std::minmax_element(smooth.begin(), smooth.end());
    const T minimum = *bounds.first, maximum = *bounds.second;
    for (std::size_t i = 0; i < target_rows * target_cols; ++i)
        output[i] = std::clamp(output[i], minimum, maximum);
}

template<class T>
Index nearest(T value) {
    return Index(value > T(0) ? value + T(.5) : value - T(.5));
}

template<class T>
OrbDescriptors extract(const T* image, Index rows, Index cols, const double* points,
                       const T* angles, std::size_t count) {
    OrbDescriptors result;
    result.mask.resize(count);
    std::vector<std::size_t> accepted;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(angles[index])) throw std::invalid_argument("ORB orientation must be finite");
        if (in_bounds(points[2 * index], points[2 * index + 1], rows, cols, 20)) {
            accepted.push_back(index);
            result.mask[index] = 1;
        }
    }
    result.data.resize(accepted.size() * orb_descriptor_size);
    detail::partition(accepted.size(), detail::active_workers(result.data.size()),
        [&](auto begin, auto end, auto) {
            for (auto i = begin; i < end; ++i) {
                const auto index = accepted[i];
                const auto* center = image + Index(points[2 * index]) * cols + Index(points[2 * index + 1]);
                const T sine = T(std::sin(double(angles[index])));
                const T cosine = T(std::cos(double(angles[index])));
                const auto sample = [&](Offset p) {
                    const Index row = nearest(sine * T(p.row) + cosine * T(p.col));
                    const Index col = nearest(cosine * T(p.row) - sine * T(p.col));
                    return center[row * cols + col];
                };
                for (std::size_t bit = 0; bit < orb_descriptor_size; ++bit)
                    result.data[i * orb_descriptor_size + bit] =
                        std::uint8_t(sample(pattern[bit].first) < sample(pattern[bit].second));
            }
        });
    return result;
}

} // namespace

bool orb_prepare(const void* input, void* output, PixelType source,
                 std::size_t rows, std::size_t cols) {
    const auto count = checked_count(input, rows, cols);
    if (!output || output == input) throw std::invalid_argument("ORB requires a separate output image");
    const auto valid = [](auto value) { return std::isfinite(value) && std::abs(value) <= 4096; };
    if (source == PixelType::float32) {
        const auto* values = static_cast<const float*>(input);
        if (!std::all_of(values, values + count, valid)) return false;
    }
    if (source == PixelType::float64) {
        const auto* values = static_cast<const double*>(input);
        if (!std::all_of(values, values + count, valid)) return false;
    }
    const auto target = source == PixelType::float32 ? source : PixelType::float64;
    gaussian_nd_into(input, output, source, target, {rows, cols}, {}, false);
    return true;
}

void orb_reduce(const void* input, void* output, bool single, std::size_t rows, std::size_t cols,
                std::size_t output_rows, std::size_t output_cols,
                const std::vector<double>& weights, bool legacy_zoom) {
    checked_count(input, rows, cols);
    if (!output || output == input || !output_rows || !output_cols
            || output_rows > rows || output_cols > cols || weights.empty() || weights.size() % 2 == 0)
        throw std::invalid_argument("invalid ORB downsample plan");
    if (single) reduce(static_cast<const float*>(input), static_cast<float*>(output),
               Index(rows), Index(cols), output_rows, output_cols, weights, legacy_zoom);
    else reduce(static_cast<const double*>(input), static_cast<double*>(output),
                Index(rows), Index(cols), output_rows, output_cols, weights, legacy_zoom);
}

OrbOctave orb_detect_octave(const void* input, bool single, std::size_t rows, std::size_t cols,
                            int fast_n, double threshold, double harris_k,
                            const std::array<double, 9>& harris_weights) {
    checked_count(input, rows, cols);
    if (fast_n < -128 || fast_n > 127 || !std::isfinite(threshold) || std::abs(threshold) > 4096
            || !std::isfinite(harris_k) || std::abs(harris_k) > 4096)
        throw std::invalid_argument("invalid ORB detector parameters");
    if (fast_n <= 0 || fast_n > 16) return {};
    if (single) return detect(static_cast<const float*>(input), Index(rows), Index(cols),
                              fast_n, threshold, harris_k, harris_weights);
    return detect(static_cast<const double*>(input), Index(rows), Index(cols),
                  fast_n, threshold, harris_k, harris_weights);
}

OrbDescriptors orb_extract_octave(const void* input, bool single, std::size_t rows, std::size_t cols,
                                  const double* points, const void* orientations, std::size_t count) {
    checked_count(input, rows, cols);
    const auto limit = std::size_t(std::numeric_limits<Index>::max()) / orb_descriptor_size;
    if (count > limit) throw std::overflow_error("ORB descriptor dimensions overflow");
    if (count && (!points || !orientations))
        throw std::invalid_argument("ORB requires keypoints and orientations");
    if (single) return extract(static_cast<const float*>(input), Index(rows), Index(cols),
                               points, static_cast<const float*>(orientations), count);
    return extract(static_cast<const double*>(input), Index(rows), Index(cols),
                   points, static_cast<const double*>(orientations), count);
}

} // namespace skimagexx
