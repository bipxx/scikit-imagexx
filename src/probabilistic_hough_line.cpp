#include "skimagexx/probabilistic_hough_line.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define SKIMAGEXX_PHL_SSE2 1
#else
#define SKIMAGEXX_PHL_SSE2 0
#endif

namespace skimagexx {
namespace {

constexpr std::size_t foreground_limit = 4096;
constexpr int dimension_limit = 256;
constexpr int fixed_unit = 1 << 16;

bool simd_allowed() {
#if !SKIMAGEXX_PHL_SSE2
    return false;
#elif defined(_MSC_VER)
    char* value = nullptr;
    std::size_t length = 0;
    const bool disabled = _dupenv_s(&value, &length, "SKIMAGEXX_DISABLE_SIMD") == 0 && value != nullptr;
    std::free(value);
    return !disabled;
#else
    return std::getenv("SKIMAGEXX_DISABLE_SIMD") == nullptr;
#endif
}

template<class T>
bool invalid_span(const T* data, std::size_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return count && (!data || address % alignof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(T));
}

int round_half_away(double value) {
    return static_cast<int>(value > 0 ? value + 0.5 : value - 0.5);
}

int fixed_floor(int value) {
    return (value < 0 ? value - (fixed_unit - 1) : value) / fixed_unit;
}

#if SKIMAGEXX_PHL_SSE2
void project_two(__m128d c_v, __m128d s_v, __m128d x_v, __m128d y_v, int& bin0, int& bin1) {
    const __m128d proj = _mm_add_pd(_mm_mul_pd(c_v, x_v), _mm_mul_pd(s_v, y_v));
    const __m128d sign_mask = _mm_set1_pd(-0.0);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128d signed_half = _mm_or_pd(_mm_and_pd(proj, sign_mask), half_v);
    const __m128i bins = _mm_cvttpd_epi32(_mm_add_pd(proj, signed_half));
    bin0 = _mm_cvtsi128_si32(bins);
    bin1 = _mm_cvtsi128_si32(_mm_srli_si128(bins, 4));
}
#endif

using Point = ProbabilisticHoughPoint;

struct Walk {
    Point start;
    Point step;
    bool x_major;

    Walk(Point origin, double cosine, double sine)
        : start(origin), x_major(std::abs(sine) > std::abs(cosine)) {
        const double a = -sine;
        const double b = cosine;
        if (x_major) {
            step = {a > 0 ? 1 : -1, round_half_away(b * fixed_unit / std::abs(a))};
            start.y = start.y * fixed_unit + fixed_unit / 2;
        } else {
            step = {round_half_away(a * fixed_unit / std::abs(b)), b > 0 ? 1 : -1};
            start.x = start.x * fixed_unit + fixed_unit / 2;
        }
    }

    Point position(Point fixed) const {
        return x_major ? Point{fixed.x, fixed_floor(fixed.y)} : Point{fixed_floor(fixed.x), fixed.y};
    }
};

template<class Visit>
void follow(const Walk& walk, int direction, Visit visit) {
    Point fixed = walk.start;
    while (visit(walk.position(fixed))) {
        fixed.x += direction * walk.step.x;
        fixed.y += direction * walk.step.y;
    }
}

}  // namespace

std::vector<ProbabilisticHoughSegment> probabilistic_hough_line_ordered(
    const std::uint8_t* image, std::size_t image_count, int height, int width,
    const double* cosines, const double* sines, std::size_t angle_count,
    const std::intptr_t* permutation, std::size_t count, int threshold, int line_length, int line_gap) {
    if (height < 1 || height > dimension_limit || width < 1 || width > dimension_limit ||
        image_count != std::size_t(height * width) ||
        count > foreground_limit || angle_count > 256 || threshold < 1 || threshold > 4096 ||
        line_length < 0 || line_length > 256 || line_gap < 0 || line_gap > 256 ||
        invalid_span(image, image_count) || invalid_span(permutation, count) ||
        invalid_span(cosines, angle_count) || invalid_span(sines, angle_count) || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("probabilistic_hough_line input is outside the bounded domain");

    const int offset = int(std::ceil(std::sqrt(double(height * height + width * width))));
    for (std::size_t k = 0; k < angle_count; ++k) {
        const double x = std::abs(cosines[k]);
        const double y = std::abs(sines[k]);
        if (!std::isfinite(x) || !std::isfinite(y) || x > 1.0 || y > 1.0 || std::max(x, y) < 0.5 ||
            x * (width - 1) + y * (height - 1) >= offset - 0.5)
            throw std::invalid_argument("probabilistic_hough_line direction exceeds the distance grid");
    }

    std::array<bool, foreground_limit> seen{};
    for (std::size_t k = 0; k < count; ++k) {
        const auto index = std::size_t(permutation[k]);
        if (index >= count || seen[index])
            throw std::invalid_argument("probabilistic_hough_line requires a foreground permutation");
        seen[index] = true;
    }
    std::vector<Point> points;
    points.reserve(count);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image[y * width + x]) {
                if (points.size() == count)
                    throw std::invalid_argument("probabilistic_hough_line foreground count differs");
                points.push_back({x, y});
            }
        }
    }
    if (points.size() != count)
        throw std::invalid_argument("probabilistic_hough_line foreground count differs");

    std::vector<ProbabilisticHoughSegment> lines;
    if (!angle_count || !count)
        return lines;

    const bool use_simd = simd_allowed();
    const std::size_t stride = std::size_t(2 * offset);
    std::vector<std::uint8_t> mask(image, image + image_count);
    std::vector<std::int16_t> accumulator(angle_count * stride, 0);
    std::int16_t* const accum_zero = accumulator.data() + offset;

    // The walk advances one pixel per step on its major axis. Each in-bounds
    // major-axis coordinate is visited once, except the shared start.
    std::array<double, dimension_limit> removed_x;
    std::array<double, dimension_limit> removed_y;

    while (count > 0) {
        const Point start = points[std::size_t(permutation[--count])];
        if (!mask[start.y * width + start.x])
            continue;
        int strongest = threshold - 1;
        std::size_t winner = 0;
        const double px = double(start.x);
        const double py = double(start.y);
        std::size_t k = 0;
#if SKIMAGEXX_PHL_SSE2
        if (use_simd && angle_count >= 2) {
            const __m128d x_v = _mm_set1_pd(px);
            const __m128d y_v = _mm_set1_pd(py);
            std::int16_t* row = accum_zero;
            for (; k + 1 < angle_count; k += 2, row += 2 * stride) {
                int b0 = 0;
                int b1 = 0;
                project_two(_mm_loadu_pd(cosines + k), _mm_loadu_pd(sines + k), x_v, y_v, b0, b1);
                const int v0 = ++row[b0];
                if (v0 > strongest) {
                    strongest = v0;
                    winner = k;
                }
                const int v1 = ++row[stride + b1];
                if (v1 > strongest) {
                    strongest = v1;
                    winner = k + 1;
                }
            }
        }
#else
        (void)use_simd;
#endif
        for (; k < angle_count; ++k) {
            const int b = round_half_away(cosines[k] * px + sines[k] * py);
            const int value = ++(accum_zero + k * stride)[b];
            if (value > strongest) {
                strongest = value;
                winner = k;
            }
        }
        if (strongest < threshold)
            continue;

        const Walk walk(start, cosines[winner], sines[winner]);
        const auto endpoint = [&](int direction) {
            Point last = start;
            int gap = 0;
            follow(walk, direction, [&](Point point) {
                if (point.x < 0 || point.x >= width || point.y < 0 || point.y >= height)
                    return false;
                if (mask[point.y * width + point.x]) {
                    last = point;
                    gap = 0;
                } else if (++gap > line_gap) {
                    return false;
                }
                return true;
            });
            return last;
        };
        const Point first = endpoint(1);
        const Point second = endpoint(-1);
        if (std::abs(first.x - second.x) < line_length && std::abs(first.y - second.y) < line_length)
            continue;

        // Collect newly cleared line pixels once, then decrement each angle row contiguously.
        std::size_t removed_count = 0;
        const auto collect_removed = [&](int direction, Point end) {
            follow(walk, direction, [&](Point point) {
                auto& marked = mask[point.y * width + point.x];
                if (marked) {
                    marked = 0;
                    removed_x[removed_count] = double(point.x);
                    removed_y[removed_count] = double(point.y);
                    ++removed_count;
                }
                return point.x != end.x || point.y != end.y;
            });
        };
        collect_removed(1, first);
        collect_removed(-1, second);

        const double* const rx = removed_x.data();
        const double* const ry = removed_y.data();
        for (std::size_t angle = 0; angle < angle_count; ++angle) {
            const double c = cosines[angle];
            const double s = sines[angle];
            std::int16_t* const row = accum_zero + angle * stride;
            std::size_t j = 0;
#if SKIMAGEXX_PHL_SSE2
            if (use_simd && removed_count >= 2) {
                const __m128d c_v = _mm_set1_pd(c);
                const __m128d s_v = _mm_set1_pd(s);
                for (; j + 1 < removed_count; j += 2) {
                    int b0 = 0;
                    int b1 = 0;
                    project_two(c_v, s_v, _mm_loadu_pd(rx + j), _mm_loadu_pd(ry + j), b0, b1);
                    --row[b0];
                    --row[b1];
                }
            }
#endif
            for (; j < removed_count; ++j)
                --row[round_half_away(c * rx[j] + s * ry[j])];
        }
        lines.push_back({first, second});
        // The original cap exceeds the maximum foreground count in this domain.
    }
    return lines;
}

}  // namespace skimagexx
