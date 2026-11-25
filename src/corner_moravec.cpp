#include <skimagexx/corner_moravec.hpp>

#include <algorithm>
#include <cfenv>
#include <limits>
#include <vector>

#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#include <skimagexx/image.hpp>
#include <immintrin.h>
#define MORAVEC_AVX2 1
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#elif defined(__clang__)
#pragma clang fp contract(off)
#endif
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace skimagexx {
namespace {

void square_shift(const double* pixels, double* squares, int height, int width,
    int dy, int dx) {
    const int begin = std::max(0, -dx);
    const int end = width - std::max(0, dx);
    for (int y = 0; y < height - dy; ++y) {
        const double* row = pixels + y * width;
        const double* shifted = row + dy * width;
        double* plane = squares + y * width;
        for (int x = begin; x < end; ++x) {
            const double difference = row[x] - shifted[x + dx];
            plane[x] = difference * difference;
        }
    }
}

double sum_three_by_three(const double* p, int width) {
    // Squares are nonnegative. Starting at p[0] is equivalent to +0 + p[0].
    // Keep the following additions in patch row-major order.
    double score = p[0];
    score += p[1];
    score += p[2];
    score += p[width];
    score += p[width + 1];
    score += p[width + 2];
    score += p[2 * width];
    score += p[2 * width + 1];
    score += p[2 * width + 2];
    return score;
}

#ifdef MORAVEC_AVX2
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4752)
#define MORAVEC_AVX_INLINE __forceinline
#define MORAVEC_AVX_ENTRY __declspec(noinline)
#else
#define MORAVEC_AVX_INLINE __attribute__((target("avx2"), always_inline)) inline
#define MORAVEC_AVX_ENTRY __attribute__((target("avx2"), noinline))
#endif

MORAVEC_AVX_INLINE __m256d sum_three_by_three_avx2(const double* p, int width) {
    auto score = _mm256_loadu_pd(p);
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + 1));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + 2));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + width));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + width + 1));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + width + 2));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + 2 * width));
    score = _mm256_add_pd(score, _mm256_loadu_pd(p + 2 * width + 1));
    return _mm256_add_pd(score, _mm256_loadu_pd(p + 2 * width + 2));
}

MORAVEC_AVX_ENTRY
void window_one_avx2(const double* sq_pp, const double* sq_pm, double* response,
    int height, int width, int end) {
    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < end; x += 4) {
            const int patch = (y - 1) * width + x - 1;
            auto best = sum_three_by_three_avx2(sq_pp + patch - width - 1, width);
            best = _mm256_min_pd(best, sum_three_by_three_avx2(sq_pm + patch - width + 1, width));
            best = _mm256_min_pd(best, sum_three_by_three_avx2(sq_pm + patch, width));
            best = _mm256_min_pd(best, sum_three_by_three_avx2(sq_pp + patch, width));
            _mm256_storeu_pd(response + y * width + x, best);
        }
    }
}

#undef MORAVEC_AVX_INLINE
#undef MORAVEC_AVX_ENTRY
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

void window_one(const double* pixels, double* response, int height, int width) {
    // In nearest rounding, reversing a difference only changes its sign.
    // Read each plane at the shifted patch origin for the opposite direction.
    std::vector<double> sq_pp(height * width);
    std::vector<double> sq_pm(height * width);
    square_shift(pixels, sq_pp.data(), height, width, 1, 1);
    square_shift(pixels, sq_pm.data(), height, width, 1, -1);
    int x_begin = 2;
#ifdef MORAVEC_AVX2
    if (width >= 8 && has_avx2()) {
        x_begin += (width - 4) / 4 * 4;
        window_one_avx2(sq_pp.data(), sq_pm.data(), response, height, width, x_begin);
    }
#endif
    for (int y = 2; y < height - 2; ++y) {
        for (int x = x_begin; x < width - 2; ++x) {
            const int patch = (y - 1) * width + x - 1;
            response[y * width + x] = std::min({
                sum_three_by_three(sq_pp.data() + patch - width - 1, width),
                sum_three_by_three(sq_pm.data() + patch - width + 1, width),
                sum_three_by_three(sq_pm.data() + patch, width),
                sum_three_by_three(sq_pp.data() + patch, width)});
        }
    }
}

void larger_window(const double* pixels, double* response, int height, int width, int window) {
    const int border = 2 * window;
    const int diameter = 2 * window + 1;
    std::vector<double> squares(height * width);  // Reuse one plane per shift pair.
    for (int y = border; y < height - border; ++y) {
        std::fill(response + y * width + border, response + y * width + width - border,
            std::numeric_limits<double>::max());
    }
    for (int dy = 1; dy <= window; ++dy) {
        for (int dx = -window; dx <= window; ++dx) {
            if (dx == 0) {
                continue;
            }
            square_shift(pixels, squares.data(), height, width, dy, dx);
            const int offset = dy * width + dx;
            for (int y = border; y < height - border; ++y) {
                for (int x = border; x < width - border; ++x) {
                    const double* p = squares.data() + (y - window) * width + x - window;
                    const double* reverse = p - offset;
                    double forward_score = 0.0;
                    double reverse_score = 0.0;
                    for (int row = 0; row < diameter; ++row) {
                        for (int col = 0; col < diameter; ++col) {
                            const int pixel = row * width + col;
                            forward_score += p[pixel];
                            reverse_score += reverse[pixel];
                        }
                    }
                    double& best = response[y * width + x];
                    best = std::min(best, std::min(forward_score, reverse_score));
                }
            }
        }
    }
}

}  // namespace

bool corner_moravec_u8_into(const std::uint8_t* image, std::size_t rows,
    std::size_t columns, int window, double* response) {
    if (!image || !response || !corner_moravec_shape_supported(rows, columns, window)
        || std::fegetround() != FE_TONEAREST) {
        return false;
    }

    constexpr double reciprocal = 1.0 / 255.0;
    std::vector<double> pixels(rows * columns);
    std::transform(image, image + pixels.size(), pixels.begin(),
        [reciprocal](std::uint8_t pixel) { return double(pixel) * reciprocal; });
    std::fill_n(response, pixels.size(), 0.0);

    const int height = static_cast<int>(rows);
    const int width = static_cast<int>(columns);
    if (window == 1) {
        window_one(pixels.data(), response, height, width);
    } else {
        larger_window(pixels.data(), response, height, width, window);
    }
    return true;
}

}  // namespace skimagexx

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
