#include "skimagexx/hough_line.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define SKIMAGEXX_HL_SSE2 1
#else
#define SKIMAGEXX_HL_SSE2 0
#endif

#if SKIMAGEXX_HL_SSE2 && defined(SKIMAGEXX_AVX2)
#include <immintrin.h>
#define SKIMAGEXX_HL_AVX2 1
#if defined(_MSC_VER) && !defined(__clang__)
#define HOUGH_LINE_AVX2 __declspec(noinline)
#define HOUGH_LINE_AVX2_INLINE __forceinline
#else
#define HOUGH_LINE_AVX2 __attribute__((target("avx2"), noinline))
#define HOUGH_LINE_AVX2_INLINE __attribute__((target("avx2"), always_inline)) inline
#endif
#else
#define SKIMAGEXX_HL_AVX2 0
#endif

#ifdef _MSC_VER
#define HOUGH_LINE_RESTRICT __restrict
#else
#define HOUGH_LINE_RESTRICT __restrict__
#endif

namespace skimagexx {
namespace {

bool simd_allowed() {
#if !SKIMAGEXX_HL_SSE2
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

struct ByteSpan {
    std::uintptr_t begin;
    std::uintptr_t end;
};

template<class T>
ByteSpan byte_span(const T* data, std::size_t count) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || begin % alignof(T) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        throw std::invalid_argument("invalid hough_line buffer");
    return {begin, begin + count * sizeof(T)};
}

bool overlaps(ByteSpan left, ByteSpan right) {
    return left.begin < right.end && right.begin < left.end;
}

#if SKIMAGEXX_HL_SSE2
void distance_two(__m128d proj, __m128i offset_v, __m128d sign_mask, __m128d half_v,
                  std::size_t& d0, std::size_t& d1) {
    const __m128d signed_half = _mm_or_pd(_mm_and_pd(proj, sign_mask), half_v);
    const __m128i bins = _mm_add_epi32(offset_v, _mm_cvttpd_epi32(_mm_add_pd(proj, signed_half)));
    d0 = static_cast<std::size_t>(static_cast<std::uint32_t>(_mm_cvtsi128_si32(bins)));
    d1 = static_cast<std::size_t>(static_cast<std::uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(bins, 4))));
}
#endif

struct NonzeroRow {
    double row;
    std::size_t count;
};

void extract_nonzeros(const std::uint8_t* image, std::size_t height, std::size_t width,
                      std::vector<NonzeroRow>& rows, std::vector<double>& columns) {
    rows.reserve(height);
    for (std::size_t row = 0; row < height; ++row) {
        const auto* pixels = image + row * width;
        const auto start = columns.size();
        std::size_t column = 0;
        for (; column + sizeof(std::uint64_t) <= width; column += sizeof(std::uint64_t)) {
            std::uint64_t word;
            std::memcpy(&word, pixels + column, sizeof(word));
            if (word) {
                for (std::size_t i = 0; i < sizeof(word); ++i) {
                    if (pixels[column + i]) columns.push_back(double(column + i));
                }
            }
        }
        for (; column < width; ++column) {
            if (pixels[column]) columns.push_back(double(column));
        }
        if (columns.size() != start) rows.push_back({double(row), columns.size() - start});
    }
}

void hough_line_sparse(const std::vector<NonzeroRow>& rows, const double* columns, int offset,
                       const double* HOUGH_LINE_RESTRICT cosines,
                       const double* HOUGH_LINE_RESTRICT sines, std::size_t angles,
                       std::uint64_t* HOUGH_LINE_RESTRICT accumulator,
                       std::size_t accumulator_count, bool use_simd) {
    std::fill_n(accumulator, accumulator_count, std::uint64_t{0});
#if SKIMAGEXX_HL_SSE2
    const __m128d sign_mask = _mm_set1_pd(-0.0);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128i offset_v = _mm_set1_epi32(offset);
#else
    (void)use_simd;
#endif
    for (const auto& entry : rows) {
        const double drow = entry.row;
        for (std::size_t i = 0; i < entry.count; ++i) {
            const double dcol = *columns++;
            std::size_t angle = 0;
#if SKIMAGEXX_HL_SSE2
            if (use_simd && angles >= 2) {
                const __m128d col_v = _mm_set1_pd(dcol);
                const __m128d row_v = _mm_set1_pd(drow);
                for (; angle + 1 < angles; angle += 2) {
                    const __m128d proj = _mm_add_pd(
                        _mm_mul_pd(_mm_loadu_pd(cosines + angle), col_v),
                        _mm_mul_pd(_mm_loadu_pd(sines + angle), row_v));
                    std::size_t d0 = 0;
                    std::size_t d1 = 0;
                    distance_two(proj, offset_v, sign_mask, half_v, d0, d1);
                    ++accumulator[d0 * angles + angle];
                    ++accumulator[d1 * angles + angle + 1];
                }
            }
#endif
            for (; angle < angles; ++angle) {
                // The pinned rounding helper separately adjusts by half before truncating.
                const double projection = cosines[angle] * dcol + sines[angle] * drow;
                const double adjusted = projection > 0.0 ? projection + 0.5 : projection - 0.5;
                const auto distance = std::size_t(offset + static_cast<std::int64_t>(adjusted));
                ++accumulator[distance * angles + angle];
            }
        }
    }
}

int rounded_distance(double projection, int offset) {
    const double adjusted = projection > 0.0 ? projection + 0.5 : projection - 0.5;
    return offset + static_cast<int>(adjusted);
}

#if SKIMAGEXX_HL_AVX2
template<bool mixed_sign>
HOUGH_LINE_AVX2_INLINE __m128i project_four(const double* columns, __m256d cosine, __m256d sin_row,
                                          __m256d half, __m256d sign) {
    const auto projection = _mm256_add_pd(_mm256_mul_pd(cosine, _mm256_loadu_pd(columns)), sin_row);
    if constexpr (mixed_sign)
        half = _mm256_or_pd(_mm256_and_pd(projection, sign), half);
    // Scale each signed rho to index an eight-counter group.
    return _mm_slli_epi32(_mm256_cvttpd_epi32(_mm256_add_pd(projection, half)), 3);
}

template<bool mixed_sign>
HOUGH_LINE_AVX2
void vote_angle_avx2(const std::vector<NonzeroRow>& rows, const double* columns,
                     double cosine, double sine, int offset, double half,
                     std::uint32_t* HOUGH_LINE_RESTRICT histogram) {
    constexpr std::size_t batch_size = 128;
    alignas(16) std::array<std::int16_t, batch_size> distances;
    const auto cos_v = _mm256_set1_pd(cosine);
    const auto sign_mask = _mm256_set1_pd(-0.0);
    const auto half_v = _mm256_set1_pd(half);
    auto* const center = histogram + 8 * offset;
    for (const auto& row : rows) {
        const double sin_row = sine * row.row;
        const auto sin_row_v = _mm256_set1_pd(sin_row);
        const auto* const row_end = columns + row.count;
        std::size_t remaining = row.count - row.count % 4;
        while (remaining) {
            const auto batch = std::min(remaining, batch_size);
            const auto paired = batch - batch % 8;
            for (std::size_t i = 0; i < paired; i += 8) {
                const auto first = project_four<mixed_sign>(
                    columns + i, cos_v, sin_row_v, half_v, sign_mask);
                const auto second = project_four<mixed_sign>(
                    columns + i + 4, cos_v, sin_row_v, half_v, sign_mask);
                _mm_store_si128(reinterpret_cast<__m128i*>(distances.data() + i),
                                _mm_packs_epi32(first, second));
            }
            if (paired < batch) {
                const auto tail = project_four<mixed_sign>(
                    columns + paired, cos_v, sin_row_v, half_v, sign_mask);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(distances.data() + paired),
                                 _mm_packs_epi32(tail, tail));
            }
            // The scaled rho grid fits in int16_t. Scatter after storing whole
            // vectors of these indices instead of extracting each SIMD lane.
            for (std::size_t i = 0; i < paired; i += 8) {
                ++center[distances[i]];
                ++center[distances[i + 1] + 1];
                ++center[distances[i + 2] + 2];
                ++center[distances[i + 3] + 3];
                ++center[distances[i + 4] + 4];
                ++center[distances[i + 5] + 5];
                ++center[distances[i + 6] + 6];
                ++center[distances[i + 7] + 7];
            }
            for (std::size_t i = paired; i < batch; ++i)
                ++center[distances[i] + static_cast<int>(i - paired)];
            columns += batch;
            remaining -= batch;
        }
        for (; columns != row_end; ++columns)
            ++histogram[8 * rounded_distance(cosine * *columns + sin_row, offset)];
    }
}

HOUGH_LINE_AVX2
void merge_vote_banks(std::uint32_t* HOUGH_LINE_RESTRICT votes,
                      const std::uint32_t* HOUGH_LINE_RESTRICT histogram, std::size_t size) {
    std::size_t i = 0;
    for (; i + 1 < size; i += 2) {
        const auto first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(histogram + i * 8));
        const auto second = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(histogram + (i + 1) * 8));
        const auto pairs = _mm256_hadd_epi32(first, second);
        const auto groups = _mm256_hadd_epi32(pairs, pairs);
        const auto sums = _mm_add_epi32(_mm256_castsi256_si128(groups), _mm256_extracti128_si256(groups, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(votes + i), sums);
    }
    if (i < size) {
        std::uint32_t sum = 0;
        for (std::size_t lane = 0; lane < 8; ++lane)
            sum += histogram[i * 8 + lane];
        votes[i] = sum;
    }
}
#endif

void vote_angle(const std::vector<NonzeroRow>& rows, const double* columns,
                double cosine, double sine, int offset, bool use_simd,
                std::uint32_t* HOUGH_LINE_RESTRICT votes,
                std::uint32_t* HOUGH_LINE_RESTRICT extra) {
#if SKIMAGEXX_HL_SSE2
    const __m128d cos_v = _mm_set1_pd(cosine);
    const __m128d sign_mask = _mm_set1_pd(-0.0);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128i offset_v = _mm_set1_epi32(offset);
#else
    (void)use_simd;
    (void)extra;
#endif
    for (const auto& row : rows) {
        const double sin_row = sine * row.row;
        std::size_t i = 0;
#if SKIMAGEXX_HL_SSE2
        if (use_simd) {
            const __m128d sin_row_v = _mm_set1_pd(sin_row);
            for (; i + 1 < row.count; i += 2) {
                const auto projection = _mm_add_pd(
                    _mm_mul_pd(cos_v, _mm_loadu_pd(columns + i)), sin_row_v);
                std::size_t d0, d1;
                distance_two(projection, offset_v, sign_mask, half_v, d0, d1);
                ++votes[d0];
                ++extra[d1];
            }
        }
#endif
        for (; i < row.count; ++i)
            ++votes[rounded_distance(cosine * columns[i] + sin_row, offset)];
        columns += row.count;
    }
}

void hough_line_dense(const std::vector<NonzeroRow>& rows, const double* columns, int offset,
                      const double* cosines, const double* sines, std::size_t angles,
                      std::uint64_t* accumulator, bool use_simd) {
    const auto num_distances = std::size_t(2 * offset + 1);
    const auto column_bytes = num_distances * sizeof(std::uint32_t);
    const auto tile_size = std::clamp(24 * 1024 / column_bytes, std::size_t(16), std::size_t(32));
#if SKIMAGEXX_HL_AVX2
    const bool use_avx2 = use_simd && offset <= std::numeric_limits<std::int16_t>::max() / 8 && has_avx2();
#else
    constexpr bool use_avx2 = false;
#endif
    // Each SIMD lane updates a separate histogram to break same-bin dependencies.
    // Only the active angle needs these extra histograms.
    const std::size_t extra_columns = use_avx2 ? 8 : use_simd ? 1 : 0;
    const auto tile_columns = std::min(angles, tile_size);
    std::vector<std::uint32_t> bins(num_distances * (tile_columns + extra_columns));
    auto* extra = bins.data() + tile_columns * num_distances;
    for (std::size_t first = 0; first < angles; first += tile_columns) {
        const auto block = std::min(tile_columns, angles - first);
        for (std::size_t local = 0; local < block; ++local) {
            const auto angle = first + local;
            auto* votes = bins.data() + local * num_distances;
#if SKIMAGEXX_HL_AVX2
            if (use_avx2) {
                const double cosine = cosines[angle];
                const double sine = sines[angle];
                if (cosine >= 0.0 && sine >= 0.0) {
                    vote_angle_avx2<false>(rows, columns, cosine, sine, offset, 0.5, extra);
                } else if (cosine <= 0.0 && sine <= 0.0) {
                    vote_angle_avx2<false>(rows, columns, cosine, sine, offset, -0.5, extra);
                } else {
                    vote_angle_avx2<true>(rows, columns, cosine, sine, offset, 0.5, extra);
                }
                merge_vote_banks(votes, extra, num_distances);
            } else
#endif
            {
                vote_angle(rows, columns, cosines[angle], sines[angle], offset, use_simd, votes, extra);
                if (use_simd) {
                    for (std::size_t distance = 0; distance < num_distances; ++distance)
                        votes[distance] += extra[distance];
                }
            }
            std::fill_n(extra, extra_columns * num_distances, std::uint32_t{0});
        }
        // Store adjacent public angle columns together.
        for (std::size_t distance = 0; distance < num_distances; ++distance) {
            auto* output = accumulator + distance * angles + first;
            for (std::size_t local = 0; local < block; ++local)
                output[local] = bins[local * num_distances + distance];
        }
        if (!use_avx2)
            std::fill_n(bins.data(), block * num_distances, std::uint32_t{0});
    }
}

}  // namespace

int hough_line_offset(std::size_t height, std::size_t width, std::size_t angles) {
    if (height == 0 || width == 0 || height > 4096 || width > 4096 ||
        angles == 0 || angles > 1024 || height * width > 1048576 ||
        height * width > 134217728 / angles)
        throw std::invalid_argument("hough_line shape is outside the bounded domain");
    return static_cast<int>(std::ceil(std::sqrt(double(height * height + width * width))));
}

void hough_line_into(const std::uint8_t* image, std::size_t image_count,
                     std::size_t height, std::size_t width,
                     const double* cosines, const double* sines, std::size_t angles,
                     std::uint64_t* accumulator, std::size_t accumulator_count) {
    if (std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("hough_line requires round-to-nearest");
    const auto offset = hough_line_offset(height, width, angles);
    if (image_count != height * width || accumulator_count != std::size_t(2 * offset + 1) * angles)
        throw std::invalid_argument("hough_line buffer counts must match the image and accumulator");
    const auto output = byte_span(accumulator, accumulator_count);
    if (overlaps(output, byte_span(image, image_count)) ||
        overlaps(output, byte_span(cosines, angles)) || overlaps(output, byte_span(sines, angles)))
        throw std::invalid_argument("hough_line output overlaps input storage");
    for (std::size_t angle = 0; angle < angles; ++angle) {
        const double cosine = cosines[angle];
        const double sine = sines[angle];
        if (!std::isfinite(cosine) || !std::isfinite(sine) ||
            std::abs(cosine) * double(width - 1) + std::abs(sine) * double(height - 1) > offset)
            throw std::invalid_argument("hough_line coefficients exceed the distance grid");
    }

    std::vector<NonzeroRow> rows;
    std::vector<double> columns;
    extract_nonzeros(image, height, width, rows, columns);
    if (columns.empty()) {
        std::fill_n(accumulator, accumulator_count, std::uint64_t{0});
        return;
    }

    const bool use_simd = simd_allowed();
    const auto num_distances = std::size_t(2 * offset + 1);
    if (columns.size() < 2 * num_distances) {
        hough_line_sparse(rows, columns.data(), offset, cosines, sines, angles,
                          accumulator, accumulator_count, use_simd);
    } else {
        hough_line_dense(rows, columns.data(), offset, cosines, sines, angles,
                         accumulator, use_simd);
    }
}

}  // namespace skimagexx
