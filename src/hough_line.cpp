#include "skimagexx/hough_line.hpp"

#include <algorithm>
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

void hough_line_dense(const std::vector<NonzeroRow>& rows,
                      const double* columns, int offset,
                      const double* HOUGH_LINE_RESTRICT cosines,
                      const double* HOUGH_LINE_RESTRICT sines, std::size_t angles,
                      std::uint64_t* HOUGH_LINE_RESTRICT accumulator, bool use_simd) {
    const auto num_distances = std::size_t(2 * offset + 1);
    constexpr auto cache_budget = 24 * 1024;
    const auto block_size = std::min(angles, std::clamp(
        cache_budget / (num_distances * sizeof(std::uint32_t)), std::size_t(1), std::size_t(8)));
    std::vector<std::uint32_t> votes(num_distances * block_size);
#if SKIMAGEXX_HL_SSE2
    const __m128d sign_mask = _mm_set1_pd(-0.0);
    const __m128d half_v = _mm_set1_pd(0.5);
    const __m128i offset_v = _mm_set1_epi32(offset);
#else
    (void)use_simd;
#endif
    for (std::size_t first = 0; first < angles; first += block_size) {
        const auto block = std::min(block_size, angles - first);
        for (std::size_t local = 0; local < block; ++local) {
            const auto angle = first + local;
            auto* HOUGH_LINE_RESTRICT col_accum = votes.data() + local * num_distances;
            const double cosine = cosines[angle];
            const double sine = sines[angle];
#if SKIMAGEXX_HL_SSE2
            const __m128d cos_v = _mm_set1_pd(cosine);
#endif
            const double* col_ptr = columns;
            for (const auto& entry : rows) {
                const double sin_row = sine * entry.row;
                const std::size_t count = entry.count;
                std::size_t i = 0;
#if SKIMAGEXX_HL_SSE2
                if (use_simd && count >= 2) {
                    const __m128d sin_row_v = _mm_set1_pd(sin_row);
                    for (; i + 1 < count; i += 2) {
                        const __m128d proj = _mm_add_pd(_mm_mul_pd(cos_v, _mm_loadu_pd(col_ptr + i)), sin_row_v);
                        std::size_t d0 = 0;
                        std::size_t d1 = 0;
                        distance_two(proj, offset_v, sign_mask, half_v, d0, d1);
                        ++col_accum[d0];
                        ++col_accum[d1];
                    }
                }
#endif
                for (; i < count; ++i) {
                    // The pinned rounding helper separately adjusts by half before truncating.
                    const double projection = cosine * col_ptr[i] + sin_row;
                    const double adjusted = projection > 0.0 ? projection + 0.5 : projection - 0.5;
                    const auto distance = std::size_t(offset + static_cast<std::int64_t>(adjusted));
                    ++col_accum[distance];
                }
                col_ptr += count;
            }
        }
        // Write adjacent public angle columns together while the block is hot.
        for (std::size_t distance = 0; distance < num_distances; ++distance) {
            auto* output = accumulator + distance * angles + first;
            for (std::size_t local = 0; local < block; ++local) {
                auto& value = votes[local * num_distances + distance];
                output[local] = value;
                value = 0;
            }
        }
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
