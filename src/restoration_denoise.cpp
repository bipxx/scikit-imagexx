#include "skimagexx/restoration_denoise.hpp"
#include "skimagexx/image.hpp"

#include "execution.hpp"
#include "fft_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define SKIMAGEXX_NLM_AVX2
#endif

namespace skimagexx {
namespace {

template<class T>
bool finite(const T* image, std::size_t size) {
    return std::all_of(image, image + size, [](T value) { return std::isfinite(value); });
}

int reflect(int position, int extent, bool symmetric = false) {
    if (extent == 1) return 0;
    const int period = 2 * (extent - static_cast<int>(!symmetric));
    int wrapped = position % period;
    if (wrapped < 0) wrapped += period;
    return wrapped < extent ? wrapped : period - wrapped - static_cast<int>(symmetric);
}

double nl_weight(double distance) {
    // Schraudolph's exponential approximation, including scikit-image's
    // RMS bias and truncation before adding the bias.
    const auto exponent = static_cast<std::int32_t>(-1512775.3951951856938 * distance);
    const auto bits = static_cast<std::uint64_t>(1072632447 + exponent) << 32;
    double weight;
    std::memcpy(&weight, &bits, sizeof(weight));
    return weight;
}

#ifdef SKIMAGEXX_NLM_AVX2
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4752)
#define NLM_AVX_INLINE __forceinline
#define NLM_AVX_ENTRY __declspec(noinline)
#else
#define NLM_AVX_INLINE __attribute__((target("avx2"), always_inline)) inline
#define NLM_AVX_ENTRY __attribute__((target("avx2"), noinline))
#endif

template<int FixedChannels, bool FirstRow, bool FirstPlane>
NLM_AVX_ENTRY int nl_prefix_avx2(const double* padded, double* sums, std::ptrdiff_t row,
                                 std::ptrdiff_t shift, int end, int channels,
                                 std::ptrdiff_t count, int width, std::ptrdiff_t plane,
                                 double noise, double& running) {
    if constexpr (FixedChannels != 0) channels = FixedChannels;
    const auto variance = _mm256_set1_pd(noise);
    int x = 0;
    for (; x + 4 <= end; x += 4) {
        const auto p = row + x;
        auto squared = _mm256_setzero_pd();
        for (int c = 0; c < channels; ++c) {
            const auto* a = padded + c * count + p;
            const auto delta = _mm256_sub_pd(_mm256_loadu_pd(a), _mm256_loadu_pd(a + shift));
            squared = _mm256_add_pd(squared, _mm256_mul_pd(delta, delta));
        }
        // Keep the exact left-to-right sum across the row and SIMD across rows/planes.
        const auto values = _mm256_sub_pd(squared, variance);
        const auto lo = _mm256_castpd256_pd128(values);
        const auto hi = _mm256_extractf128_pd(values, 1);
        const auto first = _mm_add_sd(_mm_set_sd(running), lo);
        const auto second = _mm_add_sd(first, _mm_unpackhi_pd(lo, lo));
        const auto third = _mm_add_sd(second, hi);
        const auto fourth = _mm_add_sd(third, _mm_unpackhi_pd(hi, hi));
        running = _mm_cvtsd_f64(fourth);
        const auto prefix_lo = _mm_unpacklo_pd(first, second);
        const auto prefix_hi = _mm_unpacklo_pd(third, fourth);
        auto sum = _mm256_insertf128_pd(_mm256_castpd128_pd256(prefix_lo), prefix_hi, 1);
        if constexpr (!FirstRow) sum = _mm256_add_pd(sum, _mm256_loadu_pd(sums + p - width));
        if constexpr (!FirstPlane) {
            sum = _mm256_add_pd(sum, _mm256_loadu_pd(sums + p - plane));
            if constexpr (!FirstRow)
                sum = _mm256_sub_pd(sum, _mm256_loadu_pd(sums + p - plane - width));
        }
        _mm256_storeu_pd(sums + p, sum);
    }
    return x;
}

NLM_AVX_INLINE __m256d nl_patch_sum_avx2(const double* sums, std::ptrdiff_t ry, int radius) {
    return _mm256_sub_pd(
        _mm256_sub_pd(_mm256_loadu_pd(sums + ry + radius), _mm256_loadu_pd(sums + ry - radius)),
        _mm256_sub_pd(_mm256_loadu_pd(sums - ry + radius), _mm256_loadu_pd(sums - ry - radius)));
}

NLM_AVX_INLINE __m256d nl_weight_avx2(__m256d distance) {
    const auto exponent = _mm256_cvttpd_epi32(_mm256_mul_pd(
        _mm256_set1_pd(-1512775.3951951856938), distance));
    const auto biased = _mm_add_epi32(_mm_set1_epi32(1072632447), exponent);
    return _mm256_castsi256_pd(_mm256_slli_epi64(_mm256_cvtepi32_epi64(biased), 32));
}

NLM_AVX_INLINE void nl_update_avx2(double* weights, double* accum, const double* padded,
                                   int channels, std::ptrdiff_t count, __m256d weight,
                                   __m256i mask) {
    _mm256_maskstore_pd(weights, mask, _mm256_add_pd(_mm256_loadu_pd(weights), weight));
    for (int c = 0; c < channels; ++c) {
        auto* total = accum + c * count;
        const auto product = _mm256_mul_pd(weight, _mm256_loadu_pd(padded + c * count));
        _mm256_maskstore_pd(total, mask, _mm256_add_pd(_mm256_loadu_pd(total), product));
    }
}

template<bool Volume, int FixedChannels>
NLM_AVX_ENTRY int nl_patches_avx2(const double* sums, const double* padded, double* weights,
                                  double* accum, std::ptrdiff_t count, int channels,
                                  std::ptrdiff_t row, int x, int end, std::ptrdiff_t shift,
                                  std::ptrdiff_t rz, std::ptrdiff_t ry, int radius,
                                  double inv_scale, double multiplicity) {
    if constexpr (FixedChannels != 0) channels = FixedChannels;
    const auto zero = _mm256_setzero_pd();
    const auto inv = _mm256_set1_pd(inv_scale);
    const auto factor = _mm256_set1_pd(multiplicity);
    const auto cutoff = _mm256_set1_pd(5.0);
    for (; x + 4 <= end; x += 4) {
        const auto p = row + x;
        const auto q = p + shift;
        auto squared = nl_patch_sum_avx2(sums + p + rz, ry, radius);
        if constexpr (Volume)
            squared = _mm256_sub_pd(squared, nl_patch_sum_avx2(sums + p - rz, ry, radius));
        const auto distance = _mm256_mul_pd(_mm256_max_pd(squared, zero), inv);
        const auto active = _mm256_cmp_pd(distance, cutoff, _CMP_LE_OQ);
        if (!_mm256_movemask_pd(active)) continue;
        const auto weight = _mm256_mul_pd(factor, nl_weight_avx2(_mm256_and_pd(distance, active)));
        const auto mask = _mm256_castpd_si256(active);
        // If the two ranges overlap, the higher address from the earlier lane
        // is updated first. Zero shift applies both contributions to each lane.
        if (shift > 0) {
            nl_update_avx2(weights + q, accum + q, padded + p, channels, count, weight, mask);
            nl_update_avx2(weights + p, accum + p, padded + q, channels, count, weight, mask);
        } else {
            nl_update_avx2(weights + p, accum + p, padded + q, channels, count, weight, mask);
            nl_update_avx2(weights + q, accum + q, padded + p, channels, count, weight, mask);
        }
    }
    return x;
}

#undef NLM_AVX_INLINE
#undef NLM_AVX_ENTRY
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

template<int FixedChannels, bool FirstPlane, bool FirstRow>
void nlm_prefix_row_loop(const double* padded, double* sums, int z, int y, int px,
                         std::ptrdiff_t plane, std::ptrdiff_t shift, int x_last,
                         int channels, std::ptrdiff_t count, double noise,
                         bool vectorized) {
    double running = 0;
    const std::ptrdiff_t row = z * plane + static_cast<std::ptrdiff_t>(y) * px;
    int x = 0;
#ifdef SKIMAGEXX_NLM_AVX2
    if (vectorized)
        x = nl_prefix_avx2<FixedChannels, FirstRow, FirstPlane>(
            padded, sums, row, shift, x_last, channels,
            count, px, plane, noise, running);
#else
    (void)vectorized;
#endif
    for (; x < x_last; ++x) {
        const std::ptrdiff_t p = row + x;
        double squared = 0;
        for (int c = 0; c < channels; ++c) {
            const auto channel = c * count;
            const double delta = padded[channel + p] - padded[channel + p + shift];
            squared += delta * delta;
        }
        running += squared - noise;
        double sum = running;
        if constexpr (!FirstRow) sum += sums[p - px];
        if constexpr (!FirstPlane) {
            sum += sums[p - plane];
            if constexpr (!FirstRow) sum -= sums[p - plane - px];
        }
        sums[p] = sum;
    }
}

template<int FixedChannels, bool FirstPlane>
void nlm_prefix_plane_loop(const double* padded, double* sums, int z, int y_first, int y_last,
                           int px, std::ptrdiff_t plane, std::ptrdiff_t shift, int x_last,
                           int channels, std::ptrdiff_t count, double noise, bool vectorized) {
    nlm_prefix_row_loop<FixedChannels, FirstPlane, true>(
        padded, sums, z, y_first, px, plane, shift, x_last, channels, count, noise, vectorized);
    for (int y = y_first + 1; y < y_last; ++y)
        nlm_prefix_row_loop<FixedChannels, FirstPlane, false>(
            padded, sums, z, y, px, plane, shift, x_last, channels, count, noise, vectorized);
}

template<bool Volume, int FixedChannels, class T>
void nonlocal_means(const T* image, T* out, int depth, int rows, int cols, int channels,
                    int patch, int search, double h, double sigma) {
    using Index = std::ptrdiff_t;
    const int radius = patch / 2;
    const int pad = radius + search + 1;
    const int px = cols + 2 * pad;
    const int py = rows + 2 * pad;
    const int pz = Volume ? depth + 2 * pad : 1;
    const Index plane = static_cast<Index>(px) * py;
    const Index count = plane * pz;
    if constexpr (FixedChannels != 0) channels = FixedChannels;
    std::vector<double> padded(count * channels);
    // Each shift overwrites its entire overlapping prefix-sum region before use.
    std::unique_ptr<double[]> sums(new double[count]);
    std::vector<double> weights(count);
    std::vector<double> accum(count * channels);
    std::vector<int> reflected_x(px);
    for (int x = 0; x < px; ++x) reflected_x[x] = reflect(x - pad, cols);
    for (int z = 0; z < pz; ++z) {
        const int iz = Volume ? reflect(z - pad, depth) : 0;
        for (int y = 0; y < py; ++y) {
            const int iy = reflect(y - pad, rows);
            for (int x = 0; x < px; ++x) {
                const auto src = ((static_cast<Index>(iz) * rows + iy) * cols + reflected_x[x]) * channels;
                const auto dst = z * plane + static_cast<Index>(y) * px + x;
                for (int c = 0; c < channels; ++c)
                    padded[c * count + dst] = image[src + c];
            }
        }
    }
    double scale = channels * h * h * patch * patch;
    if constexpr (Volume) scale *= patch;
    const double inv_scale = 1.0 / scale;
#ifdef SKIMAGEXX_NLM_AVX2
    const bool vectorized = has_avx2();
#endif
    const double noise = (2 * (sigma * sigma)) * channels;
    const Index ry = static_cast<Index>(radius) * px;
    const Index rz = Volume ? radius * plane : 0;
    for (int dz = Volume ? -search : 0; dz <= (Volume ? search : 0); ++dz) {
        const int z_first = Volume ? std::max(0, -dz) : 0;
        const int z_last = pz - std::max(0, dz);
        for (int dy = -search; dy <= search; ++dy) {
            const int y_first = std::max(0, -dy);
            const int y_last = py - std::max(0, dy);
            for (int dx = 0; dx <= search; ++dx) {
                const Index shift = dz * plane + static_cast<Index>(dy) * px + dx;
                const int x_last = px - dx;
                const double multiplicity = dx == 0 ? 0.5 : 1.0;

                // Inclusive prefix sum over the overlap of the padded images.
                // Each query subtracts the lower corner, so the reference's
                // fast path effectively integrates 2*radius samples per axis.
#ifdef SKIMAGEXX_NLM_AVX2
                const bool use_simd = vectorized;
#else
                const bool use_simd = false;
#endif
                nlm_prefix_plane_loop<FixedChannels, true>(
                    padded.data(), sums.get(), z_first, y_first, y_last,
                    px, plane, shift, x_last, channels, count, noise, use_simd);
                if constexpr (Volume) {
                    for (int z = z_first + 1; z < z_last; ++z)
                        nlm_prefix_plane_loop<FixedChannels, false>(
                            padded.data(), sums.get(), z, y_first, y_last,
                            px, plane, shift, x_last, channels, count, noise, use_simd);
                }
                const auto patch_sum = [&](Index p) {
                    return (sums[p + ry + radius] - sums[p + ry - radius]) -
                           (sums[p - ry + radius] - sums[p - ry - radius]);
                };
                for (int z = (Volume ? pad - std::max(0, dz) : 0);
                     z < (Volume ? pad + depth - std::min(0, dz) : 1); ++z) {
                    for (int y = pad - std::max(0, dy); y < pad + rows - std::min(0, dy); ++y) {
                        const Index row = z * plane + static_cast<Index>(y) * px;
                        int x = pad - dx;
#ifdef SKIMAGEXX_NLM_AVX2
                        if (vectorized)
                            x = nl_patches_avx2<Volume, FixedChannels>(sums.get(), padded.data(), weights.data(),
                                accum.data(), count, channels, row, x, pad + cols, shift,
                                rz, ry, radius, inv_scale, multiplicity);
#endif
                        for (; x < pad + cols; ++x) {
                            const Index p = row + x;
                            const Index q = p + shift;
                            double squared;
                            if constexpr (Volume) squared = patch_sum(p + rz) - patch_sum(p - rz);
                            else squared = patch_sum(p);
                            const double distance = std::max(0.0, squared) * inv_scale;
                            if (distance > 5) continue;
                            const double w = multiplicity * nl_weight(distance);
                            weights[p] += w;
                            weights[q] += w;
                            for (int c = 0; c < channels; ++c) {
                                const auto channel = c * count;
                                accum[channel + p] += w * padded[channel + q];
                                accum[channel + q] += w * padded[channel + p];
                            }
                        }
                    }
                }
            }
        }
    }
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < rows; ++y) {
            Index p = (Volume ? (z + pad) * plane : 0) + static_cast<Index>(y + pad) * px + pad;
            for (int x = 0; x < cols; ++x, ++p) {
                for (int c = 0; c < channels; ++c)
                    *out++ = static_cast<T>(accum[c * count + p] / weights[p]);
            }
        }
    }
}

constexpr double root_half = 0.7071067811865476;
constexpr std::array<double, 4> db2_low{
    -0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416};
constexpr std::array<double, 4> db2_high{
    -0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037};

template<class T>
struct WaveletBands {
    std::array<std::vector<T>, 4> data;
    int rows;
    int cols;
};

template<class T>
std::pair<T, T> analysis_pair(const T* line, int extent, int stride, int index, int length) {
    if (length == 2) {
        const T a = line[(2 * index) * stride];
        const T b = line[std::min(2 * index + 1, extent - 1) * stride];
        const T h = static_cast<T>(root_half);
        return {h * b + h * a, h * a - h * b};
    }
    T low = 0;
    T high = 0;
    for (int k = 0; k < 4; ++k) {
        int pos = 2 * index + 1 - k;
        if (pos < 0 || pos >= extent) pos = reflect(pos, extent, true);
        const T value = line[pos * stride];
        low += static_cast<T>(db2_low[k]) * value;
        high += static_cast<T>(db2_high[k]) * value;
    }
    return {low, high};
}

template<class T>
WaveletBands<T> analyze(const T* input, int rows, int cols, int pitch, bool plane, int length) {
    const int r = plane ? (rows + length - 1) / 2 : 1;
    const int c = (cols + length - 1) / 2;
    WaveletBands<T> bands{{}, r, c};
    for (int band = 0; band < (plane ? 4 : 2); ++band)
        bands.data[band].resize(static_cast<std::size_t>(r) * c);
    std::vector<T> low(plane ? cols : 0);
    std::vector<T> high(plane ? cols : 0);
    for (int y = 0; y < r; ++y) {
        if (plane) {
            for (int x = 0; x < cols; ++x) {
                const auto pair = analysis_pair(input + x * pitch, rows, cols * pitch, y, length);
                low[x] = pair.first;
                high[x] = pair.second;
            }
        }
        for (int x = 0; x < c; ++x) {
            const auto p = static_cast<std::size_t>(y) * c + x;
            const auto upper = analysis_pair(plane ? low.data() : input, cols, plane ? 1 : pitch, x, length);
            bands.data[0][p] = upper.first;
            bands.data[1][p] = upper.second;
            if (plane) {
                const auto lower = analysis_pair(high.data(), cols, 1, x, length);
                bands.data[2][p] = lower.first;
                bands.data[3][p] = lower.second;
            }
        }
    }
    return bands;
}

template<class T>
double noise_median(std::vector<T> details) {
    const auto last = std::remove(details.begin(), details.end(), T(0));
    if (last == details.begin()) return std::numeric_limits<double>::quiet_NaN();
    std::transform(details.begin(), last, details.begin(), [](T x) { return std::abs(x); });
    const auto n = last - details.begin();
    const auto mid = details.begin() + n / 2;
    std::nth_element(details.begin(), mid, last);
    const T median = n % 2 ? *mid : (*mid + *std::max_element(details.begin(), mid)) / T(2);
    return static_cast<double>(median) / 0.6744897501960817;
}

template<class T>
void synthesis_line(const T* lo, const T* hi, int count, int stride,
                    T* output, int extent, int step, int length) {
    if (length == 2) {
        const auto h = static_cast<T>(root_half);
        for (int i = 0; i < count; ++i) {
            const T low = h * lo[i * stride];
            const T high = h * hi[i * stride];
            if (2 * i < extent) output[2 * i * step] = low + high;
            if (2 * i + 1 < extent) output[(2 * i + 1) * step] = low - high;
        }
        return;
    }
    for (int i = 0; i < extent; ++i) {
        // Decimation chose analysis origin 2*j+1. The synthesis support
        // at sample i is therefore indexed by k=2*j+1-i.
        T low = 0;
        T high = 0;
        for (int k = (1 - i) & 1; k < 4; k += 2) {
            const int j = (i + k - 1) / 2;
            if (j >= 0 && j < count) {
                low += static_cast<T>(db2_low[k]) * lo[j * stride];
                high += static_cast<T>(db2_high[k]) * hi[j * stride];
            }
        }
        output[i * step] = low + high;
    }
}

template<class T>
std::vector<T> synthesize(const WaveletBands<T>& bands, int rows, int cols, bool plane, int length) {
    std::vector<T> result(static_cast<std::size_t>(rows) * cols);
    if (!plane) {
        synthesis_line(bands.data[0].data(), bands.data[1].data(), bands.cols, 1,
                       result.data(), cols, 1, length);
        return result;
    }
    std::vector<T> lo(static_cast<std::size_t>(rows) * bands.cols);
    std::vector<T> hi(lo.size());
    for (int x = 0; x < bands.cols; ++x) {
        synthesis_line(bands.data[0].data() + x, bands.data[2].data() + x, bands.rows,
                       bands.cols, lo.data() + x, rows, bands.cols, length);
        synthesis_line(bands.data[1].data() + x, bands.data[3].data() + x, bands.rows,
                       bands.cols, hi.data() + x, rows, bands.cols, length);
    }
    for (int y = 0; y < rows; ++y) {
        const auto row = static_cast<std::size_t>(y);
        synthesis_line(lo.data() + row * bands.cols, hi.data() + row * bands.cols,
                       bands.cols, 1, result.data() + row * cols, cols, 1, length);
    }
    return result;
}

template<class T>
void threshold(std::vector<T>& values, double sigma, double universal, bool soft, bool bayes) {
    double cutoff = universal;
    if (bayes) {
        double squares = 0;
        for (T v : values) squares += v * v;
        const double variance = sigma * sigma;
        cutoff = variance / std::sqrt(std::max(squares / static_cast<double>(values.size()) - variance,
                                                static_cast<double>(std::numeric_limits<T>::epsilon())));
    }
    const T value = static_cast<T>(cutoff);
    for (T& v : values) {
        const T magnitude = std::abs(v);
        if (soft) v = magnitude <= value ? T(0) : std::copysign(magnitude - value, v);
        else if (magnitude < value) v = 0;
    }
}

std::size_t fft_extent(std::size_t n) {
    for (;; ++n) {
        auto rest = n;
        for (std::size_t factor : {2, 3, 5})
            while (rest % factor == 0) rest /= factor;
        if (rest == 1) return n;
    }
}

template<class T>
bool separable_psf(const T* psf, std::size_t rows, std::size_t cols,
                   std::vector<T>& vertical, std::vector<T>& horizontal) {
    const auto* maximum = std::max_element(psf, psf + rows * cols);
    const auto pivot = static_cast<std::size_t>(maximum - psf);
    const auto pivot_row = pivot / cols;
    const auto pivot_col = pivot % cols;
    const double largest = *maximum;
    horizontal.assign(psf + pivot_row * cols, psf + (pivot_row + 1) * cols);
    vertical.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const double scale = psf[row * cols + pivot_col] / largest;
        vertical[row] = static_cast<T>(scale);
        for (std::size_t col = 0; col < cols; ++col) {
            const double sample = psf[row * cols + col];
            if (std::abs(scale * horizontal[col] - sample) >
                2 * std::numeric_limits<T>::epsilon() * sample) return false;
        }
    }
    return true;
}

template<class T>
void convolve_separable(const T* input, T* output, T* scratch,
                        std::size_t rows, std::size_t cols,
                        const std::vector<T>& v, const std::vector<T>& h, bool mirror) {
    const auto center_x = (h.size() - 1) / 2;
    const auto center_y = (v.size() - 1) / 2;
    // Each tap contributes along a contiguous interval, including the
    // truncated intervals for implicit zero padding.
    std::fill_n(scratch, rows * cols, T(0));
    for (std::size_t k = 0; k < h.size(); ++k) {
        const T tap = h[mirror ? h.size() - 1 - k : k];
        if (tap == 0) continue;
        const auto src = center_x > k ? center_x - k : 0;
        const auto dst = k > center_x ? k - center_x : 0;
        const auto count = cols - std::max(src, dst);
        for (std::size_t row = 0; row < rows; ++row) {
            const auto* a = input + row * cols + src;
            auto* b = scratch + row * cols + dst;
            for (std::size_t i = 0; i < count; ++i) b[i] += tap * a[i];
        }
    }
    std::fill_n(output, rows * cols, T(0));
    for (std::size_t k = 0; k < v.size(); ++k) {
        const T tap = v[mirror ? v.size() - 1 - k : k];
        if (tap == 0) continue;
        const auto src = center_y > k ? center_y - k : 0;
        const auto dst = k > center_y ? k - center_y : 0;
        const auto count = (rows - std::max(src, dst)) * cols;
        const auto* a = scratch + src * cols;
        auto* b = output + dst * cols;
        for (std::size_t i = 0; i < count; ++i) b[i] += tap * a[i];
    }
}

} // namespace

template<class T>
bool denoise_nl_means(const T* image, T* out, std::size_t depth, std::size_t rows,
                      std::size_t cols, std::size_t channels, bool volume, int patch,
                      int search, double h, double sigma) {
    if (patch <= 0 || patch > 63 || search < 0 || search > 32 ||
        !std::isfinite(h) || h < 1e-12 || h > 1e12 ||
        !std::isfinite(sigma) || sigma < 0 || sigma > 1e12 || !channels || channels > 16 ||
        !depth || !rows || !cols || (!volume && depth != 1) ||
        std::max({depth, rows, cols}) > 1000000) return false;
    const auto pad = static_cast<std::size_t>(patch / 2 + search + 1);
    const auto padded = (volume ? depth + 2 * pad : 1) * (rows + 2 * pad);
    if (padded > (std::size_t{16000000} / channels) / (cols + 2 * pad) ||
        !finite(image, depth * rows * cols * channels)) return false;
    for (std::size_t i = 0; i < depth * rows * cols * channels; ++i)
        if (std::abs(image[i]) > 1e12) return false;
    patch |= 1;
    const auto run = [&](auto dim, auto batch) {
        nonlocal_means<decltype(dim)::value, decltype(batch)::value>(
            image, out, static_cast<int>(depth), static_cast<int>(rows), static_cast<int>(cols),
            static_cast<int>(channels), patch, search, h, sigma);
    };
    const auto dispatch = [&](auto dim) {
        if (channels == 1) run(dim, std::integral_constant<int, 1>{});
        else if (channels == 3) run(dim, std::integral_constant<int, 3>{});
        else run(dim, std::integral_constant<int, 0>{});
    };
    if (volume) dispatch(std::true_type{});
    else dispatch(std::false_type{});
    return true;
}

template<class T>
double estimate_sigma(const T* image, std::size_t rows, std::size_t cols,
                      std::size_t channels, std::size_t channel, bool plane) {
    if (!rows || !cols || !channels || channel >= channels || (!plane && rows != 1) ||
        rows > 1000000 || cols > 1000000 || channels > 16 ||
        rows * cols > 16000000 || !finite(image, rows * cols * channels))
        return std::numeric_limits<double>::quiet_NaN();
    const int r = static_cast<int>(rows);
    const int c = static_cast<int>(cols);
    const int pitch = static_cast<int>(channels);
    const int out_rows = plane ? (r + 3) / 2 : 1;
    const int out_cols = (c + 3) / 2;
    std::vector<T> details(static_cast<std::size_t>(out_rows) * out_cols);
    std::vector<T> line(plane ? cols : 0);
    for (int y = 0; y < out_rows; ++y) {
        if (plane) {
            for (int x = 0; x < c; ++x)
                line[x] = analysis_pair(image + channel + x * pitch, r, c * pitch, y, 4).second;
        }
        for (int x = 0; x < out_cols; ++x) {
            details[static_cast<std::size_t>(y) * out_cols + x] =
                analysis_pair(plane ? line.data() : image + channel, c, plane ? 1 : pitch, x, 4).second;
        }
    }
    return noise_median(std::move(details));
}

template<class T>
bool denoise_wavelet(const T* image, T* out, std::size_t rows, std::size_t cols,
                     std::size_t channels, bool plane, int length, int levels,
                     double sigma, bool automatic_sigma, bool soft, bool bayes) {
    if (!rows || !cols || !channels || channels > 16 || (!plane && rows != 1) ||
        rows > 1000000 || cols > 1000000 || rows * cols > 16000000 ||
        (length != 2 && length != 4) || levels < 0 || levels > 20 ||
        (!automatic_sigma && (!std::isfinite(sigma) || sigma < 0)) ||
        !finite(image, rows * cols * channels)) return false;
    std::vector<std::pair<int, int>> shapes;
    int r = static_cast<int>(rows), c = static_cast<int>(cols);
    for (int level = 0; level < levels; ++level) {
        if (c < length - 1 || (plane && r < length - 1)) return false;
        shapes.emplace_back(r, c);
        r = plane ? (r + length - 1) / 2 : 1;
        c = (c + length - 1) / 2;
    }
    if (levels == 0) {
        if (automatic_sigma) return false;
        std::copy_n(image, rows * cols * channels, out);
        return true;
    }
    for (std::size_t ch = 0; ch < channels; ++ch) {
        std::vector<WaveletBands<T>> stack;
        for (const auto& shape : shapes) {
            const T* input = stack.empty() ? image + ch : stack.back().data[0].data();
            const int stride = stack.empty() ? static_cast<int>(channels) : 1;
            stack.push_back(analyze(input, shape.first, shape.second, stride, plane, length));
        }
        const int size = plane ? 4 : 2;
        const double noise = automatic_sigma ? noise_median(stack.front().data[size - 1]) : sigma;
        if (!std::isfinite(noise) || (soft && noise == 0)) return false;
        const double universal = noise * std::sqrt(2 * std::log(static_cast<double>(rows * cols)));
        std::vector<T> result;
        for (std::size_t level = stack.size(); level-- > 0;) {
            auto& bands = stack[level];
            if (!result.empty()) bands.data[0] = std::move(result);
            for (int band = 1; band < size; ++band)
                threshold(bands.data[band], noise, universal, soft, bayes);
            result = synthesize(bands, shapes[level].first, shapes[level].second, plane, length);
        }
        for (std::size_t pixel = 0; pixel < result.size(); ++pixel)
            out[pixel * channels + ch] = result[pixel];
    }
    return true;
}

template<class T>
bool richardson_lucy(const T* image, const T* psf, T* out, std::size_t rows, std::size_t cols,
                     std::size_t pr, std::size_t pc, int iterations, bool clip, double epsilon) {
    if (!rows || !cols || !pr || !pc || std::max({rows, cols, pr, pc}) > 1000000 ||
        pr > rows || pc > cols || rows * cols > 16000000 || iterations < 0 ||
        !std::isfinite(epsilon) || epsilon < 0 ||
        !finite(image, rows * cols) || !finite(psf, pr * pc)) return false;
    // Near-zero and signed kernels amplify roundoff when the reference picks
    // direct convolution. Use the FFT route for the positive imaging domain.
    if (*std::min_element(image, image + rows * cols) < 0 ||
        *std::min_element(psf, psf + pr * pc) < 0 ||
        std::accumulate(psf, psf + pr * pc, 0.0) < 1e-8) return false;
    std::fill_n(out, rows * cols, T(0.5));
    if (iterations == 0) return true;
    const auto safe_divisor = [epsilon](T divisor) {
        return divisor >= static_cast<T>(1e-8) && (epsilon == 0 ||
               std::abs(divisor - static_cast<T>(epsilon)) >=
               T(16) * std::numeric_limits<T>::epsilon() * divisor);
    };
    const auto finish = [&] {
        if (!finite(out, rows * cols)) return false;
        if (clip) {
            for (std::size_t i = 0; i < rows * cols; ++i)
                out[i] = std::clamp(out[i], T(-1), T(1));
        }
        return true;
    };
    std::vector<T> vertical, horizontal;
    if (pr + pc <= 48 && separable_psf(psf, pr, pc, vertical, horizontal)) {
        std::vector<T> scratch(rows * cols), work(rows * cols);
        for (int i = 0; i < iterations; ++i) {
            convolve_separable(out, work.data(), scratch.data(), rows, cols, vertical, horizontal, false);
            for (std::size_t p = 0; p < rows * cols; ++p) {
                const T divisor = work[p] + static_cast<T>(1e-12);
                if (!safe_divisor(divisor)) return false;
                work[p] = divisor < static_cast<T>(epsilon) ? T(0) : image[p] / divisor;
            }
            convolve_separable(work.data(), work.data(), scratch.data(), rows, cols, vertical, horizontal, true);
            for (std::size_t p = 0; p < rows * cols; ++p) out[p] *= work[p];
        }
        return finish();
    }
    // The convolution tail can wrap into the discarded prefix. Only the
    // centered image extent must be free of circular aliasing.
    const std::size_t height = fft_extent(rows + pr / 2);
    const std::size_t width = fft_extent(cols + pc / 2);
    const std::vector<std::size_t> shape{height, width};
    const std::vector<std::size_t> axes{0, 1};
    const auto real_size = height * width;
    const auto spectrum_size = height * (width / 2 + 1);
    const int workers = detail::active_workers(real_size);
    std::vector<T> real(real_size);
    std::unique_ptr<T[]> convolved(new T[real_size]);
    std::array<std::vector<std::complex<T>>, 2> kernels{
        std::vector<std::complex<T>>(spectrum_size), std::vector<std::complex<T>>(spectrum_size)};
    std::vector<std::complex<T>> spectrum(spectrum_size);
    for (std::size_t row = 0; row < pr; ++row)
        std::copy_n(psf + row * pc, pc, real.data() + row * width);
    detail::fft_r2c(shape, axes, real.data(), kernels[0].data(), workers);
    for (std::size_t row = 0; row < pr; ++row)
        std::reverse_copy(psf + (pr - row - 1) * pc, psf + (pr - row) * pc, real.data() + row * width);
    detail::fft_r2c(shape, axes, real.data(), kernels[1].data(), workers);
    const auto first = ((pr - 1) / 2) * width + (pc - 1) / 2;
    auto convolve = [&](std::size_t kernel) {
        detail::fft_r2c(shape, axes, real.data(), spectrum.data(), workers);
        for (std::size_t i = 0; i < spectrum_size; ++i) spectrum[i] *= kernels[kernel][i];
        detail::fft_c2r(shape, axes, spectrum.data(), convolved.get(), workers);
    };
    std::fill(real.begin(), real.end(), T(0));
    for (int i = 0; i < iterations; ++i) {
        for (std::size_t row = 0; row < rows; ++row)
            std::copy_n(out + row * cols, cols, real.data() + row * width);
        convolve(0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                const T divisor = convolved[first + row * width + col] + static_cast<T>(1e-12);
                if (!safe_divisor(divisor)) return false;
                real[row * width + col] = divisor < static_cast<T>(epsilon) ?
                    T(0) : image[row * cols + col] / divisor;
            }
        }
        convolve(1);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col)
                out[row * cols + col] *= convolved[first + row * width + col];
        }
    }
    return finish();
}

template<class T>
bool inpaint_biharmonic(const T* image, const std::uint8_t* mask, T* out,
                        std::size_t rows, std::size_t cols, std::size_t channels) {
    if (!rows || !cols || !channels || channels > 16 || rows > 1000000 || cols > 1000000 ||
        rows * cols > 16000000 || !finite(image, rows * cols * channels)) return false;
    const auto count = rows * cols;
    std::vector<int> labels(count, -1);
    std::vector<std::size_t> unknown;
    for (std::size_t p = 0; p < count; ++p) {
        if (!mask[p]) continue;
        labels[p] = static_cast<int>(unknown.size());
        unknown.push_back(p);
    }
    if (unknown.size() == count || unknown.size() > 16384) return false;
    std::copy_n(image, count * channels, out);
    if (unknown.empty()) return true;

    // Applying the reflective grid Laplacian twice gives the biharmonic
    // stencil, including its different coefficients on each image border.
    const auto laplace = [=](std::size_t p, auto emit) {
        int degree = 0;
        if (p >= cols) { emit(p - cols, 1); ++degree; }
        if (p < count - cols) { emit(p + cols, 1); ++degree; }
        if (p % cols != 0) { emit(p - 1, 1); ++degree; }
        if (p % cols != cols - 1) { emit(p + 1, 1); ++degree; }
        emit(p, -degree);
    };
    const auto stencil = [&](std::size_t p, auto emit) {
        laplace(p, [&](std::size_t q, int a) {
            laplace(q, [&](std::size_t r, int b) { emit(r, a * b); });
        });
    };
    const auto size = unknown.size();
    std::size_t bandwidth = 0;
    for (std::size_t i = 0; i < size; ++i) {
        stencil(unknown[i], [&](std::size_t p, int) {
            if (labels[p] >= 0 && static_cast<std::size_t>(labels[p]) < i)
                bandwidth = std::max(bandwidth, i - static_cast<std::size_t>(labels[p]));
        });
    }
    if (size * (bandwidth + 1) * (bandwidth + 1) > 32000000) return false;
    const auto pitch = bandwidth + 1;
    std::vector<double> lower(size * pitch);
    std::vector<double> right(size * channels);
    std::vector<T> minimum(channels, std::numeric_limits<T>::infinity());
    std::vector<T> maximum(channels, -std::numeric_limits<T>::infinity());
    for (std::size_t p = 0; p < count; ++p) {
        if (mask[p]) continue;
        for (std::size_t c = 0; c < channels; ++c) {
            minimum[c] = std::min(minimum[c], image[p * channels + c]);
            maximum[c] = std::max(maximum[c], image[p * channels + c]);
        }
    }
    for (std::size_t i = 0; i < size; ++i) {
        stencil(unknown[i], [&](std::size_t p, int value) {
            const auto j = labels[p];
            if (j < 0) {
                for (std::size_t c = 0; c < channels; ++c)
                    right[i * channels + c] -= value * static_cast<double>(image[p * channels + c]);
            } else if (static_cast<std::size_t>(j) <= i) {
                lower[i * pitch + i - static_cast<std::size_t>(j)] += value;
            }
        });
    }

    // Symmetric positive definite principal submatrix. Factor it once and
    // reuse the banded factor for all channels.
    for (std::size_t i = 0; i < size; ++i) {
        const auto begin = i > bandwidth ? i - bandwidth : 0;
        for (std::size_t j = begin; j <= i; ++j) {
            double value = lower[i * pitch + i - j];
            for (auto k = begin; k < j; ++k)
                value -= lower[i * pitch + i - k] * lower[j * pitch + j - k];
            if (i == j && !(value > 0)) return false;
            lower[i * pitch + i - j] = i == j ? std::sqrt(value) : value / lower[j * pitch];
        }
        for (std::size_t c = 0; c < channels; ++c) {
            for (auto j = begin; j < i; ++j)
                right[i * channels + c] -= lower[i * pitch + i - j] * right[j * channels + c];
            right[i * channels + c] /= lower[i * pitch];
        }
    }
    for (std::size_t i = size; i-- > 0;) {
        const auto end = std::min(size, i + pitch);
        for (std::size_t c = 0; c < channels; ++c) {
            double result = right[i * channels + c];
            for (std::size_t j = i + 1; j < end; ++j)
                result -= lower[j * pitch + j - i] * right[j * channels + c];
            result /= lower[i * pitch];
            right[i * channels + c] = result;
            out[unknown[i] * channels + c] =
                std::clamp(static_cast<T>(result), minimum[c], maximum[c]);
        }
    }
    return true;
}

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double tau = 2 * pi;

int phase_increment(double difference) {
    return (difference < -pi) - (difference > pi);
}

struct PhaseEdge {
    std::uint32_t first;
    std::uint32_t second;
    double cost;
};

void sort_phase_edges(std::vector<PhaseEdge>& edges, std::ptrdiff_t left, std::ptrdiff_t right) {
    // Keep the reference partition's tie ordering. With equal-cost edges,
    // the choice of union root can change the global multiple of 2*pi.
    while (left < right) {
        std::array<double, 3> candidates{
            edges[left].cost, edges[left + (right - left) / 2].cost, edges[right].cost};
        if (candidates[1] < candidates[0]) std::swap(candidates[0], candidates[1]);
        if (candidates[2] < candidates[1]) std::swap(candidates[1], candidates[2]);
        if (candidates[1] < candidates[0]) std::swap(candidates[0], candidates[1]);
        double pivot = candidates[1];
        if (candidates[0] == candidates[1]) {
            pivot = candidates[2];
            if (candidates[1] == candidates[2]) {
                auto p = left + 1;
                while (p <= right && edges[p].cost == candidates[0]) ++p;
                if (p > right) return;
                pivot = std::max(edges[p].cost, candidates[0]);
            }
        }
        auto a = left;
        auto b = right;
        while (a <= b) {
            while (edges[a].cost < pivot) ++a;
            while (edges[b].cost >= pivot) --b;
            if (a < b) std::swap(edges[a++], edges[b--]);
        }
        if (a - left < right - a + 1) {
            sort_phase_edges(edges, left, a - 1);
            left = a;
        } else {
            sort_phase_edges(edges, a, right);
            right = a - 1;
        }
    }
}

class PhaseGroups {
public:
    explicit PhaseGroups(std::size_t count) : parent(count), sizes(count, 1), offset(count) {
        std::iota(parent.begin(), parent.end(), std::size_t{0});
    }

    std::size_t root(std::size_t p) {
        if (parent[p] == p) return p;
        const auto old = parent[p];
        parent[p] = root(old);
        offset[p] += offset[old];
        return parent[p];
    }

    bool join(std::size_t a, std::size_t b, int jump) {
        const auto u = root(a);
        const auto v = root(b);
        if (u == v) return false;
        const int delta = offset[a] - jump - offset[b];
        if (sizes[v] == 1 || sizes[u] > sizes[v]) {
            parent[v] = u;
            sizes[u] += sizes[v];
            offset[v] = delta;
        } else {
            parent[u] = v;
            sizes[v] += sizes[u];
            offset[u] = -delta;
        }
        return true;
    }

    int increment(std::size_t p) {
        root(p);
        return offset[p];
    }

    std::size_t group_size(std::size_t p) {
        return sizes[root(p)];
    }

private:
    std::vector<std::size_t> parent;
    std::vector<std::size_t> sizes;
    std::vector<int> offset;
};

// Local copy of the default seed's stream. No mutation of the C library's
// process-global RNG and no dependency on results from previous calls.
class BoundaryCosts {
public:
    BoundaryCosts() {
#if defined(__GLIBC__)
        state[0] = 1;
        for (std::size_t i = 1; i < state.size(); ++i)
            state[i] = static_cast<std::uint32_t>((16807ULL * state[i - 1]) % 2147483647ULL);
        for (int i = 0; i < 310; ++i) next();
#endif
    }

    double next() {
#if defined(_WIN32)
        seed = seed * 214013u + 2531011u;
        return static_cast<double>((seed >> 16) & 32767u);
#elif defined(__GLIBC__)
        const auto lag = (index + 28) % state.size();
        state[index] += state[lag];
        const auto result = state[index] >> 1;
        index = (index + 1) % state.size();
        return static_cast<double>(result);
#else
        return 0;
#endif
    }

private:
#if defined(_WIN32)
    std::uint32_t seed = 0;
#elif defined(__GLIBC__)
    std::array<std::uint32_t, 31> state{};
    std::size_t index = 3;
#endif
};

} // namespace

template<class T>
bool unwrap_phase(const T* image, double* out, std::size_t depth, std::size_t rows,
                  std::size_t cols, int rank, bool deterministic_boundary) {
    if (!depth || !rows || !cols || std::max({depth, rows, cols}) > 1000000 ||
        depth * rows > 16000000 / cols ||
        rank < 1 || rank > 3 || (rank < 3 && depth != 1) || (rank < 2 && rows != 1))
        return false;
    const auto plane = cols * rows;
    const auto size = plane * depth;
    if (!finite(image, size)) return false;
    if (rank == 1) {
        int increment = 0;
        out[0] = image[0];
        for (std::size_t i = 1; i < size; ++i) {
            increment -= phase_increment(static_cast<double>(image[i - 1]) - image[i]);
            out[i] = image[i] + tau * increment;
        }
        return true;
    }
    if (rows < 2 || cols < 2 || (rank == 3 && depth < 2)) return false;
    const auto diff = [&](std::size_t a, std::size_t b) {
        return static_cast<double>(image[a]) - image[b];
    };
    const std::array<std::size_t, 3> strides{1, cols, plane};
    const std::array<std::size_t, 3> extents{cols, rows, depth};
    bool has_jump = false;
    const auto each_edge = [&](int axis, const auto& visit) {
        const auto stride = strides[axis];
        const auto slab = stride * extents[axis];
        for (std::size_t begin = 0; begin < size; begin += slab) {
            for (auto p = begin; p < begin + slab - stride; ++p) {
                if (!visit(p, p + stride)) return false;
            }
        }
        return true;
    };
    for (int axis = 0; axis < rank && !has_jump; ++axis) {
        has_jump = !each_edge(axis, [&](std::size_t p, std::size_t q) {
            return std::abs(diff(p, q)) <= pi;
        });
    }
    if (!has_jump) {
        std::copy_n(image, size, out);
        return true;
    }
#if !defined(_WIN32) && !defined(__GLIBC__)
    return false;
#endif
    if (!deterministic_boundary || size > 1000000) return false;
    for (std::size_t p = 0; p < size; ++p)
        if (std::abs(static_cast<double>(image[p])) > pi + 2e-7) return false;
    using Index = std::ptrdiff_t;
    const auto width = static_cast<Index>(cols);
    const auto step = static_cast<Index>(plane);
    std::vector<Index> directions{1, width};
    if (rank == 3) directions.push_back(step);
    directions.push_back(width + 1);
    directions.push_back(width - 1);
    if (rank == 3) {
        for (Index y = -1; y <= 1; ++y) {
            for (Index x = -1; x <= 1; ++x) {
                if (y != 0 || x != 0) directions.push_back(step + y * width + x);
            }
        }
    }
    BoundaryCosts boundary;
    std::vector<double> costs(size);
    for (std::size_t p = 0; p < size; ++p) {
        const auto x = p % cols;
        const auto y = p / cols % rows;
        const auto z = p / plane;
        const double random_cost = boundary.next();
        if (x == 0 || x + 1 == cols || y == 0 || y + 1 == rows ||
            (rank == 3 && (z == 0 || z + 1 == depth))) {
            costs[p] = random_cost;
            continue;
        }
        double cost = 0;
        for (auto d : directions) {
            const auto index = static_cast<Index>(p);
            const double before = diff(static_cast<std::size_t>(index - d), p);
            const double after = diff(p, static_cast<std::size_t>(index + d));
            const double second = (before + tau * phase_increment(before)) -
                                  (after + tau * phase_increment(after));
            cost += second * second;
        }
        costs[p] = cost;
    }
    std::vector<PhaseEdge> edges;
    edges.reserve(size * rank);
    for (int axis = 0; axis < rank; ++axis) {
        each_edge(axis, [&](std::size_t p, std::size_t q) {
            edges.push_back({static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(q),
                             costs[p] + costs[q]});
            return true;
        });
    }

    // An integrable field has a unique relative phase everywhere. If a
    // uniquely ordered prefix grows a strict majority, subsequent unions
    // cannot change its root, so the costly tail need not be ordered.
    if (rank == 2 && size >= 1024) {
        std::vector<int> increments(size);
        for (std::size_t p = 1; p < size; ++p) {
            const auto q = p % cols ? p - 1 : p - cols;
            increments[p] = increments[q] - phase_increment(diff(q, p));
        }
        const bool consistent = each_edge(1, [&](std::size_t p, std::size_t q) {
            return increments[p] - increments[q] == phase_increment(diff(p, q));
        });
        if (consistent) {
            std::array<double, 64> sample{};
            for (std::size_t i = 0; i < sample.size(); ++i)
                sample[i] = edges[i * edges.size() / sample.size()].cost;
            const auto pivot = sample.begin() + 48;
            std::nth_element(sample.begin(), pivot, sample.end());
            std::vector<PhaseEdge> prefix;
            prefix.reserve(edges.size());
            for (const auto& edge : edges)
                if (edge.cost < *pivot) prefix.push_back(edge);
            std::sort(prefix.begin(), prefix.end(), [](const PhaseEdge& a, const PhaseEdge& b) {
                return a.cost < b.cost;
            });
            PhaseGroups groups(size);
            for (std::size_t i = 0; i < prefix.size(); ++i) {
                const auto& edge = prefix[i];
                if (i + 1 < prefix.size() && edge.cost == prefix[i + 1].cost) break;
                if (groups.join(edge.first, edge.second, phase_increment(diff(edge.first, edge.second))) &&
                    groups.group_size(edge.first) > size / 2) {
                    const int origin = groups.increment(edge.first) - increments[edge.first];
                    for (std::size_t p = 0; p < size; ++p)
                        out[p] = image[p] + tau * (origin + increments[p]);
                    return true;
                }
            }
        }
    }
    sort_phase_edges(edges, 0, static_cast<Index>(edges.size()) - 1);
    PhaseGroups groups(size);
    auto components = size;
    for (const auto& edge : edges) {
        if (groups.join(edge.first, edge.second, phase_increment(diff(edge.first, edge.second))) &&
            --components == 1) break;
    }
    for (std::size_t p = 0; p < size; ++p)
        out[p] = image[p] + tau * groups.increment(p);
    return true;
}

#define SKIMAGEXX_RESTORATION_INSTANTIATE(T) \
    template bool denoise_nl_means(const T*, T*, std::size_t, std::size_t, std::size_t, std::size_t, bool, int, int, double, double); \
    template double estimate_sigma(const T*, std::size_t, std::size_t, std::size_t, std::size_t, bool); \
    template bool denoise_wavelet(const T*, T*, std::size_t, std::size_t, std::size_t, bool, int, int, double, bool, bool, bool); \
    template bool inpaint_biharmonic(const T*, const std::uint8_t*, T*, std::size_t, std::size_t, std::size_t); \
    template bool unwrap_phase(const T*, double*, std::size_t, std::size_t, std::size_t, int, bool); \
    template bool richardson_lucy(const T*, const T*, T*, std::size_t, std::size_t, std::size_t, std::size_t, int, bool, double)

SKIMAGEXX_RESTORATION_INSTANTIATE(float);
SKIMAGEXX_RESTORATION_INSTANTIATE(double);
#undef SKIMAGEXX_RESTORATION_INSTANTIATE

} // namespace skimagexx
