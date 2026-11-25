#include "skimagexx/quickshift.hpp"
#include "skimagexx/image.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(SKIMAGEXX_AVX2) && (defined(__AVX2__) || (defined(__GNUC__) && !defined(_MSC_VER)) || (defined(__clang__) && !defined(_MSC_VER)))
#include <immintrin.h>
#define QUICKSHIFT_AVX2 1
#if defined(__AVX2__)
#define QUICKSHIFT_TARGET_AVX2
#else
#define QUICKSHIFT_TARGET_AVX2 __attribute__((target("avx2")))
#endif
#if defined(__GNUC__) || defined(__clang__)
#define QUICKSHIFT_INLINE inline __attribute__((always_inline))
#else
#define QUICKSHIFT_INLINE __forceinline
#endif
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#elif defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

namespace skimagexx {
namespace {

using Index = std::ptrdiff_t;

template<class T, Index fixed_channels>
T distance(const T* a, const T* b, T row_squared, T col_squared, Index pixels, Index channels) {
    T result = 0;
    const auto count = fixed_channels ? fixed_channels : channels;
    for (Index channel = 0; channel < count; ++channel) {
        const T difference = a[channel * pixels] - b[channel * pixels];
        result += difference * difference;
    }
    result += row_squared;
    result += col_squared;
    return result;
}

template<class T>
void add_weight(T& density, T& other_density, double weight) {
    density = static_cast<T>(density + weight);
    other_density = static_cast<T>(other_density + weight);
}

#ifdef QUICKSHIFT_AVX2

unsigned first_lane(unsigned mask) {
#if defined(_MSC_VER)
    unsigned long lane;
    _BitScanForward(&lane, mask);
    return lane;
#else
    return __builtin_ctz(mask);
#endif
}

// For n = round(x / (log(2) / 16)), reduce to r in [-log(2)/32, log(2)/32].
// The table holds the rounded 2**(k/16) and the relative rounding error,
// where k = n modulo 16. Reconstruct with head + head * (tail + expm1(r)).
QUICKSHIFT_TARGET_AVX2 __m256d exp_pd(__m256d x) {
    alignas(32) static constexpr double heads[] = {
        0x1p+0, 0x1.0b5586cf9890fp+0, 0x1.172b83c7d517bp+0, 0x1.2387a6e756238p+0,
        0x1.306fe0a31b715p+0, 0x1.3dea64c123422p+0, 0x1.4bfdad5362a27p+0, 0x1.5ab07dd485429p+0,
        0x1.6a09e667f3bcdp+0, 0x1.7a11473eb0187p+0, 0x1.8ace5422aa0dbp+0, 0x1.9c49182a3f090p+0,
        0x1.ae89f995ad3adp+0, 0x1.c199bdd85529cp+0, 0x1.d5818dcfba487p+0, 0x1.ea4afa2a490dap+0,
    };
    alignas(32) static constexpr double tails[] = {
        0x0p+0, 0x1.79aa65d837b6dp-54, -0x1.01b15eaa59348p-55, 0x1.68efde3a8a894p-54,
        0x1.34d754db0abb6p-55, 0x1.59f48a72a4c6dp-55, 0x1.690cebb7aafb0p-56, 0x1.063e1e21c5409p-54,
        -0x1.3b3efbf5e2228p-54, -0x1.b32dcb94da51dp-56, 0x1.db72fc1f0eab4p-55, 0x1.1affc2b91ce27p-56,
        0x1.c1a7792cb3387p-55, 0x1.36eae30af0cb3p-56, 0x1.4a385a63d07a7p-56, -0x1.ff7128fd391f0p-55,
    };

    // Leave the underflow range and -infinity to the scalar library.
    const auto normal = _mm256_cmp_pd(x, _mm256_set1_pd(-700), _CMP_GE_OQ);
    const auto reduced_input = _mm256_and_pd(x, normal);
    const auto n = _mm256_round_pd(_mm256_mul_pd(reduced_input, _mm256_set1_pd(0x1.71547652b82fep+4)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto k = _mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(n));
    const auto index = _mm256_and_si256(k, _mm256_set1_epi64x(15));
    const auto head = _mm256_i64gather_pd(heads, index, 8);
    const auto tail = _mm256_i64gather_pd(tails, index, 8);
    const auto r = _mm256_sub_pd(
        _mm256_sub_pd(reduced_input, _mm256_mul_pd(n, _mm256_set1_pd(0x1.62e42fefa0000p-5))),
        _mm256_mul_pd(n, _mm256_set1_pd(0x1.cf79abc9e3b3ap-44)));

    // Degree-five minimax fit to (exp(r) - 1 - r) / r**2 on the reduced
    // interval. Its absolute approximation error is below 8.1e-17.
    auto p = _mm256_set1_pd(0x1.a01b65093a453p-13);
    p = _mm256_add_pd(_mm256_set1_pd(0x1.6c17ed4cc84ebp-10), _mm256_mul_pd(p, r));
    p = _mm256_add_pd(_mm256_set1_pd(0x1.11111110abadcp-7), _mm256_mul_pd(p, r));
    p = _mm256_add_pd(_mm256_set1_pd(0x1.55555554e9467p-5), _mm256_mul_pd(p, r));
    p = _mm256_add_pd(_mm256_set1_pd(0x1.5555555555557p-3), _mm256_mul_pd(p, r));
    p = _mm256_add_pd(_mm256_set1_pd(0x1.0000000000001p-1), _mm256_mul_pd(p, r));
    const auto expm1 = _mm256_add_pd(r, _mm256_mul_pd(_mm256_mul_pd(r, r), p));
    const auto value = _mm256_add_pd(head, _mm256_mul_pd(head, _mm256_add_pd(tail, expm1)));
    const auto adjust = _mm256_slli_epi64(_mm256_and_si256(k, _mm256_set1_epi64x(-16)), 48);
    const auto result = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(value), adjust));

    if (_mm256_movemask_pd(normal) == 15) return result;
    alignas(32) double arguments[4], weights[4];
    _mm256_store_pd(arguments, x);
    _mm256_store_pd(weights, result);
    for (int lane = 0; lane < 4; ++lane)
        if (!(arguments[lane] >= -700)) weights[lane] = std::exp(arguments[lane]);
    return _mm256_load_pd(weights);
}

QUICKSHIFT_TARGET_AVX2 void exp_ps(__m256 x, double* weights) {
    // The float32 reference promotes each exponent and adds its weight in
    // double precision before storing density back to float.
    _mm256_store_pd(weights, exp_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x))));
    _mm256_store_pd(weights + 4, exp_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1))));
}

template<class T> struct Vector;

template<> struct Vector<double> {
    using Pack = __m256d;
    static constexpr Index width = 4;
    static QUICKSHIFT_TARGET_AVX2 Pack load(const double* p) { return _mm256_loadu_pd(p); }
    static QUICKSHIFT_TARGET_AVX2 Pack set(double value) { return _mm256_set1_pd(value); }
    static QUICKSHIFT_TARGET_AVX2 Pack add(Pack a, Pack b) { return _mm256_add_pd(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack sub(Pack a, Pack b) { return _mm256_sub_pd(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack mul(Pack a, Pack b) { return _mm256_mul_pd(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack minimum(Pack a, Pack b) { return _mm256_min_pd(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack minimum(Pack x) {
        x = minimum(x, _mm256_permute2f128_pd(x, x, 1));
        return minimum(x, _mm256_permute_pd(x, 5));
    }
    static QUICKSHIFT_TARGET_AVX2 Pack eligible(Pack x, Pack best, Pack densities, Pack density) {
        return _mm256_and_pd(_mm256_cmp_pd(x, best, _CMP_LT_OQ),
                             _mm256_cmp_pd(densities, density, _CMP_GT_OQ));
    }
    static QUICKSHIFT_TARGET_AVX2 Pack select(Pack best, Pack x, Pack mask) {
        return _mm256_blendv_pd(best, x, mask);
    }
    static QUICKSHIFT_TARGET_AVX2 unsigned matches(Pack x, Pack best, Pack eligible) {
        return _mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(x, best, _CMP_EQ_OQ), eligible));
    }
    static QUICKSHIFT_TARGET_AVX2 double first(Pack x) { return _mm256_cvtsd_f64(x); }
};

template<> struct Vector<float> {
    using Pack = __m256;
    static constexpr Index width = 8;
    static QUICKSHIFT_TARGET_AVX2 Pack load(const float* p) { return _mm256_loadu_ps(p); }
    static QUICKSHIFT_TARGET_AVX2 Pack set(float value) { return _mm256_set1_ps(value); }
    static QUICKSHIFT_TARGET_AVX2 Pack add(Pack a, Pack b) { return _mm256_add_ps(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack sub(Pack a, Pack b) { return _mm256_sub_ps(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack mul(Pack a, Pack b) { return _mm256_mul_ps(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack minimum(Pack a, Pack b) { return _mm256_min_ps(a, b); }
    static QUICKSHIFT_TARGET_AVX2 Pack minimum(Pack x) {
        x = minimum(x, _mm256_permute2f128_ps(x, x, 1));
        x = minimum(x, _mm256_permute_ps(x, 0x4e));
        return minimum(x, _mm256_permute_ps(x, 0xb1));
    }
    static QUICKSHIFT_TARGET_AVX2 Pack eligible(Pack x, Pack best, Pack densities, Pack density) {
        return _mm256_and_ps(_mm256_cmp_ps(x, best, _CMP_LT_OQ),
                             _mm256_cmp_ps(densities, density, _CMP_GT_OQ));
    }
    static QUICKSHIFT_TARGET_AVX2 Pack select(Pack best, Pack x, Pack mask) {
        return _mm256_blendv_ps(best, x, mask);
    }
    static QUICKSHIFT_TARGET_AVX2 unsigned matches(Pack x, Pack best, Pack eligible) {
        return _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(x, best, _CMP_EQ_OQ), eligible));
    }
    static QUICKSHIFT_TARGET_AVX2 float first(Pack x) { return _mm256_cvtss_f32(x); }
};

template<class T, Index fixed_channels>
QUICKSHIFT_TARGET_AVX2 typename Vector<T>::Pack distances(
        const T* a, const T* b, T row_squared, const T* col_squared,
        Index pixels, Index channels) {
    using V = Vector<T>;
    auto result = V::set(0);
    const auto count = fixed_channels ? fixed_channels : channels;
    for (Index channel = 0; channel < count; ++channel) {
        const auto difference = V::sub(V::set(a[channel * pixels]), V::load(b + channel * pixels));
        result = V::add(result, V::mul(difference, difference));
    }
    return V::add(V::add(result, V::set(row_squared)), V::load(col_squared));
}

QUICKSHIFT_TARGET_AVX2 QUICKSHIFT_INLINE void add_density_lanes(__m256d x, double& density, double* neighbors) {
    const auto weight = exp_pd(x);
    const auto approximate = _mm256_cmp_pd(x, _mm256_set1_pd(-700), _CMP_GE_OQ);
    const auto step = _mm256_and_si256(_mm256_castpd_si256(approximate), _mm256_set1_epi64x(1));
    const auto bits = _mm256_castpd_si256(weight);
    const auto low = _mm256_castsi256_pd(_mm256_sub_epi64(bits, step));
    const auto high = _mm256_castsi256_pd(_mm256_add_epi64(bits, step));
    const auto previous = _mm256_loadu_pd(neighbors);
    const auto next = _mm256_add_pd(previous, low);
    const auto same_neighbors = _mm256_movemask_pd(
        _mm256_cmp_pd(next, _mm256_add_pd(previous, high), _CMP_EQ_OQ));
    alignas(32) double arguments[4], lows[4], highs[4], next_neighbors[4];
    _mm256_store_pd(arguments, x);
    _mm256_store_pd(lows, low);
    _mm256_store_pd(highs, high);
    _mm256_store_pd(next_neighbors, next);
    for (unsigned lane = 0; lane < 4; ++lane) {
        const auto next_density = density + lows[lane];
        // Both sums must round the same way across the approximation's
        // error interval. Advance the center one lane at a time.
        if ((same_neighbors & (1u << lane)) && next_density == density + highs[lane]) {
            density = next_density;
            neighbors[lane] = next_neighbors[lane];
        } else {
            add_weight(density, neighbors[lane], std::exp(arguments[lane]));
        }
    }
}

QUICKSHIFT_TARGET_AVX2 QUICKSHIFT_INLINE void add_density_lanes(__m256 x, float& density, float* neighbors) {
    alignas(32) float arguments[8];
    alignas(32) double weights[8];
    _mm256_store_ps(arguments, x);
    exp_ps(x, weights);
    for (Index lane = 0; lane < 8; ++lane) {
        double weight = weights[lane];
        auto& neighbor = neighbors[lane];
        // Recompute with the scalar library when an adjacent weight
        // could change either rounded sum and therefore a density tie.
        if (arguments[lane] < 0 && arguments[lane] >= -700) {
            std::uint64_t bits;
            std::memcpy(&bits, &weight, sizeof(bits));
            const auto low_bits = bits - 1, high_bits = bits + 1;
            double low, high;
            std::memcpy(&low, &low_bits, sizeof(low));
            std::memcpy(&high, &high_bits, sizeof(high));
            if (static_cast<float>(density + low) != static_cast<float>(density + high) ||
                static_cast<float>(neighbor + low) != static_cast<float>(neighbor + high)) {
                weight = std::exp(static_cast<double>(arguments[lane]));
            }
        }
        add_weight(density, neighbor, weight);
    }
}

template<class T, Index fixed_channels>
QUICKSHIFT_TARGET_AVX2 Index density_row(
        const T* a, const T* b, T& density, T* neighbors, T row_squared,
        const T* col_squared, Index count, Index pixels, Index channels, T inverse) {
    using V = Vector<T>;
    Index i = 0;
    for (; i + V::width <= count; i += V::width) {
        const auto x = V::mul(distances<T, fixed_channels>(
            a, b + i, row_squared, col_squared + i, pixels, channels), V::set(inverse));
        add_density_lanes(x, density, neighbors + i);
    }
    return i;
}

template<class T, Index fixed_channels>
QUICKSHIFT_TARGET_AVX2 Index parent_row(
        const T* a, const T* b, const T* neighbors, T density, T row_squared,
        const T* col_squared, Index count, Index pixels, Index channels, Index start,
        T& closest, Index& parent) {
    using V = Vector<T>;
    const auto center_density = V::set(density);
    auto best = V::set(closest);
    Index i = 0;
    for (; i + V::width <= count; i += V::width) {
        const auto d = distances<T, fixed_channels>(a, b + i, row_squared, col_squared + i, pixels, channels);
        const auto valid = V::eligible(d, best, V::load(neighbors + i), center_density);
        best = V::minimum(V::select(best, d, valid));
        const auto choice = V::matches(d, best, valid);
        const auto lane = first_lane(choice | (1u << V::width));
        parent = choice ? start + i + lane : parent;
    }
    closest = V::first(best);
    return i;
}

#endif

template<class T, Index fixed_channels, bool vectorized>
void segment(const T* image, const T* noise, Index* labels,
             Index height, Index width, Index channels, T kernel_size, T max_distance) {
    const auto count = height * width;
    const auto radius = static_cast<Index>(std::min(
        std::ceil(static_cast<double>(T{3} * kernel_size)),
        static_cast<double>(std::max(height, width))));
    const T inverse = static_cast<T>(-0.5 / (kernel_size * kernel_size));
    std::vector<T> channel_planes(count * channels);
    for (Index channel = 0; channel < channels; ++channel) {
        for (Index pixel = 0; pixel < count; ++pixel)
            channel_planes[channel * count + pixel] = image[pixel * channels + channel];
    }
    const auto* planes = channel_planes.data();
    std::vector<T> densities(noise, noise + count);
    std::vector<T> squared_offsets(2 * radius + 1);
    T* const squared = squared_offsets.data() + radius;
    for (Index offset = -radius; offset <= radius; ++offset) {
        const T difference = static_cast<T>(offset);
        squared[offset] = difference * difference;
    }

    // Each pair contributes to both densities. Traversing the upper triangle
    // in row-major order preserves the summation order at every pixel.
    for (Index row = 0; row < height; ++row) {
        const auto bottom = std::min(height, row + radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            auto& density = densities[pixel];
            density += T{1};
            const auto* values = planes + pixel;
            const auto left = std::max(Index{0}, col - radius);
            const auto right = std::min(width, col + radius + 1);
            for (auto other_row = row; other_row < bottom; ++other_row) {
                const auto start = other_row == row ? col + 1 : left;
                const auto row_squared = squared[other_row - row];
                const auto other = other_row * width + start;
                Index i = 0;
#ifdef QUICKSHIFT_AVX2
                if constexpr (vectorized) {
                    i = density_row<T, fixed_channels>(values, planes + other, density,
                        densities.data() + other, row_squared, squared + (start - col),
                        right - start, count, channels, inverse);
                }
#endif
                for (; i < right - start; ++i) {
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, planes + other + i, row_squared,
                        squared[start + i - col], count, channels);
                    const auto weight = std::exp(static_cast<double>(squared_distance * inverse));
                    add_weight(density, densities[other + i], weight);
                }
            }
        }
    }

    const auto cutoff = static_cast<double>(max_distance);
    const auto square_radius = static_cast<Index>(
        std::min(static_cast<double>(radius), std::floor(cutoff) + 1));
    // Widen the disk to account for rounding in the two spatial squares,
    // their sum, and the final square root cast to T. Color terms can only
    // increase the squared distance.
    const double widened_cutoff = std::nextafter(
        cutoff * (1 + 4 * static_cast<double>(std::numeric_limits<T>::epsilon())),
        std::numeric_limits<double>::infinity());
    const double cutoff_radius = std::min(2 * static_cast<double>(square_radius), widened_cutoff);
    const auto search_radius = static_cast<Index>(
        std::min(static_cast<double>(square_radius), std::floor(cutoff_radius)));
    std::vector<Index> col_radii(search_radius + 1);
    for (Index dy = 0; dy <= search_radius; ++dy) {
        const auto y = static_cast<double>(dy);
        col_radii[dy] = std::min(search_radius, static_cast<Index>(
            std::sqrt(cutoff_radius * cutoff_radius - y * y)));
    }
    for (Index row = 0; row < height; ++row) {
        const auto top = std::max(Index{0}, row - search_radius);
        const auto bottom = std::min(height, row + search_radius + 1);
        for (Index col = 0; col < width; ++col) {
            const auto pixel = row * width + col;
            const auto* values = planes + pixel;
            const auto density = densities[pixel];
            T closest = std::numeric_limits<T>::infinity();
            auto parent = pixel;
            for (auto other_row = top; other_row < bottom; ++other_row) {
                const auto dy = std::abs(other_row - row);
                const auto row_squared = squared[dy];
                if (row_squared >= closest) continue;
                const auto col_radius = col_radii[dy];
                const auto left = std::max(Index{0}, col - col_radius);
                const auto right = std::min(width, col + col_radius + 1);
                const auto other = other_row * width + left;
                Index i = 0;
#ifdef QUICKSHIFT_AVX2
                if constexpr (vectorized) {
                    i = parent_row<T, fixed_channels>(values, planes + other,
                        densities.data() + other, density, row_squared, squared + (left - col),
                        right - left, count, channels, other, closest, parent);
                }
#endif
                for (; i < right - left; ++i) {
                    if (densities[other + i] <= density) continue;
                    const auto squared_distance = distance<T, fixed_channels>(
                        values, planes + other + i, row_squared,
                        squared[left + i - col], count, channels);
                    if (squared_distance < closest) {
                        closest = squared_distance;
                        parent = other + i;
                    }
                }
            }
            labels[pixel] = static_cast<T>(std::sqrt(static_cast<double>(closest))) <= max_distance
                            ? parent : pixel;
        }
    }

    for (Index pixel = 0; pixel < count; ++pixel) {
        auto root = pixel;
        while (labels[root] != root) root = labels[root];
        auto node = pixel;
        while (labels[node] != root) {
            const auto next_node = labels[node];
            labels[node] = root;
            node = next_node;
        }
    }
    // Encode labels as negative values until all compressed root indices have
    // been resolved. Numbering the roots in pixel order preserves tie order.
    Index next = 0;
    for (Index pixel = 0; pixel < count; ++pixel) {
        if (labels[pixel] == pixel)
            labels[pixel] = -1 - next++;
    }
    for (Index pixel = 0; pixel < count; ++pixel) {
        if (labels[pixel] >= 0)
            labels[pixel] = labels[labels[pixel]];
    }
    for (Index pixel = 0; pixel < count; ++pixel)
        labels[pixel] = -1 - labels[pixel];
}

template<bool vectorized, class T>
void dispatch_channels(const T* image, const T* noise, Index* labels,
                       Index height, Index width, Index channels, T kernel_size, T max_distance) {
    switch (channels) {
    case 1:
        segment<T, 1, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    case 3:
        segment<T, 3, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        break;
    default:
        segment<T, 0, vectorized>(image, noise, labels, height, width, channels, kernel_size, max_distance);
    }
}

} // namespace

template<class T>
void quickshift_into(const T* image, const T* noise, Index* labels,
                     Index height, Index width, Index channels, T kernel_size, T max_distance) {
#ifdef QUICKSHIFT_AVX2
    if (has_avx2() && std::fegetround() == FE_TONEAREST) {
        dispatch_channels<true>(image, noise, labels, height, width, channels, kernel_size, max_distance);
        return;
    }
#endif
    dispatch_channels<false>(image, noise, labels, height, width, channels, kernel_size, max_distance);
}

template void quickshift_into(const float*, const float*, Index*, Index, Index, Index, float, float);
template void quickshift_into(const double*, const double*, Index*, Index, Index, Index, double, double);

} // namespace skimagexx

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif
