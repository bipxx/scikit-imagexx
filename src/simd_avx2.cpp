#include "simd.hpp"
#include "simd_math.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <limits>
#include <iterator>
#include <vector>

namespace skimagexx::simd {
std::ptrdiff_t felzenszwalb_rgb_cost_row(const double* a, const double* b,
                                       double* out, std::ptrdiff_t cols) {
    const auto squared_difference = [](const double* first, const double* second) {
        const auto difference = _mm256_sub_pd(_mm256_loadu_pd(first), _mm256_loadu_pd(second));
        return _mm256_mul_pd(difference, difference);
    };
    const auto end = cols - cols % 4;
    for (std::ptrdiff_t x = 0; x < end; x += 4, a += 12, b += 12) {
        // Three contiguous vectors hold the squared differences of four RGB
        // pixels. Assemble channels before adding in scalar channel order.
        const auto first = squared_difference(a, b);
        const auto middle = squared_difference(a + 4, b + 4);
        const auto last = squared_difference(a + 8, b + 8);
        const auto r = _mm256_blend_pd(_mm256_blend_pd(
            _mm256_permute4x64_pd(first, _MM_SHUFFLE(0, 0, 3, 0)),
            _mm256_permute4x64_pd(middle, _MM_SHUFFLE(0, 2, 0, 0)), 0x4),
            _mm256_permute4x64_pd(last, _MM_SHUFFLE(1, 0, 0, 0)), 0x8);
        const auto g = _mm256_blend_pd(_mm256_blend_pd(
            _mm256_permute4x64_pd(first, _MM_SHUFFLE(0, 0, 0, 1)),
            _mm256_permute4x64_pd(middle, _MM_SHUFFLE(0, 3, 0, 0)), 0x6),
            _mm256_permute4x64_pd(last, _MM_SHUFFLE(2, 0, 0, 0)), 0x8);
        const auto blue = _mm256_blend_pd(_mm256_blend_pd(
            _mm256_permute4x64_pd(first, _MM_SHUFFLE(0, 0, 0, 2)),
            _mm256_permute4x64_pd(middle, _MM_SHUFFLE(0, 0, 1, 0)), 0x2),
            _mm256_permute4x64_pd(last, _MM_SHUFFLE(3, 0, 0, 0)), 0xc);
        const auto sum = _mm256_add_pd(_mm256_add_pd(r, g), blue);
        _mm256_storeu_pd(out + x, _mm256_sqrt_pd(sum));
    }
    return end;
}

std::ptrdiff_t felzenszwalb_gray_cost_row(const double* a, const double* b,
                                         double* out, std::ptrdiff_t cols, bool paired) {
    const auto squared_difference = [](const double* first, const double* second) {
        const auto delta = _mm256_sub_pd(_mm256_loadu_pd(first), _mm256_loadu_pd(second));
        return _mm256_mul_pd(delta, delta);
    };
    const auto end = cols - cols % 4;
    if (paired) {
        for (std::ptrdiff_t x = 0; x < end; x += 4, a += 8, b += 8) {
            const auto first = squared_difference(a, b);
            const auto second = squared_difference(a + 4, b + 4);
            const auto sums = _mm256_permute4x64_pd(_mm256_hadd_pd(first, second),
                                                    _MM_SHUFFLE(3, 1, 2, 0));
            _mm256_storeu_pd(out + x, _mm256_sqrt_pd(sums));
        }
    } else {
        for (std::ptrdiff_t x = 0; x < end; x += 4)
            _mm256_storeu_pd(out + x, _mm256_sqrt_pd(squared_difference(a + x, b + x)));
    }
    return end;
}

namespace {
template<int Channels, bool Scale>
std::ptrdiff_t slic_assign_float_row(const float* image, const float* center, const float* x_distances,
                                     float* distances, std::int32_t* nearest, std::ptrdiff_t length,
                                     std::ptrdiff_t channels, std::ptrdiff_t pixels, std::int32_t label,
                                     float color_scale, float spatial_yz,
                                     float spatial_weight, bool& change) {
    const auto yz = _mm256_set1_ps(spatial_yz);
    const auto weight = _mm256_set1_ps(spatial_weight);
    const auto scale = _mm256_set1_ps(color_scale);
    const auto labels = _mm256_set1_epi32(label);
    const auto c0 = _mm256_set1_ps(center[0]);
    const auto c1 = Channels == 3 ? _mm256_set1_ps(center[1]) : _mm256_setzero_ps();
    const auto c2 = Channels == 3 ? _mm256_set1_ps(center[2]) : _mm256_setzero_ps();
    std::ptrdiff_t x = 0;
    int changed = 0;
    for (; x + 8 <= length; x += 8) {
        const auto d0 = _mm256_sub_ps(_mm256_loadu_ps(image + x), c0);
        auto color = _mm256_mul_ps(d0, d0);
        if constexpr (Channels == 3) {
            const auto d1 = _mm256_sub_ps(_mm256_loadu_ps(image + pixels + x), c1);
            const auto d2 = _mm256_sub_ps(_mm256_loadu_ps(image + 2 * pixels + x), c2);
            color = _mm256_add_ps(_mm256_add_ps(color, _mm256_mul_ps(d1, d1)), _mm256_mul_ps(d2, d2));
        } else if constexpr (Channels == 0) {
            for (std::ptrdiff_t c = 1; c < channels; ++c) {
                const auto delta = _mm256_sub_ps(_mm256_loadu_ps(image + c * pixels + x),
                                                      _mm256_set1_ps(center[c]));
                color = _mm256_add_ps(color, _mm256_mul_ps(delta, delta));
            }
        }
        if constexpr (Scale)
            color = _mm256_div_ps(color, scale);
        const auto space = _mm256_add_ps(yz, _mm256_loadu_ps(x_distances + x));
        const auto distance = _mm256_add_ps(_mm256_mul_ps(space, weight), color);
        const auto previous = _mm256_loadu_ps(distances + x);
        const auto mask = _mm256_cmp_ps(previous, distance, _CMP_GT_OQ);
        const int winners = _mm256_movemask_ps(mask);
        if (winners) {
            _mm256_storeu_ps(distances + x, _mm256_blendv_ps(previous, distance, mask));
            const auto prior_labels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(nearest + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(nearest + x),
                                _mm256_blendv_epi8(prior_labels, labels, _mm256_castps_si256(mask)));
            changed |= winners;
        }
    }
    change |= changed != 0;
    return x;
}

template<int Channels, bool Scale>
std::ptrdiff_t slic_assign_double_row(const double* image, const double* center, const double* x_distances,
                                      double* distances, std::int32_t* nearest, std::ptrdiff_t length,
                                      std::ptrdiff_t channels, std::ptrdiff_t pixels, std::int32_t label,
                                      double color_scale, double spatial_yz,
                                      double spatial_weight, bool& change) {
    const auto yz = _mm256_set1_pd(spatial_yz);
    const auto weight = _mm256_set1_pd(spatial_weight);
    const auto scale = _mm256_set1_pd(color_scale);
    const auto labels = _mm_set1_epi32(label);
    const auto label_lanes = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    const auto c0 = _mm256_set1_pd(center[0]);
    const auto c1 = Channels == 3 ? _mm256_set1_pd(center[1]) : _mm256_setzero_pd();
    const auto c2 = Channels == 3 ? _mm256_set1_pd(center[2]) : _mm256_setzero_pd();
    std::ptrdiff_t x = 0;
    int changed = 0;
    for (; x + 4 <= length; x += 4) {
        const auto d0 = _mm256_sub_pd(_mm256_loadu_pd(image + x), c0);
        auto color = _mm256_mul_pd(d0, d0);
        if constexpr (Channels == 3) {
            const auto d1 = _mm256_sub_pd(_mm256_loadu_pd(image + pixels + x), c1);
            const auto d2 = _mm256_sub_pd(_mm256_loadu_pd(image + 2 * pixels + x), c2);
            color = _mm256_add_pd(_mm256_add_pd(color, _mm256_mul_pd(d1, d1)), _mm256_mul_pd(d2, d2));
        } else if constexpr (Channels == 0) {
            for (std::ptrdiff_t c = 1; c < channels; ++c) {
                const auto delta = _mm256_sub_pd(_mm256_loadu_pd(image + c * pixels + x),
                                                      _mm256_set1_pd(center[c]));
                color = _mm256_add_pd(color, _mm256_mul_pd(delta, delta));
            }
        }
        if constexpr (Scale)
            color = _mm256_div_pd(color, scale);
        const auto space = _mm256_add_pd(yz, _mm256_loadu_pd(x_distances + x));
        const auto distance = _mm256_add_pd(_mm256_mul_pd(space, weight), color);
        const auto previous = _mm256_loadu_pd(distances + x);
        const auto mask = _mm256_cmp_pd(previous, distance, _CMP_GT_OQ);
        const int winners = _mm256_movemask_pd(mask);
        if (winners) {
            const auto label_mask = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(mask), label_lanes);
            _mm256_storeu_pd(distances + x, _mm256_blendv_pd(previous, distance, mask));
            const auto prior_labels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nearest + x));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(nearest + x),
                             _mm_blendv_epi8(prior_labels, labels, _mm256_castsi256_si128(label_mask)));
            changed |= winners;
        }
    }
    change |= changed != 0;
    return x;
}
// Accumulate across features with one lane per feature. Processing pixels in
// raster order keeps every cluster's floating-point additions in scalar order.
template<class T>
bool accumulate_slic_rgb(const T* image, const std::int32_t* nearest, T* centers,
                         std::ptrdiff_t* members, std::ptrdiff_t depth, std::ptrdiff_t height,
                         std::ptrdiff_t width, std::int32_t start) {
    const auto pixels = depth * height * width;
    const T* green = image + pixels;
    const T* blue = green + pixels;
    std::ptrdiff_t offset = 0;
    for (std::ptrdiff_t z = 0; z < depth; ++z) {
        for (std::ptrdiff_t y = 0; y < height; ++y) {
            for (std::ptrdiff_t x = 0; x < width; ++x, ++offset) {
                const auto k = nearest[offset] - start;
                if (k < 0)
                    return false;
                ++members[k];
                T* center = centers + std::ptrdiff_t(k) * 8;
                if constexpr (std::is_same_v<T, float>) {
                    const auto values = _mm256_setr_ps(T(z), T(y), T(x), image[offset],
                                                       green[offset], blue[offset], 0, 0);
                    _mm256_storeu_ps(center, _mm256_add_ps(_mm256_loadu_ps(center), values));
                } else {
                    const auto values = _mm256_setr_pd(T(z), T(y), T(x), image[offset]);
                    const auto colors = _mm_setr_pd(green[offset], blue[offset]);
                    _mm256_storeu_pd(center, _mm256_add_pd(_mm256_loadu_pd(center), values));
                    _mm_storeu_pd(center + 4, _mm_add_pd(_mm_loadu_pd(center + 4), colors));
                }
            }
        }
    }
    return true;
}
} // namespace

std::ptrdiff_t slic_assign_row(const float* image, const float* center, const float* x_distances,
                               float* distances, std::int32_t* nearest, std::ptrdiff_t length,
                               std::ptrdiff_t channels, std::ptrdiff_t pixels, std::int32_t label,
                               float color_scale, float spatial_yz, float spatial_weight, bool& change) {
    const auto assign = color_scale == float(1)
        ? (channels == 3 ? slic_assign_float_row<3, false> :
           channels == 1 ? slic_assign_float_row<1, false> : slic_assign_float_row<0, false>)
        : slic_assign_float_row<0, true>;
    return assign(image, center, x_distances, distances, nearest, length, channels, pixels, label,
                  color_scale, spatial_yz, spatial_weight, change);
}

bool slic_accumulate_rgb(const float* image, const std::int32_t* nearest, float* centers,
                         std::ptrdiff_t* members, std::ptrdiff_t depth, std::ptrdiff_t height,
                         std::ptrdiff_t width, std::int32_t start) {
    return accumulate_slic_rgb(image, nearest, centers, members, depth, height, width, start);
}

std::ptrdiff_t slic_assign_row(const double* image, const double* center, const double* x_distances,
                               double* distances, std::int32_t* nearest, std::ptrdiff_t length,
                               std::ptrdiff_t channels, std::ptrdiff_t pixels, std::int32_t label,
                               double color_scale, double spatial_yz, double spatial_weight, bool& change) {
    const auto assign = color_scale == double(1)
        ? (channels == 3 ? slic_assign_double_row<3, false> :
           channels == 1 ? slic_assign_double_row<1, false> : slic_assign_double_row<0, false>)
        : slic_assign_double_row<0, true>;
    return assign(image, center, x_distances, distances, nearest, length, channels, pixels, label,
                  color_scale, spatial_yz, spatial_weight, change);
}

bool slic_accumulate_rgb(const double* image, const std::int32_t* nearest, double* centers,
                         std::ptrdiff_t* members, std::ptrdiff_t depth, std::ptrdiff_t height,
                         std::ptrdiff_t width, std::int32_t start) {
    return accumulate_slic_rgb(image, nearest, centers, members, depth, height, width, start);
}

void slic_assign_four_rgb(const double* image, const double* centers, double* x_distances,
                          std::int32_t* nearest, std::ptrdiff_t height, std::ptrdiff_t width,
                          double sx, double sy, double spatial_weight, std::int32_t start) {
    const auto pixels = height * width;
    const auto weight = _mm256_set1_pd(spatial_weight);
    const auto label_lanes = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    for (std::ptrdiff_t k = 0; k < 4; ++k) {
        const double cx = centers[k * 8 + 2];
        for (std::ptrdiff_t x = 0; x < width; ++x) {
            const double dx = sx * (cx - static_cast<double>(x));
            x_distances[k * width + x] = dx * dx;
        }
    }
    for (std::ptrdiff_t y = 0; y < height; ++y) {
        __m256d y_distances[4];
        for (std::ptrdiff_t k = 0; k < 4; ++k) {
            const double dy = sy * (centers[k * 8 + 1] - static_cast<double>(y));
            y_distances[k] = _mm256_set1_pd(dy * dy);
        }
        for (std::ptrdiff_t x = 0; x < width; x += 4) {
            const auto offset = y * width + x;
            const auto red = _mm256_loadu_pd(image + offset);
            const auto green = _mm256_loadu_pd(image + pixels + offset);
            const auto blue = _mm256_loadu_pd(image + 2 * pixels + offset);
            const auto score = [&](std::ptrdiff_t k) {
                const double* center = centers + k * 8 + 3;
                const auto r = _mm256_sub_pd(red, _mm256_set1_pd(center[0]));
                const auto g = _mm256_sub_pd(green, _mm256_set1_pd(center[1]));
                const auto b = _mm256_sub_pd(blue, _mm256_set1_pd(center[2]));
                const auto color = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(r, r),
                                                              _mm256_mul_pd(g, g)),
                                                  _mm256_mul_pd(b, b));
                const auto space = _mm256_add_pd(y_distances[k], _mm256_loadu_pd(x_distances + k * width + x));
                return _mm256_add_pd(_mm256_mul_pd(space, weight), color);
            };
            auto best = score(0);
            auto labels = _mm256_castsi256_pd(_mm256_set1_epi64x(start));
            const auto consider = [&](std::int32_t k) {
                const auto distance = score(k);
                const auto wins = _mm256_cmp_pd(best, distance, _CMP_GT_OQ);
                best = _mm256_blendv_pd(best, distance, wins);
                labels = _mm256_blendv_pd(labels, _mm256_castsi256_pd(_mm256_set1_epi64x(start + k)), wins);
            };
            consider(1);
            consider(2);
            consider(3);
            const auto packed = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(labels), label_lanes);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(nearest + offset), _mm256_castsi256_si128(packed));
        }
    }
}

bool gaussian_fold_safe(const double* input,std::size_t count) {
    const double limit = std::numeric_limits<double>::max()/4;
    std::int64_t limit_bits;
    std::memcpy(&limit_bits,&limit,sizeof(limit_bits));
    constexpr auto magnitude = std::numeric_limits<std::int64_t>::max();
    const auto max_value = _mm256_set1_epi64x(limit_bits);
    const auto mask = _mm256_set1_epi64x(magnitude);
    std::size_t i = 0;
    for (; i+4 <= count; i += 4) {
        const auto bits = _mm256_castpd_si256(_mm256_loadu_pd(input+i));
        const auto too_large = _mm256_cmpgt_epi64(_mm256_and_si256(bits,mask),max_value);
        if (_mm256_movemask_pd(_mm256_castsi256_pd(too_large))) return false;
    }
    for (; i < count; ++i) {
        std::int64_t bits;
        std::memcpy(&bits,input+i,sizeof(bits));
        if ((bits & magnitude) > limit_bits) return false;
    }
    return true;
}
bool gamma_simple(const float* p, float* out, std::size_t n, double exponent, double gain) {
    bool negative = false;
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto a = _mm256_loadu_ps(p+i);
        negative = negative || _mm256_movemask_ps(_mm256_cmp_ps(a,_mm256_setzero_ps(),_CMP_LT_OQ)) != 0;
        const auto power = exponent == 0 ? _mm256_set1_ps(1) : exponent == 1 ? a :
                           exponent == 2 ? _mm256_mul_ps(a,a) : _mm256_sqrt_ps(a);
        _mm256_storeu_ps(out+i,_mm256_mul_ps(power,_mm256_set1_ps(float(gain))));
    }
    for (; i < n; ++i) {
        negative = negative || p[i] < 0;
        const float power = exponent == 0 ? 1.0f : exponent == 1 ? p[i] : exponent == 2 ? p[i]*p[i] : std::sqrt(p[i]);
        out[i] = power*float(gain);
    }
    return negative;
}
bool gamma_simple(const double* p, double* out, std::size_t n, double exponent, double gain) {
    bool negative = false;
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto a = _mm256_loadu_pd(p+i);
        negative = negative || _mm256_movemask_pd(_mm256_cmp_pd(a,_mm256_setzero_pd(),_CMP_LT_OQ)) != 0;
        const auto power = exponent == 0 ? _mm256_set1_pd(1) : exponent == 1 ? a :
                           exponent == 2 ? _mm256_mul_pd(a,a) : _mm256_sqrt_pd(a);
        _mm256_storeu_pd(out+i,_mm256_mul_pd(power,_mm256_set1_pd(gain)));
    }
    for (; i < n; ++i) {
        negative = negative || p[i] < 0;
        const double power = exponent == 0 ? 1.0 : exponent == 1 ? p[i] : exponent == 2 ? p[i]*p[i] : std::sqrt(p[i]);
        out[i] = power*gain;
    }
    return negative;
}
void lookup(const std::uint8_t* p, double* out, std::size_t n, const double* table) {
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        std::uint32_t packed;
        std::memcpy(&packed,p+i,sizeof(packed));
        const auto indices = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(packed)));
        _mm256_storeu_pd(out+i,_mm256_i32gather_pd(table,indices,8));
    }
    for (; i < n; ++i) out[i] = table[p[i]];
}
void lookup(const std::uint8_t* p, std::uint8_t* out, std::size_t n, const std::int32_t* table) {
    std::size_t i = 0;
    const auto shuffle = _mm_setr_epi8(0,4,8,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    for (; i+8 <= n; i += 8) {
        const auto indices = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+i)));
        const auto values = _mm256_i32gather_epi32(table,indices,4);
        const auto low = _mm_shuffle_epi8(_mm256_castsi256_si128(values),shuffle);
        const auto high = _mm_shuffle_epi8(_mm256_extracti128_si256(values,1),shuffle);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out+i),_mm_unpacklo_epi32(low,high));
    }
    for (; i < n; ++i) out[i] = static_cast<std::uint8_t>(table[p[i]]);
}
void gray2rgb(const float* p, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto a = _mm256_loadu_ps(p+i);
        _mm256_storeu_ps(out+3*i,_mm256_permutevar8x32_ps(a,_mm256_setr_epi32(0,0,0,1,1,1,2,2)));
        _mm256_storeu_ps(out+3*i+8,_mm256_permutevar8x32_ps(a,_mm256_setr_epi32(2,3,3,3,4,4,4,5)));
        _mm256_storeu_ps(out+3*i+16,_mm256_permutevar8x32_ps(a,_mm256_setr_epi32(5,5,6,6,6,7,7,7)));
    }
    for (; i < n; ++i) out[3*i] = out[3*i+1] = out[3*i+2] = p[i];
}
void gray2rgb(const double* p, double* out, std::size_t n) {
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto a = _mm256_loadu_pd(p+i);
        _mm256_storeu_pd(out+3*i,_mm256_permute4x64_pd(a,_MM_SHUFFLE(1,0,0,0)));
        _mm256_storeu_pd(out+3*i+4,_mm256_permute4x64_pd(a,_MM_SHUFFLE(2,2,1,1)));
        _mm256_storeu_pd(out+3*i+8,_mm256_permute4x64_pd(a,_MM_SHUFFLE(3,3,3,2)));
    }
    for (; i < n; ++i) out[3*i] = out[3*i+1] = out[3*i+2] = p[i];
}
void gray2rgb(const std::uint8_t* p, std::uint8_t* out, std::size_t n) {
    std::size_t i = 0;
    for (; i+16 <= n; i += 16) {
        const auto a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p+i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out+3*i),_mm_shuffle_epi8(a,_mm_setr_epi8(0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out+3*i+16),_mm_shuffle_epi8(a,_mm_setr_epi8(5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10)));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out+3*i+32),_mm_shuffle_epi8(a,_mm_setr_epi8(10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15)));
    }
    for (; i < n; ++i) out[3*i] = out[3*i+1] = out[3*i+2] = p[i];
}
void invert(const float* p, float* out, std::size_t n, bool signed_float) {
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto a = _mm256_loadu_ps(p+i);
        _mm256_storeu_ps(out+i,signed_float ? _mm256_xor_ps(a,_mm256_set1_ps(-0.0f)) : _mm256_sub_ps(_mm256_set1_ps(1),a));
    }
    for (; i < n; ++i) out[i] = signed_float ? -p[i] : 1.0f-p[i];
}
void invert(const double* p, double* out, std::size_t n, bool signed_float) {
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto a = _mm256_loadu_pd(p+i);
        _mm256_storeu_pd(out+i,signed_float ? _mm256_xor_pd(a,_mm256_set1_pd(-0.0)) : _mm256_sub_pd(_mm256_set1_pd(1),a));
    }
    for (; i < n; ++i) out[i] = signed_float ? -p[i] : 1.0-p[i];
}
void invert(const std::uint8_t* p, std::uint8_t* out, std::size_t n, bool) {
    std::size_t i = 0;
    for (; i+32 <= n; i += 32) _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)),_mm256_set1_epi8(-1)));
    for (; i < n; ++i) out[i] = std::uint8_t(255-p[i]);
}
bool uniform(const std::uint8_t* p, std::size_t n) {
    const auto value = _mm256_set1_epi8(static_cast<char>(p[0]));
    std::size_t i = 0;
    for (; i+32 <= n; i += 32)
        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)),value)) != -1) return false;
    for (; i < n; ++i) if (p[i] != p[0]) return false;
    return true;
}
void otsu_scores(const float* left, const float* right, const double* left_sum,
                 const double* right_sum, double* scores, std::size_t n) {
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto l = _mm_loadu_ps(left+i), r = _mm_loadu_ps(right+i);
        const auto mean_l = _mm256_div_pd(_mm256_loadu_pd(left_sum+i),_mm256_cvtps_pd(l));
        const auto mean_r = _mm256_div_pd(_mm256_loadu_pd(right_sum+i),_mm256_cvtps_pd(r));
        const auto delta = _mm256_sub_pd(mean_l,mean_r);
        _mm256_storeu_pd(scores+i,_mm256_mul_pd(_mm256_cvtps_pd(_mm_mul_ps(l,r)),_mm256_mul_pd(delta,delta)));
    }
    for (; i < n; ++i) {
        const double delta = left_sum[i]/left[i]-right_sum[i]/right[i];
        scores[i] = double(left[i]*right[i])*(delta*delta);
    }
}
void gray(const std::uint8_t* p, double* out, std::size_t n) {
    std::size_t i = 0;
    const auto red = _mm_setr_epi8(0,3,6,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    const auto green = _mm_setr_epi8(1,4,7,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    const auto blue = _mm_setr_epi8(2,5,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    constexpr double scale = 1.0/255.0;
    // Six remaining pixels ensure that the 16-byte load stays in bounds.
    for (; i+6 <= n; i += 4) {
        const auto packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p+3*i));
        const auto r = _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_shuffle_epi8(packed,red)));
        const auto g = _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_shuffle_epi8(packed,green)));
        const auto b = _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_shuffle_epi8(packed,blue)));
        _mm256_storeu_pd(out+i,_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(r,_mm256_set1_pd(0.2125*scale)),
            _mm256_mul_pd(g,_mm256_set1_pd(0.7154*scale))),_mm256_mul_pd(b,_mm256_set1_pd(0.0721*scale))));
    }
    for (; i < n; ++i) out[i] = double(p[3*i])*(0.2125*scale)+double(p[3*i+1])*(0.7154*scale)+double(p[3*i+2])*(0.0721*scale);
}
void gray(const float* p, float* out, std::size_t n) {
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto packed0 = _mm256_loadu_ps(p+3*i), packed1 = _mm256_loadu_ps(p+3*i+8), packed2 = _mm256_loadu_ps(p+3*i+16);
        const auto a = _mm256_permute2f128_ps(packed0,packed1,0x30);
        const auto v = _mm256_permute2f128_ps(packed0,packed2,0x21);
        const auto c = _mm256_permute2f128_ps(packed1,packed2,0x30);
        const auto t0 = _mm256_shuffle_ps(a,v,_MM_SHUFFLE(2,1,2,1));
        const auto t1 = _mm256_shuffle_ps(v,c,_MM_SHUFFLE(1,0,3,2));
        const auto r = _mm256_shuffle_ps(a,t1,_MM_SHUFFLE(3,0,3,0));
        const auto g0 = _mm256_shuffle_ps(a,v,_MM_SHUFFLE(0,0,1,1));
        const auto g1 = _mm256_shuffle_ps(v,c,_MM_SHUFFLE(2,2,3,3));
        const auto g = _mm256_shuffle_ps(g0,g1,_MM_SHUFFLE(2,0,2,0));
        const auto b = _mm256_shuffle_ps(t0,c,_MM_SHUFFLE(3,0,2,1));
        _mm256_storeu_ps(out+i, _mm256_add_ps(_mm256_add_ps(
            _mm256_mul_ps(r,_mm256_set1_ps(0.2125f)), _mm256_mul_ps(g,_mm256_set1_ps(0.7154f))),
            _mm256_mul_ps(b,_mm256_set1_ps(0.0721f))));
    }
    for (; i < n; ++i) out[i] = p[3*i]*0.2125f+p[3*i+1]*0.7154f+p[3*i+2]*0.0721f;
}
void gray(const double* p, double* out, std::size_t n) {
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto a = _mm256_loadu_pd(p+3*i), v = _mm256_loadu_pd(p+3*i+4), c = _mm256_loadu_pd(p+3*i+8);
        const auto r = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,3,0)),
            _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,2,0,0)),0x4),_mm256_permute4x64_pd(c,_MM_SHUFFLE(1,0,0,0)),0x8);
        const auto g = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,1)),
            _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,3,0,0)),0x6),_mm256_permute4x64_pd(c,_MM_SHUFFLE(2,0,0,0)),0x8);
        const auto b = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,2)),
            _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,0,1,0)),0x2),_mm256_permute4x64_pd(c,_MM_SHUFFLE(3,0,0,0)),0xc);
        _mm256_storeu_pd(out+i, _mm256_add_pd(_mm256_add_pd(
            _mm256_mul_pd(r,_mm256_set1_pd(0.2125)), _mm256_mul_pd(g,_mm256_set1_pd(0.7154))),
            _mm256_mul_pd(b,_mm256_set1_pd(0.0721))));
    }
    for (; i < n; ++i) out[i] = p[3*i]*0.2125+p[3*i+1]*0.7154+p[3*i+2]*0.0721;
}
namespace {
__m256d load4(const double* p) { return _mm256_loadu_pd(p); }
__m256d load4(const float* p) { return _mm256_cvtps_pd(_mm_loadu_ps(p)); }
__m256d load4(const std::uint8_t* p) {
    std::uint32_t bytes;
    std::memcpy(&bytes,p,sizeof(bytes));
    return _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(bytes))));
}
void store4(double* p, __m256d v) { _mm256_storeu_pd(p,v); }
void store4(float* p, __m256d v) { _mm_storeu_ps(p,_mm256_cvtpd_ps(v)); }
template<class Input, class Output>
void convolve_impl(const Input* const* sources, Output* out, std::size_t n,
                   const double* kernel, std::size_t taps, double scale) {
    std::vector<double> scaled;
    if (scale != 1.0) {
        scaled.resize(taps);
        std::transform(kernel,kernel+taps,scaled.begin(),[scale](double v) { return v*scale; });
        kernel = scaled.data();
    }
    const auto middle = taps/2;
    const auto vector_rows = [&](auto symmetry) {
        constexpr bool Symmetric = decltype(symmetry)::value;
        const auto end = Symmetric ? middle : taps;
        std::size_t x = 0;
        for (; x+8 <= n; x += 8) {
            auto a = _mm256_setzero_pd(), b = _mm256_setzero_pd();
            if constexpr (Symmetric) {
                const auto weight = _mm256_set1_pd(kernel[middle]);
                a = _mm256_add_pd(a,_mm256_mul_pd(load4(sources[middle]+x),weight));
                b = _mm256_add_pd(b,_mm256_mul_pd(load4(sources[middle]+x+4),weight));
            }
            for (std::size_t k = 0; k < end; ++k) {
                const auto weight = _mm256_set1_pd(kernel[k]);
                auto first = load4(sources[k]+x), second = load4(sources[k]+x+4);
                if constexpr (Symmetric) {
                    const auto* mirror = sources[taps-1-k]+x;
                    first = _mm256_add_pd(first,load4(mirror));
                    second = _mm256_add_pd(second,load4(mirror+4));
                }
                a = _mm256_add_pd(a,_mm256_mul_pd(first,weight));
                b = _mm256_add_pd(b,_mm256_mul_pd(second,weight));
            }
            store4(out+x,a); store4(out+x+4,b);
        }
        return x;
    };
    std::size_t x = 0;
    if (n >= 8) {
        const bool symmetric = taps%2 != 0 &&
            std::equal(kernel,kernel+middle,std::make_reverse_iterator(kernel+taps));
        x = symmetric ? vector_rows(std::true_type{}) : vector_rows(std::false_type{});
    }
    for (; x < n; ++x) {
        double sum = 0;
        for (std::size_t k = 0; k < taps; ++k) sum += double(sources[k][x])*kernel[k];
        out[x] = Output(sum);
    }
}
}
void convolve(const double* const* p, double* out, std::size_t n, const double* k, std::size_t taps, double scale) { convolve_impl(p,out,n,k,taps,scale); }
void convolve(const float* const* p, float* out, std::size_t n, const double* k, std::size_t taps, double scale) { convolve_impl(p,out,n,k,taps,scale); }
void convolve(const std::uint8_t* const* p, double* out, std::size_t n, const double* k, std::size_t taps, double scale) { convolve_impl(p,out,n,k,taps,scale); }

// Interior pixels only. Caller handles reflected borders and vector tails.
std::size_t gradient(const float* t, const float* m, const float* b, float* out, std::size_t w, double outer, double center, int axis) {
    std::size_t x = 1;
    const auto q = _mm256_set1_ps(float(outer)), half = _mm256_set1_ps(float(center));
    for (; x+8 < w; x += 8) {
        const auto tl = _mm256_loadu_ps(t+x-1), tr = _mm256_loadu_ps(t+x+1);
        const auto bl = _mm256_loadu_ps(b+x-1), br = _mm256_loadu_ps(b+x+1);
        const auto dx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(tr,tl),q),
            _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(m+x+1),_mm256_loadu_ps(m+x-1)),half)),
            _mm256_mul_ps(_mm256_sub_ps(br,bl),q));
        const auto dy = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(bl,tl),q),
            _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(b+x),_mm256_loadu_ps(t+x)),half)),
            _mm256_mul_ps(_mm256_sub_ps(br,tr),q));
        if (axis == 0) _mm256_storeu_ps(out+x,dy);
        else if (axis == 1) _mm256_storeu_ps(out+x,dx);
        else _mm256_storeu_ps(out+x, _mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(dx,dx),_mm256_mul_ps(dy,dy))),_mm256_set1_ps(float(std::sqrt(2.0)))));
    }
    return x;
}
namespace {
__m256d edge_load(const double* p) { return load4(p); }
__m256d edge_load(const std::uint8_t* p) { return load4(p); }
template<class T> std::size_t edges_double(const T* t, const T* m, const T* b, double* out, std::size_t w, double outer, double center, int axis) {
    std::size_t x = 1;
    const auto q = _mm256_set1_pd(outer), half = _mm256_set1_pd(center);
    for (; x+4 < w; x += 4) {
        const auto tl = edge_load(t+x-1), tr = edge_load(t+x+1);
        const auto bl = edge_load(b+x-1), br = edge_load(b+x+1);
        const auto dx = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(tr,tl),q),
            _mm256_mul_pd(_mm256_sub_pd(edge_load(m+x+1),edge_load(m+x-1)),half)),
            _mm256_mul_pd(_mm256_sub_pd(br,bl),q));
        const auto dy = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(bl,tl),q),
            _mm256_mul_pd(_mm256_sub_pd(edge_load(b+x),edge_load(t+x)),half)),
            _mm256_mul_pd(_mm256_sub_pd(br,tr),q));
        constexpr double scale = std::is_same_v<T,std::uint8_t> ? 255.0 : 1.0;
        if (axis == 0) _mm256_storeu_pd(out+x,_mm256_div_pd(dy,_mm256_set1_pd(scale)));
        else if (axis == 1) _mm256_storeu_pd(out+x,_mm256_div_pd(dx,_mm256_set1_pd(scale)));
        else _mm256_storeu_pd(out+x, _mm256_div_pd(_mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(dx,dx),_mm256_mul_pd(dy,dy))),_mm256_set1_pd(std::sqrt(2.0)*scale)));
    }
    return x;
}
}
std::size_t gradient(const double* t, const double* m, const double* b, double* out, std::size_t w, double outer, double center, int axis) { return edges_double(t,m,b,out,w,outer,center,axis); }
std::size_t gradient(const std::uint8_t* t, const std::uint8_t* m, const std::uint8_t* b, double* out, std::size_t w, double outer, double center, int axis) { return edges_double(t,m,b,out,w,outer,center,axis); }
namespace {
template<class T> double sum_impl(const T* p, std::size_t n) {
    auto a = _mm256_setzero_pd(), b = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) { a = _mm256_add_pd(a,load4(p+i)); b = _mm256_add_pd(b,load4(p+i+4)); }
    double lanes[4];
    _mm256_storeu_pd(lanes,_mm256_add_pd(a,b));
    double total = (lanes[0]+lanes[1])+(lanes[2]+lanes[3]);
    for (; i < n; ++i) total += double(p[i]);
    return total;
}
}
double sum(const float* p, std::size_t n) { return sum_impl(p,n); }
double sum(const double* p, std::size_t n) { return sum_impl(p,n); }
double sum(const std::uint8_t* p, std::size_t n) { return sum_impl(p,n); }
namespace {
// Float operations round at the same expression boundaries as NumPy ufuncs.
__m256d round_for_float(__m256d value, float) { return _mm256_cvtps_pd(_mm256_cvtpd_ps(value)); }
__m256d round_for_float(__m256d value, double) { return value; }
template<class T> bool exposure_impl(const T* p, T* out, std::size_t n, int transform,
                                      double parameter, double gain, bool inverse) {
    bool negative = false;
    const auto one = _mm256_set1_pd(1);
    const auto gain_vector = _mm256_set1_pd(double(T(gain)));
    const auto parameter_vector = _mm256_set1_pd(double(T(parameter)));
    auto rounded = [](auto value) { return round_for_float(value,T{}); };
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto value = load4(p+i);
        negative = negative || _mm256_movemask_pd(_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_LT_OQ)) != 0;
        __m256d corrected;
        if (transform == 0) {
            // Special values use pow so e.g. NaN**0 and signed zero agree.
            const auto regular = _mm256_and_pd(_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_GT_OQ),
                                               _mm256_cmp_pd(value,_mm256_set1_pd(INFINITY),_CMP_LT_OQ));
            if (_mm256_movemask_pd(regular) == 15) {
                corrected = rounded(math::exp4(_mm256_mul_pd(_mm256_mul_pd(math::log2_4(value),parameter_vector),
                                                             _mm256_set1_pd(0.6931471805599453094))));
            } else {
                double lanes[4]; _mm256_storeu_pd(lanes,value);
                for (auto& v : lanes) v = double(std::pow(T(v),T(parameter)));
                corrected = _mm256_loadu_pd(lanes);
            }
            corrected = rounded(_mm256_mul_pd(corrected,gain_vector));
        } else if (transform == 1) {
            corrected = inverse ? rounded(_mm256_sub_pd(rounded(math::exp4(_mm256_mul_pd(value,
                _mm256_set1_pd(0.6931471805599453094)))),one)) : rounded(math::log2_4(rounded(_mm256_add_pd(one,value))));
            corrected = rounded(_mm256_mul_pd(corrected,gain_vector));
        } else {
            const auto exponent = rounded(_mm256_mul_pd(gain_vector,rounded(_mm256_sub_pd(parameter_vector,value))));
            corrected = rounded(_mm256_div_pd(one,rounded(_mm256_add_pd(one,rounded(math::exp4(exponent))))));
            if (inverse) corrected = rounded(_mm256_sub_pd(one,corrected));
        }
        store4(out+i,corrected);
    }
    for (; i < n; ++i) {
        negative = negative || p[i] < T(0);
        if (transform == 0) out[i] = T(std::pow(p[i],T(parameter)))*T(gain);
        else if (transform == 1) out[i] = T(inverse ? T(std::pow(T(2),p[i])-T(1)) : std::log2(T(1)+p[i]))*T(gain);
        else {
            const T exponent = T(gain)*T(T(parameter)-p[i]);
            const T corrected = T(1)/T(T(1)+std::exp(exponent));
            out[i] = inverse ? T(1)-corrected : corrected;
        }
    }
    return negative;
}
}
bool exposure(const float* p, float* out, std::size_t n, int kind, double parameter, double gain, bool inverse) {
    if (kind != 2) return exposure_impl(p,out,n,kind,parameter,gain,inverse);
    bool negative = false;
    const auto one = _mm256_set1_ps(1);
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto value = _mm256_loadu_ps(p+i);
        negative = negative || _mm256_movemask_ps(_mm256_cmp_ps(value,_mm256_setzero_ps(),_CMP_LT_OQ)) != 0;
        const auto exponent = _mm256_mul_ps(_mm256_set1_ps(float(gain)),_mm256_sub_ps(_mm256_set1_ps(float(parameter)),value));
        auto corrected = _mm256_div_ps(one,_mm256_add_ps(one,math::exp8(exponent)));
        if (inverse) corrected = _mm256_sub_ps(one,corrected);
        _mm256_storeu_ps(out+i,corrected);
    }
    const bool tail_negative = exposure_impl(p+i,out+i,n-i,kind,parameter,gain,inverse);
    return negative || tail_negative;
}
bool exposure(const double* p, double* out, std::size_t n, int kind, double parameter, double gain, bool inverse) {
    return exposure_impl(p,out,n,kind,parameter,gain,inverse);
}
namespace {
template<class T> void bounds_impl(const T* p, std::size_t n, double& low, double& high) {
    const double inf = std::numeric_limits<double>::infinity();
    auto l = _mm256_set1_pd(inf), h = _mm256_set1_pd(-inf);
    bool nan = false;
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        const auto a = load4(p+i);
        nan = nan || _mm256_movemask_pd(_mm256_cmp_pd(a,a,_CMP_UNORD_Q)) != 0;
        l = _mm256_min_pd(l,a); h = _mm256_max_pd(h,a);
    }
    double lows[4], highs[4];
    _mm256_storeu_pd(lows,l); _mm256_storeu_pd(highs,h);
    low = *std::min_element(lows,lows+4); high = *std::max_element(highs,highs+4);
    for (; i < n; ++i) {
        nan = nan || std::isnan(double(p[i]));
        low = std::min(low,double(p[i])); high = std::max(high,double(p[i]));
    }
    if (nan) low = high = std::numeric_limits<double>::quiet_NaN();
}
template<class T> void rescale_double(const T* p, double* out, std::size_t n,
                                      double imin, double imax, double omin, double omax) {
    std::size_t i = 0;
    if (std::isnan(imin) || std::isnan(imax) || std::isnan(omin) || std::isnan(omax)) {
        std::fill(out,out+n,std::numeric_limits<double>::quiet_NaN()); return;
    }
    const auto lo = _mm256_set1_pd(imin), hi = _mm256_set1_pd(imax);
    const auto outlo = _mm256_set1_pd(omin), outhi = _mm256_set1_pd(omax);
    for (; i+4 <= n; i += 4) {
        const auto original = load4(p+i);
        auto value = _mm256_min_pd(_mm256_max_pd(original,lo),hi);
        if (imin != imax) value = _mm256_add_pd(_mm256_mul_pd(
            _mm256_div_pd(_mm256_sub_pd(value,lo),_mm256_set1_pd(imax-imin)),_mm256_set1_pd(omax-omin)),outlo);
        else value = _mm256_min_pd(_mm256_max_pd(value,outlo),outhi);
        value = _mm256_blendv_pd(value,original,_mm256_cmp_pd(original,original,_CMP_UNORD_Q));
        _mm256_storeu_pd(out+i,value);
    }
    for (; i < n; ++i) {
        double value = std::min(std::max(double(p[i]),imin),imax);
        out[i] = imin != imax ? ((value-imin)/(imax-imin))*(omax-omin)+omin : std::min(std::max(value,omin),omax);
    }
}
}
void bounds(const float* p, std::size_t n, double& low, double& high) { bounds_impl(p,n,low,high); }
void bounds(const double* p, std::size_t n, double& low, double& high) { bounds_impl(p,n,low,high); }
void bounds(const std::uint8_t* p, std::size_t n, double& low, double& high) { bounds_impl(p,n,low,high); }
void rescale(const double* p, double* out, std::size_t n, double a, double b, double c, double d) { rescale_double(p,out,n,a,b,c,d); }
void rescale(const std::uint8_t* p, double* out, std::size_t n, double a, double b, double c, double d) { rescale_double(p,out,n,a,b,c,d); }
void rescale(const float* p, float* out, std::size_t n, double imin, double imax, double omin, double omax) {
    if (std::isnan(imin) || std::isnan(imax) || std::isnan(omin) || std::isnan(omax)) {
        std::fill(out,out+n,std::numeric_limits<float>::quiet_NaN()); return;
    }
    const auto lo = _mm256_set1_ps(float(imin)), hi = _mm256_set1_ps(float(imax));
    const auto outlo = _mm256_set1_ps(float(omin)), outhi = _mm256_set1_ps(float(omax));
    std::size_t i = 0;
    for (; i+8 <= n; i += 8) {
        const auto original = _mm256_loadu_ps(p+i);
        auto value = _mm256_min_ps(_mm256_max_ps(original,lo),hi);
        if (imin != imax) value = _mm256_add_ps(_mm256_mul_ps(
            _mm256_div_ps(_mm256_sub_ps(value,lo),_mm256_set1_ps(float(imax-imin))),_mm256_set1_ps(float(omax-omin))),outlo);
        else value = _mm256_min_ps(_mm256_max_ps(value,outlo),outhi);
        value = _mm256_blendv_ps(value,original,_mm256_cmp_ps(original,original,_CMP_UNORD_Q));
        _mm256_storeu_ps(out+i,value);
    }
    for (; i < n; ++i) {
        const float value = std::min(std::max(p[i],float(imin)),float(imax));
        out[i] = imin != imax ? ((value-float(imin))/float(imax-imin))*float(omax-omin)+float(omin) : std::min(std::max(value,float(omin)),float(omax));
    }
}

#ifdef SKIMAGEXX_AVX2
namespace {

using Index = std::ptrdiff_t;

template<class T>
void quickshift_add_weight(T& density, T& neighbor, double weight) {
    density = static_cast<T>(density + weight);
    neighbor = static_cast<T>(neighbor + weight);
}

unsigned first_lane(unsigned mask) {
    unsigned lane = (mask & 0xfu) ? 0 : 4;
    if (!(mask & (3u << lane))) lane += 2;
    return lane + !(mask & (1u << lane));
}

__m256d finish_exp(__m256d x, __m256d normal, __m256d result) {
    if (_mm256_movemask_pd(normal) == 15) return result;
    alignas(32) double arguments[4], weights[4];
    _mm256_store_pd(arguments, x);
    _mm256_store_pd(weights, result);
    for (int lane = 0; lane < 4; ++lane)
        if (!(arguments[lane] >= -700)) weights[lane] = std::exp(arguments[lane]);
    return _mm256_load_pd(weights);
}

// For n = round(x / (log(2) / 16)), reduce to r in [-log(2)/32, log(2)/32].
// The table holds the rounded 2**(k/16) and the relative rounding error,
// where k = n modulo 16. Reconstruct with head + head * (tail + expm1(r)).
__m256d exp_pd(__m256d x) {
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

    return finish_exp(x, normal, result);
}

void exp_ps(__m256 x, double* weights) {
    // The float32 reference promotes each exponent and adds its weight in
    // double precision before storing density back to float.
    _mm256_store_pd(weights, exp_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x))));
    _mm256_store_pd(weights + 4, exp_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1))));
}

template<class T> struct Vector;

template<> struct Vector<double> {
    using Pack = __m256d;
    static constexpr Index width = 4;
    static Pack load(const double* p) { return _mm256_loadu_pd(p); }
    static Pack set(double value) { return _mm256_set1_pd(value); }
    static Pack add(Pack a, Pack b) { return _mm256_add_pd(a, b); }
    static Pack sub(Pack a, Pack b) { return _mm256_sub_pd(a, b); }
    static Pack mul(Pack a, Pack b) { return _mm256_mul_pd(a, b); }
    static Pack minimum(Pack a, Pack b) { return _mm256_min_pd(a, b); }
    static Pack minimum(Pack x) {
        x = minimum(x, _mm256_permute2f128_pd(x, x, 1));
        return minimum(x, _mm256_permute_pd(x, 5));
    }
    static Pack eligible(Pack x, Pack best, Pack densities, Pack density) {
        return _mm256_and_pd(_mm256_cmp_pd(x, best, _CMP_LT_OQ),
                             _mm256_cmp_pd(densities, density, _CMP_GT_OQ));
    }
    static Pack select(Pack best, Pack x, Pack mask) {
        return _mm256_blendv_pd(best, x, mask);
    }
    static unsigned mask(Pack x) { return _mm256_movemask_pd(x); }
    static unsigned matches(Pack x, Pack best, Pack eligible) {
        return _mm256_movemask_pd(_mm256_and_pd(_mm256_cmp_pd(x, best, _CMP_EQ_OQ), eligible));
    }
    static double first(Pack x) { return _mm256_cvtsd_f64(x); }
};

template<> struct Vector<float> {
    using Pack = __m256;
    static constexpr Index width = 8;
    static Pack load(const float* p) { return _mm256_loadu_ps(p); }
    static Pack set(float value) { return _mm256_set1_ps(value); }
    static Pack add(Pack a, Pack b) { return _mm256_add_ps(a, b); }
    static Pack sub(Pack a, Pack b) { return _mm256_sub_ps(a, b); }
    static Pack mul(Pack a, Pack b) { return _mm256_mul_ps(a, b); }
    static Pack minimum(Pack a, Pack b) { return _mm256_min_ps(a, b); }
    static Pack minimum(Pack x) {
        x = minimum(x, _mm256_permute2f128_ps(x, x, 1));
        x = minimum(x, _mm256_permute_ps(x, 0x4e));
        return minimum(x, _mm256_permute_ps(x, 0xb1));
    }
    static Pack eligible(Pack x, Pack best, Pack densities, Pack density) {
        return _mm256_and_ps(_mm256_cmp_ps(x, best, _CMP_LT_OQ),
                             _mm256_cmp_ps(densities, density, _CMP_GT_OQ));
    }
    static Pack select(Pack best, Pack x, Pack mask) {
        return _mm256_blendv_ps(best, x, mask);
    }
    static unsigned mask(Pack x) { return _mm256_movemask_ps(x); }
    static unsigned matches(Pack x, Pack best, Pack eligible) {
        return _mm256_movemask_ps(_mm256_and_ps(_mm256_cmp_ps(x, best, _CMP_EQ_OQ), eligible));
    }
    static float first(Pack x) { return _mm256_cvtss_f32(x); }
};

template<class T, Index fixed_channels>
typename Vector<T>::Pack distances(
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

// Absolute error is below 2e-14 for x <= 0, including range reduction and
// polynomial evaluation. The density pass checks ordering with this bound.
inline __m256d density_exp(__m256d x) {
    const auto normal = _mm256_cmp_pd(x, _mm256_set1_pd(-700), _CMP_GE_OQ);
    const auto input = _mm256_and_pd(x, normal);
    const auto n = _mm256_round_pd(_mm256_mul_pd(input, _mm256_set1_pd(0x1.71547652b82fep+0)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto r = _mm256_sub_pd(
        _mm256_sub_pd(input, _mm256_mul_pd(n, _mm256_set1_pd(0x1.62e42fefa0000p-1))),
        _mm256_mul_pd(n, _mm256_set1_pd(0x1.cf79abc9e3b3ap-40)));
    const auto r2 = _mm256_mul_pd(r, r);
    const auto r4 = _mm256_mul_pd(r2, r2);
    const auto r8 = _mm256_mul_pd(r4, r4);
    // Evaluate the degree-eleven Taylor polynomial for exp(r) in groups
    // on |r| <= log(2)/2. The truncation error is below 1e-14.
    const auto pair = [r] (double a, double b) {
        return _mm256_add_pd(_mm256_set1_pd(a), _mm256_mul_pd(r, _mm256_set1_pd(b)));
    };
    const auto a = _mm256_add_pd(pair(1.0 / 2, 1.0 / 6),
                                _mm256_mul_pd(r2, pair(1.0 / 24, 1.0 / 120)));
    const auto b = _mm256_add_pd(pair(1.0 / 720, 1.0 / 5040),
                                _mm256_mul_pd(r2, pair(1.0 / 40320, 1.0 / 362880)));
    const auto c = pair(1.0 / 3628800, 1.0 / 39916800);
    const auto p = _mm256_add_pd(a, _mm256_add_pd(_mm256_mul_pd(r4, b), _mm256_mul_pd(r8, c)));
    const auto value = _mm256_add_pd(_mm256_set1_pd(1), _mm256_add_pd(r, _mm256_mul_pd(r2, p)));
    const auto scale = _mm256_slli_epi64(_mm256_cvtepi32_epi64(_mm256_cvttpd_epi32(n)), 52);
    const auto result = _mm256_castsi256_pd(_mm256_add_epi64(_mm256_castpd_si256(value), scale));
    return finish_exp(x, normal, result);
}

// Factor the Gaussian into spatial and color terms. With normalized color
// and |inverse| <= 1/16, the degree-seven Taylor remainder is < 7e-15.
// Separating the rounded spatial and color exponents costs O(epsilon) in
// absolute weight: the sensitivity |x| * exp(x) for x <= 0 is at most 1/e.
inline __m256d factored_density_weight(
        double center, const double* values, const double* spatial, double inverse) {
    const auto delta = _mm256_sub_pd(_mm256_set1_pd(center), _mm256_loadu_pd(values));
    const auto r = _mm256_mul_pd(_mm256_mul_pd(delta, delta), _mm256_set1_pd(inverse));
    const auto r2 = _mm256_mul_pd(r, r);
    const auto r4 = _mm256_mul_pd(r2, r2);
    constexpr double c3 = 1.0 / 6, c4 = 1.0 / 24, c5 = 1.0 / 120;
    constexpr double c6 = 1.0 / 720, c7 = 1.0 / 5040;
    const auto a = _mm256_add_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(r, _mm256_set1_pd(c3)));
    const auto b = _mm256_add_pd(_mm256_set1_pd(c4), _mm256_mul_pd(r, _mm256_set1_pd(c5)));
    const auto c = _mm256_add_pd(_mm256_set1_pd(c6), _mm256_mul_pd(r, _mm256_set1_pd(c7)));
    const auto p = _mm256_add_pd(a, _mm256_add_pd(_mm256_mul_pd(r2, b), _mm256_mul_pd(r4, c)));
    const auto color = _mm256_add_pd(_mm256_set1_pd(1), _mm256_add_pd(r, _mm256_mul_pd(r2, p)));
    return _mm256_mul_pd(_mm256_loadu_pd(spatial), color);
}

inline void add_density_lanes(__m256 x, float& density, float* neighbors) {
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
        quickshift_add_weight(density, neighbor, weight);
    }
}

template<Index fixed_channels, bool factored>
Index density_row_double(const double* a, const double* b, double& density, double* neighbors,
                         double row_squared, const double* col_squared, Index count,
                         Index pixels, Index channels, double inverse, const double* spatial_weights) {
    const auto weights_at = [&](Index offset) {
        if constexpr (factored) {
            return factored_density_weight(*a, b + offset, spatial_weights + offset, inverse);
        } else {
            const auto d = distances<double, fixed_channels>(
                a, b + offset, row_squared, col_squared + offset, pixels, channels);
            return density_exp(_mm256_mul_pd(d, _mm256_set1_pd(inverse)));
        }
    };
    constexpr Index width = Vector<double>::width;
    auto sum = _mm256_setzero_pd();
    Index i = 0;
    for (; i + width <= count; i += width) {
        const auto weight = weights_at(i);
        _mm256_storeu_pd(neighbors + i, _mm256_add_pd(_mm256_loadu_pd(neighbors + i), weight));
        sum = _mm256_add_pd(sum, weight);
    }
    if (i == 0) return i;
    const auto halves = _mm_add_pd(_mm256_castpd256_pd128(sum), _mm256_extractf128_pd(sum, 1));
    density += _mm_cvtsd_f64(_mm_add_sd(halves, _mm_unpackhi_pd(halves, halves)));
    // Reuse a full in-bounds load for the tail and accumulate only
    // the lanes to the right of the last processed block.
    if (i < count) {
        const auto start = count - width;
        alignas(32) double weights[width];
        _mm256_store_pd(weights, weights_at(start));
        for (; i < count; ++i) quickshift_add_weight(density, neighbors[i], weights[i - start]);
    }
    return i;
}

} // namespace

template<class T, Index fixed_channels>
Index quickshift_density_row(
        const T* a, const T* b, T& density, T* neighbors, T row_squared,
        const T* col_squared, Index count, Index pixels, Index channels, T inverse,
        const double* spatial_weights) {
    if constexpr (std::is_same_v<T, double>) {
        if constexpr (fixed_channels == 1) {
            if (spatial_weights) {
                return density_row_double<1, true>(
                    a, b, density, neighbors, row_squared, col_squared, count, pixels, channels,
                    inverse, spatial_weights);
            }
        }
        return density_row_double<fixed_channels, false>(
            a, b, density, neighbors, row_squared, col_squared, count, pixels, channels,
            inverse, spatial_weights);
    } else {
        using V = Vector<T>;
        Index i = 0;
        T center = density;
        for (; i + V::width <= count; i += V::width) {
            const auto x = V::mul(distances<T, fixed_channels>(
                a, b + i, row_squared, col_squared + i, pixels, channels), V::set(inverse));
            add_density_lanes(x, center, neighbors + i);
        }
        density = center;
        return i;
    }
}

template<class T, Index fixed_channels>
Index quickshift_parent_row(
        const T* a, const T* b, const T* neighbors, T density, T row_squared,
        const T* col_squared, Index count, Index pixels, Index channels, Index start,
        T& closest, Index& parent) {
    using V = Vector<T>;
    const auto center_density = V::set(density);
    auto best = V::set(closest);
    Index i = 0;
    for (; i + V::width <= count; i += V::width) {
        const auto neighbor_densities = V::load(neighbors + i);
        const auto spatial = V::add(V::set(row_squared), V::load(col_squared + i));
        const auto possible = V::eligible(spatial, best, neighbor_densities, center_density);
        if (!V::mask(possible)) continue;
        const auto d = distances<T, fixed_channels>(a, b + i, row_squared, col_squared + i, pixels, channels);
        const auto valid = V::eligible(d, best, neighbor_densities, center_density);
        if (!V::mask(valid)) continue;
        best = V::minimum(V::select(best, d, valid));
        const auto choice = V::matches(d, best, valid);
        parent = start + i + first_lane(choice);
    }
    closest = V::first(best);
    return i;
}

template Index quickshift_density_row<float, 0>(
    const float*, const float*, float&, float*, float, const float*, Index, Index, Index, float, const double*);
template Index quickshift_parent_row<float, 0>(
    const float*, const float*, const float*, float, float, const float*,
    Index, Index, Index, Index, float&, Index&);
template Index quickshift_density_row<float, 1>(
    const float*, const float*, float&, float*, float, const float*, Index, Index, Index, float, const double*);
template Index quickshift_parent_row<float, 1>(
    const float*, const float*, const float*, float, float, const float*,
    Index, Index, Index, Index, float&, Index&);
template Index quickshift_density_row<float, 3>(
    const float*, const float*, float&, float*, float, const float*, Index, Index, Index, float, const double*);
template Index quickshift_parent_row<float, 3>(
    const float*, const float*, const float*, float, float, const float*,
    Index, Index, Index, Index, float&, Index&);
template Index quickshift_density_row<double, 0>(
    const double*, const double*, double&, double*, double, const double*, Index, Index, Index, double, const double*);
template Index quickshift_parent_row<double, 0>(
    const double*, const double*, const double*, double, double, const double*,
    Index, Index, Index, Index, double&, Index&);
template Index quickshift_density_row<double, 1>(
    const double*, const double*, double&, double*, double, const double*, Index, Index, Index, double, const double*);
template Index quickshift_parent_row<double, 1>(
    const double*, const double*, const double*, double, double, const double*,
    Index, Index, Index, Index, double&, Index&);
template Index quickshift_density_row<double, 3>(
    const double*, const double*, double&, double*, double, const double*, Index, Index, Index, double, const double*);
template Index quickshift_parent_row<double, 3>(
    const double*, const double*, const double*, double, double, const double*,
    Index, Index, Index, Index, double&, Index&);
#endif
} // namespace skimagexx::simd
