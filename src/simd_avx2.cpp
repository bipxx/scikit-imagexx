#include "simd.hpp"
#include "simd_math.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <limits>

namespace skimagexx::simd {
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
    std::size_t x = 0;
    for (; x+8 <= n; x += 8) {
        auto a = _mm256_setzero_pd(), b = _mm256_setzero_pd();
        for (std::size_t k = 0; k < taps; ++k) {
            const auto weight = _mm256_set1_pd(kernel[k]*scale);
            a = _mm256_add_pd(a,_mm256_mul_pd(load4(sources[k]+x),weight));
            b = _mm256_add_pd(b,_mm256_mul_pd(load4(sources[k]+x+4),weight));
        }
        store4(out+x,a); store4(out+x+4,b);
    }
    for (; x < n; ++x) {
        double sum = 0;
        for (std::size_t k = 0; k < taps; ++k) sum += double(sources[k][x])*(kernel[k]*scale);
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
}
