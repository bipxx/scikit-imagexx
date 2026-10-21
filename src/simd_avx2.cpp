#include "simd.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <type_traits>

namespace skimagexx::simd {
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
}
