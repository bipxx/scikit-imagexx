#include "simd.hpp"
#include "color_detail.hpp"
#include "perceptual_detail.hpp"
#include "distance_detail.hpp"
#include "distance_metrics.hpp"
#include "simd_math.hpp"
#include <cfenv>
#include <cstring>
#include <climits>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 13
#define SKIMAGEXX_LINUX_WIDE_LAB_POWER 1
#endif

namespace skimagexx::simd {
namespace {
template<class T> struct Vector;
#define COLOR_VECTOR(TYPE, REGISTER, SUFFIX, LANES) \
template<> struct Vector<TYPE> { \
    using R = REGISTER; static constexpr std::size_t lanes = LANES; \
    static R set(TYPE x) { return _mm256_set1_##SUFFIX(x); } \
    static R load(const TYPE* p) { return _mm256_loadu_##SUFFIX(p); } \
    static R add(R a, R b) { return _mm256_add_##SUFFIX(a,b); } \
    static R sub(R a, R b) { return _mm256_sub_##SUFFIX(a,b); } \
    static R negate(R a) { return _mm256_xor_##SUFFIX(a,set(TYPE(-0.0))); } \
    static R mul(R a, R b) { return _mm256_mul_##SUFFIX(a,b); } \
    static R div(R a, R b) { return _mm256_div_##SUFFIX(a,b); } \
    static R minimum(R a, R b) { return _mm256_min_##SUFFIX(a,b); } \
    static R maximum(R a, R b) { return _mm256_max_##SUFFIX(a,b); } \
    static R floor(R a) { return _mm256_floor_##SUFFIX(a); } \
    static R eq(R a, R b) { return _mm256_cmp_##SUFFIX(a,b,_CMP_EQ_OQ); } \
    static R absolute(R a) { return _mm256_andnot_##SUFFIX(set(TYPE(-0.0)),a); } \
    static R gt(R a, R b) { return _mm256_cmp_##SUFFIX(a,b,_CMP_GT_OQ); } \
    static int bits(R mask) { return _mm256_movemask_##SUFFIX(mask); } \
    static R choose(R a, R b, R mask) { return _mm256_blendv_##SUFFIX(a,b,mask); } \
    static R clip(R x) { \
        const auto bounded = _mm256_min_##SUFFIX(_mm256_max_##SUFFIX(x,set(0)),set(1)); \
        return choose(bounded,x,_mm256_cmp_##SUFFIX(x,x,_CMP_UNORD_Q)); \
    } \
};
COLOR_VECTOR(float,__m256,ps,8)
COLOR_VECTOR(double,__m256d,pd,4)
#undef COLOR_VECTOR

__m256 gather(const float* p, std::ptrdiff_t step) {
    const auto s = static_cast<int>(step);
    return _mm256_i32gather_ps(p,_mm256_setr_epi32(0,s,2*s,3*s,4*s,5*s,6*s,7*s),4);
}
__m256d gather(const double* p, std::ptrdiff_t step) {
    const auto s = static_cast<int>(step);
    return _mm256_i32gather_pd(p,_mm_setr_epi32(0,s,2*s,3*s),8);
}

void unpack(const float* p, __m256& r, __m256& g, __m256& b) {
    const auto p0 = _mm256_loadu_ps(p), p1 = _mm256_loadu_ps(p+8), p2 = _mm256_loadu_ps(p+16);
    const auto a = _mm256_permute2f128_ps(p0,p1,0x30), v = _mm256_permute2f128_ps(p0,p2,0x21), c = _mm256_permute2f128_ps(p1,p2,0x30);
    const auto t0 = _mm256_shuffle_ps(a,v,_MM_SHUFFLE(2,1,2,1)), t1 = _mm256_shuffle_ps(v,c,_MM_SHUFFLE(1,0,3,2));
    r = _mm256_shuffle_ps(a,t1,_MM_SHUFFLE(3,0,3,0));
    const auto g0 = _mm256_shuffle_ps(a,v,_MM_SHUFFLE(0,0,1,1)), g1 = _mm256_shuffle_ps(v,c,_MM_SHUFFLE(2,2,3,3));
    g = _mm256_shuffle_ps(g0,g1,_MM_SHUFFLE(2,0,2,0));
    b = _mm256_shuffle_ps(t0,c,_MM_SHUFFLE(3,0,2,1));
}
void unpack(const double* p, __m256d& r, __m256d& g, __m256d& b) {
    const auto a = _mm256_loadu_pd(p), v = _mm256_loadu_pd(p+4), c = _mm256_loadu_pd(p+8);
    r = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,3,0)),
        _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,2,0,0)),4),_mm256_permute4x64_pd(c,_MM_SHUFFLE(1,0,0,0)),8);
    g = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,1)),
        _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,3,0,0)),6),_mm256_permute4x64_pd(c,_MM_SHUFFLE(2,0,0,0)),8);
    b = _mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,2)),
        _mm256_permute4x64_pd(v,_MM_SHUFFLE(0,0,1,0)),2),_mm256_permute4x64_pd(c,_MM_SHUFFLE(3,0,0,0)),12);
}
void unpack_four_stride(const float* p,__m256& r,__m256& g,__m256& b) {
    auto r0=_mm_loadu_ps(p),g0=_mm_loadu_ps(p+4),b0=_mm_loadu_ps(p+8),unused0=_mm_loadu_ps(p+12);
    auto r1=_mm_loadu_ps(p+16),g1=_mm_loadu_ps(p+20),b1=_mm_loadu_ps(p+24),unused1=_mm_loadu_ps(p+28);
    _MM_TRANSPOSE4_PS(r0,g0,b0,unused0);
    _MM_TRANSPOSE4_PS(r1,g1,b1,unused1);
    r=_mm256_insertf128_ps(_mm256_castps128_ps256(r0),r1,1);
    g=_mm256_insertf128_ps(_mm256_castps128_ps256(g0),g1,1);
    b=_mm256_insertf128_ps(_mm256_castps128_ps256(b0),b1,1);
}
void pack(float* p, __m256 r, __m256 g, __m256 b) {
    const auto i0 = _mm256_setr_epi32(0,0,0,1,1,1,2,2), i1 = _mm256_setr_epi32(2,3,3,3,4,4,4,5), i2 = _mm256_setr_epi32(5,5,6,6,6,7,7,7);
    _mm256_storeu_ps(p,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(r,i0),_mm256_permutevar8x32_ps(g,i0),0x92),_mm256_permutevar8x32_ps(b,i0),0x24));
    _mm256_storeu_ps(p+8,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(b,i1),_mm256_permutevar8x32_ps(r,i1),0x92),_mm256_permutevar8x32_ps(g,i1),0x24));
    _mm256_storeu_ps(p+16,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(g,i2),_mm256_permutevar8x32_ps(b,i2),0x92),_mm256_permutevar8x32_ps(r,i2),0x24));
}
void pack(double* p, __m256d r, __m256d g, __m256d b) {
    _mm256_storeu_pd(p,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(r,_MM_SHUFFLE(1,0,0,0)),
        _mm256_permute4x64_pd(g,0),2),_mm256_permute4x64_pd(b,0),4));
    _mm256_storeu_pd(p+4,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(g,_MM_SHUFFLE(2,0,0,1)),
        _mm256_permute4x64_pd(b,_MM_SHUFFLE(0,0,1,0)),2),_mm256_permute4x64_pd(r,_MM_SHUFFLE(0,2,0,0)),4));
    _mm256_storeu_pd(p+8,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(b,_MM_SHUFFLE(3,0,0,2)),
        _mm256_permute4x64_pd(r,_MM_SHUFFLE(0,0,3,0)),2),_mm256_permute4x64_pd(g,_MM_SHUFFLE(0,3,0,0)),4));
}
__m256d power(__m256d x, double exponent) {
    return math::exp4(_mm256_mul_pd(math::log2_4(x),_mm256_set1_pd(exponent*0.69314718055994530942)));
}
__m256 power(__m256 x, float exponent) {
    const auto low = _mm256_cvtpd_ps(power(_mm256_cvtps_pd(_mm256_castps256_ps128(x)),double(exponent)));
    const auto high = _mm256_cvtpd_ps(power(_mm256_cvtps_pd(_mm256_extractf128_ps(x,1)),double(exponent)));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(low),high,1);
}
template<class T> typename Vector<T>::R decode(typename Vector<T>::R x) {
    using V = Vector<T>;
    const auto mask = V::gt(x,V::set(T(0.04045)));
    const auto linear=V::div(x,V::set(T(12.92)));
    if (!V::bits(mask)) return linear;
    const auto adjusted = V::choose(V::set(1),V::div(V::add(x,V::set(T(0.055))),V::set(T(1.055))),mask);
    return V::choose(linear,power(adjusted,T(2.4)),mask);
}
template<class T> typename Vector<T>::R encode(typename Vector<T>::R x) {
    using V = Vector<T>;
    const auto mask = V::gt(x,V::set(T(0.0031308)));
    const auto linear=V::mul(x,V::set(T(12.92)));
    if (!V::bits(mask)) return V::clip(linear);
    const auto adjusted = V::choose(V::set(1),x,mask);
    const auto nonlinear = V::sub(V::mul(V::set(T(1.055)),power(adjusted,T(1.0/2.4))),V::set(T(0.055)));
    return V::clip(V::choose(linear,nonlinear,mask));
}
template<class Input> std::size_t linear_double(const Input* p, double* out, std::size_t n, const detail::ColorSpec& spec) {
    // Keep triples interleaved: replicate input components directly into output
    // lanes instead of unpacking and then repacking three channel vectors.
    __m256d weights[3][3], before[3], after[3];
    for (int block = 0; block < 3; ++block) {
        for (int channel = 0; channel < 3; ++channel) {
            double values[4];
            for (int lane = 0; lane < 4; ++lane) values[lane] = spec.matrix[3*((4*block+lane)%3)+channel];
            weights[block][channel] = _mm256_loadu_pd(values);
        }
        double offsets[4];
        for (int lane = 0; lane < 4; ++lane) offsets[lane] = (4*block+lane)%3 == 0 ? 16.0 : 128.0;
        before[block] = spec.input_offset ? _mm256_loadu_pd(offsets) : _mm256_setzero_pd();
        after[block] = spec.output_offset ? _mm256_loadu_pd(offsets) : _mm256_setzero_pd();
    }
    const auto load = [](const Input* input) {
        if constexpr (std::is_same_v<Input,double>) return _mm256_loadu_pd(input);
        else {
            // Exactly four bytes: three loads cover twelve bytes/four RGB pixels.
            // Do not overread the final packet with a sixteen-byte load.
            std::int32_t packed;std::memcpy(&packed,input,sizeof(packed));
            const auto integers=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed));
            return _mm256_mul_pd(_mm256_cvtepi32_pd(integers),_mm256_set1_pd(1.0/255.0));
        }
    };
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        auto a = load(p+3*i), b = load(p+3*i+4), c = load(p+3*i+8);
        if (spec.input_offset) { a = _mm256_sub_pd(a,before[0]); b = _mm256_sub_pd(b,before[1]); c = _mm256_sub_pd(c,before[2]); }
        const auto finish = [&](int block, __m256d red, __m256d green, __m256d blue) {
            auto value = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(red,weights[block][0]),
                _mm256_mul_pd(green,weights[block][1])),_mm256_mul_pd(blue,weights[block][2]));
            if (spec.output_offset) value = _mm256_add_pd(value,after[block]);
            _mm256_storeu_pd(out+3*i+4*block,value);
        };
        finish(0,_mm256_permute4x64_pd(a,_MM_SHUFFLE(3,0,0,0)),
            _mm256_blend_pd(_mm256_permute4x64_pd(a,85),_mm256_permute4x64_pd(b,0),8),
            _mm256_blend_pd(_mm256_permute4x64_pd(a,170),_mm256_permute4x64_pd(b,85),8));
        finish(1,_mm256_blend_pd(_mm256_permute4x64_pd(a,255),_mm256_permute4x64_pd(b,170),12),
            _mm256_permute4x64_pd(b,_MM_SHUFFLE(3,3,0,0)),
            _mm256_blend_pd(_mm256_permute4x64_pd(b,85),_mm256_permute4x64_pd(c,0),12));
        finish(2,_mm256_blend_pd(_mm256_permute4x64_pd(b,170),_mm256_permute4x64_pd(c,85),14),
            _mm256_blend_pd(_mm256_permute4x64_pd(b,255),_mm256_permute4x64_pd(c,170),14),
            _mm256_permute4x64_pd(c,_MM_SHUFFLE(3,3,3,0)));
    }
    return i;
}
template<class Input> std::size_t run(const Input* p, detail::ColorOutput<Input>* out, std::size_t n, const detail::ColorSpec& spec,
                                    std::ptrdiff_t pixel_stride, std::ptrdiff_t channel_stride) {
    using T = detail::ColorOutput<Input>;
    using V = Vector<T>;
    using R = typename V::R;
    if constexpr (std::is_same_v<Input,double>)
        if (spec.transfer == 0 && pixel_stride == 3 && channel_stride == 1) return linear_double(p,out,n,spec);
    if constexpr (std::is_same_v<Input,std::uint8_t>)
        if (spec.transfer == 0 && !spec.input_offset && pixel_stride == 3 && channel_stride == 1)
            return linear_double(p,out,n,spec);
    const double* lookup = nullptr;
    if constexpr (std::is_same_v<Input,std::uint8_t>)
        if (spec.transfer == 1) lookup = detail::srgb_decode_table().data();
    R matrix[9];
    for (int j = 0; j < 9; ++j) matrix[j] = V::set(T(spec.matrix[j]));
    std::size_t i = 0;
    for (; i+V::lanes <= n; i += V::lanes) {
        R r,g,b;
        if constexpr (std::is_floating_point_v<Input>) {
            const auto* start = p+std::ptrdiff_t(i)*pixel_stride;
            if (pixel_stride == 3 && channel_stride == 1) unpack(start,r,g,b);
            else if (pixel_stride == 1) {
                r = V::load(start); g = V::load(start+channel_stride); b = V::load(start+2*channel_stride);
            } else if (pixel_stride == 0) {
                r = V::set(start[0]); g = V::set(start[channel_stride]); b = V::set(start[2*channel_stride]);
            } else if (pixel_stride >= INT_MIN/7 && pixel_stride <= INT_MAX/7) {
                r = gather(start,pixel_stride); g = gather(start+channel_stride,pixel_stride); b = gather(start+2*channel_stride,pixel_stride);
            } else {
                T converted[3*V::lanes];
                for (std::size_t j = 0; j < 3*V::lanes; ++j)
                    converted[j] = start[std::ptrdiff_t(j/3)*pixel_stride+std::ptrdiff_t(j%3)*channel_stride];
                unpack(converted,r,g,b);
            }
            if (spec.input_offset) { r = V::sub(r,V::set(16)); g = V::sub(g,V::set(128)); b = V::sub(b,V::set(128)); }
        } else {
            // Bounded stack block: no full converted image or extra memory pass.
            double converted[12];
            for (int j = 0; j < 12; ++j) {
                if constexpr (std::is_same_v<Input,bool>) converted[j] = reinterpret_cast<const unsigned char*>(p+3*i)[j] != 0;
                else converted[j] = detail::color_value(p[3*i+j],j%3,spec.input_offset);
                if constexpr (std::is_same_v<Input,std::uint8_t>)
                    if (lookup) converted[j] = lookup[p[3*i+j]];
            }
            unpack(converted,r,g,b);
        }
        if (spec.transfer == 1 && !std::is_same_v<Input,std::uint8_t>) { r = decode<T>(r); g = decode<T>(g); b = decode<T>(b); }
        R values[3];
        for (int j = 0; j < 3; ++j) {
            values[j] = V::add(V::add(V::mul(r,matrix[3*j]),V::mul(g,matrix[3*j+1])),V::mul(b,matrix[3*j+2]));
            if (spec.output_offset) values[j] = V::add(values[j],V::set(T(j == 0 ? 16 : 128)));
            if (spec.transfer == 2) values[j] = encode<T>(values[j]);
        }
        pack(out+3*i,values[0],values[1],values[2]);
    }
    return i;
}
}
std::size_t color_transform(const void* source, void* out, std::size_t n, PixelType type, const detail::ColorSpec& spec,
                           std::ptrdiff_t pixel_stride, std::ptrdiff_t channel_stride) {
    switch (type) {
        case PixelType::boolean: return run(static_cast<const bool*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::uint8: return run(static_cast<const std::uint8_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::int8: return run(static_cast<const std::int8_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::uint16: return run(static_cast<const std::uint16_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::int16: return run(static_cast<const std::int16_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::uint32: return run(static_cast<const std::uint32_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::int32: return run(static_cast<const std::int32_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::uint64: return run(static_cast<const std::uint64_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::int64: return run(static_cast<const std::int64_t*>(source),static_cast<double*>(out),n,spec,3,1);
        case PixelType::float32: return run(static_cast<const float*>(source),static_cast<float*>(out),n,spec,pixel_stride,channel_stride);
        case PixelType::float64: return run(static_cast<const double*>(source),static_cast<double*>(out),n,spec,pixel_stride,channel_stride);
        default: return 0;
    }
}
namespace {
bool all_finite(__m256 x) { return _mm256_movemask_ps(_mm256_cmp_ps(Vector<float>::absolute(x),_mm256_set1_ps(INFINITY),_CMP_LT_OQ)) == 255; }
bool all_finite(__m256d x) { return _mm256_movemask_pd(_mm256_cmp_pd(Vector<double>::absolute(x),_mm256_set1_pd(INFINITY),_CMP_LT_OQ)) == 15; }
int mask_bits(__m256 x) { return _mm256_movemask_ps(x); }
int mask_bits(__m256d x) { return _mm256_movemask_pd(x); }
__m256 bytes_mod6(__m256 x) {
    auto ints = _mm256_and_si256(_mm256_cvttps_epi32(x),_mm256_set1_epi32(255));
    const auto bytes = _mm256_cvtepi32_ps(ints);
    return _mm256_sub_ps(bytes,_mm256_mul_ps(_mm256_floor_ps(_mm256_div_ps(bytes,_mm256_set1_ps(6))),_mm256_set1_ps(6)));
}
__m256d bytes_mod6(__m256d x) {
    const auto ints = _mm_and_si128(_mm256_cvttpd_epi32(x),_mm_set1_epi32(255));
    const auto bytes = _mm256_cvtepi32_pd(ints);
    return _mm256_sub_pd(bytes,_mm256_mul_pd(_mm256_floor_pd(_mm256_div_pd(bytes,_mm256_set1_pd(6))),_mm256_set1_pd(6)));
}
__m256d cube_root(__m256d x) {
    if (mask_bits(_mm256_cmp_pd(x,_mm256_set1_pd(1e8),_CMP_GT_OQ))) {
        double v[4]; _mm256_storeu_pd(v,x); for (auto& a : v) a = std::cbrt(a); return _mm256_loadu_pd(v);
    }
    return power(x,1.0/3.0);
}
__m256 cube_root(__m256 x) {
    const auto lo = _mm256_cvtpd_ps(cube_root(_mm256_cvtps_pd(_mm256_castps256_ps128(x))));
    const auto hi = _mm256_cvtpd_ps(cube_root(_mm256_cvtps_pd(_mm256_extractf128_ps(x,1))));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1);
}
__m256d luv_cube(__m256d x,__m256d) { return _mm256_mul_pd(_mm256_mul_pd(x,x),x); }
__m256 luv_cube(__m256 x,__m256 nonlinear) {
    // One float32 power rounding affects the cancellation in Luv->XYZ and is
    // amplified by the sRGB transfer near black. Match NumPy's powf stage;
    // the remaining inverse arithmetic and transfer stay vectorized.
    float values[8]; _mm256_storeu_ps(values,x);
    const auto mask=static_cast<unsigned>(mask_bits(nonlinear));
    for (unsigned j=0;j<8;++j) if (mask&(1u<<j)) values[j]=std::pow(values[j],3.0f);
    return _mm256_loadu_ps(values);
}
__m256d multiply_white(__m256d x, double white) { return _mm256_mul_pd(x,_mm256_set1_pd(white)); }
__m256 multiply_white(__m256 x, double white) {
    const auto lo = _mm256_cvtpd_ps(multiply_white(_mm256_cvtps_pd(_mm256_castps256_ps128(x)),white));
    const auto hi = _mm256_cvtpd_ps(multiply_white(_mm256_cvtps_pd(_mm256_extractf128_ps(x,1)),white));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1);
}
void polar4(__m256d x, __m256d y, __m256d& radius, __m256d& angle) {
    using V = Vector<double>;
    const auto ax = V::absolute(x), ay = V::absolute(y), high = V::maximum(ax,ay), low = V::minimum(ax,ay);
    const auto unusual = _mm256_or_pd(V::gt(high,V::set(1e150)),_mm256_andnot_pd(V::eq(high,V::set(0)),V::gt(V::set(1e-150),high)));
    if (!all_finite(x) || !all_finite(y) || mask_bits(unusual)) {
        double a[4], b[4], r[4], h[4]; _mm256_storeu_pd(a,x); _mm256_storeu_pd(b,y);
        for (int j = 0; j < 4; ++j) { r[j] = std::hypot(a[j],b[j]); const auto t = std::atan2(b[j],a[j]); h[j] = t+(t < 0 ? 2*detail::pi : 0); }
        radius = _mm256_loadu_pd(r); angle = _mm256_loadu_pd(h); return;
    }
    const auto ratio = V::div(low,V::choose(high,V::set(1),V::eq(high,V::set(0))));
    radius = V::mul(high,_mm256_sqrt_pd(V::add(V::set(1),V::mul(ratio,ratio))));
    const auto reduce = V::gt(ratio,V::set(0.4142135623730950488));
    const auto z = V::choose(ratio,V::div(V::sub(ratio,V::set(1)),V::add(ratio,V::set(1))),reduce);
    const auto square = V::mul(z,z);
    auto series = V::set(-1.0/31);
    for (int k = 14; k >= 0; --k) series = V::add(V::set((k%2 ? -1.0 : 1.0)/double(2*k+1)),V::mul(square,series));
    angle = V::add(V::mul(z,series),V::choose(V::set(0),V::set(detail::pi/4),reduce));
    angle = V::choose(angle,V::sub(V::set(detail::pi/2),angle),V::gt(ay,ax));
    angle = V::choose(angle,V::sub(V::set(detail::pi),angle),x); // Sign bit includes -0.
    angle = _mm256_xor_pd(angle,_mm256_and_pd(y,V::set(-0.0)));
    angle = V::add(angle,V::choose(V::set(0),V::set(2*detail::pi),V::gt(V::set(0),angle)));
}
void polar(__m256d x, __m256d y, __m256d& r, __m256d& h) { polar4(x,y,r,h); }
void polar(__m256 x, __m256 y, __m256& r, __m256& h) {
    __m256d r0,h0,r1,h1;
    polar4(_mm256_cvtps_pd(_mm256_castps256_ps128(x)),_mm256_cvtps_pd(_mm256_castps256_ps128(y)),r0,h0);
    polar4(_mm256_cvtps_pd(_mm256_extractf128_ps(x,1)),_mm256_cvtps_pd(_mm256_extractf128_ps(y,1)),r1,h1);
    r = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(r0)),_mm256_cvtpd_ps(r1),1);
    h = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(h0)),_mm256_cvtpd_ps(h1),1);
}
void sincos4(__m256d h, __m256d& sine, __m256d& cosine) {
    using V = Vector<double>;
    const auto k = _mm256_round_pd(V::mul(h,V::set(2/detail::pi)),_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto r = V::sub(h,V::mul(k,V::set(1.57079632673412561417)));
    r = V::sub(r,V::mul(k,V::set(6.07710050650619224932e-11)));
    if (!all_finite(h) || mask_bits(V::gt(V::absolute(h),V::set(1e6))) || mask_bits(V::gt(V::set(1e-8),V::absolute(r)))) {
        double v[4], s[4], c[4]; _mm256_storeu_pd(v,h);
        for (int j = 0; j < 4; ++j) { s[j] = std::sin(v[j]); c[j] = std::cos(v[j]); }
        sine = _mm256_loadu_pd(s); cosine = _mm256_loadu_pd(c); return;
    }
    constexpr double scoeff[] = {1,-1.0/6,1.0/120,-1.0/5040,1.0/362880,-1.0/39916800,1.0/6227020800,-1.0/1307674368000,1.0/355687428096000};
    constexpr double ccoeff[] = {1,-1.0/2,1.0/24,-1.0/720,1.0/40320,-1.0/3628800,1.0/479001600,-1.0/87178291200,1.0/20922789888000};
    const auto square = V::mul(r,r);
    auto s = V::set(scoeff[8]), c = V::set(ccoeff[8]);
    for (int j = 7; j >= 0; --j) { s = V::add(V::set(scoeff[j]),V::mul(square,s)); c = V::add(V::set(ccoeff[j]),V::mul(square,c)); }
    s = V::mul(r,s);
    const auto quadrant = _mm256_cvtepi32_pd(_mm_and_si128(_mm256_cvtpd_epi32(k),_mm_set1_epi32(3)));
    const auto ns = _mm256_xor_pd(s,V::set(-0.0)), nc = _mm256_xor_pd(c,V::set(-0.0));
    sine = V::choose(V::choose(V::choose(s,c,V::eq(quadrant,V::set(1))),ns,V::eq(quadrant,V::set(2))),nc,V::eq(quadrant,V::set(3)));
    cosine = V::choose(V::choose(V::choose(c,ns,V::eq(quadrant,V::set(1))),nc,V::eq(quadrant,V::set(2))),s,V::eq(quadrant,V::set(3)));
}
void sincos(__m256d h, __m256d& s, __m256d& c) { sincos4(h,s,c); }
void sincos(__m256 h, __m256& s, __m256& c) {
    __m256d s0,c0,s1,c1;
    sincos4(_mm256_cvtps_pd(_mm256_castps256_ps128(h)),s0,c0);
    sincos4(_mm256_cvtps_pd(_mm256_extractf128_ps(h,1)),s1,c1);
    s = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(s0)),_mm256_cvtpd_ps(s1),1);
    c = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(c0)),_mm256_cvtpd_ps(c1),1);
}
void unpack_uint8_triples(const std::uint8_t* input, __m128i& x, __m128i& y, __m128i& z) {
    // Load exactly twelve bytes, including the final four-pixel packet.
    std::int64_t low;
    std::int32_t high;
    std::memcpy(&low,input,sizeof(low));
    std::memcpy(&high,input+sizeof(low),sizeof(high));
    const auto pixels = _mm_unpacklo_epi64(_mm_cvtsi64_si128(low),_mm_cvtsi32_si128(high));
    x = _mm_cvtepu8_epi32(_mm_shuffle_epi8(pixels,_mm_setr_epi8(0,3,6,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)));
    y = _mm_cvtepu8_epi32(_mm_shuffle_epi8(pixels,_mm_setr_epi8(1,4,7,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)));
    z = _mm_cvtepu8_epi32(_mm_shuffle_epi8(pixels,_mm_setr_epi8(2,5,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1)));
}

std::size_t xyz8_to_lab(const std::uint8_t* input, double* output, std::size_t n,
    const std::array<double,3>& white) {
    using V = Vector<double>;
    double curves[3][256];
    const auto scale=V::set(1.0/255.0);
    // Cache the existing SIMD curve for each possible byte, within this call.
    // The admitted white range keeps all lanes on the same cube-root implementation.
    for (int j=0;j<3;++j) {
        const auto w=V::set(white[j]);
        for (int k=0;k<256;k+=4) {
            const auto bytes=_mm_setr_epi32(k,k+1,k+2,k+3);
            const auto scaled=V::div(V::mul(_mm256_cvtepi32_pd(bytes),scale),w);
            const auto mask=V::gt(scaled,V::set(0.008856));
            auto curve=V::add(V::mul(V::set(7.787),scaled),V::set(16.0/116.0));
            if (V::bits(mask)) curve=V::choose(curve,cube_root(V::choose(V::set(1),scaled,mask)),mask);
            _mm256_storeu_pd(curves[j]+k,curve);
        }
    }
    std::size_t i=0;
    for (;i+4<=n;i+=4) {
        __m128i a,b,c;
        unpack_uint8_triples(input+3*i,a,b,c);
        const auto x=_mm256_i32gather_pd(curves[0],a,8);
        const auto y=_mm256_i32gather_pd(curves[1],b,8);
        const auto z=_mm256_i32gather_pd(curves[2],c,8);
        pack(output+3*i,V::sub(V::mul(V::set(116),y),V::set(16)),
            V::mul(V::set(500),V::sub(x,y)),V::mul(V::set(200),V::sub(y,z)));
    }
    return i;
}

std::size_t lab8_to_lch(const std::uint8_t* input, double* output, std::size_t n) {
    const auto scale=_mm256_set1_pd(1.0/255.0);
    std::size_t i=0;
    for (;i+4<=n;i+=4) {
        __m128i x,y,z;
        unpack_uint8_triples(input+3*i,x,y,z);
        __m256d radius,angle;
        polar(_mm256_mul_pd(_mm256_cvtepi32_pd(y),scale),
              _mm256_mul_pd(_mm256_cvtepi32_pd(z),scale),radius,angle);
        pack(output+3*i,_mm256_mul_pd(_mm256_cvtepi32_pd(x),scale),radius,angle);
    }
    return i;
}

template<class Input> std::size_t run_perceptual(const Input* p, detail::ColorOutput<Input>* out, std::size_t n,
    const detail::PerceptualSpec& spec, std::ptrdiff_t pixels, std::ptrdiff_t channels, PerceptualInfo& info) {
    using T = detail::ColorOutput<Input>; using V = Vector<T>; using R = typename V::R; using Op = PerceptualTransform;
    const auto op = spec.transform;
    const bool rgb_forward=op==Op::rgb2lab || op==Op::rgb2luv;
    const auto* lut = rgb_forward && std::is_same_v<Input,std::uint8_t> ? detail::srgb_decode_table().data() : nullptr;
    const auto& matrix = detail::color_spec(rgb_forward ? ColorTransform::rgb2xyz : ColorTransform::xyz2rgb).matrix;
    R weights[9]; for (int j = 0; j < 9; ++j) weights[j] = V::set(T(matrix[j]));
    std::size_t i = 0;
    for (; i+V::lanes <= n; i += V::lanes) {
        const auto* start = p+std::ptrdiff_t(i)*pixels;
        R x,y,z;
        if constexpr (std::is_floating_point_v<Input>) {
            if (pixels == 3 && channels == 1) unpack(start,x,y,z);
            else if (pixels == 1) { x=V::load(start); y=V::load(start+channels); z=V::load(start+2*channels); }
            else if (pixels == 0) { x=V::set(start[0]); y=V::set(start[channels]); z=V::set(start[2*channels]); }
            else if (pixels >= INT_MIN/7 && pixels <= INT_MAX/7) { x=gather(start,pixels); y=gather(start+channels,pixels); z=gather(start+2*channels,pixels); }
            else {
                T block[3*V::lanes];
                for (std::size_t j=0;j<3*V::lanes;++j) block[j]=start[std::ptrdiff_t(j/3)*pixels+std::ptrdiff_t(j%3)*channels];
                unpack(block,x,y,z);
            }
        } else {
            double block[12];
            for (int j=0;j<12;++j) {
                const auto offset=std::ptrdiff_t(j/3)*pixels+std::ptrdiff_t(j%3)*channels;
                if constexpr (std::is_same_v<Input,bool>) block[j]=reinterpret_cast<const unsigned char*>(start)[offset] != 0;
                else block[j]=detail::color_value(start[offset],0,false);
                if constexpr (std::is_same_v<Input,std::uint8_t>) if (lut) block[j]=lut[start[offset]];
            }
            unpack(block,x,y,z);
        }
        bool scalar_block = false;
        if (op == Op::rgb2hsv || op == Op::hsv2rgb || static_cast<int>(op)>=8) scalar_block = !all_finite(x) || !all_finite(y) || !all_finite(z);
        if (!scalar_block && op == Op::rgb2hsv) {
            const auto high=V::maximum(V::maximum(x,y),z), low=V::minimum(V::minimum(x,y),z), delta=V::sub(high,low);
            const auto zero=V::eq(delta,V::set(0)), divisor=V::choose(delta,V::set(1),zero);
            auto h=V::div(V::sub(y,z),divisor);
            h=V::choose(h,V::add(V::set(2),V::div(V::sub(z,x),divisor)),V::eq(y,high));
            h=V::choose(h,V::add(V::set(4),V::div(V::sub(x,y),divisor)),V::eq(z,high));
            h=V::div(h,V::set(6)); h=V::sub(h,V::floor(h));
            const auto s=V::choose(V::div(delta,V::choose(high,V::set(1),zero)),V::set(0),zero);
            x=V::choose(h,V::set(0),zero); y=s; z=high;
            scalar_block = !all_finite(delta) || !all_finite(x) || !all_finite(y);
        } else if (!scalar_block && op == Op::hsv2rgb) {
            const auto h6=V::mul(x,V::set(6)), hi=V::floor(h6);
            scalar_block = !all_finite(hi) || mask_bits(V::gt(V::set(T(-2147483648.0)),hi)) ||
                           mask_bits(V::gt(hi,V::set(T(2147483648.0)))) || mask_bits(V::eq(hi,V::set(T(2147483648.0))));
            if (!scalar_block) {
                const auto f=V::sub(h6,hi), value=z, s=y;
                const auto p0=V::mul(value,V::sub(V::set(1),s)), q=V::mul(value,V::sub(V::set(1),V::mul(f,s))),
                           t=V::mul(value,V::sub(V::set(1),V::mul(V::sub(V::set(1),f),s)));
                const auto sector=bytes_mod6(hi);
                x=value; y=t; z=p0;
                x=V::choose(x,q,V::eq(sector,V::set(1))); y=V::choose(y,value,V::eq(sector,V::set(1)));
                for (int j=2;j<=5;++j) {
                    const auto mask=V::eq(sector,V::set(T(j)));
                    const auto r=j==2 || j==3 ? p0 : j==4 ? t : value;
                    const auto g=j==2 ? value : j==3 ? q : p0;
                    const auto b=j==2 ? t : j==3 || j==4 ? value : q;
                    x=V::choose(x,r,mask); y=V::choose(y,g,mask); z=V::choose(z,b,mask);
                }
            }
        } else if (!scalar_block && (op==Op::xyz2luv || op==Op::rgb2luv)) {
            if (op==Op::rgb2luv) {
                if constexpr (!std::is_same_v<Input,std::uint8_t>) { x=decode<T>(x); y=decode<T>(y); z=decode<T>(z); }
                const auto a=x,b=y,c=z;
                x=V::add(V::add(V::mul(a,weights[0]),V::mul(b,weights[1])),V::mul(c,weights[2]));
                y=V::add(V::add(V::mul(a,weights[3]),V::mul(b,weights[4])),V::mul(c,weights[5]));
                z=V::add(V::add(V::mul(a,weights[6]),V::mul(b,weights[7])),V::mul(c,weights[8]));
            }
            const auto scaled=V::div(y,V::set(T(spec.white[1]))),mask=V::gt(scaled,V::set(T(.008856)));
            auto light=V::mul(V::set(T(903.3)),scaled);
            if (V::bits(mask)) {
                const auto positive=V::choose(V::set(1),scaled,mask);
                light=V::choose(light,V::sub(V::mul(V::set(116),cube_root(positive)),V::set(16)),mask);
            }
            const auto denominator=V::add(V::add(V::add(x,V::mul(V::set(15),y)),V::mul(V::set(3),z)),V::set(std::numeric_limits<T>::epsilon()));
            const auto factor=V::mul(V::set(13),light);
            const auto u=V::mul(factor,V::sub(V::div(V::mul(V::set(4),x),denominator),V::set(T(spec.uv[0]))));
            const auto v=V::mul(factor,V::sub(V::div(V::mul(V::set(9),y),denominator),V::set(T(spec.uv[1]))));
            x=light; y=u; z=v;
        } else if (!scalar_block && (op==Op::luv2xyz || op==Op::luv2rgb)) {
            const auto light=x,nonlinear=V::gt(light,V::set(T(7.999625)));
            auto luminance=V::div(light,V::set(T(903.3)));
            if (mask_bits(nonlinear)) {
                const auto curve=V::div(V::add(light,V::set(16)),V::set(116));
                luminance=V::choose(luminance,luv_cube(curve,nonlinear),nonlinear);
            }
            luminance=V::mul(luminance,V::set(T(spec.white[1])));
            const auto denominator=V::add(V::mul(V::set(13),light),V::set(std::numeric_limits<T>::epsilon()));
            const auto a=V::add(V::set(T(spec.uv[0])),V::div(y,denominator)),b=V::add(V::set(T(spec.uv[1])),V::div(z,denominator));
            const auto c=V::mul(V::mul(V::set(3),luminance),V::sub(V::mul(V::set(5),b),V::set(3)));
            z=V::div(V::sub(V::mul(V::sub(a,V::set(4)),c),V::mul(V::mul(V::mul(V::set(15),a),b),luminance)),V::mul(V::set(12),b));
            x=V::negate(V::add(V::div(c,b),V::mul(V::set(3),z))); y=luminance;
            if (op==Op::luv2rgb) {
                const auto a0=x,b0=y,c0=z;
                x=encode<T>(V::add(V::add(V::mul(a0,weights[0]),V::mul(b0,weights[1])),V::mul(c0,weights[2])));
                y=encode<T>(V::add(V::add(V::mul(a0,weights[3]),V::mul(b0,weights[4])),V::mul(c0,weights[5])));
                z=encode<T>(V::add(V::add(V::mul(a0,weights[6]),V::mul(b0,weights[7])),V::mul(c0,weights[8])));
            }
        } else if (op == Op::xyz2lab || op == Op::rgb2lab) {
            if (op == Op::rgb2lab) {
                if constexpr (!std::is_same_v<Input,std::uint8_t>) { x=decode<T>(x); y=decode<T>(y); z=decode<T>(z); }
                const auto a=x,b=y,c=z;
                x=V::add(V::add(V::mul(a,weights[0]),V::mul(b,weights[1])),V::mul(c,weights[2]));
                y=V::add(V::add(V::mul(a,weights[3]),V::mul(b,weights[4])),V::mul(c,weights[5]));
                z=V::add(V::add(V::mul(a,weights[6]),V::mul(b,weights[7])),V::mul(c,weights[8]));
            }
            R values[]={x,y,z};
            for (int j=0;j<3;++j) {
                auto v=V::div(values[j],V::set(T(spec.white[j])));
                const auto mask=V::gt(v,V::set(T(0.008856)));
                values[j]=V::add(V::mul(V::set(T(7.787)),v),V::set(T(16.0/116.0)));
                if (V::bits(mask)) {
                    const auto positive=V::choose(V::set(1),v,mask);
                    values[j]=V::choose(values[j],cube_root(positive),mask);
                }
            }
            x=V::sub(V::mul(V::set(116),values[1]),V::set(16));
            y=V::mul(V::set(500),V::sub(values[0],values[1])); z=V::mul(V::set(200),V::sub(values[1],values[2]));
        } else if (op == Op::lab2xyz || op == Op::lab2rgb) {
            const auto fy=V::div(V::add(x,V::set(16)),V::set(116)), fx=V::add(V::div(y,V::set(500)),fy);
            auto fz=V::sub(fy,V::div(z,V::set(200)));
            const auto invalid=V::gt(V::set(0),fz); auto bits=static_cast<unsigned>(mask_bits(invalid));
            while (bits) { ++info.negative_z; bits &= bits-1; }
            fz=V::choose(fz,V::set(0),invalid);
            R values[]={fx,fy,fz};
            for (int j=0;j<3;++j) {
                const auto v=values[j], mask=V::gt(v,V::set(T(0.2068966)));
                values[j]=multiply_white(V::choose(V::div(V::sub(v,V::set(T(16.0/116.0))),V::set(T(7.787))),V::mul(V::mul(v,v),v),mask),spec.white[j]);
            }
            x=values[0]; y=values[1]; z=values[2];
            if (op == Op::lab2rgb) {
                const auto a=x,b=y,c=z;
                x=encode<T>(V::add(V::add(V::mul(a,weights[0]),V::mul(b,weights[1])),V::mul(c,weights[2])));
                y=encode<T>(V::add(V::add(V::mul(a,weights[3]),V::mul(b,weights[4])),V::mul(c,weights[5])));
                z=encode<T>(V::add(V::add(V::mul(a,weights[6]),V::mul(b,weights[7])),V::mul(c,weights[8])));
            }
        } else if (op == Op::lab2lch) { R radius,angle; polar(y,z,radius,angle); y=radius; z=angle; }
        else if (op == Op::lch2lab) { R sine,cosine; sincos(z,sine,cosine); z=V::mul(y,sine); y=V::mul(y,cosine); }
        if (scalar_block) {
            for (std::size_t j=0;j<V::lanes;++j) detail::perceptual_pixel(start+std::ptrdiff_t(j)*pixels,out+(i+j)*spec.channels,spec,channels,info);
        } else if (spec.channels == 3) pack(out+3*i,x,y,z);
        else {
            T block[3*V::lanes]; pack(block,x,y,z);
            for (std::size_t j=0;j<V::lanes;++j) {
                std::copy(block+3*j,block+3*j+3,out+(i+j)*spec.channels);
                for (std::size_t c=3;c<spec.channels;++c) {
                    const auto offset=std::ptrdiff_t(j)*pixels+std::ptrdiff_t(c)*channels;
                    if constexpr (std::is_same_v<Input,bool>) out[(i+j)*spec.channels+c]=reinterpret_cast<const unsigned char*>(start)[offset] != 0;
                    else out[(i+j)*spec.channels+c]=detail::color_value(start[offset],0,false);
                }
            }
        }
    }
    return i;
}
template<class T> bool ordinary_lab_span(const T* input, std::size_t elements) {
    // Integer comparisons preserve all floating flags and reject negative zero.
    constexpr bool single = std::is_same_v<T,float>;
    using Bits = std::conditional_t<single,std::uint32_t,std::uint64_t>;
    constexpr Bits one = single ? 0x3f800000ULL : 0x3ff0000000000000ULL;
    constexpr Bits minimum = single ? 0x00800000ULL : 0x0010000000000000ULL;
    constexpr auto lanes = Vector<T>::lanes;
    const auto set = [](Bits v) {
        if constexpr (single) return _mm256_set1_epi32(static_cast<std::int32_t>(v));
        else return _mm256_set1_epi64x(static_cast<std::int64_t>(v));
    };
    const auto gt = [](__m256i a,__m256i b) {
        if constexpr (single) return _mm256_cmpgt_epi32(a,b);
        else return _mm256_cmpgt_epi64(a,b);
    };
    const auto zero = set(0), below_normal = set(minimum-1), upper = set(one);
    std::size_t i=0;
    for (;i+lanes<=elements;i+=lanes) {
        const auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto is_zero = single ? _mm256_cmpeq_epi32(bits,zero) : _mm256_cmpeq_epi64(bits,zero);
        const auto valid = _mm256_andnot_si256(gt(bits,upper),_mm256_or_si256(gt(bits,below_normal),is_zero));
        if (_mm256_movemask_epi8(valid)!=-1) return false;
    }
    for (;i<elements;++i) {
        Bits bits;
        std::memcpy(&bits,input+i,sizeof(bits));
        if (bits>one || (bits && bits<minimum)) return false;
    }
    return true;
}

using LabPair = __m256d[2];
constexpr double lab_log_coeff[] = {1,1.0/3,1.0/5,1.0/7,1.0/9,1.0/11,1.0/13,1.0/15,1.0/17,1.0/19};
constexpr double lab_exp_coeff[] = {1,1,1.0/2,1.0/6,1.0/24,1.0/120,1.0/720,1.0/5040,
    1.0/40320,1.0/362880,1.0/3628800,1.0/39916800,1.0/479001600,1.0/6227020800};

// The same reduced log2 and exp polynomials as math::log2_4 and math::exp4.
// This caller only supplies bounded positive normal inputs or the neutral 1,
// so neither libm fallback is reachable. Evaluate both halves at each degree.
template<bool prescaled> void ordinary_lab_power(LabPair& x, double exponent) {
    __m256d e[2], t[2], square[2], polynomial[2], k[2];
    const auto one = _mm256_set1_pd(1);
    for (int i=0;i<2;++i) {
        const auto bits = _mm256_castpd_si256(x[i]);
        const auto exponents = _mm256_srli_epi64(bits,52);
        e[i] = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(exponents,_mm256_set1_epi64x(0x4330000000000000LL))),
            _mm256_set1_pd(4503599627371519.0));
        auto m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(bits,_mm256_set1_epi64x(0x000fffffffffffffLL)),
            _mm256_set1_epi64x(0x3ff0000000000000LL)));
        const auto upper = _mm256_cmp_pd(m,_mm256_set1_pd(1.4142135623730951),_CMP_GT_OQ);
        m = _mm256_blendv_pd(m,_mm256_mul_pd(m,_mm256_set1_pd(0.5)),upper);
        e[i] = _mm256_add_pd(e[i],_mm256_and_pd(upper,one));
        t[i] = _mm256_div_pd(_mm256_sub_pd(m,one),_mm256_add_pd(m,one));
        square[i] = _mm256_mul_pd(t[i],t[i]);
        polynomial[i] = _mm256_set1_pd(lab_log_coeff[9]);
    }
    for (int degree=8;degree>=0;--degree)
        for (int i=0;i<2;++i)
            polynomial[i] = _mm256_add_pd(_mm256_set1_pd(lab_log_coeff[degree]),_mm256_mul_pd(square[i],polynomial[i]));
    const auto scale = _mm256_set1_pd(prescaled ? exponent : exponent*0.69314718055994530942);
    for (int i=0;i<2;++i) {
        const auto log = _mm256_add_pd(e[i],_mm256_mul_pd(_mm256_mul_pd(t[i],polynomial[i]),_mm256_set1_pd(2.8853900817779268147)));
        const auto y = _mm256_mul_pd(log,scale);
        k[i] = _mm256_round_pd(_mm256_mul_pd(y,_mm256_set1_pd(1.4426950408889634074)),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const auto r = _mm256_sub_pd(y,_mm256_mul_pd(k[i],_mm256_set1_pd(0.69314718055994528623)));
        x[i] = _mm256_sub_pd(r,_mm256_mul_pd(k[i],_mm256_set1_pd(2.3190468138462995584e-17)));
        polynomial[i] = _mm256_set1_pd(lab_exp_coeff[13]);
    }
    for (int degree=12;degree>=0;--degree)
        for (int i=0;i<2;++i)
            polynomial[i] = _mm256_add_pd(_mm256_set1_pd(lab_exp_coeff[degree]),_mm256_mul_pd(x[i],polynomial[i]));
    for (int i=0;i<2;++i) {
        const auto integers = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k[i]));
        const auto bits = _mm256_slli_epi64(_mm256_add_epi64(integers,_mm256_set1_epi64x(1023)),52);
        x[i] = _mm256_mul_pd(polynomial[i],_mm256_castsi256_pd(bits));
    }
}

#ifdef SKIMAGEXX_LINUX_WIDE_LAB_POWER
bool ordinary_lab_wide_available() {
    // GCC reports AVX-512 CPU bits only when the OS has enabled ZMM state.
    // The two halves must not reorder delivery of an unmasked SIMD exception.
    return (_mm_getcsr() & _MM_MASK_MASK) == _MM_MASK_MASK && __builtin_cpu_supports("x86-64-v4");
}

// Pointer ABI leaves all ZMM values inside the target function. This is the
// positive-normal polynomial above, with each half in the same order.
__attribute__((target("avx512f"), noinline))
void ordinary_lab_power_wide(void* values, double scale) {
    __m512d input;
    std::memcpy(&input,values,sizeof(input));
    const auto one = _mm512_set1_pd(1);
    const auto bits = _mm512_castpd_si512(input);
    const auto exponents = _mm512_srli_epi64(bits,52);
    auto e = _mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(exponents,_mm512_set1_epi64(0x4330000000000000LL))),
        _mm512_set1_pd(4503599627371519.0));
    auto m = _mm512_castsi512_pd(_mm512_or_si512(_mm512_and_si512(bits,_mm512_set1_epi64(0x000fffffffffffffLL)),
        _mm512_set1_epi64(0x3ff0000000000000LL)));
    const auto upper = _mm512_cmp_pd_mask(m,_mm512_set1_pd(1.4142135623730951),_CMP_GT_OQ);
    m = _mm512_mask_blend_pd(upper,m,_mm512_mul_pd(m,_mm512_set1_pd(0.5)));
    e = _mm512_add_pd(e,_mm512_maskz_mov_pd(upper,one));
    const auto t = _mm512_div_pd(_mm512_sub_pd(m,one),_mm512_add_pd(m,one));
    const auto square = _mm512_mul_pd(t,t);
    auto polynomial = _mm512_set1_pd(lab_log_coeff[9]);
    for (int degree=8;degree>=0;--degree)
        polynomial = _mm512_add_pd(_mm512_set1_pd(lab_log_coeff[degree]),_mm512_mul_pd(square,polynomial));
    const auto log = _mm512_add_pd(e,_mm512_mul_pd(_mm512_mul_pd(t,polynomial),_mm512_set1_pd(2.8853900817779268147)));
    const auto y = _mm512_mul_pd(log,_mm512_set1_pd(scale));
    const auto k = _mm512_roundscale_pd(_mm512_mul_pd(y,_mm512_set1_pd(1.4426950408889634074)),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const auto r = _mm512_sub_pd(y,_mm512_mul_pd(k,_mm512_set1_pd(0.69314718055994528623)));
    const auto x = _mm512_sub_pd(r,_mm512_mul_pd(k,_mm512_set1_pd(2.3190468138462995584e-17)));
    polynomial = _mm512_set1_pd(lab_exp_coeff[13]);
    for (int degree=12;degree>=0;--degree)
        polynomial = _mm512_add_pd(_mm512_set1_pd(lab_exp_coeff[degree]),_mm512_mul_pd(x,polynomial));
    const auto integers = _mm512_cvtepi32_epi64(_mm512_cvtpd_epi32(k));
    const auto power = _mm512_slli_epi64(_mm512_add_epi64(integers,_mm512_set1_epi64(1023)),52);
    const auto result = _mm512_mul_pd(polynomial,_mm512_castsi512_pd(power));
    std::memcpy(values,&result,sizeof(result));
}

// Keep the f32 exponent multiply live, including its FE_INEXACT on neutral
// lanes. Callers supply constant exponents, but the original path scales them
// at runtime. The f64 path still calls the prescaled helper directly.
__attribute__((noipa))
void ordinary_lab_power_wide_from_exponent(void* values, double exponent) {
    ordinary_lab_power_wide(values,exponent*0.69314718055994530942);
}
#endif

template<bool wide_power=false> __m256 ordinary_lab_power(__m256 input, double exponent) {
    LabPair x = {_mm256_cvtps_pd(_mm256_castps256_ps128(input)),
                 _mm256_cvtps_pd(_mm256_extractf128_ps(input,1))};
#ifdef SKIMAGEXX_LINUX_WIDE_LAB_POWER
    if constexpr (wide_power) ordinary_lab_power_wide_from_exponent(x,exponent);
    else
#endif
        ordinary_lab_power<false>(x,exponent);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(x[0])),_mm256_cvtpd_ps(x[1]),1);
}

template<bool wide_power> __m256 ordinary_rgb_decode(__m256 input) {
    using V = Vector<float>;
    const auto mask=V::gt(input,V::set(0.04045f));
    const auto linear=V::div(input,V::set(12.92f));
    if (!V::bits(mask)) return linear;
    const auto adjusted=V::div(V::add(input,V::set(0.055f)),V::set(1.055f));
    return V::choose(linear,ordinary_lab_power<wide_power>(V::choose(V::set(1),adjusted,mask),double(2.4f)),mask);
}

template<bool wide_power> __m256 ordinary_lab_curve(__m256 input, __m256 white) {
    using V = Vector<float>;
    const auto scaled=V::div(input,white);
    const auto mask=V::gt(scaled,V::set(0.008856f));
    const auto linear=V::add(V::mul(V::set(7.787f),scaled),V::set(float(16.0/116.0)));
    if (!V::bits(mask)) return linear;
    return V::choose(linear,ordinary_lab_power<wide_power>(V::choose(V::set(1),scaled,mask),1.0/3.0),mask);
}

template<bool rgb,bool wide_power=false> std::size_t ordinary_lab_transform(const float* input, float* output, std::size_t n,
    const std::array<double,3>& white) {
    using V = Vector<float>;
    __m256 weights[9];
    if constexpr (rgb) {
        const auto& matrix=detail::color_spec(ColorTransform::rgb2xyz).matrix;
        for (int j=0;j<9;++j) weights[j]=V::set(float(matrix[j]));
    }
    const auto wx=V::set(float(white[0])), wy=V::set(float(white[1])), wz=V::set(float(white[2]));
    std::size_t i=0;
    for (;i+V::lanes<=n;i+=V::lanes) {
        __m256 x,y,z;
        unpack(input+3*i,x,y,z);
        if constexpr (rgb) {
            const auto a=ordinary_rgb_decode<wide_power>(x), b=ordinary_rgb_decode<wide_power>(y), c=ordinary_rgb_decode<wide_power>(z);
            x=V::add(V::add(V::mul(a,weights[0]),V::mul(b,weights[1])),V::mul(c,weights[2]));
            y=V::add(V::add(V::mul(a,weights[3]),V::mul(b,weights[4])),V::mul(c,weights[5]));
            z=V::add(V::add(V::mul(a,weights[6]),V::mul(b,weights[7])),V::mul(c,weights[8]));
        }
        const auto a=ordinary_lab_curve<wide_power>(x,wx), b=ordinary_lab_curve<wide_power>(y,wy), c=ordinary_lab_curve<wide_power>(z,wz);
        pack(output+3*i,V::sub(V::mul(V::set(116),b),V::set(16)),
            V::mul(V::set(500),V::sub(a,b)),V::mul(V::set(200),V::sub(b,c)));
    }
    return i;
}
template<bool decode,bool wide_power=false> void ordinary_lab_curve(LabPair& values, __m256d white) {
    using V = Vector<double>;
    LabPair linear, mask, nonlinear;
    bool active=false;
    for (int lane=0;lane<2;++lane) {
        const auto scaled = decode ? values[lane] : V::div(values[lane],white);
        mask[lane]=V::gt(scaled,V::set(decode ? 0.04045 : 0.008856));
        if constexpr (decode) linear[lane]=V::div(scaled,V::set(12.92));
        else linear[lane]=V::add(V::mul(V::set(7.787),scaled),V::set(16.0/116.0));
        nonlinear[lane]=V::set(1);
        if (V::bits(mask[lane])) {
            const auto adjusted = decode ? V::div(V::add(scaled,V::set(0.055)),V::set(1.055)) : scaled;
            nonlinear[lane]=V::choose(nonlinear[lane],adjusted,mask[lane]);
            active=true;
        }
    }
    // Match the constant-folded double scale without adding FE_INEXACT
    // for packets such as XYZ=(1,1,1) with unity white.
    constexpr double exponent = decode ? 2.4 : 1.0/3.0;
    constexpr double scale = exponent*0.69314718055994530942;
    if (active) {
#ifdef SKIMAGEXX_LINUX_WIDE_LAB_POWER
        if constexpr (wide_power) ordinary_lab_power_wide(nonlinear,scale);
        else
#endif
            ordinary_lab_power<true>(nonlinear,scale);
    }
    for (int lane=0;lane<2;++lane) values[lane]=V::choose(linear[lane],nonlinear[lane],mask[lane]);
}

template<bool rgb,bool wide_power=false> std::size_t ordinary_lab_transform(const double* input, double* output, std::size_t n,
    const std::array<double,3>& white) {
    using V = Vector<double>;
    constexpr auto block = 2*V::lanes;
    __m256d weights[9];
    if constexpr (rgb) {
        const auto& matrix=detail::color_spec(ColorTransform::rgb2xyz).matrix;
        for (int j=0;j<9;++j) weights[j]=V::set(matrix[j]);
    }
    const auto wx=V::set(white[0]), wy=V::set(white[1]), wz=V::set(white[2]);
    for (std::size_t i=0;i<n;i+=block) {
        LabPair x,y,z;
        for (std::size_t lane=0;lane<2;++lane) unpack(input+3*(i+lane*V::lanes),x[lane],y[lane],z[lane]);
        if constexpr (rgb) {
            ordinary_lab_curve<true,wide_power>(x,wx); ordinary_lab_curve<true,wide_power>(y,wy); ordinary_lab_curve<true,wide_power>(z,wz);
            for (std::size_t lane=0;lane<2;++lane) {
                const auto a=x[lane], b=y[lane], c=z[lane];
                x[lane]=V::add(V::add(V::mul(a,weights[0]),V::mul(b,weights[1])),V::mul(c,weights[2]));
                y[lane]=V::add(V::add(V::mul(a,weights[3]),V::mul(b,weights[4])),V::mul(c,weights[5]));
                z[lane]=V::add(V::add(V::mul(a,weights[6]),V::mul(b,weights[7])),V::mul(c,weights[8]));
            }
        }
        ordinary_lab_curve<false,wide_power>(x,wx); ordinary_lab_curve<false,wide_power>(y,wy); ordinary_lab_curve<false,wide_power>(z,wz);
        for (std::size_t lane=0;lane<2;++lane)
            pack(output+3*(i+lane*V::lanes),V::sub(V::mul(V::set(116),y[lane]),V::set(16)),
                V::mul(V::set(500),V::sub(x[lane],y[lane])),V::mul(V::set(200),V::sub(y[lane],z[lane])));
    }
    return n;
}

template<class T> std::size_t packed_lab_or_legacy(const T* input, T* output, std::size_t n,
    const detail::PerceptualSpec& spec, std::ptrdiff_t pixels, std::ptrdiff_t channels, PerceptualInfo& info) {
    using Op = PerceptualTransform;
    const auto op=spec.transform;
    // The paired double path uses complete groups and round-to-nearest.
    // Other double spans use the original four-lane loop and tail.
    if (n>=64 && pixels==3 && channels==1 && spec.channels==3 && (op==Op::xyz2lab || op==Op::rgb2lab) &&
        (std::is_same_v<T,float> || (n%8==0 && _MM_GET_ROUNDING_MODE()==_MM_ROUND_NEAREST)) &&
        std::all_of(spec.white.begin(),spec.white.end(),[](double v) { return v>=0.25 && v<=2; }) &&
        ordinary_lab_span(input,3*n)) {
#ifdef SKIMAGEXX_LINUX_WIDE_LAB_POWER
        if constexpr (std::is_same_v<T,double>) {
            if (ordinary_lab_wide_available()) {
                if (op==Op::xyz2lab) return ordinary_lab_transform<false,true>(input,output,n,spec.white);
                return ordinary_lab_transform<true,true>(input,output,n,spec.white);
            }
        } else if (_MM_GET_ROUNDING_MODE()==_MM_ROUND_NEAREST && ordinary_lab_wide_available()) {
            if (op==Op::xyz2lab) return ordinary_lab_transform<false,true>(input,output,n,spec.white);
            return ordinary_lab_transform<true,true>(input,output,n,spec.white);
        }
#endif
        if (op==Op::xyz2lab) return ordinary_lab_transform<false>(input,output,n,spec.white);
        return ordinary_lab_transform<true>(input,output,n,spec.white);
    }
    return run_perceptual(input,output,n,spec,pixels,channels,info);
}
}
std::size_t perceptual_transform(const void* source, void* out, std::size_t n, PixelType type,
    const detail::PerceptualSpec& spec, std::ptrdiff_t pixels, std::ptrdiff_t channels, PerceptualInfo& info) {
#define PERCEPTUAL_CASE(KIND,TYPE,OUTPUT) case PixelType::KIND: return run_perceptual(static_cast<const TYPE*>(source),static_cast<OUTPUT*>(out),n,spec,pixels,channels,info)
    switch(type) {
        PERCEPTUAL_CASE(boolean,bool,double);
        case PixelType::uint8: {
            const auto* p=static_cast<const std::uint8_t*>(source);
            auto* dst=static_cast<double*>(out);
            if (pixels==3 && channels==1 && spec.channels==3) {
                if (spec.transform==PerceptualTransform::xyz2lab && n>=1024 &&
                    std::all_of(spec.white.begin(),spec.white.end(),[](double v) { return v>=0.25 && v<=2; }))
                    return xyz8_to_lab(p,dst,n,spec.white);
                if (spec.transform==PerceptualTransform::lab2lch) return lab8_to_lch(p,dst,n);
            }
            return run_perceptual(p,dst,n,spec,pixels,channels,info);
        } PERCEPTUAL_CASE(int8,std::int8_t,double);
        PERCEPTUAL_CASE(uint16,std::uint16_t,double); PERCEPTUAL_CASE(int16,std::int16_t,double);
        PERCEPTUAL_CASE(uint32,std::uint32_t,double); PERCEPTUAL_CASE(int32,std::int32_t,double);
        PERCEPTUAL_CASE(uint64,std::uint64_t,double); PERCEPTUAL_CASE(int64,std::int64_t,double);
        case PixelType::float32:
            return packed_lab_or_legacy(static_cast<const float*>(source),static_cast<float*>(out),n,spec,pixels,channels,info);
        case PixelType::float64:
            return packed_lab_or_legacy(static_cast<const double*>(source),static_cast<double*>(out),n,spec,pixels,channels,info);
        default: return 0;
    }
#undef PERCEPTUAL_CASE
}
namespace {
template<class T,class Input> void distance_load_other(const void* source,std::ptrdiff_t pixels,std::ptrdiff_t channels,
    typename Vector<T>::R (&values)[3]) {
    using V=Vector<T>;
    if constexpr(std::is_same_v<T,double> && std::is_integral_v<Input> && sizeof(Input)<=4) {
        using Storage=std::conditional_t<std::is_same_v<Input,bool>,unsigned char,Input>;
        const auto* p=static_cast<const Storage*>(source);
        if constexpr(sizeof(Input)==1) {
            if(pixels==3 && (channels==1 || channels==-1)) {
                // Exactly twelve bytes: no overread at a four-pixel tail.
                const auto* start=p+(channels<0 ? 2*channels : 0);
                std::uint64_t low; std::uint32_t high;
                std::memcpy(&low,start,8); std::memcpy(&high,start+8,4);
                auto packed=_mm_insert_epi32(_mm_cvtsi64_si128(static_cast<long long>(low)),static_cast<int>(high),2);
                if constexpr(std::is_same_v<Input,bool>)
                    packed=_mm_andnot_si128(_mm_cmpeq_epi8(packed,_mm_setzero_si128()),_mm_set1_epi8(1));
                for(int c=0;c<3;++c) {
                    const int k=channels==1 ? c : 2-c;
                    const auto bytes=_mm_shuffle_epi8(packed,_mm_setr_epi8(char(k),char(k+3),char(k+6),char(k+9),
                        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
                    const auto ints=std::is_signed_v<Input> ? _mm_cvtepi8_epi32(bytes) : _mm_cvtepu8_epi32(bytes);
                    values[c]=_mm256_cvtepi32_pd(ints);
                }
                return;
            }
        }
        for(int c=0;c<3;++c) {
            const auto* channel=p+c*channels;
            if(!pixels) { values[c]=V::set(T(std::is_same_v<Input,bool> ? (*channel!=0) : *channel)); continue; }
            __m128i ints;
            if(pixels==1 && sizeof(Input)==1) {
                std::uint32_t word; std::memcpy(&word,channel,4);
                auto bytes=_mm_cvtsi32_si128(static_cast<int>(word));
                if constexpr(std::is_same_v<Input,bool>)
                    bytes=_mm_andnot_si128(_mm_cmpeq_epi8(bytes,_mm_setzero_si128()),_mm_set1_epi8(1));
                ints=std::is_signed_v<Input> ? _mm_cvtepi8_epi32(bytes) : _mm_cvtepu8_epi32(bytes);
            } else {
                const auto lane=[&](int j) { const auto v=channel[std::ptrdiff_t(j)*pixels];
                    if constexpr(std::is_same_v<Input,bool>) return int(v!=0);
                    else return static_cast<int>(v); };
                ints=_mm_setr_epi32(lane(0),lane(1),lane(2),lane(3));
            }
            if constexpr(std::is_same_v<Input,std::uint32_t>)
                values[c]=_mm256_add_pd(_mm256_cvtepi32_pd(_mm_and_si128(ints,_mm_set1_epi32(0x7fffffff))),
                    _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(ints,31)),_mm256_set1_pd(2147483648.0)));
            else values[c]=_mm256_cvtepi32_pd(ints);
        }
        return;
    }
    if constexpr(std::is_same_v<T,double> && std::is_same_v<Input,float>) {
        const auto* p=static_cast<const float*>(source);
        const int s=static_cast<int>(pixels);
        for(int c=0;c<3;++c) {
            const auto* channel=p+c*channels;
            const auto lanes=pixels==0 ? _mm_set1_ps(*channel) : pixels==1 ? _mm_loadu_ps(channel) :
                _mm_i32gather_ps(channel,_mm_setr_epi32(0,s,2*s,3*s),4);
            values[c]=_mm256_cvtps_pd(lanes);
        }
        return;
    }
    for(int c=0;c<3;++c) {
        T lanes[V::lanes];
        for(std::size_t j=0;j<V::lanes;++j)
            lanes[j]=detail::distance_read<T,Input>(source,std::ptrdiff_t(j)*pixels+c*channels);
        values[c]=V::load(lanes);
    }
}
template<class T> void distance_load(const void* source,PixelType type,std::ptrdiff_t pixels,std::ptrdiff_t channels,
    typename Vector<T>::R (&values)[3],bool following_pixel=false) {
    using V=Vector<T>;
    const bool same=(std::is_same_v<T,float> && type==PixelType::float32) ||
                    (std::is_same_v<T,double> && type==PixelType::float64);
    if(same) {
        const auto* p=static_cast<const T*>(source);
        if constexpr(std::is_same_v<T,float>) {
            if(pixels==-3 && channels==1) {
                unpack(p-21,values[0],values[1],values[2]);
                const auto reverse=_mm256_setr_epi32(7,6,5,4,3,2,1,0);
                for(auto& value:values)value=_mm256_permutevar8x32_ps(value,reverse);
                return;
            }
            // A later logical pixel proves the fourth load lane lies inside
            // the input span. The final packet keeps the exact three-channel
            // gathers; SDK inputs need not contain a trailing fourth channel.
            if(pixels==4 && channels==1 && following_pixel) {
                unpack_four_stride(p,values[0],values[1],values[2]);
                return;
            }
        }
        if(pixels==3 && channels==1) unpack(p,values[0],values[1],values[2]);
        else for(int c=0;c<3;++c) values[c]=pixels==0 ? V::set(p[c*channels]) :
            pixels==1 ? V::load(p+c*channels) : gather(p+c*channels,pixels);
    } else {
#define DISTANCE_LOAD(KIND,TYPE) case PixelType::KIND: return distance_load_other<T,TYPE>(source,pixels,channels,values)
        switch(type) {
            DISTANCE_LOAD(boolean,bool);
            DISTANCE_LOAD(uint8,std::uint8_t); DISTANCE_LOAD(int8,std::int8_t);
            DISTANCE_LOAD(uint16,std::uint16_t); DISTANCE_LOAD(int16,std::int16_t);
            DISTANCE_LOAD(uint32,std::uint32_t); DISTANCE_LOAD(int32,std::int32_t);
            DISTANCE_LOAD(uint64,std::uint64_t); DISTANCE_LOAD(int64,std::int64_t);
            DISTANCE_LOAD(float32,float); DISTANCE_LOAD(float64,double);
            default: return;
        }
#undef DISTANCE_LOAD
    }
}
__m256 distance_sqrt(__m256 v){return _mm256_sqrt_ps(v);}
__m256d distance_sqrt(__m256d v){return _mm256_sqrt_pd(v);}
void distance_store(float* p,__m256 v){_mm256_storeu_ps(p,v);}
void distance_store(double* p,__m256d v){_mm256_storeu_pd(p,v);}
template<class T> std::size_t distance_run(const void* a,const void* b,T* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    using V=Vector<T>;
    const auto safe=std::numeric_limits<int>::max()/std::ptrdiff_t(V::lanes);
    if(ap>safe || ap < -safe || bp>safe || bp < -safe) return 0;
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    std::size_t i=0;
    for(;i+V::lanes<=n;i+=V::lanes) {
        typename V::R av[3],bv[3];
        distance_load<T>(static_cast<const unsigned char*>(a)+std::ptrdiff_t(i)*ap*ab,at,ap,ac,av,i+V::lanes<n);
        distance_load<T>(static_cast<const unsigned char*>(b)+std::ptrdiff_t(i)*bp*bb,bt,bp,bc,bv,i+V::lanes<n);
        const auto x=V::sub(bv[0],av[0]),y=V::sub(bv[1],av[1]),z=V::sub(bv[2],av[2]);
        distance_store(out+i,distance_sqrt(V::add(V::add(V::mul(x,x),V::mul(y,y)),V::mul(z,z))));
    }
    return i;
}
}
std::size_t delta_e76(const void* a,const void* b,float* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    return distance_run(a,b,out,n,at,bt,ap,ac,bp,bc);
}
std::size_t delta_e76(const void* a,const void* b,double* out,std::size_t n,PixelType at,PixelType bt,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc) {
    return distance_run(a,b,out,n,at,bt,ap,ac,bp,bc);
}
namespace {
bool distance_fma_available() {
    static const bool available=[] {
#if defined(_MSC_VER)
        int regs[4];__cpuidex(regs,1,0);return (regs[2]&(1<<12))!=0;
#elif defined(__GNUC__) || defined(__clang__)
        return bool(__builtin_cpu_supports("fma"));
#else
        return false;
#endif
    }();
    return available;
}
template<class T,bool Svml=false> struct VectorDistanceMath {
    using R=__m256d;
    using Wide=VectorDistanceMath<double,Svml>;
    static R round(R x){if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm256_cvtpd_ps(x));else return x;}
    static R c(double x){return _mm256_set1_pd(T(x));}
    static R add(R a,R b){
        if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm_add_ps(_mm256_cvtpd_ps(a),_mm256_cvtpd_ps(b)));
        else return _mm256_add_pd(a,b);
    }
    static R sub(R a,R b){
        if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm_sub_ps(_mm256_cvtpd_ps(a),_mm256_cvtpd_ps(b)));
        else return _mm256_sub_pd(a,b);
    }
    static R mul(R a,R b){
        if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm_mul_ps(_mm256_cvtpd_ps(a),_mm256_cvtpd_ps(b)));
        else return _mm256_mul_pd(a,b);
    }
    static R div(R a,R b){
        if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm_div_ps(_mm256_cvtpd_ps(a),_mm256_cvtpd_ps(b)));
        else return _mm256_div_pd(a,b);
    }
    static R square(R x){return mul(x,x);}
    static R sqrt(R x){
        if constexpr(std::is_same_v<T,float>) return _mm256_cvtps_pd(_mm_sqrt_ps(_mm256_cvtpd_ps(x)));
        else return _mm256_sqrt_pd(x);
    }
    static R power(R x,int n){
        const auto x2=_mm256_mul_pd(x,x),x4=_mm256_mul_pd(x2,x2);
        return round(n==4 ? x4 : _mm256_mul_pd(_mm256_mul_pd(x4,x2),x));
    }
    static R abs(R x){return _mm256_andnot_pd(_mm256_set1_pd(-0.),x);}
    static R degrees(R x){return mul(x,c(T(180)/T(detail::distance_pi)));}
    static R lt(R a,R b){return _mm256_cmp_pd(a,b,_CMP_LT_OQ);}
    static R gt(R a,R b){return _mm256_cmp_pd(a,b,_CMP_GT_OQ);}
    static R eq(R a,R b){return _mm256_cmp_pd(a,b,_CMP_EQ_OQ);}
    static R le(R a,R b){return _mm256_cmp_pd(a,b,_CMP_LE_OQ);}
    static R ge(R a,R b){return _mm256_cmp_pd(a,b,_CMP_GE_OQ);}
    static R both(R a,R b){return _mm256_and_pd(a,b);}
    static R choose(R no,R yes,R mask){return _mm256_blendv_pd(no,yes,mask);}
    static R nonnegative(R x){return choose(x,c(0),le(x,c(0)));}
    static R hypot(R a,R b){
        const auto high=_mm256_max_pd(abs(a),abs(b)),low=_mm256_min_pd(abs(a),abs(b));
        if(!all_finite(a) || !all_finite(b) || mask_bits(gt(high,_mm256_set1_pd(1e150))) ||
           mask_bits(both(gt(high,c(0)),lt(high,_mm256_set1_pd(1e-150))))) {
            double x[4],y[4]; _mm256_storeu_pd(x,a); _mm256_storeu_pd(y,b);
            for(int j=0;j<4;++j) x[j]=std::hypot(T(x[j]),T(y[j]));
            return _mm256_loadu_pd(x);
        }
        const auto ratio=_mm256_div_pd(low,choose(high,_mm256_set1_pd(1),eq(high,_mm256_setzero_pd())));
        return round(_mm256_mul_pd(high,_mm256_sqrt_pd(_mm256_add_pd(_mm256_set1_pd(1),_mm256_mul_pd(ratio,ratio)))));
    }
    static R angle(R a,R b,bool =false){
        // Preserve the float atan2 rounding before the double 2*pi in-place add.
        double x[4],y[4]; _mm256_storeu_pd(x,a); _mm256_storeu_pd(y,b);
        if constexpr(Svml && std::is_same_v<T,double>) {
            if(distance_fma_available())distance_svml_angles(x,y,x);
            else for(int j=0;j<4;++j){const auto h=detail::svml_atan2(y[j],x[j]);x[j]=h+(h<0?2*detail::distance_pi:0);}
        } else for(int j=0;j<4;++j) {const auto h=std::atan2(T(y[j]),T(x[j]));x[j]=T(double(h)+(h<0?2*detail::distance_pi:0));}
        return _mm256_loadu_pd(x);
    }
    static R sin(R x){R s,c; sincos4(x,s,c);return round(s);}
    static R cos(R x){R s,c; sincos4(x,s,c);return round(c);}
    static R exp(R x){return round(math::exp4(x));}
    static R hue_squared(R a,R b,R x,R y){
        // This cancellation-sensitive stage deliberately uses libm hypot in
        // double, as the upstream implementation does, even for float32 Lab.
        double aa[4],bb[4],xx[4],yy[4],ca[4],cb[4];
        _mm256_storeu_pd(aa,a);_mm256_storeu_pd(bb,b);_mm256_storeu_pd(xx,x);_mm256_storeu_pd(yy,y);
        for(int j=0;j<4;++j){ca[j]=std::hypot(aa[j],bb[j]);cb[j]=std::hypot(xx[j],yy[j]);}
        return round(_mm256_mul_pd(_mm256_set1_pd(2),_mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(ca),_mm256_loadu_pd(cb)),
            _mm256_add_pd(_mm256_mul_pd(a,x),_mm256_mul_pd(b,y)))));
    }
    static R hue_squared_prepared(R a,R b,R x,R y,R second_chroma){
        double aa[4],bb[4],ca[4];
        _mm256_storeu_pd(aa,a);_mm256_storeu_pd(bb,b);
        for(int j=0;j<4;++j)ca[j]=std::hypot(aa[j],bb[j]);
        return round(_mm256_mul_pd(_mm256_set1_pd(2),_mm256_sub_pd(_mm256_mul_pd(_mm256_loadu_pd(ca),second_chroma),
            _mm256_add_pd(_mm256_mul_pd(a,x),_mm256_mul_pd(b,y)))));
    }
};
bool reusable_float_color(const void* source,std::ptrdiff_t channels) {
    const auto* bytes=static_cast<const unsigned char*>(source);
    for(int c=0;c<3;++c) {
        std::uint32_t bits;
        std::memcpy(&bits,bytes+std::ptrdiff_t(c)*channels*std::ptrdiff_t(sizeof(float)),sizeof(bits));
        const auto magnitude=bits&0x7fffffffu;
        // Keep exceptional, subnormal and overflow-adjacent input graphs on
        // the old path, without classifying a signaling NaN arithmetically.
        if(magnitude && (magnitude<0x00800000u || magnitude>=0x7f000000u))return false;
    }
    return true;
}
template<class T,class Output,bool Svml=false> std::size_t metric_run(const void* a,const void* b,Output* out,std::size_t n,
    PixelType at,PixelType bt,DistanceMetric metric,const DistanceParameters& p,bool wide,bool scalar_cmc,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc,bool legacy_ciede2000) {
    if(ap>INT_MAX/4 || ap< -INT_MAX/4 || bp>INT_MAX/4 || bp< -INT_MAX/4) return 0;
    const auto ab=std::ptrdiff_t(pixel_size(at)),bb=std::ptrdiff_t(pixel_size(bt));
    using Math=VectorDistanceMath<T,Svml>;
    detail::DistanceSecondChroma<Math> prepared{};
    const detail::DistanceSecondChroma<Math>* second_chroma=nullptr;
    __m256d constant[3];
    if constexpr(std::is_same_v<T,float>) {
        if(metric==DistanceMetric::cmc && bp==0 && n>=4 && at==PixelType::float32 && bt==at &&
           !detail::distance_math_callbacks && reusable_float_color(b,bc)) {
            const auto control=_mm_getcsr();
            if((control&0x7f80u)==0x1f80u && std::fegetround()==FE_TONEAREST) {
                distance_load<double>(b,bt,bp,bc,constant);
                prepared.ordinary=Math::hypot(constant[1],constant[2]);
                prepared.hue=_mm256_set1_pd(std::hypot(_mm256_cvtsd_f64(constant[1]),_mm256_cvtsd_f64(constant[2])));
                second_chroma=&prepared;
            }
        }
    }
    std::size_t i=0;
    for(;i+4<=n;i+=4) {
        __m256d av[3],bv[3];
        distance_load<double>(static_cast<const unsigned char*>(a)+std::ptrdiff_t(i)*ap*ab,at,ap,ac,av);
        if(second_chroma)for(int c=0;c<3;++c)bv[c]=constant[c];
        else distance_load<double>(static_cast<const unsigned char*>(b)+std::ptrdiff_t(i)*bp*bb,bt,bp,bc,bv);
        const auto result=detail::distance_metric<Math>(av,bv,metric,p,wide,scalar_cmc,legacy_ciede2000,second_chroma);
        if constexpr(std::is_same_v<Output,float>) _mm_storeu_ps(out+i,_mm256_cvtpd_ps(result));
        else _mm256_storeu_pd(out+i,result);
        if(metric==DistanceMetric::ciede2000) {
            // Chroma approximations are harmless away from the 180-degree
            // discontinuity. Refine only near-antipodal lanes with libm hypot
            // and power so the SVML angle provider sees the right inputs.
            const auto xy=_mm256_mul_pd(av[1],bv[2]),yx=_mm256_mul_pd(av[2],bv[1]);
            const auto cross=_mm256_sub_pd(xy,yx);
            const auto dot=_mm256_add_pd(_mm256_mul_pd(av[1],bv[1]),_mm256_mul_pd(av[2],bv[2]));
            const auto sign=_mm256_set1_pd(-0.);
            const auto magnitude=_mm256_add_pd(_mm256_andnot_pd(sign,xy),_mm256_andnot_pd(sign,yx));
            const auto bound=_mm256_mul_pd(magnitude,_mm256_set1_pd(32*std::numeric_limits<T>::epsilon()));
            const auto near=_mm256_and_pd(_mm256_cmp_pd(dot,_mm256_setzero_pd(),_CMP_LT_OQ),
                _mm256_cmp_pd(_mm256_andnot_pd(sign,cross),bound,_CMP_LE_OQ));
            const auto mask=_mm256_movemask_pd(near);
            if(mask) {
                double aa[3][4],bb[3][4];
                for(int channel=0;channel<3;++channel){_mm256_storeu_pd(aa[channel],av[channel]);_mm256_storeu_pd(bb[channel],bv[channel]);}
                for(int lane=0;lane<4;++lane)if(mask&(1<<lane)) {
                    double first[]={aa[0][lane],aa[1][lane],aa[2][lane]},second[]={bb[0][lane],bb[1][lane],bb[2][lane]};
                    out[i+std::size_t(lane)]=Output(detail::distance_metric<detail::ScalarDistanceMath<T,Svml>>(
                        first,second,metric,p,wide,scalar_cmc,legacy_ciede2000));
                }
            }
        }
    }
    return i;
}
}
std::size_t metric_distance(const void* a,const void* b,void* out,std::size_t n,PixelType at,PixelType bt,
    DistanceMetric metric,const DistanceParameters& p,bool wide,bool scalar_cmc,
    std::ptrdiff_t ap,std::ptrdiff_t ac,std::ptrdiff_t bp,std::ptrdiff_t bc,bool legacy_ciede2000,bool svml_angles) {
    if(svml_angles) {
        if(at==PixelType::float32 && bt==PixelType::float32)
            return metric_run<float,float,true>(a,b,static_cast<float*>(out),n,at,bt,metric,p,wide,scalar_cmc,ap,ac,bp,bc,legacy_ciede2000);
        return metric_run<double,double,true>(a,b,static_cast<double*>(out),n,at,bt,metric,p,wide,scalar_cmc,ap,ac,bp,bc,legacy_ciede2000);
    }
    if(at==PixelType::float32 && bt==PixelType::float32) {
        if(scalar_cmc) return metric_run<float>(a,b,static_cast<double*>(out),n,at,bt,metric,p,wide,scalar_cmc,ap,ac,bp,bc,legacy_ciede2000);
        return metric_run<float>(a,b,static_cast<float*>(out),n,at,bt,metric,p,wide,scalar_cmc,ap,ac,bp,bc,legacy_ciede2000);
    }
    return metric_run<double>(a,b,static_cast<double*>(out),n,at,bt,metric,p,wide,scalar_cmc,ap,ac,bp,bc,legacy_ciede2000);
}
} // namespace skimagexx::simd
