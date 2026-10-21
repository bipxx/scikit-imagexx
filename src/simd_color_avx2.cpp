#include "simd.hpp"
#include "color_detail.hpp"
#include "simd_math.hpp"
#include <cstring>
#include <climits>

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
    static R mul(R a, R b) { return _mm256_mul_##SUFFIX(a,b); } \
    static R div(R a, R b) { return _mm256_div_##SUFFIX(a,b); } \
    static R gt(R a, R b) { return _mm256_cmp_##SUFFIX(a,b,_CMP_GT_OQ); } \
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
    const auto adjusted = V::choose(V::set(1),V::div(V::add(x,V::set(T(0.055))),V::set(T(1.055))),mask);
    return V::choose(V::div(x,V::set(T(12.92))),power(adjusted,T(2.4)),mask);
}
template<class T> typename Vector<T>::R encode(typename Vector<T>::R x) {
    using V = Vector<T>;
    const auto mask = V::gt(x,V::set(T(0.0031308)));
    const auto adjusted = V::choose(V::set(1),x,mask);
    const auto nonlinear = V::sub(V::mul(V::set(T(1.055)),power(adjusted,T(1.0/2.4))),V::set(T(0.055)));
    return V::clip(V::choose(V::mul(x,V::set(T(12.92))),nonlinear,mask));
}
std::size_t linear_double(const double* p, double* out, std::size_t n, const detail::ColorSpec& spec) {
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
    std::size_t i = 0;
    for (; i+4 <= n; i += 4) {
        auto a = _mm256_loadu_pd(p+3*i), b = _mm256_loadu_pd(p+3*i+4), c = _mm256_loadu_pd(p+3*i+8);
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
} // namespace skimagexx::simd
