#include "simd.hpp"
#include "stain_detail.hpp"
#include "simd_math.hpp"
#include <climits>

namespace skimagexx::simd {
namespace {
__m256d clip(__m256d x,bool upper) {
    auto value=_mm256_max_pd(x,_mm256_setzero_pd());
    if (upper) value=_mm256_min_pd(value,_mm256_set1_pd(1));
    return _mm256_blendv_pd(value,x,_mm256_cmp_pd(x,x,_CMP_UNORD_Q));
}
void pack(double* p,__m256d r,__m256d g,__m256d b) {
    _mm256_storeu_pd(p,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(r,_MM_SHUFFLE(1,0,0,0)),_mm256_permute4x64_pd(g,0),2),_mm256_permute4x64_pd(b,0),4));
    _mm256_storeu_pd(p+4,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(g,_MM_SHUFFLE(2,0,0,1)),_mm256_permute4x64_pd(b,_MM_SHUFFLE(0,0,1,0)),2),_mm256_permute4x64_pd(r,_MM_SHUFFLE(0,2,0,0)),4));
    _mm256_storeu_pd(p+8,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(b,_MM_SHUFFLE(3,0,0,2)),_mm256_permute4x64_pd(r,_MM_SHUFFLE(0,0,3,0)),2),_mm256_permute4x64_pd(g,_MM_SHUFFLE(0,3,0,0)),4));
}
void unpack(__m256d a,__m256d b,__m256d c,__m256d* out) {
    out[0]=_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,3,0)),
        _mm256_permute4x64_pd(b,_MM_SHUFFLE(0,2,0,0)),4),_mm256_permute4x64_pd(c,_MM_SHUFFLE(1,0,0,0)),8);
    out[1]=_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,1)),_mm256_permute4x64_pd(b,_MM_SHUFFLE(0,3,0,0)),6);
    out[1]=_mm256_blend_pd(out[1],_mm256_permute4x64_pd(c,_MM_SHUFFLE(2,0,0,0)),8);
    out[2]=_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(a,_MM_SHUFFLE(0,0,0,2)),
        _mm256_permute4x64_pd(b,_MM_SHUFFLE(0,0,1,0)),2),_mm256_permute4x64_pd(c,_MM_SHUFFLE(3,0,0,0)),12);
}
template<class Input> __m256d load(const Input* p,std::ptrdiff_t step) {
    if constexpr (std::is_same_v<Input,double>) {
        if (!step) return _mm256_set1_pd(*p);
        if (step==1) return _mm256_loadu_pd(p);
        if (step>=INT_MIN/3 && step<=INT_MAX/3) {
            const auto s=static_cast<int>(step);
            return _mm256_i32gather_pd(p,_mm_setr_epi32(0,s,2*s,3*s),8);
        }
    } else if constexpr (std::is_same_v<Input,float>) {
        if (!step) return _mm256_set1_pd(double(*p));
        if (step==1) return _mm256_cvtps_pd(_mm_loadu_ps(p));
        if (step>=INT_MIN/3 && step<=INT_MAX/3) {
            const auto s=static_cast<int>(step);
            return _mm256_cvtps_pd(_mm_i32gather_ps(p,_mm_setr_epi32(0,s,2*s,3*s),4));
        }
    }
    double values[4]; for (int j=0;j<4;++j) values[j]=double(detail::stain_value(p,std::ptrdiff_t(j)*step));
    return _mm256_loadu_pd(values);
}
template<class Input> std::size_t run(const Input* p,double* output,std::size_t n,const detail::StainSpec& spec,
    std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    const bool separate=spec.transform==StainTransform::separate;
    const auto adjustment=_mm256_set1_pd(spec.adjust),negative=_mm256_set1_pd(-0.0);
    const double* lookup=nullptr;
    if constexpr (std::is_same_v<Input,std::uint8_t>) if (separate) lookup=detail::stain_log_table().data();
    std::size_t i=0;
    for (;i+4<=n;i+=4) {
        __m256d values[3];
        bool contiguous=false;
        if constexpr (std::is_floating_point_v<Input>) {
            if (pixels==3 && channels==1) {
                const auto* start=p+3*i;
                if constexpr (std::is_same_v<Input,double>)
                    unpack(_mm256_loadu_pd(start),_mm256_loadu_pd(start+4),_mm256_loadu_pd(start+8),values);
                else unpack(_mm256_cvtps_pd(_mm_loadu_ps(start)),_mm256_cvtps_pd(_mm_loadu_ps(start+4)),_mm256_cvtps_pd(_mm_loadu_ps(start+8)),values);
                contiguous=true;
            }
        }
        for (int c=0;c<3;++c) {
            const auto* start=p+std::ptrdiff_t(i)*pixels+std::ptrdiff_t(c)*channels;
            auto value=_mm256_setzero_pd();
            if constexpr (std::is_same_v<Input,std::uint8_t>) {
                if (!separate) value=load(start,pixels);
            } else value=contiguous ? values[c] : load(start,pixels);
            if (separate) {
                if constexpr (std::is_same_v<Input,std::uint8_t>) {
                    double logs[4]; for (int j=0;j<4;++j) logs[j]=lookup[start[std::ptrdiff_t(j)*pixels]];
                    value=_mm256_loadu_pd(logs);
                } else if constexpr (std::is_same_v<Input,float>) {
                    const auto floor=_mm256_set1_pd(double(float(1e-6)));
                    value=_mm256_blendv_pd(value,floor,_mm256_cmp_pd(value,floor,_CMP_LE_OQ));
                    value=_mm256_mul_pd(math::log2_4(value),_mm256_set1_pd(0.69314718055994530942));
                    value=_mm256_cvtps_pd(_mm256_cvtpd_ps(value));
                } else {
                    const auto floor=_mm256_set1_pd(1e-6);
                    value=_mm256_blendv_pd(value,floor,_mm256_cmp_pd(value,floor,_CMP_LE_OQ));
                    value=_mm256_mul_pd(math::log2_4(value),_mm256_set1_pd(0.69314718055994530942));
                }
            }
            if (spec.intermediate32) {
                const auto v=_mm256_cvtpd_ps(value),adjust=_mm_set1_ps(float(spec.adjust));
                value=_mm256_cvtps_pd(separate ? _mm_div_ps(v,_mm_sub_ps(_mm_setzero_ps(),adjust)) :
                    _mm_xor_ps(_mm_mul_ps(v,adjust),_mm_set1_ps(-0.0f)));
            } else value=separate ? _mm256_div_pd(value,_mm256_sub_pd(_mm256_setzero_pd(),adjustment)) :
                _mm256_xor_pd(_mm256_mul_pd(value,adjustment),negative);
            values[c]=value;
        }
        __m256d result[3];
        for (std::size_t c=0;c<spec.channels;++c) {
            const auto* w=spec.weights+3*c;
            auto value=_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(values[0],_mm256_set1_pd(w[0])),
                _mm256_mul_pd(values[1],_mm256_set1_pd(w[1]))),_mm256_mul_pd(values[2],_mm256_set1_pd(w[2])));
            if (!separate) value=math::exp4(value);
            value=clip(value,!separate);
            if (spec.channels==3) result[c]=value;
            else {
                double lanes[4]; _mm256_storeu_pd(lanes,value);
                for (int j=0;j<4;++j) output[(i+std::size_t(j))*spec.channels+c]=lanes[j];
            }
        }
        if (spec.channels==3) pack(output+3*i,result[0],result[1],result[2]);
    }
    return i;
}
}
std::size_t stain_transform(const void* input,double* output,std::size_t n,PixelType type,
    const detail::StainSpec& spec,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
#define STAIN(KIND,TYPE) case PixelType::KIND: return run(static_cast<const TYPE*>(input),output,n,spec,pixels,channels)
    switch(type) {
        STAIN(boolean,bool); STAIN(uint8,std::uint8_t); STAIN(int8,std::int8_t);
        STAIN(uint16,std::uint16_t); STAIN(int16,std::int16_t); STAIN(uint32,std::uint32_t);
        STAIN(int32,std::int32_t); STAIN(uint64,std::uint64_t); STAIN(int64,std::int64_t);
        STAIN(float32,float); STAIN(float64,double);
    }
#undef STAIN
    return 0;
}
}
