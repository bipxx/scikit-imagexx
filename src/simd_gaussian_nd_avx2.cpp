#include "simd.hpp"
#include <immintrin.h>
#include <cstring>
#include <limits>
#include <type_traits>
namespace skimagexx::simd {
namespace {
__m128i low32(__m256i x) {x=_mm256_shuffle_epi32(x,_MM_SHUFFLE(2,0,2,0));return _mm_unpacklo_epi64(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));}
__m256d unsigned32(__m128i x) {
    return _mm256_add_pd(_mm256_cvtepi32_pd(x),_mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(x,31)),_mm256_set1_pd(4294967296.0)));
}
template<class T> __m256d raw(const T* p) {
    if constexpr(sizeof(T)==1) {
        std::uint32_t value;std::memcpy(&value,p,4);const auto bytes=_mm_cvtsi32_si128(int(value));
        return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi8_epi32(bytes) : _mm_cvtepu8_epi32(bytes));
    } else if constexpr(sizeof(T)==2) {
        const auto words=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi16_epi32(words) : _mm_cvtepu16_epi32(words));
    } else if constexpr(sizeof(T)==4) {
        const auto values=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return std::is_signed_v<T> ? _mm256_cvtepi32_pd(values) : unsigned32(values);
    } else {
        const auto values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const auto lo=unsigned32(low32(values));const auto high=low32(_mm256_srli_epi64(values,32));
        const auto hi=std::is_signed_v<T> ? _mm256_cvtepi32_pd(high) : unsigned32(high);
        return _mm256_add_pd(_mm256_mul_pd(hi,_mm256_set1_pd(4294967296.0)),lo);
    }
}
template<class T,bool Boolean=false> std::size_t convert(const void* source,double* output,std::size_t count,bool preserve) {
    const auto* input=static_cast<const T*>(source);std::size_t i=0;
    const auto scale=_mm256_set1_pd(1.0/double(std::numeric_limits<T>::max())),negative_one=_mm256_set1_pd(-1);
    for(;i+4<=count;i+=4) {
        auto value=raw(input+i);
        if constexpr(Boolean)value=_mm256_and_pd(_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_NEQ_OQ),_mm256_set1_pd(1));
        else if(!preserve) {value=_mm256_mul_pd(value,scale);if constexpr(std::is_signed_v<T>)value=_mm256_max_pd(value,negative_one);}
        _mm256_storeu_pd(output+i,value);
    }
    return i;
}
__m256d load(const void* data,std::size_t offset,bool single) {
    if(single)return _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(data)+offset));
    return _mm256_loadu_pd(static_cast<const double*>(data)+offset);
}
void store(void* data,std::size_t offset,bool single,__m256d value) {
    if(single)_mm_storeu_ps(static_cast<float*>(data)+offset,_mm256_cvtpd_ps(value));
    else _mm256_storeu_pd(static_cast<double*>(data)+offset,value);
}
}
std::size_t gaussian_prepare(const void* input,double* output,std::size_t count,PixelType type,bool preserve) {
    switch(type) {
    case PixelType::boolean:return convert<std::uint8_t,true>(input,output,count,preserve);
#define RAW(tag,T) case PixelType::tag:return convert<T>(input,output,count,preserve);
    RAW(uint8,std::uint8_t) RAW(int8,std::int8_t) RAW(uint16,std::uint16_t) RAW(int16,std::int16_t)
    RAW(uint32,std::uint32_t) RAW(int32,std::int32_t) RAW(uint64,std::uint64_t) RAW(int64,std::int64_t)
    default:return 0;
    }
#undef RAW
}
std::size_t gaussian_rows(const void* const* left,const void* const* right,const void* center,void* output,
    std::size_t count,std::size_t radius,const double* weights,double center_weight,double cval,bool input_single,bool output_single) {
    const auto middle=_mm256_set1_pd(center_weight),constant=_mm256_set1_pd(cval);std::size_t x=0;
    for(;x+16<=count;x+=16) {
        auto a=_mm256_mul_pd(load(center,x,input_single),middle),b=_mm256_mul_pd(load(center,x+4,input_single),middle);
        auto c=_mm256_mul_pd(load(center,x+8,input_single),middle),d=_mm256_mul_pd(load(center,x+12,input_single),middle);
        for(std::size_t j=0;j<radius;++j) {
            const auto weight=_mm256_set1_pd(weights[j]);
            auto pair=[&](std::size_t offset){return _mm256_add_pd(left[j] ? load(left[j],offset,input_single) : constant,right[j] ? load(right[j],offset,input_single) : constant);};
            a=_mm256_add_pd(a,_mm256_mul_pd(pair(x),weight));b=_mm256_add_pd(b,_mm256_mul_pd(pair(x+4),weight));
            c=_mm256_add_pd(c,_mm256_mul_pd(pair(x+8),weight));d=_mm256_add_pd(d,_mm256_mul_pd(pair(x+12),weight));
        }
        store(output,x,output_single,a);store(output,x+4,output_single,b);store(output,x+8,output_single,c);store(output,x+12,output_single,d);
    }
    for(;x+4<=count;x+=4) {
        auto sum=_mm256_mul_pd(load(center,x,input_single),middle);
        for(std::size_t j=0;j<radius;++j) {
            const auto first=left[j] ? load(left[j],x,input_single) : constant,second=right[j] ? load(right[j],x,input_single) : constant;
            sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_add_pd(first,second),_mm256_set1_pd(weights[j])));
        }
        store(output,x,output_single,sum);
    }
    return x;
}
namespace {
__m256d clip_double(__m256d value,double lower) {
    const auto low=_mm256_set1_pd(lower),one=_mm256_set1_pd(1);
    value=_mm256_blendv_pd(value,low,_mm256_cmp_pd(value,low,_CMP_LT_OQ));
    return _mm256_blendv_pd(value,one,_mm256_cmp_pd(value,one,_CMP_GT_OQ));
}
}
std::size_t gaussian_difference(const void* first,const void* second,void* output,std::size_t count,bool single) {
    std::size_t i=0;
    if(single)for(;i+8<=count;i+=8)_mm256_storeu_ps(static_cast<float*>(output)+i,_mm256_sub_ps(_mm256_loadu_ps(static_cast<const float*>(first)+i),_mm256_loadu_ps(static_cast<const float*>(second)+i)));
    else for(;i+4<=count;i+=4)_mm256_storeu_pd(static_cast<double*>(output)+i,_mm256_sub_pd(_mm256_loadu_pd(static_cast<const double*>(first)+i),_mm256_loadu_pd(static_cast<const double*>(second)+i)));
    return i;
}
std::size_t gaussian_unsharp(const void* image,const void* blur,void* output,std::size_t count,bool input_single,bool output_single,
    double amount,bool wide,bool clip,double lower) {
    std::size_t i=0;const auto scale=_mm256_set1_pd(amount);
    if(input_single && !wide) {
        const auto factor=_mm256_set1_ps(float(amount)),lo=_mm256_set1_ps(float(lower)),hi=_mm256_set1_ps(1);
        for(;i+8<=count;i+=8) {
            const auto original=_mm256_loadu_ps(static_cast<const float*>(image)+i);
            const auto delta=_mm256_sub_ps(original,_mm256_loadu_ps(static_cast<const float*>(blur)+i));
            auto value=_mm256_add_ps(original,_mm256_mul_ps(delta,factor));
            if(clip) {value=_mm256_blendv_ps(value,lo,_mm256_cmp_ps(value,lo,_CMP_LT_OQ));value=_mm256_blendv_ps(value,hi,_mm256_cmp_ps(value,hi,_CMP_GT_OQ));}
            if(output_single)_mm256_storeu_ps(static_cast<float*>(output)+i,value);
            else {_mm256_storeu_pd(static_cast<double*>(output)+i,_mm256_cvtps_pd(_mm256_castps256_ps128(value)));_mm256_storeu_pd(static_cast<double*>(output)+i+4,_mm256_cvtps_pd(_mm256_extractf128_ps(value,1)));}
        }
    } else {
        for(;i+4<=count;i+=4) {
            __m256d original,delta;
            if(input_single) {
                const auto value=_mm_loadu_ps(static_cast<const float*>(image)+i);
                delta=_mm256_cvtps_pd(_mm_sub_ps(value,_mm_loadu_ps(static_cast<const float*>(blur)+i)));original=_mm256_cvtps_pd(value);
            } else {original=_mm256_loadu_pd(static_cast<const double*>(image)+i);delta=_mm256_sub_pd(original,_mm256_loadu_pd(static_cast<const double*>(blur)+i));}
            auto value=_mm256_add_pd(original,_mm256_mul_pd(delta,scale));if(clip)value=clip_double(value,lower);
            store(output,i,output_single,value);
        }
    }
    return i;
}

}
