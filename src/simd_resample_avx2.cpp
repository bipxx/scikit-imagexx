// Independent output lanes preserve ndimage's axis/corner multiplication and
// summation order. BSD reference attribution: scipy-ndimage-LICENSE.txt.
#include "resample_detail.hpp"
#include <immintrin.h>
#include "resize_cast.hpp"
#include <cstring>
#include <type_traits>

namespace skimagexx::simd {
namespace {
__m256d unsigned32(__m128i value) {
    const auto signed_value=_mm256_cvtepi32_pd(value);
    const auto negative=_mm256_cmp_pd(signed_value,_mm256_setzero_pd(),_CMP_LT_OQ);
    return _mm256_add_pd(signed_value,_mm256_and_pd(negative,_mm256_set1_pd(0x1p32)));
}
__m128i words32(__m256i value,bool high) {
    const auto lo=_mm256_castsi256_si128(value),hi=_mm256_extracti128_si256(value,1);
    return high ? _mm_unpacklo_epi64(_mm_shuffle_epi32(lo,_MM_SHUFFLE(3,1,3,1)),_mm_shuffle_epi32(hi,_MM_SHUFFLE(3,1,3,1)))
                : _mm_unpacklo_epi64(_mm_shuffle_epi32(lo,_MM_SHUFFLE(2,0,2,0)),_mm_shuffle_epi32(hi,_MM_SHUFFLE(2,0,2,0)));
}
__m128 mask32(__m256i valid) {
    const int mask=_mm256_movemask_pd(_mm256_castsi256_pd(valid));
    return _mm_castsi128_ps(_mm_set_epi32(mask&8 ? -1 : 0,mask&4 ? -1 : 0,mask&2 ? -1 : 0,mask&1 ? -1 : 0));
}
template<class T> __m256d gather(const T* input,__m256i index,__m256i valid,double constant) {
    const auto mask=_mm256_castsi256_pd(valid),cval=_mm256_set1_pd(constant);
    if constexpr(std::is_same_v<T,double>)return _mm256_mask_i64gather_pd(cval,input,index,mask,8);
    else if constexpr(std::is_same_v<T,float>) {
        const auto values=_mm256_mask_i64gather_ps(_mm_setzero_ps(),input,index,mask32(valid),4);
        return _mm256_blendv_pd(cval,_mm256_cvtps_pd(values),mask);
    } else if constexpr(sizeof(T)==4) {
        const auto values=_mm256_mask_i64gather_epi32(_mm_setzero_si128(),reinterpret_cast<const int*>(input),index,_mm_castps_si128(mask32(valid)),4);
        const auto converted=std::is_signed_v<T> ? _mm256_cvtepi32_pd(values) : unsigned32(values);
        return _mm256_blendv_pd(cval,converted,mask);
    } else if constexpr(sizeof(T)==8) {
        const auto values=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),reinterpret_cast<const long long*>(input),index,valid,8);
        const auto high=words32(values,true),low=words32(values,false);
        const auto high_value=std::is_signed_v<T> ? _mm256_cvtepi32_pd(high) : unsigned32(high);
        const auto converted=_mm256_add_pd(_mm256_mul_pd(high_value,_mm256_set1_pd(0x1p32)),unsigned32(low));
        return _mm256_blendv_pd(cval,converted,mask);
    } else {
        // AVX2 has no byte/word gather. Read exactly the selected elements;
        // widening a read here would overrun the final source element.
        alignas(32) std::int64_t offsets[4];alignas(32) double values[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(offsets),index);
        const auto bits=_mm256_movemask_pd(mask);
        for(int lane=0;lane<4;++lane)values[lane]=(bits&(1<<lane)) ? double(input[offsets[lane]]) : constant;
        return _mm256_load_pd(values);
    }
}
template<class T,bool Boolean> void store(T* output,__m256d value) {
    if constexpr(std::is_same_v<T,double>)_mm256_storeu_pd(output,value);
    else if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(output,_mm256_cvtpd_ps(value));
    else if constexpr(sizeof(T)==8) {
        alignas(32) double values[4];_mm256_store_pd(values,value);
        for(int lane=0;lane<4;++lane)output[lane]=detail::resize_sample_cast<T,Boolean>(values[lane]);
    } else {
        if constexpr(!Boolean) {
            const auto positive=_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_GT_OQ);
            if constexpr(std::is_unsigned_v<T>)value=_mm256_blendv_pd(_mm256_setzero_pd(),_mm256_add_pd(value,_mm256_set1_pd(.5)),positive);
            else value=_mm256_blendv_pd(_mm256_sub_pd(value,_mm256_set1_pd(.5)),_mm256_add_pd(value,_mm256_set1_pd(.5)),positive);
            value=_mm256_min_pd(_mm256_max_pd(value,_mm256_set1_pd(double(std::numeric_limits<T>::min()))),_mm256_set1_pd(double(std::numeric_limits<T>::max())));
        }
        auto high_bit=_mm_setzero_si128();
        if constexpr(std::is_same_v<T,std::uint32_t>) {
            const auto upper=_mm256_cmp_pd(value,_mm256_set1_pd(0x1p31),_CMP_GE_OQ);
            // Keep the adjusted operand nonnegative: subtracting 2**32
            // would change truncation of the sampler's fractional +.5.
            value=_mm256_sub_pd(value,_mm256_and_pd(upper,_mm256_set1_pd(0x1p31)));
            high_bit=_mm_and_si128(_mm_castps_si128(mask32(_mm256_castpd_si256(upper))),_mm_set1_epi32(std::numeric_limits<std::int32_t>::min()));
        }
        auto integers=_mm_xor_si128(_mm256_cvttpd_epi32(value),high_bit);
        if constexpr(sizeof(T)==4)_mm_storeu_si128(reinterpret_cast<__m128i*>(output),integers);
        else if constexpr(sizeof(T)==2) {
            integers=_mm_and_si128(integers,_mm_set1_epi32(65535));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(output),_mm_packus_epi32(integers,_mm_setzero_si128()));
        } else {
            integers=_mm_and_si128(integers,_mm_set1_epi32(255));
            const auto words=_mm_packus_epi32(integers,_mm_setzero_si128());
            const auto bytes=_mm_packus_epi16(words,_mm_setzero_si128());
            const auto packed=_mm_cvtsi128_si32(bytes);std::memcpy(output,&packed,4);
        }
    }
}
template<class Input,class Output,bool Boolean=false,bool Paired=false>
std::size_t sample(const Input* input,Output* output,const detail::ZoomPlan& plan,
                        std::size_t begin,std::size_t end) {
    // NumPy has at most 64 image axes. This fixed coordinate scratch stays on
    // each worker stack and avoids allocations inside the parallel region.
    if(plan.axes.size()>64)return begin;
    const auto rank=plan.axes.size();
    const detail::ZoomPoint* points[64][4];
    const auto zero=_mm256_setzero_si256();
    for(;end-begin>=4;begin+=4) {
        for(std::size_t axis=0;axis<rank;++axis)for(std::size_t lane=0;lane<4;++lane) {
            if constexpr(Paired)points[axis][lane]=&plan.axes[axis][lane];
            else points[axis][lane]=&plan.axes[axis][((begin+lane)/plan.output_strides[axis])%plan.axes[axis].size()];
        }
        auto total=_mm256_setzero_pd();
        for(std::size_t corner=0;corner<plan.corners;++corner) {
            const auto* taps=plan.footprint.data()+corner*rank;
            auto index=zero,valid=_mm256_cmpeq_epi64(zero,zero);
            for(std::size_t axis=0;axis<rank;++axis) {
                const auto tap=taps[axis];
                const auto offsets=_mm256_set_epi64x(points[axis][3]->offsets[tap],points[axis][2]->offsets[tap],
                    points[axis][1]->offsets[tap],points[axis][0]->offsets[tap]);
                const auto positive=_mm256_cmpgt_epi64(offsets,_mm256_set1_epi64x(-1));
                valid=_mm256_and_si256(valid,positive);
                index=_mm256_add_epi64(index,_mm256_and_si256(offsets,positive));
            }
            auto coefficient=gather(input,index,valid,plan.cval);
            if(plan.order)for(std::size_t axis=0;axis<rank;++axis) {
                const auto tap=taps[axis];
                const auto weights=_mm256_set_pd(points[axis][3]->weights[tap],points[axis][2]->weights[tap],
                    points[axis][1]->weights[tap],points[axis][0]->weights[tap]);
                coefficient=_mm256_mul_pd(coefficient,weights);
            }
            total=_mm256_add_pd(total,coefficient);
        }
        store<Output,Boolean>(output+begin,total);
    }
    return begin;
}
} // namespace
std::size_t zoom_sample_typed(const void* input,void* output,const detail::ZoomPlan& plan,
                             std::size_t begin,std::size_t end,PixelType source,PixelType target) {
    if(source==PixelType::float64 && target==PixelType::float32)
        return sample(static_cast<const double*>(input),static_cast<float*>(output),plan,begin,end);
    if(source!=target)return begin;
    switch(source) {
        case PixelType::boolean:return sample<std::uint8_t,std::uint8_t,true>(static_cast<const std::uint8_t*>(input),static_cast<std::uint8_t*>(output),plan,begin,end);
        #define SAMPLE(tag,T) case PixelType::tag:return sample(static_cast<const T*>(input),static_cast<T*>(output),plan,begin,end);
        SAMPLE(uint8,std::uint8_t) SAMPLE(int8,std::int8_t) SAMPLE(uint16,std::uint16_t) SAMPLE(int16,std::int16_t)
        SAMPLE(uint32,std::uint32_t) SAMPLE(int32,std::int32_t) SAMPLE(uint64,std::uint64_t) SAMPLE(int64,std::int64_t)
        SAMPLE(float32,float) SAMPLE(float64,double)
        #undef SAMPLE
    }
    return begin;
}
std::size_t zoom_sample(const double* input,double* output,const detail::ZoomPlan& plan,
                        std::size_t begin,std::size_t end) {
    return zoom_sample_typed(input,output,plan,begin,end,PixelType::float64,PixelType::float64);
}
bool profile_all_finite(const void* source,std::size_t count,bool single) {
    std::size_t i=0;
    if(single) {
        const auto* input=static_cast<const float*>(source);
        const auto mask=_mm256_set1_epi32(0x7f800000);
        for(;count-i>=8;i+=8) {
            const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
            if(_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_and_si256(words,mask),mask)))return false;
        }
        for(;i<count;++i){std::uint32_t word;std::memcpy(&word,input+i,4);if((word&0x7f800000u)==0x7f800000u)return false;}
    } else {
        const auto* input=static_cast<const double*>(source);
        const auto mask=_mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
        for(;count-i>=4;i+=4) {
            const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
            if(_mm256_movemask_epi8(_mm256_cmpeq_epi64(_mm256_and_si256(words,mask),mask)))return false;
        }
        for(;i<count;++i){std::uint64_t word;std::memcpy(&word,input+i,8);if((word&UINT64_C(0x7ff0000000000000))==UINT64_C(0x7ff0000000000000))return false;}
    }
    return true;
}
std::size_t profile_sample_typed(const void* input,void* output,const detail::ZoomPlan& plan,
                                std::size_t lanes,PixelType type) {
    switch(type) {
        case PixelType::boolean:return sample<std::uint8_t,std::uint8_t,true,true>(static_cast<const std::uint8_t*>(input),static_cast<std::uint8_t*>(output),plan,0,lanes);
#define PROFILE_SAMPLE(tag,T) case PixelType::tag:return sample<T,T,false,true>(static_cast<const T*>(input),static_cast<T*>(output),plan,0,lanes);
        PROFILE_SAMPLE(uint8,std::uint8_t) PROFILE_SAMPLE(float32,float) PROFILE_SAMPLE(float64,double)
#undef PROFILE_SAMPLE
        default:return 0;
    }
}
} // namespace skimagexx::simd
