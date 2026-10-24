#include "skimagexx/metrics.hpp"
#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <type_traits>
namespace skimagexx::simd {
namespace {
__m128i low32(__m256i x){x=_mm256_shuffle_epi32(x,_MM_SHUFFLE(2,0,2,0));return _mm_unpacklo_epi64(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));}
__m256d unsigned32(__m128i value){return _mm256_add_pd(_mm256_cvtepi32_pd(value),_mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(value,31)),_mm256_set1_pd(4294967296.)));}
__m256d wide(__m256i value,bool is_signed) {
    const auto high=low32(_mm256_srli_epi64(value,32));
    return _mm256_add_pd(_mm256_mul_pd(is_signed ? _mm256_cvtepi32_pd(high):unsigned32(high),_mm256_set1_pd(4294967296.)),unsigned32(low32(value)));
}
__m128 half4(__m128i packed,bool swap) {
    if(swap)packed=_mm_shuffle_epi8(packed,_mm_setr_epi8(1,0,3,2,5,4,7,6,-1,-1,-1,-1,-1,-1,-1,-1));
    const auto value=_mm_cvtepu16_epi32(packed),fraction=_mm_and_si128(value,_mm_set1_epi32(1023)),
               exponent=_mm_srli_epi32(_mm_and_si128(value,_mm_set1_epi32(0x7c00)),10),
               sign=_mm_slli_epi32(_mm_and_si128(value,_mm_set1_epi32(0x8000)),16);
    const auto normal=_mm_or_si128(sign,_mm_or_si128(_mm_slli_epi32(_mm_add_epi32(exponent,_mm_set1_epi32(112)),23),_mm_slli_epi32(fraction,13)));
    const auto small=_mm_or_si128(sign,_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(fraction),_mm_set1_ps(0x1p-24f))));
    const auto special=_mm_or_si128(sign,_mm_or_si128(_mm_set1_epi32(0x7f800000),_mm_slli_epi32(fraction,13)));
    auto result=_mm_blendv_epi8(normal,small,_mm_cmpeq_epi32(exponent,_mm_setzero_si128()));
    result=_mm_blendv_epi8(result,special,_mm_cmpeq_epi32(exponent,_mm_set1_epi32(31)));
    return _mm_castsi128_ps(result);
}
__m128i swap16(__m128i value) {
    return _mm_shuffle_epi8(value,_mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14));
}
__m128i swap32(__m128i value) {
    return _mm_shuffle_epi8(value,_mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12));
}
__m256i swap64(__m256i value) {
    return _mm256_shuffle_epi8(value,_mm256_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,
                                                        7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8));
}
template<class T> __m256d raw(const unsigned char* p,bool swap=false) {
    if constexpr(sizeof(T)==1) {std::uint32_t word;std::memcpy(&word,p,4);const auto v=_mm_cvtsi32_si128(int(word));return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi8_epi32(v):_mm_cvtepu8_epi32(v));}
    else if constexpr(sizeof(T)==2){auto v=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));if(swap)v=swap16(v);return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi16_epi32(v):_mm_cvtepu16_epi32(v));}
    else if constexpr(sizeof(T)==4){auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));if(swap)v=swap32(v);return std::is_signed_v<T> ? _mm256_cvtepi32_pd(v):unsigned32(v);}
    else {auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));if(swap)v=swap64(v);return wide(v,std::is_signed_v<T>);}
}
template<class T,bool Boolean=false> __m256d narrow(const unsigned char* p,std::ptrdiff_t step,bool swap) {
    // Four independent byte-safe reads: no widened gather can cross a row,
    // allocation or guard page for 8/16-bit elements or negative strides.
    const auto lane=[&](int i) {
        T value;
        if constexpr(sizeof(T)==2) {
            const auto* source=p+std::ptrdiff_t(i)*step;
            if(swap) {const unsigned char bytes[2]{source[1],source[0]};std::memcpy(&value,bytes,2);}
            else std::memcpy(&value,source,2);
        } else std::memcpy(&value,p+std::ptrdiff_t(i)*step,1);
        if constexpr(Boolean)return int(value!=0);
        else return int(value);
    };
    return _mm256_cvtepi32_pd(_mm_setr_epi32(lane(0),lane(1),lane(2),lane(3)));
}
__m256d load(const MetricImageView& view,std::ptrdiff_t offset,std::ptrdiff_t step) {
    const auto* p=static_cast<const unsigned char*>(view.data)+offset;
    // A four-lane caller has already bounded this logical run to its row.
    // Read exactly those four elements in ascending physical order, then
    // restore logical order. Avoid recursive generic dtype dispatch for the
    // common native floating reverse; keep conversion before permutation as
    // in the generic path (including signaling-NaN payloads/status).
    if(!view.half && !view.swap) {
        if(view.type==PixelType::float64 && step==-8)
            return _mm256_permute4x64_pd(_mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p-24))),_MM_SHUFFLE(0,1,2,3));
        if(view.type==PixelType::float32 && step==-4)
            return _mm256_permute4x64_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p-12)))),_MM_SHUFFLE(0,1,2,3));
    }
    if(step==0)return _mm256_set1_pd(detail::metric_read(view,offset));
    if(view.half && (step==2 || step==-2)) {
        auto value=half4(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(step==2 ? p:p-6)),view.swap);
        if(step<0)value=_mm_shuffle_ps(value,value,_MM_SHUFFLE(0,1,2,3));
        return _mm256_cvtps_pd(value);
    }
    if(view.half) {
        // AVX2 has no 16-bit gather, and widened gathers may cross the final
        // element/allocation. Pack four exact byte-safe reads, then share the
        // existing integer/float binary16 decoder (no F16C requirement).
        std::uint16_t words[4];
        for(int i=0;i<4;++i)std::memcpy(words+i,p+std::ptrdiff_t(i)*step,2);
        const auto packed=_mm_setr_epi16(short(words[0]),short(words[1]),short(words[2]),short(words[3]),0,0,0,0);
        return _mm256_cvtps_pd(half4(packed,view.swap));
    }
    const auto bytes=std::ptrdiff_t(pixel_size(view.type));
    if(!view.half && step==-bytes)
        return _mm256_permute4x64_pd(load(view,offset-3*bytes,bytes),_MM_SHUFFLE(0,1,2,3));
    if(!view.half && step!=bytes && bytes<=2) {
        switch(view.type) {
            case PixelType::boolean:return narrow<std::uint8_t,true>(p,step,false);
            case PixelType::uint8:return narrow<std::uint8_t>(p,step,false);
            case PixelType::int8:return narrow<std::int8_t>(p,step,false);
            case PixelType::uint16:return narrow<std::uint16_t>(p,step,view.swap);
            case PixelType::int16:return narrow<std::int16_t>(p,step,view.swap);
            default:break;
        }
    }
    if(!view.swap && !view.half && step!=bytes && bytes>=4) {
        const auto indices=_mm256_set_epi64x(3*step,2*step,step,0);
        switch(view.type) {
            case PixelType::float32:return _mm256_cvtps_pd(_mm256_i64gather_ps(reinterpret_cast<const float*>(p),indices,1));
            case PixelType::float64:return _mm256_i64gather_pd(reinterpret_cast<const double*>(p),indices,1);
            case PixelType::int32:return _mm256_cvtepi32_pd(_mm256_i64gather_epi32(reinterpret_cast<const int*>(p),indices,1));
            case PixelType::uint32:return unsigned32(_mm256_i64gather_epi32(reinterpret_cast<const int*>(p),indices,1));
            case PixelType::int64:return wide(_mm256_i64gather_epi64(reinterpret_cast<const long long*>(p),indices,1),true);
            case PixelType::uint64:return wide(_mm256_i64gather_epi64(reinterpret_cast<const long long*>(p),indices,1),false);
            default:break;
        }
    }
    if(step!=bytes) {
        alignas(32) double values[4];for(int i=0;i<4;++i)values[i]=detail::metric_read(view,offset+std::ptrdiff_t(i)*step);return _mm256_load_pd(values);
    }
    switch(view.type) {
        case PixelType::boolean:{std::uint32_t word;std::memcpy(&word,p,4);const auto v=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(word)));return _mm256_cvtepi32_pd(_mm_andnot_si128(_mm_cmpeq_epi32(v,_mm_setzero_si128()),_mm_set1_epi32(1)));}
        #define LOAD(tag,T) case PixelType::tag:return raw<T>(p,view.swap);
        LOAD(uint8,std::uint8_t) LOAD(int8,std::int8_t) LOAD(uint16,std::uint16_t) LOAD(int16,std::int16_t)
        LOAD(uint32,std::uint32_t) LOAD(int32,std::int32_t) LOAD(uint64,std::uint64_t) LOAD(int64,std::int64_t)
        #undef LOAD
        case PixelType::float32:{auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));if(view.swap)v=swap32(v);return _mm256_cvtps_pd(_mm_castsi128_ps(v));}
        case PixelType::float64:{auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));if(view.swap)v=swap64(v);return _mm256_castsi256_pd(v);}
        default:return _mm256_setzero_pd();
    }
}
}
bool metric_leaf_classify(const double* error,const double* energy,const double* values,std::size_t count,
                         MetricNormalization mode,MetricSums& result) {
    // Inspect exponent bits before any floating range comparison. Exceptional
    // values retain the original scalar classification and flagged replay.
    const auto exponent=_mm256_set1_epi64x(0x7ff0000000000000LL);
    const auto magnitude=_mm256_set1_epi64x(0x7fffffffffffffffLL);
    const bool range=mode==MetricNormalization::min_max,euclidean=mode==MetricNormalization::euclidean;
    auto invalid=_mm256_setzero_si256(),zeros=_mm256_setzero_si256();
    auto low=_mm256_set1_pd(INFINITY),high=_mm256_set1_pd(-INFINITY);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto value=_mm256_loadu_pd(values+i),err=_mm256_loadu_pd(error+i);
        const auto exceptional=[&](const __m256d v){return _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v),exponent),exponent);};
        invalid=_mm256_or_si256(invalid,_mm256_or_si256(exceptional(value),exceptional(err)));
        if(euclidean)invalid=_mm256_or_si256(invalid,exceptional(_mm256_loadu_pd(energy+i)));
        if(range) {
            zeros=_mm256_or_si256(zeros,_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(value),magnitude),_mm256_setzero_si256()));
            // No externally visible flags are consumed on this speculative
            // path. NaNs below still force the untouched scalar replay.
            low=_mm256_min_pd(low,value);high=_mm256_max_pd(high,value);
        }
    }
    if(_mm256_movemask_pd(_mm256_castsi256_pd(invalid)) || (range && _mm256_movemask_pd(_mm256_castsi256_pd(zeros))))return false;
    double minimum=INFINITY,maximum=-INFINITY;
    if(range) {
        alignas(32) double lows[4],highs[4];_mm256_store_pd(lows,low);_mm256_store_pd(highs,high);
        for(int j=0;j<4;++j){if(lows[j]<minimum)minimum=lows[j];if(highs[j]>maximum)maximum=highs[j];}
    }
    for(;i<count;++i) {
        if(!std::isfinite(values[i]) || !std::isfinite(error[i]) || (euclidean && !std::isfinite(energy[i])))return false;
        if(range){if(values[i]==0)return false;if(values[i]<minimum)minimum=values[i];if(values[i]>maximum)maximum=values[i];}
    }
    if(range && count){result.low=minimum;result.high=maximum;result.valid=true;}
    return true;
}
std::size_t metric_stage_values(const MetricImageView& a,const MetricImageView& b,std::ptrdiff_t first,std::ptrdiff_t second,
    std::ptrdiff_t step0,std::ptrdiff_t step1,std::size_t count,bool single,int operation,void* output) {
    // Half conversion preserves signaling payloads without raising invalid;
    // arithmetic may promote through double and correctly reports invalid.
    if(operation==0 && a.half)return 0;
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto x=load(a,first+std::ptrdiff_t(i)*step0,step0);
        if(single) {
            const auto value=_mm256_cvtpd_ps(x);
            const auto result=operation==0 ? value:operation==1 ? _mm_sub_ps(value,_mm256_cvtpd_ps(load(b,second+std::ptrdiff_t(i)*step1,step1))):_mm_mul_ps(value,value);
            _mm_storeu_ps(static_cast<float*>(output)+i,result);
        } else {
            const auto result=operation==0 ? x:operation==1 ? _mm256_sub_pd(x,load(b,second+std::ptrdiff_t(i)*step1,step1)):_mm256_mul_pd(x,x);
            _mm256_storeu_pd(static_cast<double*>(output)+i,result);
        }
    }
    return i;
}
std::size_t metric_values(const MetricImageView& a,const MetricImageView& b,std::ptrdiff_t first,std::ptrdiff_t second,
    std::ptrdiff_t step0,std::ptrdiff_t step1,std::size_t count,bool single,MetricNormalization mode,double* error,double* energy,double* values) {
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto x=load(a,first+std::ptrdiff_t(i)*step0,step0),y=load(b,second+std::ptrdiff_t(i)*step1,step1);
        if(single) {
            const auto xf=_mm256_cvtpd_ps(x),yf=_mm256_cvtpd_ps(y),delta=_mm_sub_ps(xf,yf);
            _mm256_storeu_pd(error+i,_mm256_cvtps_pd(_mm_mul_ps(delta,delta)));
            _mm256_storeu_pd(values+i,_mm256_cvtps_pd(xf));
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_cvtps_pd(_mm_mul_ps(xf,xf)));
        } else {
            const auto delta=_mm256_sub_pd(x,y);_mm256_storeu_pd(error+i,_mm256_mul_pd(delta,delta));_mm256_storeu_pd(values+i,x);
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_mul_pd(x,x));
        }
    }
    return i;
}
}
