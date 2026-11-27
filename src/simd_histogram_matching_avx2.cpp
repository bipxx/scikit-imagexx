#include "simd.hpp"
#include <algorithm>
#include <immintrin.h>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::simd {
std::size_t rank_dense_labels(const void* input,void* output,std::size_t count,std::uint64_t low,
    const std::size_t* table,PixelType type,std::size_t entry_bytes) {
    if(sizeof(std::size_t)!=8)return 0;
    if(type!=PixelType::uint16 && type!=PixelType::uint32 && type!=PixelType::uint64)return 0;
    const auto step=static_cast<long long>(entry_bytes);
    const auto offsets=_mm256_set_epi64x(3*step,2*step,step,0),base=_mm256_set1_epi64x(static_cast<long long>(low));
    const auto key_mask=_mm256_set1_epi64x(entry_bytes==8 ? UINT32_MAX : -1LL);
    const auto* entries=static_cast<const unsigned char*>(input);std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto keys=_mm256_and_si256(key_mask,
            _mm256_i64gather_epi64(reinterpret_cast<const long long*>(entries+i*entry_bytes),offsets,1));
        const auto ranks=_mm256_i64gather_epi64(reinterpret_cast<const long long*>(table),_mm256_sub_epi64(keys,base),8);
        if(type==PixelType::uint64)_mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<std::uint64_t*>(output)+i),ranks);
        else {
            const auto first=_mm_shuffle_epi32(_mm256_castsi256_si128(ranks),_MM_SHUFFLE(2,0,2,0));
            const auto second=_mm_shuffle_epi32(_mm256_extracti128_si256(ranks,1),_MM_SHUFFLE(2,0,2,0));
            const auto packed=_mm_unpacklo_epi64(first,second);
            if(type==PixelType::uint32)_mm_storeu_si128(reinterpret_cast<__m128i*>(static_cast<std::uint32_t*>(output)+i),packed);
            else _mm_storel_epi64(reinterpret_cast<__m128i*>(static_cast<std::uint16_t*>(output)+i),_mm_packus_epi32(packed,_mm_setzero_si128()));
        }
    }
    return i;
}
std::size_t rank_dense_labels(const void* input,void* output,std::size_t count,std::uint64_t low,const std::size_t* table,PixelType type) {
    return rank_dense_labels(input,output,count,low,table,type,16);
}
bool rank_uniform_bits(const void* input,std::size_t count,std::size_t bytes) {
    const auto* source=static_cast<const unsigned char*>(input);
    std::uint64_t first=0;std::memcpy(&first,source,bytes);
    const auto pattern=bytes==1 ? _mm256_set1_epi8(char(first)) : bytes==2 ? _mm256_set1_epi16(short(first)) :
        bytes==4 ? _mm256_set1_epi32(int(first)) : _mm256_set1_epi64x(static_cast<long long>(first));
    const auto lanes=32/bytes;std::size_t i=0;
    for(;i+lanes<=count;i+=lanes)if(_mm256_movemask_epi8(_mm256_cmpeq_epi8(pattern,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source+i*bytes))))!=-1)return false;
    for(;i<count;++i)if(std::memcmp(source,source+i*bytes,bytes))return false;
    return true;
}
namespace {
template<class T> __m128i small_load(const unsigned char* input,std::size_t stride) {
    if(stride==1) {
        if constexpr(sizeof(T)==1){std::uint32_t bits;std::memcpy(&bits,input,4);return _mm_cvtsi32_si128(int(bits));}
        else return _mm_loadl_epi64(reinterpret_cast<const __m128i*>(input));
    }
    alignas(16) unsigned char bytes[16]={};
    for(int j=0;j<4;++j)std::memcpy(bytes+j*sizeof(T),input+std::size_t(j)*stride*sizeof(T),sizeof(T));
    return _mm_load_si128(reinterpret_cast<const __m128i*>(bytes));
}
template<class T> std::size_t small_keys(const void* source,std::uint64_t* output,std::size_t count,std::size_t stride) {
    const auto* input=static_cast<const unsigned char*>(source);std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto raw=small_load<T>(input+i*stride*sizeof(T),stride);
        __m256i keys;
        if constexpr(sizeof(T)==1 && std::is_signed_v<T>)keys=_mm256_cvtepi8_epi64(raw);
        else if constexpr(sizeof(T)==1)keys=_mm256_cvtepu8_epi64(raw);
        else if constexpr(std::is_signed_v<T>)keys=_mm256_cvtepi16_epi64(raw);
        else keys=_mm256_cvtepu16_epi64(raw);
        if constexpr(std::is_signed_v<T>)keys=_mm256_xor_si256(keys,_mm256_set1_epi64x(std::numeric_limits<long long>::min()));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),keys);
    }
    return i;
}
template<class T> std::size_t lookup(const std::uint64_t* input,void* output,std::size_t count,std::uint64_t low,
    const double* table,std::size_t stride) {
    long long low_bits;std::memcpy(&low_bits,&low,8);const auto base=_mm256_set1_epi64x(low_bits);
    auto* target=static_cast<T*>(output);std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto indices=_mm256_sub_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),base);
        const auto values=_mm256_i64gather_pd(table,indices,8);
        if(stride==1) {
            if constexpr(std::is_same_v<T,double>) {
                _mm256_storeu_pd(target+i,values);
                continue;
            } else if constexpr(std::is_same_v<T,float>) {
                _mm_storeu_ps(target+i,_mm256_cvtpd_ps(values));
                continue;
            }
        }
        alignas(32) T lanes[4];
        if constexpr(std::is_same_v<T,double>)_mm256_store_pd(lanes,values);
        else if constexpr(std::is_same_v<T,float>)_mm_store_ps(lanes,_mm256_cvtpd_ps(values));
        else if constexpr(std::is_same_v<T,bool>) {
            const auto mask=_mm256_movemask_pd(_mm256_cmp_pd(values,_mm256_setzero_pd(),_CMP_NEQ_UQ));
            for(int j=0;j<4;++j)lanes[j]=(mask&(1<<j))!=0;
        } else {
            const auto integers=_mm256_cvttpd_epi32(values);
            const auto words=std::is_unsigned_v<T> ? _mm_packus_epi32(integers,_mm_setzero_si128()) : _mm_packs_epi32(integers,_mm_setzero_si128());
            if constexpr(sizeof(T)==2)_mm_storel_epi64(reinterpret_cast<__m128i*>(lanes),words);
            else {
                const auto bytes=std::is_unsigned_v<T> ? _mm_packus_epi16(words,_mm_setzero_si128()) : _mm_packs_epi16(words,_mm_setzero_si128());
                const auto bits=_mm_cvtsi128_si32(bytes);std::memcpy(lanes,&bits,4);
            }
        }
        if(stride==1)std::memcpy(target+i,lanes,sizeof(lanes));
        else for(int j=0;j<4;++j)target[(i+std::size_t(j))*stride]=lanes[j];
    }
    return i;
}

template<bool DetectRankEdges>
void rank_edges(__m256i zero,__m256i nan,__m256i negative,
    __m256i& nans,__m256i& negative_zeros,__m256i& positive_zeros) {
    if constexpr(DetectRankEdges) {
        nans=_mm256_or_si256(nans,nan);
        negative_zeros=_mm256_or_si256(negative_zeros,_mm256_and_si256(zero,negative));
        positive_zeros=_mm256_or_si256(positive_zeros,_mm256_andnot_si256(negative,zero));
    }
}

template<bool DetectRankEdges>
__m256i double_keys(const double* source,std::size_t stride,__m256i offsets,
    __m256i& nans,__m256i& negative_zeros,__m256i& positive_zeros) {
    const auto zeros=_mm256_setzero_si256();
    const auto sign=_mm256_set1_epi64x(INT64_MIN),magnitude=_mm256_set1_epi64x(INT64_MAX);
    auto bits=_mm256_castpd_si256(stride==1 ? _mm256_loadu_pd(source) : _mm256_i64gather_pd(source,offsets,8));
    const auto absolute=_mm256_and_si256(bits,magnitude);
    const auto zero=_mm256_cmpeq_epi64(absolute,zeros);
    const auto negative=_mm256_cmpgt_epi64(zeros,bits);
    const auto nan=_mm256_cmpgt_epi64(absolute,_mm256_set1_epi64x(INT64_C(0x7ff0000000000000)));
    rank_edges<DetectRankEdges>(zero,nan,negative,nans,negative_zeros,positive_zeros);
    bits=_mm256_andnot_si256(zero,bits);
    return _mm256_or_si256(nan,_mm256_xor_si256(bits,_mm256_or_si256(_mm256_andnot_si256(zero,negative),sign)));
}

void report_rank_edges(__m256i nans,__m256i negative_zeros,__m256i positive_zeros,unsigned& flags) {
    if(_mm256_movemask_epi8(nans))flags|=1;
    if(_mm256_movemask_epi8(negative_zeros))flags|=2;
    if(_mm256_movemask_epi8(positive_zeros))flags|=4;
}

template<class T>
std::size_t integer_keys(const void* source,std::uint64_t* output,std::size_t count,std::size_t stride) {
    const auto* input=static_cast<const unsigned char*>(source);
    const auto step=static_cast<long long>(stride);
    const auto offsets64=_mm256_setr_epi64x(0,step,2*step,3*step);
    const auto offsets32=_mm_setr_epi32(0,int(step),int(2*step),int(3*step));
    const auto sign=_mm256_set1_epi64x(INT64_MIN);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto* block=input+i*stride*sizeof(T);
        __m256i keys;
        if constexpr(sizeof(T)==4) {
            const auto bits=stride==1 ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(block)) :
                _mm_i32gather_epi32(reinterpret_cast<const int*>(block),offsets32,4);
            if constexpr(std::is_signed_v<T>)keys=_mm256_cvtepi32_epi64(bits);
            else keys=_mm256_cvtepu32_epi64(bits);
        } else {
            keys=stride==1 ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block)) :
                _mm256_i64gather_epi64(reinterpret_cast<const long long*>(block),offsets64,8);
        }
        if constexpr(std::is_signed_v<T>)keys=_mm256_xor_si256(keys,sign);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),keys);
    }
    return i;
}

template<bool DetectRankEdges>
std::size_t histogram_keys_impl(const void* input,std::uint64_t* output,std::size_t count,PixelType type,
    std::size_t stride,unsigned* flags) {
    if(stride>std::size_t(INT32_MAX)/8)return 0;
    switch(type) {
    case PixelType::uint8:return small_keys<std::uint8_t>(input,output,count,stride);
    case PixelType::int8:return small_keys<std::int8_t>(input,output,count,stride);
    case PixelType::uint16:return small_keys<std::uint16_t>(input,output,count,stride);
    case PixelType::int16:return small_keys<std::int16_t>(input,output,count,stride);
    case PixelType::uint32:return integer_keys<std::uint32_t>(input,output,count,stride);
    case PixelType::int32:return integer_keys<std::int32_t>(input,output,count,stride);
    case PixelType::uint64:return integer_keys<std::uint64_t>(input,output,count,stride);
    case PixelType::int64:return integer_keys<std::int64_t>(input,output,count,stride);
    default:break;
    }
    std::size_t i=0;
    const auto zeros=_mm256_setzero_si256();
    auto nans=zeros,negative_zeros=zeros,positive_zeros=zeros;
    if(type==PixelType::float32) {
        const auto step=int(stride);
        const auto offsets=_mm256_setr_epi32(0,step,2*step,3*step,4*step,5*step,6*step,7*step);
        const auto sign=_mm256_set1_epi32(INT32_MIN),magnitude=_mm256_set1_epi32(INT32_MAX);
        for(;i+8<=count;i+=8) {
            const auto* source=static_cast<const float*>(input)+i*stride;
            auto bits=_mm256_castps_si256(stride==1 ? _mm256_loadu_ps(source) : _mm256_i32gather_ps(source,offsets,4));
            const auto absolute=_mm256_and_si256(bits,magnitude);
            const auto zero=_mm256_cmpeq_epi32(absolute,zeros);
            const auto negative=_mm256_cmpgt_epi32(zeros,bits);
            const auto nan=_mm256_cmpgt_epi32(absolute,_mm256_set1_epi32(0x7f800000));
            rank_edges<DetectRankEdges>(zero,nan,negative,nans,negative_zeros,positive_zeros);
            bits=_mm256_andnot_si256(zero,bits);
            const auto keys=_mm256_or_si256(nan,_mm256_xor_si256(bits,_mm256_or_si256(_mm256_andnot_si256(zero,negative),sign)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_cvtepu32_epi64(_mm256_castsi256_si128(keys)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i+4),_mm256_cvtepu32_epi64(_mm256_extracti128_si256(keys,1)));
        }
    } else if(type==PixelType::float64) {
        const auto step=static_cast<long long>(stride);
        const auto offsets=_mm256_setr_epi64x(0,step,2*step,3*step);
        for(;i+4<=count;i+=4) {
            const auto keys=double_keys<DetectRankEdges>(static_cast<const double*>(input)+i*stride,stride,
                offsets,nans,negative_zeros,positive_zeros);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),keys);
        }
    }
    if constexpr(DetectRankEdges)report_rank_edges(nans,negative_zeros,positive_zeros,*flags);
    return i;
}
}
std::size_t histogram_keys(const void* input,std::uint64_t* output,std::size_t count,PixelType type,std::size_t stride) {
    return histogram_keys_impl<false>(input,output,count,type,stride,nullptr);
}
std::size_t histogram_keys(const void* input,std::uint64_t* output,std::size_t count,PixelType type,
    std::size_t stride,unsigned& flags) {
    return histogram_keys_impl<true>(input,output,count,type,stride,&flags);
}
std::size_t indexed_double_prefixes(const double* input,std::uint64_t* output,std::size_t count,
    std::size_t stride,std::uint32_t start,unsigned* flags) {
    if(stride>std::size_t(INT32_MAX)/8)return 0;
    const auto step=static_cast<long long>(stride);
    const auto offsets=_mm256_setr_epi64x(0,step,2*step,3*step);
    const auto increment=_mm256_set1_epi64x(4);
    const auto emit=[&](auto detect) {
        auto positions=_mm256_setr_epi64x(start,start+1,start+2,start+3);
        auto nans=_mm256_setzero_si256(),negative_zeros=nans,positive_zeros=nans;
        std::size_t i=0;
        for(;i+4<=count;i+=4) {
            const auto keys=double_keys<decltype(detect)::value>(input+i*stride,stride,offsets,
                nans,negative_zeros,positive_zeros);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_blend_epi32(keys,positions,0x55));
            positions=_mm256_add_epi64(positions,increment);
        }
        if constexpr(decltype(detect)::value)report_rank_edges(nans,negative_zeros,positive_zeros,*flags);
        return i;
    };
    return flags ? emit(std::true_type{}) : emit(std::false_type{});
}

void histogram_key_bounds(const std::uint64_t* keys,std::size_t count,std::uint64_t& low,std::uint64_t& high) {
    const auto sign=_mm256_set1_epi64x(INT64_MIN);
    auto minimum=_mm256_set1_epi64x(INT64_MAX),maximum=sign;
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto biased=_mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(keys+i)),sign);
        minimum=_mm256_blendv_epi8(minimum,biased,_mm256_cmpgt_epi64(minimum,biased));
        maximum=_mm256_blendv_epi8(maximum,biased,_mm256_cmpgt_epi64(biased,maximum));
    }
    alignas(32) std::uint64_t lows[4],highs[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lows),_mm256_xor_si256(minimum,sign));
    _mm256_store_si256(reinterpret_cast<__m256i*>(highs),_mm256_xor_si256(maximum,sign));
    low=*std::min_element(lows,lows+4);
    high=*std::max_element(highs,highs+4);
    for(;i<count;++i) {
        low=std::min(low,keys[i]);
        high=std::max(high,keys[i]);
    }
}
std::size_t histogram_lookup(const std::uint64_t* keys,void* output,std::size_t count,std::uint64_t low,
    const double* table,PixelType type,std::size_t stride) {
    switch(type) {
#define LOOKUP(tag,T) case PixelType::tag:return lookup<T>(keys,output,count,low,table,stride);
    LOOKUP(boolean,bool) LOOKUP(uint8,std::uint8_t) LOOKUP(int8,std::int8_t) LOOKUP(uint16,std::uint16_t)
    LOOKUP(int16,std::int16_t) LOOKUP(float32,float) LOOKUP(float64,double)
#undef LOOKUP
    default:return 0;
    }
}
}
