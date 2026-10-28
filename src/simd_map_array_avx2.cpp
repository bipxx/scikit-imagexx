#include "map_array.hpp"
#include <immintrin.h>
#include <cstring>

namespace skimagexx::simd {
namespace {
template<std::size_t Bytes>
void strided_store(__m256i first,__m256i second,char* output,std::ptrdiff_t stride) {
    std::uint64_t low[4],high[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(low),first);
    if constexpr(Bytes==16)_mm256_storeu_si256(reinterpret_cast<__m256i*>(high),second);
    for(std::size_t lane=0;lane<4;++lane) {
        auto* target=output+std::ptrdiff_t(lane)*stride;
        std::memcpy(target,low+lane,Bytes<8?Bytes:8);
        if constexpr(Bytes==16)std::memcpy(target+8,high+lane,8);
    }
}
template<std::size_t Bytes>
std::size_t small(const detail::MapArrayInput& input,const unsigned char* values,
    std::size_t begin,std::size_t count,detail::MapArrayOutput output) {
    const auto mask=input.index_bits==64?~std::uint64_t(0):(std::uint64_t(1)<<input.index_bits)-1;
    const auto width=_mm256_set1_epi64x(static_cast<long long>(mask));
    __m256i keys[8],low[8],high[8];
    for(std::size_t k=0;k<input.key_count;++k) {
        keys[k]=_mm256_set1_epi64x(static_cast<long long>(input.keys[k]&mask));
        std::uint64_t a=0,b=0;std::memcpy(&a,values+k*Bytes,Bytes<8?Bytes:8);
        if constexpr(Bytes==16)std::memcpy(&b,values+k*Bytes+8,8);
        low[k]=_mm256_set1_epi64x(static_cast<long long>(a));
        if constexpr(Bytes==16)high[k]=_mm256_set1_epi64x(static_cast<long long>(b));
    }
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto labels=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input.indices+begin+i)),width);
        auto a=_mm256_setzero_si256(),b=_mm256_setzero_si256();
        for(std::size_t k=0;k<input.key_count;++k) {
            const auto equal=_mm256_cmpeq_epi64(labels,keys[k]);
            a=_mm256_blendv_epi8(a,low[k],equal);
            if constexpr(Bytes==16)b=_mm256_blendv_epi8(b,high[k],equal);
        }
        strided_store<Bytes>(a,b,static_cast<char*>(output.data)+std::ptrdiff_t(begin+i)*output.stride,output.stride);
    }
    return i;
}
}
std::size_t map_array_small(const detail::MapArrayInput& input,const void* values,
    std::size_t begin,std::size_t count,detail::MapArrayOutput output) {
    const auto* raw=static_cast<const unsigned char*>(values);
    switch(input.itemsize) {
        case 1:return small<1>(input,raw,begin,count,output);
        case 2:return small<2>(input,raw,begin,count,output);
        case 4:return small<4>(input,raw,begin,count,output);
        case 8:return small<8>(input,raw,begin,count,output);
        case 16:return small<16>(input,raw,begin,count,output);
    }
    return 0;
}

std::size_t map_array_dense(const std::uint64_t* indices,std::size_t count,
    std::uint64_t mask,std::uint64_t minimum,std::size_t cells,
    const void* table,std::size_t bytes,void* output,std::ptrdiff_t output_stride) {
    const auto width=_mm256_set1_epi64x(static_cast<long long>(mask));
    const auto low=_mm256_set1_epi64x(static_cast<long long>(minimum));
    const auto sign=_mm256_set1_epi64x(static_cast<long long>(std::uint64_t(1)<<63));
    const auto bound=_mm256_xor_si256(_mm256_set1_epi64x(static_cast<long long>(cells)),sign);
    const auto lanes=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
    std::size_t i=0;auto* target=static_cast<unsigned char*>(output);
    for(;i+4<=count;i+=4) {
        const auto labels=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(indices+i)),width);
        const auto index=_mm256_sub_epi64(labels,low);
        const auto valid=_mm256_cmpgt_epi64(bound,_mm256_xor_si256(index,sign));
        if(output_stride!=std::ptrdiff_t(bytes)) {
            auto* destination=reinterpret_cast<char*>(target)+std::ptrdiff_t(i)*output_stride;
            if(bytes<=4) {
                const auto mask32=_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(valid,lanes));
                const auto value=_mm256_mask_i64gather_epi32(_mm_setzero_si128(),static_cast<const int*>(table),index,mask32,4);
                const auto expanded=_mm256_cvtepu32_epi64(value);
                if(bytes==1)strided_store<1>(expanded,_mm256_setzero_si256(),destination,output_stride);
                else if(bytes==2)strided_store<2>(expanded,_mm256_setzero_si256(),destination,output_stride);
                else strided_store<4>(expanded,_mm256_setzero_si256(),destination,output_stride);
            } else if(bytes==8) {
                const auto value=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table),index,valid,8);
                strided_store<8>(value,_mm256_setzero_si256(),destination,output_stride);
            } else {
                const auto doubled=_mm256_slli_epi64(index,1);
                const auto first=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table),doubled,valid,8);
                const auto second=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table)+1,doubled,valid,8);
                strided_store<16>(first,second,destination,output_stride);
            }
            continue;
        }
        if(bytes<=4) {
            const auto mask32=_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(valid,lanes));
            const auto value=_mm256_mask_i64gather_epi32(_mm_setzero_si128(),static_cast<const int*>(table),index,mask32,4);
            if(bytes==4)_mm_storeu_si128(reinterpret_cast<__m128i*>(target+i*4),value);
            else if(bytes==2) {
                const auto packed=_mm_shuffle_epi8(value,_mm_setr_epi8(0,1,4,5,8,9,12,13,-1,-1,-1,-1,-1,-1,-1,-1));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(target+i*2),packed);
            } else {
                const auto packed=_mm_shuffle_epi8(value,_mm_setr_epi8(0,4,8,12,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1));
                const auto word=_mm_cvtsi128_si32(packed);std::memcpy(target+i,&word,4);
            }
        } else if(bytes==8) {
            const auto value=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table),index,valid,8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(target+i*8),value);
        } else {
            const auto doubled=_mm256_slli_epi64(index,1);
            const auto first=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table),doubled,valid,8);
            const auto second=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),static_cast<const long long*>(table)+1,doubled,valid,8);
            const auto a=_mm256_unpacklo_epi64(first,second),b=_mm256_unpackhi_epi64(first,second);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(target+i*16),_mm256_permute2x128_si256(a,b,0x20));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(target+i*16+32),_mm256_permute2x128_si256(a,b,0x31));
        }
    }
    return i;
}
}
