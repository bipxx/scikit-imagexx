#include "skimagexx/rank.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
namespace skimagexx::simd {
namespace {
__m256i prefix4(__m256i value) {
    const auto zero=_mm256_setzero_si256();
    const auto previous=_mm256_blend_epi32(_mm256_permute4x64_epi64(value,_MM_SHUFFLE(2,1,0,0)),zero,0x03);
    const auto pairs=_mm256_add_epi64(value,previous);
    const auto previous_pairs=_mm256_blend_epi32(_mm256_permute4x64_epi64(pairs,_MM_SHUFFLE(1,0,0,0)),zero,0x0f);
    return _mm256_add_epi64(pairs,previous_pairs);
}
__m256i gather4(const std::int64_t* counts,const std::uint32_t* indices) {
    return _mm256_i32gather_epi64(reinterpret_cast<const long long*>(counts),_mm_loadu_si128(reinterpret_cast<const __m128i*>(indices)),8);
}
}
void rank_decision_prefix4(const std::int64_t* counts,const std::uint32_t* indices,std::int64_t q,std::int64_t moment,std::int64_t* qout,std::int64_t* mout) {
    const auto values=gather4(counts,indices);
    const auto bins=_mm256_cvtepu32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(indices)));
    const auto low=_mm256_mul_epu32(values,bins);
    const auto high=_mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(values,32),bins),32);
    const auto weighted=_mm256_add_epi64(low,high);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(qout),_mm256_add_epi64(prefix4(values),_mm256_set1_epi64x(q)));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(mout),_mm256_add_epi64(prefix4(weighted),_mm256_set1_epi64x(moment)));
}
void rank_majority4(const std::int64_t* counts,const std::uint32_t* indices,std::int64_t& votes,std::int64_t& candidate) {
    const auto values=gather4(counts,indices);
    auto maximum=values;
    auto other=_mm256_permute4x64_epi64(maximum,_MM_SHUFFLE(1,0,3,2));
    maximum=_mm256_blendv_epi8(maximum,other,_mm256_cmpgt_epi64(other,maximum));
    other=_mm256_permute4x64_epi64(maximum,_MM_SHUFFLE(2,3,0,1));
    maximum=_mm256_blendv_epi8(maximum,other,_mm256_cmpgt_epi64(other,maximum));
    const auto best=std::int64_t(_mm_cvtsi128_si64(_mm256_castsi256_si128(maximum)));
    if(best>votes) {
        const unsigned mask=unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(values,maximum))));
        unsigned lane=0;while(!(mask&(1u<<lane)))++lane;
        votes=best;candidate=indices[lane]; // indices arrive strictly ascending
    }
}
std::size_t rank_histogram_normalized(const std::int64_t* counts,void* output,std::size_t count,double scale,bool narrow) {
    std::size_t i=0;const auto factor=_mm256_set1_pd(scale);
    const auto order=_mm256_setr_epi32(0,2,4,6,0,0,0,0);
    for(;i+4<=count;i+=4) {
        const auto words=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(counts+i));
        const auto integers=_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(words,order));
        const auto values=_mm256_mul_pd(_mm256_cvtepi32_pd(integers),factor);
        if(narrow)_mm_storeu_ps(reinterpret_cast<float*>(output)+i,_mm256_cvtpd_ps(values));
        else _mm256_storeu_pd(reinterpret_cast<double*>(output)+i,values);
    }
    return i;
}
std::int64_t rank_count_prefix(const std::int64_t* data,std::size_t count) {
    auto sums=_mm256_setzero_si256();std::size_t i=0;
    for(;i+4<=count;i+=4)sums=_mm256_add_epi64(sums,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i)));
    alignas(32) std::int64_t lanes[4];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),sums);
    std::int64_t result=lanes[0]+lanes[1]+lanes[2]+lanes[3];
    for(;i<count;++i)result+=data[i];
    return result;
}
void rank_weighted_prefix(const std::int64_t* data,std::size_t count,std::size_t first,std::int64_t& population,std::int64_t& weighted) {
    auto sums=_mm256_setzero_si256(),products=_mm256_setzero_si256();std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));
        const auto indices=_mm256_setr_epi64x(first+i,first+i+1,first+i+2,first+i+3);
        // AVX2 has no packed 64-bit multiply. Reconstruct the low64 product
        // from unsigned32 halves; this also preserves signed count bits.
        const auto low=_mm256_mul_epu32(values,indices);
        const auto high=_mm256_slli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(values,32),indices),32);
        sums=_mm256_add_epi64(sums,values);products=_mm256_add_epi64(products,_mm256_add_epi64(low,high));
    }
    alignas(32) std::int64_t counts[4],weights[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(counts),sums);_mm256_store_si256(reinterpret_cast<__m256i*>(weights),products);
    for(int lane=0;lane<4;++lane){population+=counts[lane];weighted+=weights[lane];}
    for(;i<count;++i){population+=data[i];weighted+=data[i]*std::int64_t(first+i);}
}
std::uint64_t rank_sum(const void* input,std::size_t count,bool wide) {
    std::uint64_t result=0;std::size_t i=0;
    if(!wide) {
        const auto* data=static_cast<const std::uint8_t*>(input);auto sums=_mm256_setzero_si256();
        for(;i+32<=count;i+=32)sums=_mm256_add_epi64(sums,_mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i)),_mm256_setzero_si256()));
        alignas(32) std::uint64_t lanes[4];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),sums);for(auto value:lanes)result+=value;
        for(;i<count;++i)result+=data[i];
    } else {
        const auto* data=static_cast<const std::uint16_t*>(input);
        for(;i+16<=count;i+=16) {
            const auto v=_mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i)),_mm256_set1_epi16(short(-32768)));
            const auto pairs=_mm256_madd_epi16(v,_mm256_set1_epi16(1));
            alignas(32) std::int32_t lanes[8];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),pairs);
            std::int64_t sum=16*32768;for(auto value:lanes)sum+=value;result+=std::uint64_t(sum);
        }
        for(;i<count;++i){std::uint16_t value;std::memcpy(&value,reinterpret_cast<const unsigned char*>(data)+i*2,2);result+=value;}
    }
    return result;
}
bool rank_uniform(const void* data,bool wide,std::uint32_t& value) {
    if(wide){std::uint16_t first;std::memcpy(&first,data,2);value=first;}else value=*static_cast<const std::uint8_t*>(data);
    const auto repeated=wide ? _mm256_set1_epi16(short(value)):_mm256_set1_epi8(char(value));
    return _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(static_cast<const __m256i*>(data)),repeated))==-1;
}
std::uint16_t rank_max_u16(const std::uint16_t* data,std::size_t count) {
    auto maximum=_mm256_setzero_si256();std::size_t i=0;
    for(;i+16<=count;i+=16)maximum=_mm256_max_epu16(maximum,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i)));
    alignas(32) std::uint16_t lanes[16];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),maximum);
    std::uint16_t result=0;for(auto value:lanes)result=std::max(result,value);for(;i<count;++i){std::uint16_t value;std::memcpy(&value,reinterpret_cast<const unsigned char*>(data)+i*2,2);result=std::max(result,value);}return result;
}
std::size_t rank_positive(const void* source,std::uint8_t* output,std::size_t count,PixelType type) {
    std::size_t i=0;
    if(type==PixelType::uint8 || type==PixelType::boolean) {
        const auto* data=static_cast<const std::uint8_t*>(source);for(;i+32<=count;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),_mm256_xor_si256(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i)),_mm256_setzero_si256()),_mm256_set1_epi8(-1)));
    }
    return i;
}
}
