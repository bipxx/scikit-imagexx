#include "skimagexx/component_consumers.hpp"
#include <immintrin.h>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd {
namespace {
template<class T> __m256i fill_word(T value) {
    if constexpr(sizeof(T)==1)return _mm256_set1_epi8(char(value));
    else if constexpr(sizeof(T)==2)return _mm256_set1_epi16(short(value));
    else if constexpr(sizeof(T)==4)return _mm256_set1_epi32(int(value));
    else return _mm256_set1_epi64x(static_cast<long long>(value));
}
template<class T> __m256i expand_mask(const std::uint8_t* mask) {
    if constexpr(sizeof(T)==1)return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask));
    else if constexpr(sizeof(T)==2)return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask)));
    else if constexpr(sizeof(T)==4)return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask)));
    else {std::uint32_t word;std::memcpy(&word,mask,4);return _mm256_cvtepi8_epi64(_mm_cvtsi32_si128(int(word)));}
}
template<class T,class Select> std::size_t selected_store(const std::intptr_t* labels,std::size_t count,const void* fill,void* output,Select select) {
    T word;std::memcpy(&word,fill,sizeof(T));const auto value=fill_word(word);
    constexpr auto lanes=32/sizeof(T);std::size_t i=0;
    for(;i+lanes<=count;i+=lanes) {
        alignas(32) std::uint8_t mask[32];
        for(std::size_t j=0;j<lanes;++j)mask[j]=select(labels[i+j]) ? 255:0;
        auto* destination=reinterpret_cast<__m256i*>(static_cast<char*>(output)+i*sizeof(T));
        const auto old=_mm256_loadu_si256(destination);
        _mm256_storeu_si256(destination,_mm256_blendv_epi8(old,value,expand_mask<T>(mask)));
    }return i;
}
template<class F> std::size_t select_width(std::size_t bytes,F call) {
    switch(bytes){case 1:return call(std::uint8_t{});case 2:return call(std::uint16_t{});case 4:return call(std::uint32_t{});case 8:return call(std::uint64_t{});default:return 0;}
}
}
std::size_t consumer_filter_store(const std::intptr_t* labels,std::size_t count,const std::intptr_t* sizes,std::size_t bins,double threshold,void* output,std::size_t bytes) {
    return select_width(bytes,[&](auto zero){using T=decltype(zero);return selected_store<T>(labels,count,&zero,output,[&](std::intptr_t label){const auto key=std::uint64_t(label);return double(key<bins ? sizes[std::size_t(key)]:std::intptr_t{0})<=threshold;});});
}
std::size_t consumer_clear_store(const std::intptr_t* labels,std::size_t count,const std::uint8_t* remove,std::size_t number,const void* fill,void* output,std::size_t bytes,std::uint8_t& invalid) {
    return select_width(bytes,[&](auto word){using T=decltype(word);return selected_store<T>(labels,count,fill,output,[&](std::intptr_t label){if(label<0 || std::uint64_t(label)>number){invalid=1;return false;}return remove[std::size_t(label)]!=0;});});
}
std::size_t consumer_truth(const std::intptr_t* input,std::uint8_t* output,std::size_t count,bool invert) {
    if constexpr(sizeof(std::intptr_t)!=8)return 0;
    const auto zero=_mm256_setzero_si256();std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto equal=_mm256_cmpeq_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),zero);
        const auto bits=unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(equal)));
        for(unsigned lane=0;lane<4;++lane)output[i+lane]=std::uint8_t(bool(bits&(1u<<lane))==invert);
    }return i;
}
std::size_t consumer_bool_write(const std::uint8_t* input,std::uint8_t* output,std::size_t count,bool invert) {
    const auto toggle=_mm256_set1_epi8(invert ? 1:0);std::size_t i=0;
    for(;i+32<=count;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),
        _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),toggle));
    return i;
}
std::size_t consumer_or(const std::uint8_t* input,std::uint8_t* output,std::size_t count) {
    std::size_t i=0;
    for(;i+32<=count;i+=32)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),
        _mm256_or_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i)),
                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(output+i))));
    return i;
}
}
