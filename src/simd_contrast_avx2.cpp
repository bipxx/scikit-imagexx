#include "simd.hpp"
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>

namespace skimagexx::simd {
namespace {
template<std::size_t Lanes> struct alignas(32) Permutations {
    std::array<std::array<int,8>,std::size_t{1}<<Lanes> indices{};
    std::array<unsigned char,std::size_t{1}<<Lanes> counts{};
};
template<std::size_t Lanes> constexpr Permutations<Lanes> permutations() {
    Permutations<Lanes> output;
    for(std::size_t mask=0;mask<output.indices.size();++mask) {
        std::size_t position=0;
        for(int selected=1;selected>=0;--selected)for(std::size_t lane=0;lane<Lanes;++lane) {
            if(bool(mask&(std::size_t{1}<<lane))!=bool(selected))continue;
            for(std::size_t word=0;word<8/Lanes;++word)
                output.indices[mask][position++]=int(lane*(8/Lanes)+word);
            if(selected)++output.counts[mask];
        }
    }
    return output;
}

template<class T> struct Operations {
    static constexpr std::size_t lanes=32/sizeof(T);
    static __m256i broadcast(T value) {
        if constexpr(std::is_same_v<T,float>)return _mm256_castps_si256(_mm256_set1_ps(value));
        else if constexpr(std::is_same_v<T,double>)return _mm256_castpd_si256(_mm256_set1_pd(value));
        else if constexpr(sizeof(T)==4) {int bits;std::memcpy(&bits,&value,4);return _mm256_set1_epi32(bits);}
        else {long long bits;std::memcpy(&bits,&value,8);return _mm256_set1_epi64x(bits);}
    }
    template<bool Greater> static unsigned compare(__m256i value,__m256i pivot) {
        if constexpr(std::is_same_v<T,float>) {
            const auto comparison=_mm256_cmp_ps(_mm256_castsi256_ps(value),_mm256_castsi256_ps(pivot),Greater ? _CMP_GT_OQ : _CMP_LT_OQ);
            return unsigned(_mm256_movemask_ps(comparison));
        } else if constexpr(std::is_same_v<T,double>) {
            const auto comparison=_mm256_cmp_pd(_mm256_castsi256_pd(value),_mm256_castsi256_pd(pivot),Greater ? _CMP_GT_OQ : _CMP_LT_OQ);
            return unsigned(_mm256_movemask_pd(comparison));
        } else if constexpr(sizeof(T)==4) {
            if constexpr(std::is_unsigned_v<T>) {
                const auto sign=_mm256_set1_epi32(std::numeric_limits<int>::min());
                value=_mm256_xor_si256(value,sign);pivot=_mm256_xor_si256(pivot,sign);
            }
            const auto comparison=Greater ? _mm256_cmpgt_epi32(value,pivot) : _mm256_cmpgt_epi32(pivot,value);
            return unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(comparison)));
        } else {
            if constexpr(std::is_unsigned_v<T>) {
                const auto sign=_mm256_set1_epi64x(std::numeric_limits<long long>::min());
                value=_mm256_xor_si256(value,sign);pivot=_mm256_xor_si256(pivot,sign);
            }
            const auto comparison=Greater ? _mm256_cmpgt_epi64(value,pivot) : _mm256_cmpgt_epi64(pivot,value);
            return unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(comparison)));
        }
    }
};

template<class T> std::pair<std::size_t,std::size_t> partition_values(const T* input,T* output,
                                                                  std::size_t count,T pivot) {
    using Op=Operations<T>;
    static constexpr auto order=permutations<Op::lanes>();
    constexpr auto all=unsigned((std::size_t{1}<<Op::lanes)-1);
    const auto repeated=Op::broadcast(pivot);
    std::size_t i=0,left=0,right=count;
    // Whole-register stores put unused lanes into the free middle. Keeping
    // at least two registers unread guarantees those stores cannot overlap
    // either valid partition, including zero/equal masks and small tails.
    for(;i+2*Op::lanes<=count;i+=Op::lanes) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto less=Op::template compare<false>(value,repeated);
        const auto greater=Op::template compare<true>(value,repeated);
        const auto front_order=_mm256_load_si256(reinterpret_cast<const __m256i*>(order.indices[less].data()));
        const auto back_order=_mm256_load_si256(reinterpret_cast<const __m256i*>(order.indices[all^greater].data()));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+left),_mm256_permutevar8x32_epi32(value,front_order));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+right-Op::lanes),_mm256_permutevar8x32_epi32(value,back_order));
        left+=order.counts[less];right-=order.counts[greater];
    }
    for(;i<count;++i) {
        const auto value=input[i];
        if(value<pivot)output[left++]=value;
        else if(value>pivot)output[--right]=value;
    }
    return {left,right};
}

template<class T> T median(T a,T b,T c) {
    if(a<b) {if(b<c)return b;return a<c ? c : a;}
    if(a<c)return a;
    return b<c ? c : b;
}

template<class T> void select_values(T* source,T* scratch,T* result,std::size_t begin,std::size_t end,
    const std::size_t* ranks,std::size_t first,std::size_t last,unsigned depth) {
    if(first==last)return;
    if(end-begin<64 || !depth) {
        for(auto j=first;j<last;++j) {
            const auto rank=ranks[j];
            std::nth_element(source+begin,source+rank,source+end);
            result[rank]=source[rank];begin=rank+1;
        }
        return;
    }
    const auto pivot=median(source[begin],source[begin+(end-begin)/2],source[end-1]);
    const auto split=partition_values(source+begin,scratch+begin,end-begin,pivot);
    const auto low=begin+split.first,high=begin+split.second;
    const auto middle=std::size_t(std::lower_bound(ranks+first,ranks+last,low)-ranks);
    const auto upper=std::size_t(std::lower_bound(ranks+middle,ranks+last,high)-ranks);
    for(auto j=middle;j<upper;++j)result[ranks[j]]=pivot;
    // Equal ranks are finished. Alternating the two buffers avoids copying
    // partitions back; sibling recursion owns disjoint ranges in both buffers.
    select_values(scratch,source,result,begin,low,ranks,first,middle,depth-1);
    select_values(scratch,source,result,high,end,ranks,upper,last,depth-1);
}

template<class T> bool select(void* source,void* scratch,std::size_t count,const std::size_t* ranks,std::size_t length) {
    unsigned depth=0;for(auto n=count;n>1;n/=2)depth+=2;
    auto* values=static_cast<T*>(source);
    select_values(values,static_cast<T*>(scratch),values,0,count,ranks,0,length,depth);
    return true;
}
}

std::size_t contrast_widen16(const void* source,std::int32_t* output,std::size_t count,bool signed_input,bool swap) {
    const auto* input=static_cast<const unsigned char*>(source);
    const auto reverse=_mm256_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,
                                      1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    std::size_t i=0;
    for(;i+16<=count;i+=16) {
        auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+2*i));
        if(swap)value=_mm256_shuffle_epi8(value,reverse);
        const auto low=_mm256_castsi256_si128(value),high=_mm256_extracti128_si256(value,1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),signed_input ? _mm256_cvtepi16_epi32(low) : _mm256_cvtepu16_epi32(low));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i+8),signed_input ? _mm256_cvtepi16_epi32(high) : _mm256_cvtepu16_epi32(high));
    }
    return i;
}

bool contrast_select(void* source,void* scratch,std::size_t count,PixelType type,
                     const std::size_t* ranks,std::size_t length) {
    switch(type) {
    case PixelType::float32:return select<float>(source,scratch,count,ranks,length);
    case PixelType::float64:return select<double>(source,scratch,count,ranks,length);
    case PixelType::uint32:return select<std::uint32_t>(source,scratch,count,ranks,length);
    case PixelType::int32:return select<std::int32_t>(source,scratch,count,ranks,length);
    case PixelType::uint64:return select<std::uint64_t>(source,scratch,count,ranks,length);
    case PixelType::int64:return select<std::int64_t>(source,scratch,count,ranks,length);
    default:return false;
    }
}

bool contrast_finite(const void* source,std::size_t count,bool single) {
    std::size_t i=0;
    if(single) {
        const auto* input=static_cast<const float*>(source);
        const auto bound=_mm256_set1_ps(std::numeric_limits<float>::max());
        const auto sign=_mm256_set1_ps(-0.f);
        for(;i+8<=count;i+=8) {
            const auto value=_mm256_andnot_ps(sign,_mm256_loadu_ps(input+i));
            if(_mm256_movemask_ps(_mm256_cmp_ps(value,bound,_CMP_NLE_UQ)))return false;
        }
        for(;i<count;++i)if(!std::isfinite(input[i]))return false;
    } else {
        const auto* input=static_cast<const double*>(source);
        const auto bound=_mm256_set1_pd(std::numeric_limits<double>::max());
        const auto sign=_mm256_set1_pd(-0.);
        for(;i+4<=count;i+=4) {
            const auto value=_mm256_andnot_pd(sign,_mm256_loadu_pd(input+i));
            if(_mm256_movemask_pd(_mm256_cmp_pd(value,bound,_CMP_NLE_UQ)))return false;
        }
        for(;i<count;++i)if(!std::isfinite(input[i]))return false;
    }
    return true;
}

unsigned char contrast_truth(const unsigned char* input,std::size_t count) {
    const auto zero=_mm256_setzero_si256();unsigned char found=0;std::size_t i=0;
    for(;i+32<=count && found!=3;i+=32) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto bits=std::uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value,zero)));
        if(bits)found|=2;
        if(bits!=0xffffffffu)found|=1;
    }
    for(;i<count && found!=3;++i)found|=input[i] ? 1 : 2;
    return found;
}
}
