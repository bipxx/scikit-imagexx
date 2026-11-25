#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd {
// Separate optional TU. Caller has authenticated spans and runtime AVX2.
// Boolean storage is nonzero truth, never raw-byte bitwise conjunction.
std::uint64_t coloc_bool_count(const unsigned char* a,const unsigned char* b,std::size_t n) {
    const auto zero=_mm256_setzero_si256();std::uint64_t count=0;std::size_t i=0;
    for(;n-i>=32;i+=32) {
        auto empty=_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i)),zero);
        if(b)empty=_mm256_or_si256(empty,_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i)),zero));
        auto bits=~std::uint32_t(_mm256_movemask_epi8(empty));
        // Avoid assuming a separate POPCNT CPU feature or compiler ISA flag.
        bits=bits-((bits>>1)&0x55555555u);bits=(bits&0x33333333u)+((bits>>2)&0x33333333u);
        count+=(((bits+(bits>>4))&0x0f0f0f0fu)*0x01010101u)>>24;
    }
    for(;i<n;++i)count+=(a[i]!=0)&&(!b || b[i]!=0);
    return count;
}

// A NumPy pairwise leaf starts eight independent chains, then folds them
// in a fixed tree before adding the tail. The caller owns the recursive
// splits and the floating environment. This TU is built with contraction off.
template<class T> using MandersVec = std::conditional_t<sizeof(T) == 8, __m256d, __m256>;
template<class T> inline MandersVec<T> manders_load(const unsigned char* p) {
    if constexpr (sizeof(T) == 8) return _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    else return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
}
template<class T> inline MandersVec<T> manders_add(MandersVec<T> a, MandersVec<T> b) {
    if constexpr (sizeof(T) == 8) return _mm256_add_pd(a, b);
    else return _mm256_add_ps(a, b);
}
template<class T> inline MandersVec<T> manders_mul(MandersVec<T> a, MandersVec<T> b) {
    if constexpr (sizeof(T) == 8) return _mm256_mul_pd(a, b);
    else return _mm256_mul_ps(a, b);
}
template<class T> inline void manders_store(T* dest, MandersVec<T> value) {
    if constexpr (sizeof(T) == 8) _mm256_storeu_pd(dest, value);
    else _mm256_storeu_ps(dest, value);
}
template<class T> inline void manders_classify(MandersVec<T> value, __m256i& nonfinite, __m256i& negative) {
    if constexpr (sizeof(T) == 8) {
        const auto raw = _mm256_castpd_si256(value);
        const auto sign = _mm256_set1_epi64x(INT64_MIN);
        const auto exp = _mm256_set1_epi64x(0x7ff0000000000000LL);
        nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi64(_mm256_and_si256(raw, exp), exp));
        negative = _mm256_or_si256(negative, _mm256_andnot_si256(_mm256_cmpeq_epi64(raw, sign), raw));
    } else {
        const auto raw = _mm256_castps_si256(value);
        const auto sign = _mm256_set1_epi32(INT32_MIN);
        const auto exp = _mm256_set1_epi32(0x7f800000);
        nonfinite = _mm256_or_si256(nonfinite, _mm256_cmpeq_epi32(_mm256_and_si256(raw, exp), exp));
        negative = _mm256_or_si256(negative, _mm256_andnot_si256(_mm256_cmpeq_epi32(raw, sign), raw));
    }
}

template<class T,bool Overlap> void manders_leaf(const unsigned char* first,
    const unsigned char* second,std::size_t n,T* sums,unsigned& status) {
    constexpr std::size_t lanes=sizeof(T)==8 ? 4:8;
    constexpr std::size_t vectors=8/lanes;
    using Vec=MandersVec<T>;
    const auto zero=_mm256_setzero_si256();
    auto nonfinite=zero,negative0=zero,negative1=zero;
    Vec acc[Overlap ? 3:2][vectors];
    for(std::size_t i=0;i<n;i+=8) {
        for(std::size_t j=0;j<vectors;++j) {
            const auto k=i+j*lanes;
            const auto x=manders_load<T>(first+k*sizeof(T));
            manders_classify<T>(x,nonfinite,negative0);
            Vec y;
            if constexpr(Overlap) {
                y=manders_load<T>(second+k*sizeof(T));
                manders_classify<T>(y,nonfinite,negative1);
            } else {
                __m128i bytes;
                if constexpr(sizeof(T)==8) {
                    int word;std::memcpy(&word,second+k,4);
                    bytes=_mm_cvtsi32_si128(word);
                    const auto truth=_mm_cmpgt_epi32(_mm_cvtepu8_epi32(bytes),_mm_setzero_si128());
                    y=_mm256_cvtepi32_pd(_mm_and_si128(truth,_mm_set1_epi32(1)));
                } else {
                    bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(second+k));
                    const auto truth=_mm256_cmpgt_epi32(_mm256_cvtepu8_epi32(bytes),zero);
                    y=_mm256_cvtepi32_ps(_mm256_and_si256(truth,_mm256_set1_epi32(1)));
                }
            }
            const auto value0=Overlap ? manders_mul<T>(x,x):x;
            const auto value1=Overlap ? manders_mul<T>(y,y):manders_mul<T>(x,y);
            acc[0][j]=i ? manders_add<T>(acc[0][j],value0):value0;
            acc[1][j]=i ? manders_add<T>(acc[1][j],value1):value1;
            if constexpr(Overlap) {
                const auto dot=manders_mul<T>(x,y);
                acc[2][j]=i ? manders_add<T>(acc[2][j],dot):dot;
            }
        }
    }
    status|=unsigned(!_mm256_testz_si256(nonfinite,nonfinite));
    status|=unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(negative0)) &
                     (sizeof(T)==8 ? 0xaa:0xff)) ? 2u:0u;
    status|=unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(negative1)) &
                     (sizeof(T)==8 ? 0xaa:0xff)) ? 4u:0u;
    for(std::size_t j=0;j<(Overlap ? 3:2);++j) {
        T values[8];
        for(std::size_t k=0;k<vectors;++k)manders_store<T>(values+k*lanes,acc[j][k]);
        sums[j]=((values[0]+values[1])+(values[2]+values[3]))+
                ((values[4]+values[5])+(values[6]+values[7]));
    }
}

void coloc_manders_leaf(const unsigned char* a,const unsigned char* b,std::size_t n,
                       bool overlap,double* sums,unsigned& status) {
    if(overlap)manders_leaf<double,true>(a,b,n,sums,status);
    else manders_leaf<double,false>(a,b,n,sums,status);
}
void coloc_manders_leaf(const unsigned char* a,const unsigned char* b,std::size_t n,
                       bool overlap,float* sums,unsigned& status) {
    if(overlap)manders_leaf<float,true>(a,b,n,sums,status);
    else manders_leaf<float,false>(a,b,n,sums,status);
}
}
