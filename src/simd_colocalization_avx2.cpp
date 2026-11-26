#include <immintrin.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace {
void store_counts(__m256i total,__m256i joint,std::array<std::uint64_t,3>& counts) {
    std::uint64_t lanes[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes),total);
    for(const auto lane:lanes)counts[0]+=lane;
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes),joint);
    for(const auto lane:lanes)counts[1]+=lane;
}
}

void coloc_intersection_reverse_avx2(const unsigned char* first,const unsigned char* second,
    std::size_t rows,std::size_t width,std::array<std::uint64_t,3>& counts) {
    const auto zero=_mm256_setzero_si256(),one=_mm256_set1_epi8(1);
    auto total=zero,joint=zero;
    for(std::size_t row=0;row<rows;++row) {
        const auto* a=first+row*width;
        const auto* b=second-row*width;
        for(std::size_t i=0;i<width;i+=32) {
            const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
            const auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
            const auto selected=_mm256_andnot_si256(_mm256_cmpeq_epi8(x,zero),one);
            total=_mm256_add_epi64(total,_mm256_sad_epu8(selected,zero));
            joint=_mm256_add_epi64(joint,_mm256_sad_epu8(
                _mm256_andnot_si256(_mm256_cmpeq_epi8(y,zero),selected),zero));
        }
    }
    store_counts(total,joint,counts);
}

std::size_t coloc_intersection_avx2(const unsigned char* first,const unsigned char* second,
    const unsigned char* mask,std::size_t n,unsigned integer_inputs,std::array<std::uint64_t,3>& counts) {
    const auto zero=_mm256_setzero_si256(),one=_mm256_set1_epi8(1),invalid_bits=_mm256_set1_epi8(-2);
    auto total=zero,joint=zero,invalid=zero;
    std::size_t i=0;
    for(;n-i>=32;i+=32) {
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(first+i));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(second+i));
        const auto selection=mask ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask+i)):one;
        if(integer_inputs&1)invalid=_mm256_or_si256(invalid,a);
        if(integer_inputs&2)invalid=_mm256_or_si256(invalid,b);
        if(integer_inputs&4)invalid=_mm256_or_si256(invalid,selection);
        const auto excluded=_mm256_or_si256(_mm256_cmpeq_epi8(a,zero),_mm256_cmpeq_epi8(selection,zero));
        const auto selected=_mm256_andnot_si256(excluded,one);
        total=_mm256_add_epi64(total,_mm256_sad_epu8(selected,zero));
        joint=_mm256_add_epi64(joint,_mm256_sad_epu8(_mm256_andnot_si256(_mm256_cmpeq_epi8(b,zero),selected),zero));
    }
    store_counts(total,joint,counts);
    counts[2]=!_mm256_testz_si256(invalid,invalid_bits);
    return i;
}

namespace {
template<class U> __m256i equal_words(__m256i a,__m256i b) {
    if constexpr(sizeof(U)==8)return _mm256_cmpeq_epi64(a,b);
    else if constexpr(sizeof(U)==4)return _mm256_cmpeq_epi32(a,b);
    else return _mm256_cmpeq_epi16(a,b);
}
template<class U> __m256i broadcast(U value) {
    if constexpr(sizeof(U)==8)return _mm256_set1_epi64x(static_cast<std::int64_t>(value));
    else if constexpr(sizeof(U)==4)return _mm256_set1_epi32(static_cast<std::int32_t>(value));
    else return _mm256_set1_epi16(static_cast<std::int16_t>(value));
}
template<class U> std::size_t intersection_wide(const unsigned char* a,const unsigned char* b,
    std::size_t n,U one_bits,U value_bits,std::array<std::uint64_t,3>& counts) {
    constexpr std::size_t width=32/sizeof(U);
    const auto zero=_mm256_setzero_si256(),one=broadcast(one_bits),mask=broadcast(value_bits);
    const auto unit=broadcast(U(1));
    auto total=zero,joint=zero,invalid=zero;
    std::size_t i=0;
    for(;n-i>=width;i+=width) {
        const auto x=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i*sizeof(U))),mask);
        const auto y=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i*sizeof(U))),mask);
        const auto hit0=equal_words<U>(x,one),hit1=equal_words<U>(y,one);
        invalid=_mm256_or_si256(invalid,_mm256_or_si256(
            _mm256_andnot_si256(hit0,x),_mm256_andnot_si256(hit1,y)));
        const auto selected=_mm256_and_si256(hit0,unit);
        total=_mm256_add_epi64(total,_mm256_sad_epu8(selected,zero));
        joint=_mm256_add_epi64(joint,_mm256_sad_epu8(_mm256_and_si256(selected,hit1),zero));
    }
    store_counts(total,joint,counts);
    counts[2]=!_mm256_testz_si256(invalid,invalid);
    return i;
}
}

std::size_t coloc_intersection_wide_avx2(const unsigned char* a,const unsigned char* b,
    std::size_t n,std::size_t item,std::uint64_t one,std::uint64_t mask,
    std::array<std::uint64_t,3>& counts) {
    if(item==8)return intersection_wide(a,b,n,one,mask,counts);
    if(item==4)return intersection_wide(a,b,n,std::uint32_t(one),std::uint32_t(mask),counts);
    return intersection_wide(a,b,n,std::uint16_t(one),std::uint16_t(mask),counts);
}

// A NumPy pairwise leaf starts eight independent chains, then folds them
// in a fixed tree before adding the tail. The caller owns the recursive
// splits and the floating environment. This TU is built with contraction off.
template<class T> struct MandersVector;
template<> struct MandersVector<double> {using Type=__m256d;};
template<> struct MandersVector<float> {using Type=__m256;};
template<class T> using MandersVec=typename MandersVector<T>::Type;
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
    Vec acc[Overlap ? 3:2][vectors]{};
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
