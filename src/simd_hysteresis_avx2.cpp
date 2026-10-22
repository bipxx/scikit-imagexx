#include "simd.hpp"
#include <immintrin.h>
#include <type_traits>
#include <cstring>
namespace skimagexx::simd {
namespace {
template<class T> std::size_t integers(const void* a,const void* b,void* output,std::size_t count,std::ptrdiff_t ap,std::ptrdiff_t bp,bool minimum,bool logical) {
    const auto* x=static_cast<const T*>(a);const auto* y=static_cast<const T*>(b);constexpr auto lanes=32/sizeof(T);
    auto scalar=[](T value){
        if constexpr(sizeof(T)==1)return _mm256_set1_epi8(char(value));
        else if constexpr(sizeof(T)==2)return _mm256_set1_epi16(short(value));
        else if constexpr(sizeof(T)==4)return _mm256_set1_epi32(int(value));
        else return _mm256_set1_epi64x(static_cast<long long>(value));
    };
    using U=std::make_unsigned_t<T>;const auto sign=U(U(1)<<(8*sizeof(T)-1));T sign_value;std::memcpy(&sign_value,&sign,sizeof(T));
    const auto bias=scalar(sign_value),one=_mm256_set1_epi8(1),zero=_mm256_setzero_si256();
    std::size_t i=0;
    for(;i+lanes<=count;i+=lanes){auto av=ap ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x+i)) : scalar(x[0]);auto bv=bp ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y+i)) : scalar(y[0]);
        if(logical){av=_mm256_andnot_si256(_mm256_cmpeq_epi8(av,zero),one);bv=_mm256_andnot_si256(_mm256_cmpeq_epi8(bv,zero),one);}
        auto ca=av,cb=bv;if constexpr(!std::is_signed_v<T>){ca=_mm256_xor_si256(ca,bias);cb=_mm256_xor_si256(cb,bias);}
        __m256i greater;
        if constexpr(sizeof(T)==1)greater=_mm256_cmpgt_epi8(ca,cb);
        else if constexpr(sizeof(T)==2)greater=_mm256_cmpgt_epi16(ca,cb);
        else if constexpr(sizeof(T)==4)greater=_mm256_cmpgt_epi32(ca,cb);
        else greater=_mm256_cmpgt_epi64(ca,cb);
        if(minimum)_mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<T*>(output)+i),_mm256_blendv_epi8(av,bv,greater));
        else {const auto bits=std::uint32_t(_mm256_movemask_epi8(greater));for(std::size_t k=0;k<lanes;++k)static_cast<std::uint8_t*>(output)[i+k]=(bits>>(k*sizeof(T)))&1;}
    }return i;
}
}
std::size_t hysteresis_map(const void* a,const void* b,void* output,std::size_t count,PixelType type,
    std::ptrdiff_t ap,std::ptrdiff_t bp,bool minimum) {
    std::size_t i=0;
    if(type==PixelType::float32) {
        const auto* x=static_cast<const float*>(a);const auto* y=static_cast<const float*>(b);
        for(;i+8<=count;i+=8){const auto av=ap ? _mm256_loadu_ps(x+i) : _mm256_set1_ps(x[0]),bv=bp ? _mm256_loadu_ps(y+i) : _mm256_set1_ps(y[0]);
            if(minimum)_mm256_storeu_ps(static_cast<float*>(output)+i,_mm256_blendv_ps(_mm256_min_ps(av,bv),av,_mm256_cmp_ps(av,av,_CMP_UNORD_Q)));
            else {const auto bits=_mm256_movemask_ps(_mm256_cmp_ps(av,bv,_CMP_GT_OQ));for(int k=0;k<8;++k)static_cast<std::uint8_t*>(output)[i+k]=(bits>>k)&1;}
        }
    } else if(type==PixelType::float64) {
        const auto* x=static_cast<const double*>(a);const auto* y=static_cast<const double*>(b);
        for(;i+4<=count;i+=4){const auto av=ap ? _mm256_loadu_pd(x+i) : _mm256_set1_pd(x[0]),bv=bp ? _mm256_loadu_pd(y+i) : _mm256_set1_pd(y[0]);
            if(minimum)_mm256_storeu_pd(static_cast<double*>(output)+i,_mm256_blendv_pd(_mm256_min_pd(av,bv),av,_mm256_cmp_pd(av,av,_CMP_UNORD_Q)));
            else {const auto bits=_mm256_movemask_pd(_mm256_cmp_pd(av,bv,_CMP_GT_OQ));for(int k=0;k<4;++k)static_cast<std::uint8_t*>(output)[i+k]=(bits>>k)&1;}
        }
    } else {
        switch(type) {
#define INTEGER(tag,T) case PixelType::tag:return integers<T>(a,b,output,count,ap,bp,minimum,type==PixelType::boolean);
        INTEGER(boolean,std::uint8_t) INTEGER(uint8,std::uint8_t) INTEGER(int8,std::int8_t) INTEGER(uint16,std::uint16_t) INTEGER(int16,std::int16_t)
        INTEGER(uint32,std::uint32_t) INTEGER(int32,std::int32_t) INTEGER(uint64,std::uint64_t) INTEGER(int64,std::int64_t)
#undef INTEGER
        default:break;
        }
    }
    return i;
}
}
