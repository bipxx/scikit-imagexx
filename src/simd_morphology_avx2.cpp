#include "simd.hpp"
#include "morphology_detail.hpp"
#include <immintrin.h>

namespace skimagexx::simd {
bool boolean_pack(const std::uint8_t* input,std::uint64_t* output,std::size_t n,std::size_t offset) {
    std::size_t i=0;
    const auto zero=_mm256_setzero_si256(),ones=_mm256_set1_epi8(1);
    for (;i+32<=n;i+=32) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto is_zero=_mm256_cmpeq_epi8(value,zero),is_one=_mm256_cmpeq_epi8(value,ones);
        if (_mm256_movemask_epi8(_mm256_or_si256(is_zero,is_one))!=-1) return false;
        const auto bits=std::uint64_t(~std::uint32_t(_mm256_movemask_epi8(is_zero)));
        const auto position=offset+i,word=position/64,shift=position%64;
        output[word]|=bits<<shift;
        if (shift>32) output[word+1]|=bits>>(64-shift);
    }
    for (;i<n;++i) {
        if (input[i]>1) return false;
        if (input[i]) output[(offset+i)/64]|=std::uint64_t(1)<<((offset+i)%64);
    }
    return true;
}
std::size_t boolean_unpack(const std::uint64_t* input,std::uint8_t* output,std::size_t n) {
    const auto indices=_mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,
                                      2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const auto weights=_mm256_setr_epi8(1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128,
                                      1,2,4,8,16,32,64,-128,1,2,4,8,16,32,64,-128);
    const auto zero=_mm256_setzero_si256(),ones=_mm256_set1_epi8(1);
    std::size_t i=0;
    for (;i+32<=n;i+=32) {
        const auto bits=std::uint32_t(input[i/64]>>(i%64));
        const auto bytes=_mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)),indices);
        const auto value=_mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes,weights),zero),ones);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),value);
    }
    return i;
}
std::size_t boolean_combine(const std::uint64_t* a,const std::uint64_t* b,std::uint64_t* output,std::size_t n,bool erode) {
    std::size_t i=0;
    for (;i+4<=n;i+=4) {
        const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i)),y=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i),erode ? _mm256_and_si256(x,y) : _mm256_or_si256(x,y));
    }
    return i;
}
std::size_t boolean_shift_reduce(std::uint64_t* data,std::size_t n,std::size_t shift,bool erode) {
    const auto words=shift/64,bits=shift%64;
    const auto right=_mm_cvtsi32_si128(static_cast<int>(bits)),left=_mm_cvtsi32_si128(static_cast<int>(64-bits));
    std::size_t i=0;
    for (;i+words+4+(bits ? 1 : 0)<=n;i+=4) {
        const auto original=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));
        auto shifted=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i+words));
        if (bits) shifted=_mm256_or_si256(_mm256_srl_epi64(shifted,right),
            _mm256_sll_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i+words+1)),left));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data+i),erode ? _mm256_and_si256(original,shifted) : _mm256_or_si256(original,shifted));
    }
    return i;
}
namespace {
template<class T> __m256i integer_extrema(__m256i a,__m256i b,bool erode) {
    if constexpr (sizeof(T)==1 && std::is_unsigned_v<T>) return erode ? _mm256_min_epu8(a,b) : _mm256_max_epu8(a,b);
    else if constexpr (sizeof(T)==1) return erode ? _mm256_min_epi8(a,b) : _mm256_max_epi8(a,b);
    else if constexpr (sizeof(T)==2 && std::is_unsigned_v<T>) return erode ? _mm256_min_epu16(a,b) : _mm256_max_epu16(a,b);
    else if constexpr (sizeof(T)==2) return erode ? _mm256_min_epi16(a,b) : _mm256_max_epi16(a,b);
    else if constexpr (sizeof(T)==4 && std::is_unsigned_v<T>) return erode ? _mm256_min_epu32(a,b) : _mm256_max_epu32(a,b);
    else if constexpr (sizeof(T)==4) return erode ? _mm256_min_epi32(a,b) : _mm256_max_epi32(a,b);
    else {
        auto x=a,y=b;
        if constexpr (std::is_unsigned_v<T>) { const auto sign=_mm256_set1_epi64x(INT64_MIN); x=_mm256_xor_si256(x,sign); y=_mm256_xor_si256(y,sign); }
        const auto select=erode ? _mm256_cmpgt_epi64(x,y) : _mm256_cmpgt_epi64(y,x);
        return _mm256_blendv_epi8(a,b,select);
    }
}
template<class T> auto load(const T* p) {
    if constexpr (std::is_same_v<T,float>) return _mm256_loadu_ps(p);
    else if constexpr (std::is_same_v<T,double>) return _mm256_loadu_pd(p);
    else return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
template<class T,class Vector> void store(T* p,Vector value) {
    if constexpr (std::is_same_v<T,float>) _mm256_storeu_ps(p,value);
    else if constexpr (std::is_same_v<T,double>) _mm256_storeu_pd(p,value);
    else _mm256_storeu_si256(reinterpret_cast<__m256i*>(p),value);
}
template<class T,class Vector> Vector vector_extrema(Vector a,Vector b,bool erode) {
    // MIN/MAX return their second operand on a tie. Keep the accumulator's
    // sign when comparing equal zeros, as the scalar reduction does.
    if constexpr (std::is_same_v<T,float>) return erode ? _mm256_min_ps(b,a) : _mm256_max_ps(b,a);
    else if constexpr (std::is_same_v<T,double>) return erode ? _mm256_min_pd(b,a) : _mm256_max_pd(b,a);
    else return integer_extrema<T>(a,b,erode);
}
template<class T,bool Erode> void column_block(const void* input,void* output,std::size_t length,
    std::size_t step,const std::size_t* rows,std::size_t window,void* scratch,double fill) {
    constexpr std::size_t lanes=32/sizeof(T);
    const auto* source=static_cast<const T*>(input);
    auto* out=static_cast<T*>(output);
    auto* suffix=static_cast<T*>(scratch);
    alignas(32) T constants[lanes];
    std::fill_n(constants,lanes,detail::morphology_cast<T>(fill));
    const auto constant=load(constants);
    const auto sample=[&](std::size_t j) {
        const auto row=rows[j];
        return row==std::numeric_limits<std::size_t>::max() ? constant : load(source+row);
    };
    const auto extended=length+window-1;
    for (std::size_t block=0;block<extended;block+=window) {
        const auto stop=std::min(extended,block+window);
        auto backward=sample(stop-1);
        store(suffix+(stop-1)*lanes,backward);
        for (auto j=stop-1;j>block;--j) {
            backward=vector_extrema<T>(backward,sample(j-1),Erode);
            store(suffix+(j-1)*lanes,backward);
        }

        // The suffix for each output is in this block or the previous one.
        // Combine it as the prefix scan advances, without a prefix buffer.
        auto prefix=sample(block);
        auto j=block;
        for (;j<window-1;++j) prefix=vector_extrema<T>(prefix,sample(j+1),Erode);
        for (;;) {
            const auto row=j-(window-1);
            store(out+row*step,vector_extrema<T>(load(suffix+row*lanes),prefix,Erode));
            if (++j==stop) break;
            prefix=vector_extrema<T>(prefix,sample(j),Erode);
        }
    }
}
template<class T> std::size_t line(const void* const* pointers,std::size_t count,void* output,std::size_t n,bool erode,double fill,bool binary) {
    constexpr std::size_t lanes=32/sizeof(T);
    auto* out=static_cast<T*>(output); const T constant=detail::morphology_cast<T>(fill);
    std::size_t i=0;
    for (;i+lanes<=n;i+=lanes) {
        if constexpr (std::is_floating_point_v<T>) {
            const auto fill_vector=[&] {
                if constexpr (std::is_same_v<T,float>) return _mm256_set1_ps(constant);
                else return _mm256_set1_pd(constant);
            }();
            auto value=pointers[0] ? load(static_cast<const T*>(pointers[0])+i) : fill_vector;
            for (std::size_t k=1;k<count;++k) {
                const auto b=pointers[k] ? load(static_cast<const T*>(pointers[k])+i) : fill_vector;
                value=vector_extrema<T>(value,b,erode);
            }
            store(out+i,value);
        } else {
            alignas(32) T repeated[lanes]; std::fill(repeated,repeated+lanes,constant);
            const auto c=_mm256_load_si256(reinterpret_cast<const __m256i*>(repeated));
            auto value=pointers[0] ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const T*>(pointers[0])+i)) : c;
            for (std::size_t k=1;k<count;++k) {
                if constexpr (sizeof(T)==1) {
                    if (binary) {
                        const auto terminal=_mm256_set1_epi8(erode ? 0 : 1);
                        if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(value,terminal))==-1) break;
                    }
                }
                const auto b=pointers[k] ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const T*>(pointers[k])+i)) : c;
                value=integer_extrema<T>(value,b,erode);
            }
            if constexpr (sizeof(T)==8) {
                alignas(32) T values[lanes]; _mm256_store_si256(reinterpret_cast<__m256i*>(values),value);
                for (std::size_t j=0;j<lanes;++j) out[i+j]=detail::morphology_round(values[j]);
            } else _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),value);
        }
    }
    return i;
}
}
void morphology_column_block(const void* input,void* output,std::size_t length,std::size_t step,
    const std::size_t* rows,std::size_t window,void* suffix,PixelType type,bool erode,double fill) {
    const auto run=[&](auto value) {
        using T=decltype(value);
        if (erode) column_block<T,true>(input,output,length,step,rows,window,suffix,fill);
        else column_block<T,false>(input,output,length,step,rows,window,suffix,fill);
    };
#define MORPH(KIND,TYPE) case PixelType::KIND: run(TYPE{}); break
    switch (type) {
        MORPH(boolean,std::uint8_t); MORPH(uint8,std::uint8_t); MORPH(int8,std::int8_t);
        MORPH(uint16,std::uint16_t); MORPH(int16,std::int16_t); MORPH(uint32,std::uint32_t);
        MORPH(int32,std::int32_t); MORPH(float32,float); MORPH(float64,double);
        case PixelType::uint64: case PixelType::int64: break; // Wide-integer scans use a double buffer.
    }
#undef MORPH
}
std::size_t morphology_difference(const void* left,const void* right,void* output,std::size_t n,PixelType type,bool& invalid,bool& overflow) {
    if (type==PixelType::float32 || type==PixelType::float64) {
        const bool single=type==PixelType::float32;
        const std::size_t lanes=single ? 8 : 4;
        std::size_t i=0;
        for (;i+lanes<=n;i+=lanes) {
            if (single) {
                const auto a=_mm256_loadu_ps(static_cast<const float*>(left)+i),b=_mm256_loadu_ps(static_cast<const float*>(right)+i);
                const auto v=_mm256_sub_ps(a,b),infinity=_mm256_set1_ps(INFINITY),sign=_mm256_set1_ps(-0.0f);
                _mm256_storeu_ps(static_cast<float*>(output)+i,v);
                invalid=invalid || _mm256_movemask_ps(_mm256_cmp_ps(v,v,_CMP_UNORD_Q))!=0;
                const auto finite=_mm256_and_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign,a),infinity,_CMP_LT_OQ),_mm256_cmp_ps(_mm256_andnot_ps(sign,b),infinity,_CMP_LT_OQ));
                overflow=overflow || _mm256_movemask_ps(_mm256_and_ps(finite,_mm256_cmp_ps(_mm256_andnot_ps(sign,v),infinity,_CMP_EQ_OQ)))!=0;
            } else {
                const auto a=_mm256_loadu_pd(static_cast<const double*>(left)+i),b=_mm256_loadu_pd(static_cast<const double*>(right)+i);
                const auto v=_mm256_sub_pd(a,b),infinity=_mm256_set1_pd(INFINITY),sign=_mm256_set1_pd(-0.0);
                _mm256_storeu_pd(static_cast<double*>(output)+i,v);
                invalid=invalid || _mm256_movemask_pd(_mm256_cmp_pd(v,v,_CMP_UNORD_Q))!=0;
                const auto finite=_mm256_and_pd(_mm256_cmp_pd(_mm256_andnot_pd(sign,a),infinity,_CMP_LT_OQ),_mm256_cmp_pd(_mm256_andnot_pd(sign,b),infinity,_CMP_LT_OQ));
                overflow=overflow || _mm256_movemask_pd(_mm256_and_pd(finite,_mm256_cmp_pd(_mm256_andnot_pd(sign,v),infinity,_CMP_EQ_OQ)))!=0;
            }
        }
        return i;
    }
    const std::size_t bytes=type==PixelType::uint64 || type==PixelType::int64 ? 8 :
        type==PixelType::uint32 || type==PixelType::int32 ? 4 : type==PixelType::uint16 || type==PixelType::int16 ? 2 : 1;
    std::size_t i=0;
    for (;i+32/bytes<=n;i+=32/bytes) {
        const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(left)+i*bytes));
        const auto b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const char*>(right)+i*bytes));
        __m256i v;
        if (type==PixelType::boolean) {
            const auto zero=_mm256_setzero_si256();
            v=_mm256_and_si256(_mm256_xor_si256(_mm256_cmpeq_epi8(a,zero),_mm256_cmpeq_epi8(b,zero)),_mm256_set1_epi8(1));
        } else v=bytes==8 ? _mm256_sub_epi64(a,b) : bytes==4 ? _mm256_sub_epi32(a,b) : bytes==2 ? _mm256_sub_epi16(a,b) : _mm256_sub_epi8(a,b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<char*>(output)+i*bytes),v);
    }
    return i;
}
bool morphology_has_nan(const void* input,std::size_t n,PixelType type) {
    std::size_t i=0;
    if (type==PixelType::float32) {
        const auto* p=static_cast<const float*>(input);
        for (;i+8<=n;i+=8) { const auto v=_mm256_loadu_ps(p+i); if (_mm256_movemask_ps(_mm256_cmp_ps(v,v,_CMP_UNORD_Q))) return true; }
        for (;i<n;++i) if (std::isnan(p[i])) return true;
    } else {
        const auto* p=static_cast<const double*>(input);
        for (;i+4<=n;i+=4) { const auto v=_mm256_loadu_pd(p+i); if (_mm256_movemask_pd(_mm256_cmp_pd(v,v,_CMP_UNORD_Q))) return true; }
        for (;i<n;++i) if (std::isnan(p[i])) return true;
    }
    return false;
}
std::size_t morphology_line(const void* const* p,std::size_t count,void* out,std::size_t n,PixelType type,bool erode,double fill,bool binary) {
#define MORPH(KIND,TYPE) case PixelType::KIND: return line<TYPE>(p,count,out,n,erode,fill,binary)
    switch(type) {
        MORPH(boolean,std::uint8_t); MORPH(uint8,std::uint8_t); MORPH(int8,std::int8_t);
        MORPH(uint16,std::uint16_t); MORPH(int16,std::int16_t); MORPH(uint32,std::uint32_t);
        MORPH(int32,std::int32_t); MORPH(uint64,std::uint64_t); MORPH(int64,std::int64_t);
        MORPH(float32,float); MORPH(float64,double);
    }
#undef MORPH
    return 0;
}
}
