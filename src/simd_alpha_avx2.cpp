#include "simd.hpp"
#include "alpha_detail.hpp"
#include <immintrin.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <vector>
namespace skimagexx::simd {
namespace {
void pack(float* p, __m256 r, __m256 g, __m256 b) {
    const auto i0 = _mm256_setr_epi32(0,0,0,1,1,1,2,2), i1 = _mm256_setr_epi32(2,3,3,3,4,4,4,5), i2 = _mm256_setr_epi32(5,5,6,6,6,7,7,7);
    _mm256_storeu_ps(p,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(r,i0),_mm256_permutevar8x32_ps(g,i0),0x92),_mm256_permutevar8x32_ps(b,i0),0x24));
    _mm256_storeu_ps(p+8,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(b,i1),_mm256_permutevar8x32_ps(r,i1),0x92),_mm256_permutevar8x32_ps(g,i1),0x24));
    _mm256_storeu_ps(p+16,_mm256_blend_ps(_mm256_blend_ps(_mm256_permutevar8x32_ps(g,i2),_mm256_permutevar8x32_ps(b,i2),0x92),_mm256_permutevar8x32_ps(r,i2),0x24));
}
void pack(double* p, __m256d r, __m256d g, __m256d b) {
    _mm256_storeu_pd(p,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(r,_MM_SHUFFLE(1,0,0,0)),
        _mm256_permute4x64_pd(g,0),2),_mm256_permute4x64_pd(b,0),4));
    _mm256_storeu_pd(p+4,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(g,_MM_SHUFFLE(2,0,0,1)),
        _mm256_permute4x64_pd(b,_MM_SHUFFLE(0,0,1,0)),2),_mm256_permute4x64_pd(r,_MM_SHUFFLE(0,2,0,0)),4));
    _mm256_storeu_pd(p+8,_mm256_blend_pd(_mm256_blend_pd(_mm256_permute4x64_pd(b,_MM_SHUFFLE(3,0,0,2)),
        _mm256_permute4x64_pd(r,_MM_SHUFFLE(0,0,3,0)),2),_mm256_permute4x64_pd(g,_MM_SHUFFLE(0,3,0,0)),4));
}

template<class T> __m128i raw128(const unsigned char* p,std::ptrdiff_t step) {
    if(step==std::ptrdiff_t(sizeof(T)))return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    if(step==-std::ptrdiff_t(sizeof(T))) {
        const auto x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p-(16-sizeof(T))));
        if constexpr(sizeof(T)==1)return _mm_shuffle_epi8(x,_mm_setr_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
        else return _mm_shuffle_epi8(x,_mm_setr_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1));
    }
    if(step==2*std::ptrdiff_t(sizeof(T))) {
        // Overlap loads by one item: the last load ends at the final selected
        // element, so tightly bounded strided views need no readable padding.
        const auto lo=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const auto hi=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+16-sizeof(T)));
        if constexpr(sizeof(T)==1)return _mm_unpacklo_epi64(
            _mm_shuffle_epi8(lo,_mm_setr_epi8(0,2,4,6,8,10,12,14,-1,-1,-1,-1,-1,-1,-1,-1)),
            _mm_shuffle_epi8(hi,_mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-1,-1,-1,-1,-1,-1,-1)));
        else return _mm_unpacklo_epi64(
            _mm_shuffle_epi8(lo,_mm_setr_epi8(0,1,4,5,8,9,12,13,-1,-1,-1,-1,-1,-1,-1,-1)),
            _mm_shuffle_epi8(hi,_mm_setr_epi8(2,3,6,7,10,11,14,15,-1,-1,-1,-1,-1,-1,-1,-1)));
    }
    if(step==0) {
        const auto value=detail::alpha_load<T>(p);
        if constexpr(sizeof(T)==1)return _mm_set1_epi8(static_cast<char>(value));
        else return _mm_set1_epi16(static_cast<short>(value));
    }
    alignas(16) T values[16/sizeof(T)];
    for(std::size_t j=0;j<16/sizeof(T);++j)std::memcpy(values+j,p+std::ptrdiff_t(j)*step,sizeof(T));
    return _mm_load_si128(reinterpret_cast<const __m128i*>(values));
}
template<class T> __m256i raw256(const unsigned char* p,std::ptrdiff_t step) {
    if(step==std::ptrdiff_t(sizeof(T)))return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    if(step==-std::ptrdiff_t(sizeof(T))) {
        const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p-(32-sizeof(T))));
        if constexpr(sizeof(T)==4)return _mm256_permutevar8x32_epi32(x,_mm256_setr_epi32(7,6,5,4,3,2,1,0));
        else return _mm256_permute4x64_epi64(x,_MM_SHUFFLE(0,1,2,3));
    }
    if(step==2*std::ptrdiff_t(sizeof(T))) {
        const auto lo=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const auto hi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+32-sizeof(T)));
        if constexpr(sizeof(T)==4)return _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(lo,_mm256_setr_epi32(0,2,4,6,0,2,4,6)),
            _mm256_permutevar8x32_epi32(hi,_mm256_setr_epi32(1,3,5,7,1,3,5,7)),0xf0);
        else return _mm256_blend_epi32(_mm256_permute4x64_epi64(lo,_MM_SHUFFLE(2,0,2,0)),
            _mm256_permute4x64_epi64(hi,_MM_SHUFFLE(3,1,3,1)),0xf0);
    }
    if(step==0) {
        const T value=detail::alpha_load<T>(p);
        if constexpr(sizeof(T)==4)return _mm256_set1_epi32(static_cast<int>(value));
        else return _mm256_set1_epi64x(static_cast<long long>(value));
    }
    if(step>=INT_MIN/7 && step<=INT_MAX/7) {
        const auto s=int(step);
        if constexpr(sizeof(T)==4)return _mm256_i32gather_epi32(reinterpret_cast<const int*>(p),_mm256_setr_epi32(0,s,2*s,3*s,4*s,5*s,6*s,7*s),1);
        else return _mm256_i32gather_epi64(reinterpret_cast<const long long*>(p),_mm_setr_epi32(0,s,2*s,3*s),1);
    }
    alignas(32) T values[32/sizeof(T)];
    for(std::size_t j=0;j<32/sizeof(T);++j)std::memcpy(values+j,p+std::ptrdiff_t(j)*step,sizeof(T));
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(values));
}
__m256 clip(__m256 x) {
    const auto zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1);
    const auto low=_mm256_blendv_ps(x,zero,_mm256_cmp_ps(x,zero,_CMP_LT_OQ));
    return _mm256_blendv_ps(low,one,_mm256_cmp_ps(x,one,_CMP_GT_OQ));
}
__m256d clip(__m256d x) {
    const auto zero=_mm256_setzero_pd(),one=_mm256_set1_pd(1);
    const auto low=_mm256_blendv_pd(x,zero,_mm256_cmp_pd(x,zero,_CMP_LT_OQ));
    return _mm256_blendv_pd(low,one,_mm256_cmp_pd(x,one,_CMP_GT_OQ));
}
void rgba_float(const unsigned char* p,std::ptrdiff_t pixels,std::ptrdiff_t channels,__m256* out) {
    if(pixels==16 && channels==4) {
        __m128 a[4],b[4];
        for(int j=0;j<4;++j) {a[j]=_mm_loadu_ps(reinterpret_cast<const float*>(p+16*j));b[j]=_mm_loadu_ps(reinterpret_cast<const float*>(p+64+16*j));}
        _MM_TRANSPOSE4_PS(a[0],a[1],a[2],a[3]);_MM_TRANSPOSE4_PS(b[0],b[1],b[2],b[3]);
        for(int j=0;j<4;++j)out[j]=_mm256_insertf128_ps(_mm256_castps128_ps256(a[j]),b[j],1);
    } else for(int j=0;j<4;++j)out[j]=_mm256_castsi256_ps(raw256<std::uint32_t>(p+std::ptrdiff_t(j)*channels,pixels));
}
void rgba_double(const unsigned char* p,std::ptrdiff_t pixels,std::ptrdiff_t channels,__m256d* out) {
    if(pixels==32 && channels==8) {
        const auto a=_mm256_loadu_pd(reinterpret_cast<const double*>(p)),b=_mm256_loadu_pd(reinterpret_cast<const double*>(p+32));
        const auto c=_mm256_loadu_pd(reinterpret_cast<const double*>(p+64)),d=_mm256_loadu_pd(reinterpret_cast<const double*>(p+96));
        const auto ab0=_mm256_unpacklo_pd(a,b),ab1=_mm256_unpackhi_pd(a,b),cd0=_mm256_unpacklo_pd(c,d),cd1=_mm256_unpackhi_pd(c,d);
        out[0]=_mm256_permute2f128_pd(ab0,cd0,0x20);out[1]=_mm256_permute2f128_pd(ab1,cd1,0x20);
        out[2]=_mm256_permute2f128_pd(ab0,cd0,0x31);out[3]=_mm256_permute2f128_pd(ab1,cd1,0x31);
    } else for(int j=0;j<4;++j)out[j]=_mm256_castsi256_pd(raw256<std::uint64_t>(p+std::ptrdiff_t(j)*channels,pixels));
}
__m128i low_words(__m256i value) {
    const auto lanes=_mm256_shuffle_epi32(value,_MM_SHUFFLE(2,0,2,0));
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(lanes),_mm256_extracti128_si256(lanes,1));
}
template<class Input,bool Boolean=false> void rgba_integer(const unsigned char* p,std::ptrdiff_t pixels,std::ptrdiff_t channels,__m256d* out) {
    if constexpr(sizeof(Input)<=2) {
        if(pixels==std::ptrdiff_t(4*sizeof(Input)) && channels==std::ptrdiff_t(sizeof(Input))) {
            if constexpr(sizeof(Input)==1) {
                const auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
                for(int j=0;j<4;++j) {
                    __m128i values;
                    if(j==0)values=packed;else if(j==1)values=_mm_srli_epi32(packed,8);
                    else if(j==2)values=_mm_srli_epi32(packed,16);else values=_mm_srli_epi32(packed,24);
                    values=_mm_and_si128(values,_mm_set1_epi32(255));
                    if constexpr(Boolean)values=_mm_and_si128(_mm_cmpgt_epi32(values,_mm_setzero_si128()),_mm_set1_epi32(1));
                    else if constexpr(std::is_signed_v<Input>)values=_mm_srai_epi32(_mm_slli_epi32(values,24),24);
                    out[j]=_mm256_cvtepi32_pd(values);
                }
            } else {
                const auto packed=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
                for(int j=0;j<4;++j) {
                    __m256i shifted;
                    if(j==0)shifted=packed;else if(j==1)shifted=_mm256_srli_epi64(packed,16);
                    else if(j==2)shifted=_mm256_srli_epi64(packed,32);else shifted=_mm256_srli_epi64(packed,48);
                    auto values=low_words(_mm256_and_si256(shifted,_mm256_set1_epi64x(65535)));
                    if constexpr(std::is_signed_v<Input>)values=_mm_srai_epi32(_mm_slli_epi32(values,16),16);
                    out[j]=_mm256_cvtepi32_pd(values);
                }
            }
            if constexpr(!Boolean)for(int j=0;j<4;++j) {
                out[j]=_mm256_mul_pd(out[j],_mm256_set1_pd(1.0/double(std::numeric_limits<Input>::max())));
                if constexpr(std::is_signed_v<Input>)out[j]=_mm256_max_pd(out[j],_mm256_set1_pd(-1));
            }
            return;
        }
    }
    for(int channel=0;channel<4;++channel) {
        double values[4];
        for(int i=0;i<4;++i)values[i]=detail::alpha_normalized<Input,Boolean>(p+std::ptrdiff_t(i)*pixels+std::ptrdiff_t(channel)*channels);
        out[channel]=_mm256_loadu_pd(values);
    }
}
template<class Input,bool Boolean=false> std::size_t composite(const unsigned char* p,void* output,std::size_t n,const double* bg,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    std::size_t i=0;
    if constexpr(std::is_same_v<Input,float>) {
        __m256 background[3];for(int j=0;j<3;++j)background[j]=_mm256_set1_ps(float(bg[j]));
        for(;i+8<=n;i+=8) {
            __m256 v[4];rgba_float(p+std::ptrdiff_t(i)*pixels,pixels,channels,v);
            const auto remaining=_mm256_sub_ps(_mm256_set1_ps(1),v[3]);
            for(int j=0;j<3;++j)v[j]=clip(_mm256_add_ps(_mm256_mul_ps(remaining,background[j]),_mm256_mul_ps(v[3],v[j])));
            pack(static_cast<float*>(output)+3*i,v[0],v[1],v[2]);
        }
    } else {
        __m256d background[3];for(int j=0;j<3;++j)background[j]=_mm256_set1_pd(bg[j]);
        for(;i+4<=n;i+=4) {
            __m256d v[4];
            if constexpr(std::is_same_v<Input,double>)rgba_double(p+std::ptrdiff_t(i)*pixels,pixels,channels,v);
            else rgba_integer<Input,Boolean>(p+std::ptrdiff_t(i)*pixels,pixels,channels,v);
            const auto remaining=_mm256_sub_pd(_mm256_set1_pd(1),v[3]);
            for(int j=0;j<3;++j)v[j]=clip(_mm256_add_pd(_mm256_mul_pd(remaining,background[j]),_mm256_mul_pd(v[3],v[j])));
            pack(static_cast<double*>(output)+3*i,v[0],v[1],v[2]);
        }
    }
    return i;
}
}
std::size_t rgba_composite(const void* input,void* output,std::size_t count,PixelType type,const double* background,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    const auto* p=static_cast<const unsigned char*>(input);
    if(type==PixelType::boolean)return composite<std::uint8_t,true>(p,output,count,background,pixels,channels);
#define RGBA_CASE(tag,ctype) case PixelType::tag:return composite<ctype>(p,output,count,background,pixels,channels);
    switch(type) {
    RGBA_CASE(uint8,std::uint8_t) RGBA_CASE(int8,std::int8_t) RGBA_CASE(uint16,std::uint16_t) RGBA_CASE(int16,std::int16_t)
    RGBA_CASE(uint32,std::uint32_t) RGBA_CASE(int32,std::int32_t) RGBA_CASE(uint64,std::uint64_t) RGBA_CASE(int64,std::int64_t)
    RGBA_CASE(float32,float) RGBA_CASE(float64,double)
    default:return 0;
    }
#undef RGBA_CASE
}

namespace {
__m128i swapped_bytes(__m128i x,std::size_t width) {
    if(!width)return x;
    if(width==2)return _mm_shuffle_epi8(x,_mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14));
    if(width==4)return _mm_shuffle_epi8(x,_mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12));
    return _mm_shuffle_epi8(x,_mm_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8));
}
__m256i swapped_bytes(__m256i x,std::size_t width) {
    if(!width)return x;
    if(!width)return x;
    const auto mask=swapped_bytes(_mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),width);
    return _mm256_shuffle_epi8(x,_mm256_broadcastsi128_si256(mask));
}
}
namespace {
struct PermutationGroup {std::size_t chunk;bool alpha;__m128i mask;};
struct PermutationBlock {std::vector<PermutationGroup> groups;std::size_t bytes;};
struct alignas(16) ByteBlock {__m128i value;};
__m128i six_bytes(const unsigned char* p) {
    std::uint32_t low;std::uint16_t high;std::memcpy(&low,p,4);std::memcpy(&high,p+4,2);
    return _mm_cvtsi64_si128(static_cast<long long>(low)|(static_cast<long long>(high)<<32));
}

__m128i item_chunk(const unsigned char* p,std::size_t bytes,std::size_t chunk,std::ptrdiff_t step,std::size_t remaining) {
    const auto offset=16*chunk,tail=std::min(std::size_t(16),bytes-offset);
    if(tail==16)return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p+offset));
    // Borrow readable bytes only from within this item, or adjacent items in
    // a contiguous row. General strided/broadcast tails load exact storage.
    if(bytes>=16) {
        const auto value=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+bytes-16));
        alignas(16) unsigned char storage[16];_mm_store_si128(reinterpret_cast<__m128i*>(storage),value);
        __m128i result=_mm_setzero_si128();std::memcpy(&result,storage+16-tail,tail);return result;
    }
    if(step==std::ptrdiff_t(bytes) && remaining>=(16+bytes-1)/bytes)
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    __m128i result=_mm_setzero_si128();std::memcpy(&result,p+offset,tail);return result;
}
}
std::size_t gray_expand_mapped(const unsigned char* input,const unsigned char* alpha,unsigned char* output,
    std::size_t count,std::size_t bytes,int channels,std::ptrdiff_t source_step,std::ptrdiff_t alpha_step,const std::size_t* order) {
    if(!bytes)return count;
    const auto chunks=(bytes+15)/16,out_bytes=bytes*std::size_t(channels);
    std::vector<PermutationBlock> plan;
    for(std::size_t start=0;start<out_bytes;start+=16) {
        PermutationBlock block;block.bytes=std::min(std::size_t(16),out_bytes-start);
        struct Mask {std::size_t chunk;bool alpha; signed char bytes[16];};
        std::vector<Mask> masks;masks.reserve(2);
        for(std::size_t lane=0;lane<block.bytes;++lane) {
            const auto channel=(start+lane)/bytes,index=order[(start+lane)%bytes];
            const auto chunk=index/16;const bool opacity=channel==3;
            auto group=std::find_if(masks.begin(),masks.end(),[&](const Mask& entry){return entry.chunk==chunk && entry.alpha==opacity;});
            if(group==masks.end()) {
                Mask entry;entry.chunk=chunk;entry.alpha=opacity;
                std::fill(entry.bytes,entry.bytes+16,static_cast<signed char>(-128));
                masks.push_back(entry);group=masks.end()-1;
            }
            group->bytes[lane]=static_cast<signed char>(index%16);
        }
        for(const auto& mask:masks)block.groups.push_back({mask.chunk,mask.alpha,_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask.bytes))});
        plan.push_back(std::move(block));
    }
    if(bytes==6) {
        const auto first=plan[0].groups[0].mask;
        auto last_gray=_mm_set1_epi8(static_cast<char>(-128)),last_alpha=last_gray;
        for(const auto& group:plan[1].groups) {
            if(group.alpha)last_alpha=group.mask;else last_gray=group.mask;
        }
        const auto constant_alpha=channels==4 && alpha_step==0 ? six_bytes(alpha) : _mm_setzero_si128();
        for(std::size_t i=0;i<count;++i) {
            const auto gray=six_bytes(input+std::ptrdiff_t(i)*source_step);
            auto* out=output+i*std::size_t(channels)*6;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_shuffle_epi8(gray,first));
            auto tail=_mm_shuffle_epi8(gray,last_gray);
            if(channels==3) {
                const auto value=static_cast<std::uint16_t>(_mm_extract_epi16(tail,0));std::memcpy(out+16,&value,2);
            } else {
                const auto a=alpha_step==0 ? constant_alpha : six_bytes(alpha+std::ptrdiff_t(i)*alpha_step);
                tail=_mm_or_si128(tail,_mm_shuffle_epi8(a,last_alpha));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(out+16),tail);
            }
        }
        return count;
    }
    std::vector<ByteBlock> gray(chunks),opacity(channels==4 ? chunks : 0);
    for(std::size_t i=0;i<count;++i) {
        for(std::size_t chunk=0;chunk<chunks;++chunk) {
            gray[chunk].value=item_chunk(input+std::ptrdiff_t(i)*source_step,bytes,chunk,source_step,count-i);
            if(channels==4 && (alpha_step!=0 || i==0))opacity[chunk].value=item_chunk(alpha+std::ptrdiff_t(i)*alpha_step,bytes,chunk,alpha_step,count-i);
        }
        std::size_t position=0;
        for(const auto& block:plan) {
            auto value=_mm_setzero_si128();
            for(const auto& group:block.groups)value=_mm_or_si128(value,_mm_shuffle_epi8(group.alpha ? opacity[group.chunk].value : gray[group.chunk].value,group.mask));
            if(block.bytes==16)_mm_storeu_si128(reinterpret_cast<__m128i*>(output+i*out_bytes+position),value);
            else std::memcpy(output+i*out_bytes+position,&value,block.bytes);
            position+=block.bytes;
        }
    }
    return count;
}
std::size_t gray_expand(const unsigned char* input,const unsigned char* alpha,unsigned char* output,std::size_t count,
    std::size_t bytes,int channels,std::ptrdiff_t source_step,std::ptrdiff_t alpha_step,std::size_t swap_component) {
    std::size_t i=0;
    if(bytes==1 || bytes==2) {
        const auto lanes=16/bytes;
        for(;i+lanes<=count;i+=lanes) {
            const auto* p=input+std::ptrdiff_t(i)*source_step;
            const auto g=swapped_bytes(bytes==1 ? raw128<std::uint8_t>(p,source_step) : raw128<std::uint16_t>(p,source_step),swap_component);
            auto* out=output+i*std::size_t(channels)*bytes;
            if(channels==3) {
                if(bytes==1) {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_shuffle_epi8(g,_mm_setr_epi8(0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5)));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+16),_mm_shuffle_epi8(g,_mm_setr_epi8(5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10)));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+32),_mm_shuffle_epi8(g,_mm_setr_epi8(10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15)));
                } else {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_shuffle_epi8(g,_mm_setr_epi8(0,1,0,1,0,1,2,3,2,3,2,3,4,5,4,5)));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+16),_mm_shuffle_epi8(g,_mm_setr_epi8(4,5,6,7,6,7,6,7,8,9,8,9,8,9,10,11)));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+32),_mm_shuffle_epi8(g,_mm_setr_epi8(10,11,10,11,12,13,12,13,12,13,14,15,14,15,14,15)));
                }
            } else {
                const auto* a=alpha+std::ptrdiff_t(i)*alpha_step;
                const auto v=swapped_bytes(bytes==1 ? raw128<std::uint8_t>(a,alpha_step) : raw128<std::uint16_t>(a,alpha_step),swap_component);
                __m128i gg0,ga0,gg1,ga1;
                if(bytes==1) {
                    gg0=_mm_unpacklo_epi8(g,g);ga0=_mm_unpacklo_epi8(g,v);gg1=_mm_unpackhi_epi8(g,g);ga1=_mm_unpackhi_epi8(g,v);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_unpacklo_epi16(gg0,ga0));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+16),_mm_unpackhi_epi16(gg0,ga0));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+32),_mm_unpacklo_epi16(gg1,ga1));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+48),_mm_unpackhi_epi16(gg1,ga1));
                } else {
                    gg0=_mm_unpacklo_epi16(g,g);ga0=_mm_unpacklo_epi16(g,v);gg1=_mm_unpackhi_epi16(g,g);ga1=_mm_unpackhi_epi16(g,v);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),_mm_unpacklo_epi32(gg0,ga0));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+16),_mm_unpackhi_epi32(gg0,ga0));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+32),_mm_unpacklo_epi32(gg1,ga1));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(out+48),_mm_unpackhi_epi32(gg1,ga1));
                }
            }
        }
    } else if(bytes==4) {
        for(;i+8<=count;i+=8) {
            const auto g=swapped_bytes(raw256<std::uint32_t>(input+std::ptrdiff_t(i)*source_step,source_step),swap_component);
            auto* out=output+i*std::size_t(channels)*4;
            if(channels==3) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm256_permutevar8x32_epi32(g,_mm256_setr_epi32(0,0,0,1,1,1,2,2)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),_mm256_permutevar8x32_epi32(g,_mm256_setr_epi32(2,3,3,3,4,4,4,5)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permutevar8x32_epi32(g,_mm256_setr_epi32(5,5,6,6,6,7,7,7)));
            } else {
                const auto a=swapped_bytes(raw256<std::uint32_t>(alpha+std::ptrdiff_t(i)*alpha_step,alpha_step),swap_component);
                const auto gg0=_mm256_unpacklo_epi32(g,g),ga0=_mm256_unpacklo_epi32(g,a),gg1=_mm256_unpackhi_epi32(g,g),ga1=_mm256_unpackhi_epi32(g,a);
                const auto p0=_mm256_unpacklo_epi64(gg0,ga0),p1=_mm256_unpackhi_epi64(gg0,ga0),p2=_mm256_unpacklo_epi64(gg1,ga1),p3=_mm256_unpackhi_epi64(gg1,ga1);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm256_permute2x128_si256(p0,p1,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),_mm256_permute2x128_si256(p2,p3,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permute2x128_si256(p0,p1,0x31));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+96),_mm256_permute2x128_si256(p2,p3,0x31));
            }
        }
    } else if(bytes==8) {
        for(;i+4<=count;i+=4) {
            const auto g=swapped_bytes(raw256<std::uint64_t>(input+std::ptrdiff_t(i)*source_step,source_step),swap_component);
            auto* out=output+i*std::size_t(channels)*8;
            if(channels==3) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm256_permute4x64_epi64(g,_MM_SHUFFLE(1,0,0,0)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),_mm256_permute4x64_epi64(g,_MM_SHUFFLE(2,2,1,1)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permute4x64_epi64(g,_MM_SHUFFLE(3,3,3,2)));
            } else {
                const auto a=swapped_bytes(raw256<std::uint64_t>(alpha+std::ptrdiff_t(i)*alpha_step,alpha_step),swap_component);
                const auto gg0=_mm256_unpacklo_epi64(g,g),ga0=_mm256_unpacklo_epi64(g,a),gg1=_mm256_unpackhi_epi64(g,g),ga1=_mm256_unpackhi_epi64(g,a);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm256_permute2x128_si256(gg0,ga0,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),_mm256_permute2x128_si256(gg1,ga1,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permute2x128_si256(gg0,ga0,0x31));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+96),_mm256_permute2x128_si256(gg1,ga1,0x31));
            }
        }
    } else if(bytes==16) {
        for(;i+2<=count;i+=2) {
            const auto* p=input+std::ptrdiff_t(i)*source_step;
            const auto g=swapped_bytes(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))),_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+source_step)),1),swap_component);
            auto* out=output+i*std::size_t(channels)*16;
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out),_mm256_permute2x128_si256(g,g,0));
            if(channels==3) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),g);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permute2x128_si256(g,g,0x11));
            } else {
                const auto* p_alpha=alpha+std::ptrdiff_t(i)*alpha_step;
                const auto a=swapped_bytes(_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p_alpha))),_mm_loadu_si128(reinterpret_cast<const __m128i*>(p_alpha+alpha_step)),1),swap_component);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+32),_mm256_permute2x128_si256(g,a,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+64),_mm256_permute2x128_si256(g,g,0x11));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+96),_mm256_permute2x128_si256(g,a,0x31));
            }
        }
    }
    return i;
}
}
