#include "simd.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd {
namespace {
__m256d load(const double* p) {return _mm256_loadu_pd(p);}
__m256d load(const float* p) {return _mm256_cvtps_pd(_mm_loadu_ps(p));}
__m256d load(const std::uint8_t* p) {
    std::uint32_t bits;std::memcpy(&bits,p,4);
    return _mm256_mul_pd(_mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(bits)))),_mm256_set1_pd(1.0/255.0));
}
void store(double* p,__m256d x) {_mm256_storeu_pd(p,x);}
void store(float* p,__m256d x) {_mm_storeu_ps(p,_mm256_cvtpd_ps(x));}
template<class Input,class Output> std::size_t run(const Input* top,const Input* mid,const Input* bottom,Output* output,
    std::size_t width,std::size_t x,std::size_t end,int operation) {
    const auto divisor=_mm256_set1_pd(std::sqrt(2.0));
    for(;x+4<=end && x+4<width;x+=4) {
        if(operation==0) {
            auto sum=_mm256_sub_pd(_mm256_setzero_pd(),load(top+x));
            sum=_mm256_sub_pd(sum,load(mid+x-1));
            sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_set1_pd(4),load(mid+x)));
            sum=_mm256_sub_pd(sum,load(mid+x+1));sum=_mm256_sub_pd(sum,load(bottom+x));
            store(output+x,sum);
        } else {
            auto a=_mm256_sub_pd(_mm256_add_pd(_mm256_setzero_pd(),load(bottom+x+1)),load(mid+x));
            auto b=_mm256_sub_pd(_mm256_add_pd(_mm256_setzero_pd(),load(bottom+x)),load(mid+x+1));
            a=_mm256_andnot_pd(_mm256_cmp_pd(a,_mm256_setzero_pd(),_CMP_EQ_OQ),a);
            b=_mm256_andnot_pd(_mm256_cmp_pd(b,_mm256_setzero_pd(),_CMP_EQ_OQ),b);
            if(operation==2)store(output+x,a);
            else if(operation==3)store(output+x,b);
            else if constexpr(std::is_same_v<Output,float>) {
                const auto af=_mm256_cvtpd_ps(a),bf=_mm256_cvtpd_ps(b);
                const auto sum=_mm_add_ps(_mm_mul_ps(af,af),_mm_mul_ps(bf,bf));
                store(output+x,_mm256_div_pd(_mm256_cvtps_pd(_mm_sqrt_ps(sum)),divisor));
            } else {
                const auto sum=_mm256_add_pd(_mm256_mul_pd(a,a),_mm256_mul_pd(b,b));
                store(output+x,_mm256_div_pd(_mm256_sqrt_pd(sum),divisor));
            }
        }
    }
    return x;
}
}
#define SMALL_EDGE(Input,Output) std::size_t small_edge(const Input* top,const Input* mid,const Input* bottom,Output* output,std::size_t width,std::size_t begin,std::size_t end,int operation) {return run(top,mid,bottom,output,width,begin,end,operation);}
SMALL_EDGE(float,float)
SMALL_EDGE(double,double)
SMALL_EDGE(std::uint8_t,double)
#undef SMALL_EDGE
namespace {
template<class Input,class Output> std::size_t column_run(const Input* input,Output* output,
    std::size_t height,std::size_t y,std::size_t end,int operation) {
    const auto divisor=_mm256_set1_pd(std::sqrt(2.0));
    for(;y+4<=end && y+4<height;y+=4) {
        const auto current=load(input+y),next=load(input+y+1);
        if(operation==0) {
            auto sum=_mm256_sub_pd(_mm256_setzero_pd(),load(input+y-1));
            sum=_mm256_sub_pd(sum,current);sum=_mm256_add_pd(sum,_mm256_mul_pd(_mm256_set1_pd(4),current));
            sum=_mm256_sub_pd(sum,current);sum=_mm256_sub_pd(sum,next);
            sum=_mm256_andnot_pd(_mm256_cmp_pd(sum,_mm256_setzero_pd(),_CMP_EQ_OQ),sum);
            store(output+y,sum);
        } else {
            auto gradient=_mm256_sub_pd(_mm256_add_pd(_mm256_setzero_pd(),next),current);
            gradient=_mm256_andnot_pd(_mm256_cmp_pd(gradient,_mm256_setzero_pd(),_CMP_EQ_OQ),gradient);
            if(operation!=1)store(output+y,gradient);
            else if constexpr(std::is_same_v<Output,float>) {
                const auto g=_mm256_cvtpd_ps(gradient),square=_mm_mul_ps(g,g);
                store(output+y,_mm256_div_pd(_mm256_cvtps_pd(_mm_sqrt_ps(_mm_add_ps(square,square))),divisor));
            } else {
                const auto square=_mm256_mul_pd(gradient,gradient);
                store(output+y,_mm256_div_pd(_mm256_sqrt_pd(_mm256_add_pd(square,square)),divisor));
            }
        }
    }
    return y;
}
}
#define COLUMN_EDGE(Input,Output) std::size_t small_edge_column(const Input* input,Output* output,std::size_t height,std::size_t begin,std::size_t end,int operation) {return column_run(input,output,height,begin,end,operation);}
COLUMN_EDGE(float,float)
COLUMN_EDGE(double,double)
COLUMN_EDGE(std::uint8_t,double)
#undef COLUMN_EDGE
namespace {
template<class Input,class Output> std::size_t laplace_run(const Input* const* negative,const Input* const* positive,
    const Input* mid,Output* output,std::size_t width,std::size_t x,std::size_t end,std::size_t outer,std::size_t singletons) {
    const auto center=_mm256_set1_pd(double(2*(outer+singletons+1)));
    for(;x+4<=end && x+4<width;x+=4) {
        auto sum=_mm256_setzero_pd();
        for(std::size_t axis=0;axis<outer;++axis)sum=_mm256_sub_pd(sum,load(negative[axis]+x));
        sum=_mm256_sub_pd(sum,load(mid+x-1));
        const auto pixel=load(mid+x);
        for(std::size_t axis=0;axis<singletons;++axis)sum=_mm256_sub_pd(sum,pixel);
        sum=_mm256_add_pd(sum,_mm256_mul_pd(center,pixel));
        for(std::size_t axis=0;axis<singletons;++axis)sum=_mm256_sub_pd(sum,pixel);
        sum=_mm256_sub_pd(sum,load(mid+x+1));
        for(std::size_t axis=outer;axis>0;--axis)sum=_mm256_sub_pd(sum,load(positive[axis-1]+x));
        sum=_mm256_andnot_pd(_mm256_cmp_pd(sum,_mm256_setzero_pd(),_CMP_EQ_OQ),sum);
        store(output+x,sum);
    }
    return x;
}
}
#define LAPLACE_ND(Input,Output) std::size_t laplace_nd(const Input* const* negative,const Input* const* positive,const Input* mid,Output* output,std::size_t width,std::size_t begin,std::size_t end,std::size_t outer,std::size_t singletons) {return laplace_run(negative,positive,mid,output,width,begin,end,outer,singletons);}
LAPLACE_ND(float,float)
LAPLACE_ND(double,double)
LAPLACE_ND(std::uint8_t,double)
#undef LAPLACE_ND

std::size_t edge_half(const std::uint16_t* input,float* output,std::size_t count) {
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto raw=_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i)));
        const auto sign=_mm256_slli_epi32(_mm256_and_si256(raw,_mm256_set1_epi32(0x8000)),16);
        const auto exponent=_mm256_and_si256(_mm256_srli_epi32(raw,10),_mm256_set1_epi32(31));
        const auto fraction=_mm256_and_si256(raw,_mm256_set1_epi32(1023));
        auto bits=_mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(exponent,_mm256_set1_epi32(112)),23),_mm256_slli_epi32(fraction,13));
        const auto small=_mm256_castps_si256(_mm256_mul_ps(_mm256_cvtepi32_ps(fraction),_mm256_set1_ps(0.000000059604644775390625f)));
        bits=_mm256_blendv_epi8(bits,small,_mm256_cmpeq_epi32(exponent,_mm256_setzero_si256()));
        const auto special=_mm256_or_si256(_mm256_set1_epi32(0x7f800000),_mm256_slli_epi32(fraction,13));
        bits=_mm256_blendv_epi8(bits,special,_mm256_cmpeq_epi32(exponent,_mm256_set1_epi32(31)));
        _mm256_storeu_ps(output+i,_mm256_castsi256_ps(_mm256_or_si256(sign,bits)));
    }
    return i;
}
std::size_t edge_swap(const unsigned char* input,unsigned char* output,std::size_t count,std::size_t bytes) {
    const auto shuffle128=bytes==2 ? _mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14) :
        bytes==4 ? _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12) :
        _mm_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8);
    const auto shuffle=_mm256_broadcastsi128_si256(shuffle128);
    const auto lanes=32/bytes;std::size_t i=0;
    for(;i+lanes<=count;i+=lanes) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i*bytes));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output+i*bytes),_mm256_shuffle_epi8(value,shuffle));
    }
    return i;
}
std::size_t edge_mask(float* output,const unsigned char* mask,std::size_t count,std::ptrdiff_t stride) {
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        __m256i values;
        if(stride==0)values=_mm256_set1_epi32(mask[0]!=0);
        else if(stride==1)values=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask+i)));
        else {
            alignas(32) std::int32_t gathered[8];
            for(int j=0;j<8;++j)gathered[j]=mask[std::ptrdiff_t(i+std::size_t(j))*stride]!=0;
            values=_mm256_load_si256(reinterpret_cast<const __m256i*>(gathered));
        }
        values=_mm256_andnot_si256(_mm256_cmpeq_epi32(values,_mm256_setzero_si256()),_mm256_set1_epi32(1));
        _mm256_storeu_ps(output+i,_mm256_mul_ps(_mm256_loadu_ps(output+i),_mm256_cvtepi32_ps(values)));
    }
    return i;
}
std::size_t edge_mask(double* output,const unsigned char* mask,std::size_t count,std::ptrdiff_t stride) {
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        __m128i values;
        if(stride==0)values=_mm_set1_epi32(mask[0]!=0);
        else if(stride==1) {
            std::uint32_t bytes;std::memcpy(&bytes,mask+i,4);
            values=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(bytes)));
        } else values=_mm_setr_epi32(mask[std::ptrdiff_t(i)*stride]!=0,mask[std::ptrdiff_t(i+1)*stride]!=0,
            mask[std::ptrdiff_t(i+2)*stride]!=0,mask[std::ptrdiff_t(i+3)*stride]!=0);
        values=_mm_andnot_si128(_mm_cmpeq_epi32(values,_mm_setzero_si128()),_mm_set1_epi32(1));
        _mm256_storeu_pd(output+i,_mm256_mul_pd(_mm256_loadu_pd(output+i),_mm256_cvtepi32_pd(values)));
    }
    return i;
}
}
