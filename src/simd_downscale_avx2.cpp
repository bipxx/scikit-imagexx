#include "simd.hpp"
#include "half.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

namespace skimagexx::simd {
namespace {
template<class Value> struct Vector;
template<> struct Vector<float> {
    using Register=__m256;static constexpr std::size_t lanes=8;
    static Register load(const float* p){return _mm256_loadu_ps(p);}
    static void store(float* p,Register x){_mm256_storeu_ps(p,x);}
    static Register fill(float x){return _mm256_set1_ps(x);}
    static Register add(Register a,Register b){return _mm256_add_ps(a,b);}
    static Register divide(Register a,float b){return _mm256_div_ps(a,fill(b));}
    static Register pairs(Register a,Register b) {
        return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(_mm256_hadd_ps(a,b)),0xd8));
    }
    static bool bad(Register a,float bound) {
        const auto absolute=_mm256_andnot_ps(_mm256_set1_ps(-0.f),a);
        return _mm256_movemask_ps(_mm256_cmp_ps(absolute,fill(bound),_CMP_NLE_UQ))!=0;
    }
};
template<> struct Vector<double> {
    using Register=__m256d;static constexpr std::size_t lanes=4;
    static Register load(const double* p){return _mm256_loadu_pd(p);}
    static void store(double* p,Register x){_mm256_storeu_pd(p,x);}
    static Register fill(double x){return _mm256_set1_pd(x);}
    static Register add(Register a,Register b){return _mm256_add_pd(a,b);}
    static Register divide(Register a,double b){return _mm256_div_pd(a,fill(b));}
    static Register pairs(Register a,Register b){return _mm256_permute4x64_pd(_mm256_hadd_pd(a,b),0xd8);}
    static bool bad(Register a,double bound) {
        const auto absolute=_mm256_andnot_pd(_mm256_set1_pd(-0.),a);
        return _mm256_movemask_pd(_mm256_cmp_pd(absolute,fill(bound),_CMP_NLE_UQ))!=0;
    }
};

template<class T> double scalar_value(const unsigned char* p) {
    T value;std::memcpy(&value,p,sizeof(T));return double(value);
}
double scalar_value(const unsigned char* p,PixelType type,bool half) {
    if(half) {std::uint16_t word;std::memcpy(&word,p,2);return detail::decode_half(word);}
    switch(type) {
    case PixelType::boolean:return scalar_value<std::uint8_t>(p)!=0 ? 1. : 0.;
#define RAW(tag,T) case PixelType::tag:return scalar_value<T>(p);
    RAW(uint8,std::uint8_t) RAW(int8,std::int8_t) RAW(uint16,std::uint16_t) RAW(int16,std::int16_t)
    RAW(uint32,std::uint32_t) RAW(int32,std::int32_t) RAW(uint64,std::uint64_t) RAW(int64,std::int64_t)
    RAW(float32,float) RAW(float64,double)
#undef RAW
    }
    return 0;
}

template<class Value> Value span_sum(const unsigned char* source,std::size_t count,
    PixelType type,bool half,bool swap,Value bound,bool& supported) {
    using Op=Vector<Value>;constexpr auto width=Op::lanes;
    const auto bytes=pixel_size(type);
    if(count>128) {
        auto split=count/2;split-=split%8;
        return Value(span_sum<Value>(source,split,type,half,swap,bound,supported)+
            span_sum<Value>(source+split*bytes,count-split,type,half,swap,bound,supported));
    }
    alignas(32) unsigned char normalized[128*8];
    if(swap && bytes>1) {
        const auto done=edge_swap(source,normalized,count,bytes);
        for(auto i=done;i<count;++i)for(std::size_t byte=0;byte<bytes;++byte)
            normalized[i*bytes+byte]=source[i*bytes+bytes-1-byte];
        source=normalized;
    }
    alignas(32) Value prepared[128];const Value* values=nullptr;
    std::size_t prepared_count=count;
    if constexpr(std::is_same_v<Value,float>) {
        if(half) {
            prepared_count=edge_half(reinterpret_cast<const std::uint16_t*>(source),prepared,count);
            values=prepared;
        } else values=reinterpret_cast<const float*>(source);
    } else {
        if(type==PixelType::float64)values=reinterpret_cast<const double*>(source);
        else {prepared_count=gaussian_prepare(source,prepared,count,type,true);values=prepared;}
    }
    for(auto i=prepared_count;i<count;++i)prepared[i]=Value(scalar_value(source+i*bytes,type,half));
    const bool floating=half || type==PixelType::float32 || type==PixelType::float64;
    Value sum=Value(-0.0);std::size_t position=0;
    if(count>=8) {
        typename Op::Register accumulators[8/width];
        for(std::size_t j=0;j<8/width;++j) {
            accumulators[j]=Op::load(values+j*width);
            if(floating && Op::bad(accumulators[j],bound))supported=false;
        }
        position=8;
        for(;position+8<=count;position+=8)for(std::size_t j=0;j<8/width;++j) {
            const auto value=Op::load(values+position+j*width);
            if(floating && Op::bad(value,bound))supported=false;
            accumulators[j]=Op::add(accumulators[j],value);
        }
        alignas(32) Value lanes[8];
        for(std::size_t j=0;j<8/width;++j)Op::store(lanes+j*width,accumulators[j]);
        sum=Value(Value(Value(lanes[0]+lanes[1])+Value(lanes[2]+lanes[3]))+
                  Value(Value(lanes[4]+lanes[5])+Value(lanes[6]+lanes[7])));
    }
    for(;position<count;++position) {
        Value value;
        if(values==prepared)value=prepared[position];
        else std::memcpy(&value,reinterpret_cast<const unsigned char*>(values)+position*sizeof(Value),sizeof(Value));
        if(floating && (!std::isfinite(value) || std::abs(value)>bound))supported=false;
        sum=Value(sum+value);
    }
    return sum;
}

template<class Value,bool Half=false> std::size_t row(const void* input,void* output,
    std::size_t height,std::size_t width,std::ptrdiff_t row_stride,std::size_t output_row,
    std::size_t factor_y,std::size_t factor_x,std::size_t begin,std::size_t end,
    PixelType type,double constant,double bound,bool& supported) {
    using Op=Vector<Value>;constexpr auto lanes=Op::lanes;
    const auto bytes=pixel_size(type),output_bytes=Half ? 2 : sizeof(Value);
    const auto* source=static_cast<const unsigned char*>(input);
    end=std::min(end,width/factor_x);
    const auto top=output_row*factor_y;
    for(;begin+lanes<=end;begin+=lanes) {
        auto sum=Op::fill(Value(0));
        for(std::size_t y=0;y<factor_y;++y) {
            auto block=Op::fill(Value(Value(constant)*Value(factor_x)));
            if(top+y<height) {
                const auto* pixels=source+std::ptrdiff_t(top+y)*row_stride+begin*factor_x*bytes;
                alignas(32) Value prepared[lanes*4];const Value* values=nullptr;
                if constexpr(Half) {
                    edge_half(reinterpret_cast<const std::uint16_t*>(pixels),prepared,lanes*factor_x);values=prepared;
                } else if constexpr(std::is_same_v<Value,float>)values=reinterpret_cast<const float*>(pixels);
                else {
                    if(type==PixelType::float64)values=reinterpret_cast<const double*>(pixels);
                    else {gaussian_prepare(pixels,prepared,lanes*factor_x,type,true);values=prepared;}
                }
                typename Op::Register parts[4];
                for(std::size_t i=0;i<factor_x;++i) {
                    parts[i]=Op::load(values+i*lanes);
                    if(Op::bad(parts[i],Value(bound)))supported=false;
                }
                block=factor_x==1 ? parts[0] : factor_x==2 ? Op::pairs(parts[0],parts[1])
                    : Op::pairs(Op::pairs(parts[0],parts[1]),Op::pairs(parts[2],parts[3]));
            }
            sum=Op::add(sum,block);
        }
        if(Op::bad(sum,std::numeric_limits<Value>::max()))supported=false;
        const auto mean=Op::divide(sum,Value(factor_y*factor_x));
        auto* target=static_cast<unsigned char*>(output)+begin*output_bytes;
        if constexpr(Half) {
            alignas(32) float values[lanes];Op::store(values,mean);
            for(std::size_t i=0;i<lanes;++i) {
                const auto word=detail::encode_half(values[i]);std::memcpy(target+2*i,&word,2);
            }
        } else Op::store(reinterpret_cast<Value*>(target),mean);
    }
    return begin;
}
}
double downscale_sum(const void* input,std::size_t count,std::ptrdiff_t stride,PixelType type,
    bool half,bool swap,double bound,bool& supported) {
    if(!count)return 0.;
    const auto* source=static_cast<const unsigned char*>(input);
    if(stride<0)source+=std::ptrdiff_t(count-1)*stride;
    if(half || type==PixelType::float32)return span_sum<float>(source,count,type,half,swap,float(bound),supported);
    return span_sum<double>(source,count,type,half,swap,bound,supported);
}
std::size_t downscale_row(const void* input,void* output,std::size_t height,std::size_t width,
    std::ptrdiff_t row_stride,std::size_t output_row,std::size_t factor_y,std::size_t factor_x,
    std::size_t begin,std::size_t end,PixelType type,bool half,double constant,double bound,bool& supported) {
    if((factor_x!=1 && factor_x!=2 && factor_x!=4) || factor_y>(std::size_t(1)<<24)/factor_x)return begin;
    if(half)return row<float,true>(input,output,height,width,row_stride,output_row,factor_y,factor_x,begin,end,type,constant,bound,supported);
    if(type==PixelType::float32)return row<float>(input,output,height,width,row_stride,output_row,factor_y,factor_x,begin,end,type,constant,bound,supported);
    return row<double>(input,output,height,width,row_stride,output_row,factor_y,factor_x,begin,end,type,constant,bound,supported);
}
}
