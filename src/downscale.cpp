#include "skimagexx/downscale.hpp"
#include "execution.hpp"
#include "half.hpp"
#include "simd.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
struct Plan {
    std::vector<std::size_t> blocks,axes,output_axes;
    std::size_t count=1,volume=1,work=0;
    bool padded=false;
};
Plan plan(const LocalMeanView& input,const std::vector<std::size_t>& factors,
          void* output,const std::vector<std::size_t>& strides,const void* constant) {
    const auto rank=input.shape.size(),limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(factors.size()!=rank || input.strides.size()!=rank || strides.size()!=rank)
        throw std::invalid_argument("one local-mean factor and stride per axis required");
    if(input.half && input.type!=PixelType::uint16)throw std::invalid_argument("binary16 local-mean storage required");
    Plan result;result.blocks.resize(rank);result.axes.resize(rank);result.output_axes.resize(rank);
    const auto bytes=input.half ? 2 : input.type==PixelType::float32 ? 4 : 8;
    bool empty=false;
    for(std::size_t axis=0;axis<rank;++axis) {
        const auto length=input.shape[axis],factor=factors[axis];
        if(!factor)throw std::invalid_argument("positive local-mean factors required");
        if(length>limit || factor>limit || factor>limit/result.volume)
            throw std::overflow_error("local-mean dimensions overflow");
        result.volume*=factor;
        result.blocks[axis]=length ? (length-1)/factor+1 : 0;
        empty|=!length;result.padded|=length%factor!=0;result.axes[axis]=axis;result.output_axes[axis]=axis;
    }
    if(empty) {result.count=0;return result;}
    for(auto length:result.blocks) {
        if(length>limit/bytes/result.count)throw std::overflow_error("local-mean output dimensions overflow");
        result.count*=length;
    }
    if(!input.data || !output || (result.padded && !constant))throw std::invalid_argument("nonnull local-mean storage required");
    std::size_t positive=pixel_size(input.type)-1,negative=0;
    for(std::size_t axis=0;axis<rank;++axis) {
        const auto length=input.shape[axis];if(length<=1)continue;
        const auto stride=input.strides[axis];
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("local-mean stride overflow");
        const auto magnitude=std::size_t(stride<0 ? -stride : stride);
        auto& span=stride<0 ? negative : positive;
        if(magnitude>(limit-span)/(length-1))throw std::overflow_error("local-mean input span overflow");
        span+=magnitude*(length-1);
    }
    std::vector<std::pair<std::size_t,std::size_t>> order;
    for(std::size_t axis=0;axis<rank;++axis)if(result.blocks[axis]>1)order.emplace_back(strides[axis],result.blocks[axis]);
    std::sort(order.begin(),order.end());std::size_t extent=std::size_t(bytes);
    for(const auto& axis:order) {
        if(axis.first!=extent)throw std::invalid_argument("dense local-mean output strides required");
        extent*=axis.second;
    }
    // Reduce the physically fastest axis at the leaves. Ties retain logical
    // axis order; singleton strides never participate in pointer arithmetic.
    auto magnitude=[&](std::size_t axis) {
        if(input.shape[axis]<=1)return std::size_t(0);
        const auto stride=input.strides[axis];return std::size_t(stride<0 ? -stride : stride);
    };
    std::stable_sort(result.axes.begin(),result.axes.end(),[&](auto a,auto b){return magnitude(a)>magnitude(b);});
    std::stable_sort(result.output_axes.begin(),result.output_axes.end(),[&](auto a,auto b){return strides[a]>strides[b];});
    result.work=result.count>limit/result.volume ? limit : result.count*result.volume;
    return result;
}

template<class Value,class Getter> Value pairwise(std::size_t begin,std::size_t count,const Getter& get) {
    if(count<8) {Value sum=Value(-0.0);for(std::size_t i=0;i<count;++i)sum=Value(sum+get(begin+i));return sum;}
    if(count<=128) {
        Value lanes[8];for(std::size_t j=0;j<8;++j)lanes[j]=get(begin+j);
        std::size_t i=8;
        for(;i+8<=count;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=Value(lanes[j]+get(begin+i+j));
        Value sum=Value(Value(Value(lanes[0]+lanes[1])+Value(lanes[2]+lanes[3]))+
                        Value(Value(lanes[4]+lanes[5])+Value(lanes[6]+lanes[7])));
        for(;i<count;++i)sum=Value(sum+get(begin+i));return sum;
    }
    auto half=count/2;half-=half%8;
    return Value(pairwise<Value>(begin,half,get)+pairwise<Value>(begin+half,count-half,get));
}

template<class Input,bool Boolean,bool Half,class Value> Value read(const unsigned char* source,bool swap) {
    Input item;
    if(swap && sizeof(Input)>1) {
        std::array<unsigned char,sizeof(Input)> bytes;
        for(std::size_t i=0;i<sizeof(Input);++i)bytes[i]=source[sizeof(Input)-1-i];
        std::memcpy(&item,bytes.data(),sizeof(Input));
    } else std::memcpy(&item,source,sizeof(Input));
    if constexpr(Half)return Value(detail::decode_half(std::uint16_t(item)));
    else if constexpr(Boolean)return Value(item!=0);
    else return Value(item);
}

template<class Input,class Value,bool Boolean=false,bool Half=false> bool run(const LocalMeanView& image,
    const std::vector<std::size_t>& factors,const void* constant,void* output,
    const std::vector<std::size_t>& strides,const Plan& geometry) {
    if(!geometry.count)return true;
    const auto rank=image.shape.size();const auto* bytes=static_cast<const unsigned char*>(image.data);
    const auto fill=geometry.padded ? read<Input,Boolean,Half,Value>(static_cast<const unsigned char*>(constant),image.swap) : Value(0);
    const auto bound=std::numeric_limits<Value>::max()/Value(geometry.volume);
    if(!std::isfinite(fill) || std::abs(fill)>bound)return false;
    const auto workers=detail::active_workers(geometry.work);
    bool linear=geometry.volume==1;std::size_t input_extent=sizeof(Input);
    if(linear)for(std::size_t position=rank;position--;) {
        const auto axis=geometry.output_axes[position];
        if(image.shape[axis]>1 && image.strides[axis]!=std::ptrdiff_t(input_extent))linear=false;
        input_extent*=image.shape[axis];
    }
    const bool vectorized=has_avx2();(void)vectorized;(void)linear;
    std::vector<unsigned char> supported(std::size_t(workers),1);
    detail::partition(geometry.count,workers,[&](auto begin,auto end,auto worker) {
        std::vector<std::size_t> start(rank);
        for(auto index=begin;index<end;++index) {
#ifdef SKIMAGEXX_AVX2
            if(linear && !image.swap && vectorized) {
                bool valid=true;
                const auto done=simd::downscale_row(image.data,output,1,geometry.count,0,0,1,1,
                    index,end,image.type,Half,0.,double(bound),valid);
                if(!valid)supported[worker]=0;
                if(done>index) {index=done-1;continue;}
            }
            if(rank==2 && !image.swap && vectorized) {
                const auto inner=geometry.output_axes.back(),outer=geometry.output_axes.front();
                const auto output_columns=geometry.blocks[inner];
                if(image.strides[inner]==std::ptrdiff_t(sizeof(Input)) && strides[inner]==(Half ? 2 : sizeof(Value))) {
                    const auto row=index/output_columns,column=index%output_columns;
                    const auto stop=std::min(output_columns,column+end-index);
                    bool valid=true;
                    const auto done=simd::downscale_row(image.data,static_cast<unsigned char*>(output)+row*strides[outer],
                        image.shape[outer],image.shape[inner],image.strides[outer],row,factors[outer],factors[inner],
                        column,stop,image.type,Half,double(fill),double(bound),valid);
                    if(!valid)supported[worker]=0;
                    if(done>column) {index+=done-column-1;continue;}
                }
            }
#endif
            auto remainder=index;std::size_t target=0;
            for(std::size_t position=rank;position--;) {
                const auto axis=geometry.output_axes[position];
                const auto coordinate=remainder%geometry.blocks[axis];remainder/=geometry.blocks[axis];
                start[axis]=coordinate*factors[axis];target+=coordinate*strides[axis];
            }
            auto sample=[&](std::ptrdiff_t offset) {
                const auto value=read<Input,Boolean,Half,Value>(bytes+offset,image.swap);
                if constexpr(std::is_floating_point_v<Input> || Half)
                    if(!std::isfinite(value) || std::abs(value)>bound)supported[worker]=0;
                return value;
            };
            auto reduce=[&](auto&& self,std::size_t depth,std::ptrdiff_t offset,bool valid)->Value {
                if(depth==rank)return valid ? sample(offset) : fill;
                const auto axis=geometry.axes[depth];
                return pairwise<Value>(0,factors[axis],[&](std::size_t position) {
                    const auto coordinate=start[axis]+position;
                    const bool inside=valid && coordinate<image.shape[axis];
                    const auto next=inside ? offset+std::ptrdiff_t(coordinate)*image.strides[axis] : offset;
                    return self(self,depth+1,next,inside);
                });
            };
            Value block;
            if(rank==2) {
                const auto outer=geometry.axes[0],inner=geometry.axes[1];
                block=pairwise<Value>(0,factors[outer],[&](std::size_t y) {
                    const auto row=start[outer]+y;
                    const bool valid_row=row<image.shape[outer];
                    const auto base=valid_row ? std::ptrdiff_t(row)*image.strides[outer] : 0;
                    return pairwise<Value>(0,factors[inner],[&](std::size_t x) {
                        const auto column=start[inner]+x;
                        return valid_row && column<image.shape[inner]
                            ? sample(base+std::ptrdiff_t(column)*image.strides[inner]) : fill;
                    });
                });
            } else if(rank==1) {
                block=pairwise<Value>(0,factors[0],[&](std::size_t x) {
                    const auto coordinate=start[0]+x;
                    return coordinate<image.shape[0] ? sample(std::ptrdiff_t(coordinate)*image.strides[0]) : fill;
                });
            } else block=reduce(reduce,0,0,true);
            const auto sum=Value(Value(0)+block);
            const auto mean=Value(double(sum)/double(geometry.volume));
            if(!std::isfinite(mean))supported[worker]=0;
            if constexpr(Half) {
                const auto word=detail::encode_half(float(mean));
                std::memcpy(static_cast<unsigned char*>(output)+target,&word,2);
            } else std::memcpy(static_cast<unsigned char*>(output)+target,&mean,sizeof(Value));
        }
    });
    return std::all_of(supported.begin(),supported.end(),[](auto value){return value!=0;});
}
}
bool downscale_local_mean_into(const LocalMeanView& image,const std::vector<std::size_t>& factors,
    const void* constant,void* output,const std::vector<std::size_t>& strides) {
    const auto geometry=plan(image,factors,output,strides,constant);
    if(image.half)return run<std::uint16_t,float,false,true>(image,factors,constant,output,strides,geometry);
    switch(image.type) {
    case PixelType::boolean:return run<std::uint8_t,double,true>(image,factors,constant,output,strides,geometry);
#define RAW(tag,Input,Value) case PixelType::tag:return run<Input,Value>(image,factors,constant,output,strides,geometry);
    RAW(uint8,std::uint8_t,double) RAW(int8,std::int8_t,double)
    RAW(uint16,std::uint16_t,double) RAW(int16,std::int16_t,double)
    RAW(uint32,std::uint32_t,double) RAW(int32,std::int32_t,double)
    RAW(uint64,std::uint64_t,double) RAW(int64,std::int64_t,double)
    RAW(float32,float,float) RAW(float64,double,double)
#undef RAW
    }
    throw std::invalid_argument("unsupported local-mean input type");
}
}
