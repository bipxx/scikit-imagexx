#include "skimagexx/component_consumers.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

namespace skimagexx {
namespace {
std::size_t count_of(const std::vector<std::size_t>& shape) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t count=1;
    for(auto n:shape) {if(n>std::size_t(PTRDIFF_MAX)/8/count)throw std::overflow_error("consumer dimensions overflow");count*=n;}
    return count;
}
template<class F> void raster(const ConsumerRaster& view,F call) {
    if(view.shape.size()!=view.strides.size() || (view.itemsize!=1 && view.itemsize!=2 && view.itemsize!=4 && view.itemsize!=8))throw std::invalid_argument("invalid consumer raster");
    const auto count=count_of(view.shape);if(!count)return;
    if(!view.data)throw std::invalid_argument("nonnull consumer raster required");
    auto workers=detail::active_workers(count);
    std::vector<std::pair<std::size_t,std::size_t>> axes;
    for(std::size_t axis=0;axis<view.shape.size();++axis)if(view.shape[axis]>1)axes.emplace_back(std::size_t(std::abs(view.strides[axis])),view.shape[axis]);
    std::sort(axes.begin(),axes.end());std::size_t span=view.itemsize;
    for(const auto& axis:axes){if(axis.first<span){workers=1;break;}span+=(axis.second-1)*axis.first;}
    std::vector<std::vector<std::size_t>> positions(std::size_t(workers),std::vector<std::size_t>(view.shape.size()));
    std::vector<std::size_t> cstrides(view.shape.size());std::size_t stride=1;bool contiguous=true;
    for(auto axis=view.shape.size();axis-->0;) {cstrides[axis]=stride;if(view.shape[axis]>1 && view.strides[axis]!=std::ptrdiff_t(stride*view.itemsize))contiguous=false;stride*=view.shape[axis];}
    detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
        auto invoke=[&](std::size_t i,char* p){if constexpr(std::is_invocable_v<F,std::size_t,char*,std::size_t>)call(i,p,worker);else call(i,p);};
        if(contiguous) {for(auto i=begin;i<end;++i)invoke(i,static_cast<char*>(view.data)+i*view.itemsize);return;}
        auto& position=positions[worker];std::ptrdiff_t offset=0;
        for(std::size_t axis=0;axis<view.shape.size();++axis) {position[axis]=begin/cstrides[axis]%view.shape[axis];offset+=std::ptrdiff_t(position[axis])*view.strides[axis];}
        for(auto i=begin;i<end;++i) {
            invoke(i,static_cast<char*>(view.data)+offset);
            for(auto axis=view.shape.size();axis-->0;) {
                if(position[axis]+1<view.shape[axis]) {++position[axis];offset+=view.strides[axis];break;}
                offset-=std::ptrdiff_t(view.shape[axis]-1)*view.strides[axis];position[axis]=0;
            }
        }
    });
}
std::uint64_t word(const void* pointer,std::size_t bytes,bool swap) {
    std::uint64_t value=0;std::memcpy(&value,pointer,bytes);
    if(swap) {std::uint64_t result=0;for(std::size_t b=0;b<bytes;++b){result=(result<<8)|(value&255);value>>=8;}return result;}
    return value;
}
void sort_keys(std::vector<std::uint64_t>& keys) {
    const auto count=keys.size();const auto workers=detail::active_workers(count);
    if(workers==1){std::sort(keys.begin(),keys.end());return;}
    const auto block=(count+std::size_t(workers)-1)/std::size_t(workers);std::vector<std::uint64_t> scratch(count);
    detail::partition(std::size_t(workers),workers,[&](auto begin,auto end,auto) {
        for(auto job=begin;job<end;++job){const auto first=std::min(count,job*block),last=std::min(count,(job+1)*block);std::sort(keys.begin()+first,keys.begin()+last);}
    });
    for(std::size_t step=1;step<std::size_t(workers);step*=2) {
        const auto jobs=(std::size_t(workers)+2*step-1)/(2*step);
        detail::partition(jobs,workers,[&](auto begin,auto end,auto) {
            for(auto job=begin;job<end;++job) {
                const auto first=std::min(count,job*2*step*block),middle=std::min(count,(job*2*step+step)*block),last=std::min(count,(job+1)*2*step*block);
                std::merge(keys.begin()+first,keys.begin()+middle,keys.begin()+middle,keys.begin()+last,scratch.begin()+first);
            }
        });keys.swap(scratch);
    }
}
}
bool consumer_min_negative(const ComponentsView& input) {
    const auto count=count_of(input.shape);if(!count)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    std::size_t bytes;bool floating=false,signed_integer=false;
    using P=PixelType;
    if(input.half){bytes=2;floating=true;}
    else switch(input.type) {
        case P::boolean:case P::uint8:bytes=1;break;case P::int8:bytes=1;signed_integer=true;break;
        case P::uint16:bytes=2;break;case P::int16:bytes=2;signed_integer=true;break;
        case P::uint32:bytes=4;break;case P::int32:bytes=4;signed_integer=true;break;
        case P::uint64:bytes=8;break;case P::int64:bytes=8;signed_integer=true;break;
        case P::float32:bytes=4;floating=true;break;case P::float64:bytes=8;floating=true;break;
        default:throw std::invalid_argument("real consumer input required");
    }
    ConsumerRaster view{const_cast<void*>(input.data),input.shape,input.strides,bytes};
    const auto workers=detail::active_workers(count);struct alignas(64) Flags{bool negative=false,nan=false;};
    std::vector<Flags> flags(std::size_t(workers),Flags{});
    // Atomic-free reduction per logical chunk, including dynamic OpenMP teams.
    raster(view,[&](auto,const char* p,auto id) {
        const auto bits=word(p,bytes,input.swap),sign=std::uint64_t{1}<<(bytes*8-1);
        if(!floating){flags[id].negative|=signed_integer && (bits&sign);return;}
        const auto exponent=bytes==2 ? 0x7c00ULL : bytes==4 ? 0x7f800000ULL : 0x7ff0000000000000ULL;
        const auto mantissa=bytes==2 ? 0x3ffULL : bytes==4 ? 0x7fffffULL : 0xfffffffffffffULL;
        const bool nan=(bits&exponent)==exponent && (bits&mantissa);
        flags[id].nan|=nan;flags[id].negative|=(bits&sign) && (bits&~sign) && !nan;
    });
    bool has_negative=false,has_nan=false;for(const auto& f:flags){has_negative|=f.negative;has_nan|=f.nan;}return has_negative && !has_nan;
}
void consumer_truth_into(const std::intptr_t* input,std::uint8_t* output,std::size_t count,bool invert) {
    if(!count)return;
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::consumer_truth(input+begin,output+begin,end-begin,invert);
#endif
        for(auto i=begin;i<end;++i)output[i]=std::uint8_t((input[i]!=0)!=invert);
    });
}
void consumer_nonzero_into(const ComponentsView& input,std::uint8_t* output) {
    std::size_t bytes=1;bool floating=input.half;using P=PixelType;
    if(input.half)bytes=2;
    else switch(input.type) {
        case P::uint16:case P::int16:bytes=2;break;case P::uint32:case P::int32:bytes=4;break;
        case P::uint64:case P::int64:bytes=8;break;case P::float32:bytes=4;floating=true;break;
        case P::float64:bytes=8;floating=true;break;default:break;
    }
    ConsumerRaster source{const_cast<void*>(input.data),input.shape,input.strides,bytes};
    raster(source,[&](auto i,const char* p){auto bits=word(p,bytes,input.swap);if(floating)bits&=~(std::uint64_t{1}<<(8*bytes-1));output[i]=bits!=0;});
}
void consumer_bool_write(const std::uint8_t* input,const ConsumerRaster& output,bool invert) {
    if(output.itemsize!=1)throw std::invalid_argument("boolean consumer output required");
    if(!count_of(output.shape))return;
    bool contiguous=true;std::size_t stride=1;for(auto axis=output.shape.size();axis-->0;){if(output.shape[axis]>1 && output.strides[axis]!=std::ptrdiff_t(stride))contiguous=false;stride*=output.shape[axis];}
    if(contiguous) {
        detail::partition(count_of(output.shape),detail::active_workers(count_of(output.shape)),[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())begin+=simd::consumer_bool_write(input+begin,static_cast<std::uint8_t*>(output.data)+begin,end-begin,invert);
#endif
            for(auto i=begin;i<end;++i)static_cast<std::uint8_t*>(output.data)[i]=std::uint8_t(bool(input[i])!=invert);
        });return;
    }
    raster(output,[&](auto i,char* p){*p=char(bool(input[i])!=invert);});
}
ConsumerCounts consumer_count(const std::intptr_t* input,std::size_t count) {
    ConsumerCounts result;if(!count)return result;
    const auto workers=detail::active_workers(count);struct alignas(64) Stats{std::uint64_t maximum=0;bool negative=false;};
    std::vector<Stats> stats(std::size_t(workers),Stats{});
    detail::partition(count,workers,[&](auto begin,auto end,auto id){for(auto i=begin;i<end;++i){stats[id].negative|=input[i]<0;stats[id].maximum=std::max(stats[id].maximum,std::uint64_t(input[i]));}});
    for(const auto& s:stats){if(s.negative)throw std::invalid_argument("negative consumer labels");result.maximum=std::max(result.maximum,s.maximum);}
    result.empty=false;
    if(result.maximum<=16*1024*1024 && result.maximum<=count*4+1024) {
        const auto bins=std::size_t(result.maximum)+1;
        const auto histogram_workers=std::min(workers,std::max(1,int((64*1024*1024)/(bins*sizeof(std::intptr_t)))));
        std::vector<std::vector<std::intptr_t>> shards(static_cast<std::size_t>(histogram_workers),std::vector<std::intptr_t>(bins,0));
        detail::partition(count,histogram_workers,[&](auto begin,auto end,auto worker) {
            auto& counts=shards[worker];auto i=begin;
            while(i<end) {
                const auto value=input[i];std::size_t n=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())n=simd::components_equal_prefix(input+i,end-i,&value,sizeof(value));
#endif
                while(i+n<end && input[i+n]==value)++n;counts[std::size_t(value)]+=std::intptr_t(n);i+=n;
            }
        });
        result.keys.resize(bins);result.sizes.resize(bins);
        detail::partition(bins,workers,[&](auto begin,auto end,auto){for(auto k=begin;k<end;++k){result.keys[k]=k;for(const auto& shard:shards)result.sizes[k]+=shard[k];}});
    } else {
        std::vector<std::uint64_t> sorted(count);detail::partition(count,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)sorted[i]=std::uint64_t(input[i]);});sort_keys(sorted);
        result.keys.reserve(count);result.sizes.reserve(count);
        for(const auto key:sorted){if(result.keys.empty() || result.keys.back()!=key){result.keys.push_back(key);result.sizes.push_back(1);}else ++result.sizes.back();}
    }return result;
}
void consumer_filter_into(const std::intptr_t* labels,const ConsumerCounts& counts,double threshold,const ConsumerRaster& out) {
    if(counts.keys.size()!=counts.sizes.size())throw std::invalid_argument("invalid consumer counts");
    const bool dense=!counts.keys.empty() && counts.keys.back()+1==counts.keys.size();
    raster(out,[&](auto i,char* p) {
        const auto key=std::uint64_t(labels[i]);std::intptr_t size=0;
        if(dense) {if(key<counts.sizes.size())size=counts.sizes[std::size_t(key)];}
        else {const auto found=std::lower_bound(counts.keys.begin(),counts.keys.end(),key);if(found!=counts.keys.end() && *found==key)size=counts.sizes[std::size_t(found-counts.keys.begin())];}
        if(double(size)<=threshold)std::memset(p,0,out.itemsize);
    });
}
void consumer_border_into(std::uint8_t* out,const std::vector<std::size_t>& shape,const std::vector<std::size_t>& stops,const std::vector<std::size_t>& starts) {
    if(shape.size()!=stops.size() || shape.size()!=starts.size())throw std::invalid_argument("one border interval per axis required");
    const auto count=count_of(shape);std::vector<std::size_t> strides(shape.size());std::size_t stride=1;
    for(auto axis=shape.size();axis-->0;){strides[axis]=stride;stride*=shape[axis];}
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){bool border=false;for(std::size_t axis=0;axis<shape.size();++axis){const auto x=i/strides[axis]%shape[axis];border|=x<stops[axis] || x>=starts[axis];}out[i]=border;}});
}
std::vector<std::uint8_t> consumer_border_ids(const std::intptr_t* labels,const std::uint8_t* border,std::size_t count,std::size_t number) {
    if(number>count || (count && (!labels || !border)))throw std::invalid_argument("invalid consumer component labels");
    const auto workers=detail::active_workers(count);std::vector<std::vector<std::uint8_t>> shards(std::size_t(workers),std::vector<std::uint8_t>(number+1));
    std::vector<std::uint8_t> invalid(std::size_t(workers),0);
    detail::partition(count,workers,[&](auto begin,auto end,auto id){for(auto i=begin;i<end;++i){if(labels[i]<0 || std::uint64_t(labels[i])>number)invalid[id]=1;else if(border[i])shards[id][std::size_t(labels[i])]=1;}});
    if(std::find(invalid.begin(),invalid.end(),1)!=invalid.end())throw std::invalid_argument("invalid consumer component labels");
    std::vector<std::uint8_t> result(number+1);
    for(const auto& shard:shards){std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())i=simd::consumer_or(shard.data(),result.data(),result.size());
#endif
        for(;i<result.size();++i)result[i]|=shard[i];
    }return result;
}
void consumer_clear_into(const std::intptr_t* labels,const std::uint8_t* remove,std::size_t number,const void* value,const ConsumerRaster& out) {
    const auto count=count_of(out.shape);if(number>count || (count && (!labels || !remove || !value)))throw std::invalid_argument("invalid consumer clear storage");
    std::vector<std::uint8_t> invalid(std::size_t(detail::active_workers(count)),0);
    raster(out,[&](auto i,char* p,auto id){if(labels[i]<0 || std::uint64_t(labels[i])>number)invalid[id]=1;else if(remove[std::size_t(labels[i])])std::memcpy(p,value,out.itemsize);});
    if(std::find(invalid.begin(),invalid.end(),1)!=invalid.end())throw std::invalid_argument("invalid consumer component labels");
}
std::vector<std::uint64_t> consumer_sorted_unique(const std::uint64_t* input,std::size_t count) {
    if(!count)return {};
    std::vector<std::uint64_t> result(count);detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){std::copy(input+begin,input+end,result.begin()+begin);});
    sort_keys(result);result.erase(std::unique(result.begin(),result.end()),result.end());return result;
}
void consumer_map_into(const std::uint64_t* index,std::size_t count,const std::uint64_t* keys,const void* values,std::size_t key_count,std::size_t bytes,unsigned bits,void* output) {
    if(bits!=8 && bits!=16 && bits!=32 && bits!=64)throw std::invalid_argument("integer map index width required");
    if((count && (!index || !output)) || (key_count && (!keys || !values)) || !bytes)throw std::invalid_argument("invalid consumer map storage");
    const auto mask=bits==64 ? ~std::uint64_t{0} : (std::uint64_t{1}<<bits)-1;
    std::unordered_map<std::uint64_t,std::size_t> map;map.reserve(key_count);
    for(std::size_t k=0;k<key_count;++k)map[keys[k]&mask]=k;
    const auto& lookup=map;
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){auto* destination=static_cast<char*>(output)+i*bytes;const auto found=lookup.find(index[i]&mask);if(found==lookup.end())std::memset(destination,0,bytes);else std::memcpy(destination,static_cast<const char*>(values)+found->second*bytes,bytes);}});
}
void consumer_dense_into(const std::uint64_t* keys,const void* values,std::size_t key_count,std::size_t bytes,void* output,std::size_t length) {
    if(length)detail::partition(length,detail::active_workers(length),[&](auto begin,auto end,auto){std::memset(static_cast<char*>(output)+begin*bytes,0,(end-begin)*bytes);});
    // Serial duplicate-key scatter preserves upstream last-write ordering.
    for(std::size_t k=0;k<key_count;++k){if(keys[k]>=length)throw std::out_of_range("dense map index out of bounds");std::memcpy(static_cast<char*>(output)+keys[k]*bytes,static_cast<const char*>(values)+k*bytes,bytes);}
}
void consumer_scatter_into(const std::uint64_t* index,const void* values,std::size_t count,std::size_t bytes,void* output,std::size_t length) {
    for(std::size_t i=0;i<count;++i){if(index[i]>=length)throw std::out_of_range("scatter index out of bounds");std::memcpy(static_cast<char*>(output)+index[i]*bytes,static_cast<const char*>(values)+i*bytes,bytes);}
}
}
