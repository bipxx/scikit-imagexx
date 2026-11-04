#include "skimagexx/regionprops_state.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>

namespace skimagexx::detail {
namespace {
template<class F> void integer_type(PixelType type,F call) {
    switch(type) {
        case PixelType::uint8:call(std::uint8_t{});break;
        case PixelType::int8:call(std::int8_t{});break;
        case PixelType::uint16:call(std::uint16_t{});break;
        case PixelType::int16:call(std::int16_t{});break;
        case PixelType::uint32:call(std::uint32_t{});break;
        case PixelType::int32:call(std::int32_t{});break;
        case PixelType::uint64:call(std::uint64_t{});break;
        case PixelType::int64:call(std::int64_t{});break;
        default:break; // Excluded once by prepare.
    }
}
template<class T>T word(const char* pointer) {
    T value;std::memcpy(&value,pointer,sizeof(T));return value;
}
std::array<std::size_t,3> position(std::size_t i,const std::vector<std::size_t>& shape) {
    std::array<std::size_t,3> p{};
    for(auto axis=shape.size();axis-->0;){p[axis]=i%shape[axis];i/=shape[axis];}
    return p;
}
std::ptrdiff_t offset(const ComponentsView& view,const std::array<std::size_t,3>& p) {
    std::ptrdiff_t result=0;
    for(std::size_t a=0;a<view.shape.size();++a)result+=std::ptrdiff_t(p[a])*view.strides[a];
    return result;
}
}

bool prepare_regionprops_state(const ComponentsView& view,RegionPropsState& state) {
    const auto ndim=view.shape.size();
    if((ndim!=2 && ndim!=3) || ndim!=view.strides.size() || view.half || view.swap || !view.data)return false;
    std::size_t bytes=0;
    integer_type(view.type,[&](auto value){bytes=sizeof(value);});
    if(!bytes)return false;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t count=1,positive=0,negative=0;
    for(std::size_t a=0;a<ndim;++a) {
        const auto n=view.shape[a];const auto step=view.strides[a];
        if(!n || n>limit/count || step==std::numeric_limits<std::ptrdiff_t>::min())return false;
        count*=n;
        const auto magnitude=std::size_t(step<0?-step:step);
        auto& span=step<0?negative:positive;
        if(magnitude && n-1>(limit-span)/magnitude)return false;
        span+=(n-1)*magnitude;
    }
    if(positive>limit-(bytes-1) || count>limit/(ndim*sizeof(std::intptr_t)))return false;
    RegionPropsState prepared;prepared.labels=view;prepared.workers=get_num_threads();
    std::map<std::uint64_t,RegionPropsBounds> bounds;
    integer_type(view.type,[&](auto tag) {
        using T=decltype(tag);
        for(std::size_t i=0;i<count;++i) {
            const auto p=position(i,view.shape);
            const auto value=word<T>(static_cast<const char*>(view.data)+offset(view,p));
            if(value<=0)continue;
            const auto label=std::uint64_t(value);
            auto found=bounds.find(label);
            if(found==bounds.end()) {
                RegionPropsBounds first;first.label=label;first.start=p;first.stop=p;
                for(std::size_t a=0;a<ndim;++a)++first.stop[a];
                found=bounds.emplace(label,first).first;
            }
            auto& region=found->second;++region.initial_population;
            for(std::size_t a=0;a<ndim;++a) {
                region.start[a]=std::min(region.start[a],p[a]);
                region.stop[a]=std::max(region.stop[a],p[a]+1);
            }
        }
    });
    prepared.regions.reserve(bounds.size());
    for(const auto& entry:bounds)prepared.regions.push_back(entry.second);
    state=std::move(prepared);return true;
}

std::size_t regionprops_mask_into(const RegionPropsState& state,std::size_t index,std::uint8_t* output) {
    const auto& view=state.labels;const auto& region=state.regions[index];
    const auto ndim=view.shape.size(),last=ndim-1,width=region.stop[last]-region.start[last];
    const auto height=region.stop[ndim-2]-region.start[ndim-2];
    const auto rows=height*(ndim==3?region.stop[0]-region.start[0]:1);
    const auto workers=rows*width>=parallel_min?std::min(std::size_t(state.workers),rows):1;
    std::vector<std::size_t> populations(workers,0);
    integer_type(view.type,[&](auto tag) {
        using T=decltype(tag);const auto label=static_cast<T>(region.label);
        partition(rows,int(workers),[&](auto begin,auto end,auto worker) {
            std::size_t found=0;
            for(auto row=begin;row<end;++row) {
                auto p=region.start;
                p[ndim-2]+=row%height;if(ndim==3)p[0]+=row/height;
                const auto* input=static_cast<const char*>(view.data)+offset(view,p);
                auto* target=output+row*width;
                if(view.strides[last]==std::ptrdiff_t(sizeof(T))) {
                    // Caller-certified fresh mask, disjoint from label storage.
                    // Independent spans; OpenMP SIMD keeps integer comparison exact.
                    #pragma omp simd reduction(+:found)
                    for(std::size_t c=0;c<width;++c){const auto hit=word<T>(input+c*sizeof(T))==label;target[c]=std::uint8_t(hit);found+=hit;}
                } else {
                    for(std::size_t c=0;c<width;++c){const auto hit=word<T>(input+std::ptrdiff_t(c)*view.strides[last])==label;target[c]=std::uint8_t(hit);found+=hit;}
                }
            }
            populations[worker]=found;
        });
    });
    std::size_t count=0;for(auto n:populations)count+=n;return count;
}

std::size_t regionprops_mask_population(const std::uint8_t* mask,std::size_t count) noexcept {
    std::size_t found=0;for(std::size_t i=0;i<count;++i)found+=mask[i]!=0;return found;
}

void regionprops_coords_into(const RegionPropsState& state,std::size_t index,const std::uint8_t* mask,std::intptr_t* output) noexcept {
    const auto& region=state.regions[index];const auto ndim=state.labels.shape.size();
    std::array<std::size_t,3> shape{};std::size_t count=1;
    for(std::size_t a=0;a<ndim;++a){shape[a]=region.stop[a]-region.start[a];count*=shape[a];}
    for(std::size_t i=0;i<count;++i)if(mask[i]) {
        auto local=i;std::array<std::size_t,3> p{};
        for(auto a=ndim;a-->0;){p[a]=local%shape[a];local/=shape[a];}
        for(std::size_t a=0;a<ndim;++a)*output++=std::intptr_t(region.start[a]+p[a]);
    }
}
}
