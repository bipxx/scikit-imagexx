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
std::ptrdiff_t offset(const ComponentsView& view,const std::array<std::size_t,3>& p) {
    std::ptrdiff_t result=0;
    for(std::size_t a=0;a<view.shape.size();++a)result+=std::ptrdiff_t(p[a])*view.strides[a];
    return result;
}
std::array<std::size_t,3> region_shape(const RegionPropsState& state,std::size_t index) {
    const auto& r=state.regions[index];std::array<std::size_t,3> shape{};
    for(std::size_t a=0;a<state.labels.shape.size();++a)shape[a]=r.stop[a]-r.start[a];
    return shape;
}
std::array<std::ptrdiff_t,3> c_strides(const std::array<std::size_t,3>& shape,std::size_t ndim) {
    std::array<std::ptrdiff_t,3> strides{};std::ptrdiff_t step=1;
    for(auto a=ndim;a-->0;){strides[a]=step;step*=std::ptrdiff_t(shape[a]);}
    return strides;
}
std::array<std::size_t,3> locality_order(const std::array<std::ptrdiff_t,3>& strides,std::size_t ndim) {
    std::array<std::size_t,3> axes{0,1,2};
    const auto magnitude=[](std::ptrdiff_t value){return value<0?std::size_t(-(value+1))+1:std::size_t(value);};
    // Last axis is the inner loop. At most three fixed axes, no per-pixel work.
    if(magnitude(strides[axes[0]])<magnitude(strides[axes[1]]))std::swap(axes[0],axes[1]);
    if(ndim==3) {
        if(magnitude(strides[axes[1]])<magnitude(strides[axes[2]]))std::swap(axes[1],axes[2]);
        if(magnitude(strides[axes[0]])<magnitude(strides[axes[1]]))std::swap(axes[0],axes[1]);
    }
    return axes;
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
        const auto consume=[&](const char* pointer,const std::array<std::size_t,3>& p) {
            const auto value=word<T>(pointer);
            if(value<=0)return;
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
        };
        std::array<std::ptrdiff_t,3> strides{};
        for(std::size_t a=0;a<ndim;++a)strides[a]=view.strides[a];
        const auto axes=locality_order(strides,ndim);
        const auto outer=axes[0],inner=axes[ndim-1];
        std::array<std::size_t,3> p{};
        const auto* base=static_cast<const char*>(view.data);
        for(p[outer]=0;p[outer]<view.shape[outer];++p[outer]) {
            const auto* plane=base+std::ptrdiff_t(p[outer])*strides[outer];
            if(ndim==2) {
                for(p[inner]=0;p[inner]<view.shape[inner];++p[inner])
                    consume(plane+std::ptrdiff_t(p[inner])*strides[inner],p);
            } else {
                const auto middle=axes[1];
                for(p[middle]=0;p[middle]<view.shape[middle];++p[middle]) {
                    const auto* row=plane+std::ptrdiff_t(p[middle])*strides[middle];
                    for(p[inner]=0;p[inner]<view.shape[inner];++p[inner])
                        consume(row+std::ptrdiff_t(p[inner])*strides[inner],p);
                }
            }
        }
    });
    prepared.regions.reserve(bounds.size());
    for(const auto& entry:bounds)prepared.regions.push_back(entry.second);
    state=std::move(prepared);return true;
}

std::size_t regionprops_mask_into(const RegionPropsState& state,std::size_t index,std::uint8_t* output) {
    return regionprops_mask_strided_into(state,index,output,c_strides(region_shape(state,index),state.labels.shape.size()));
}

std::size_t regionprops_mask_strided_into(const RegionPropsState& state,std::size_t index,
    std::uint8_t* output,const std::array<std::ptrdiff_t,3>& output_strides) {
    const auto& view=state.labels;const auto& region=state.regions[index];
    const auto ndim=view.shape.size();const auto shape=region_shape(state,index);
    const auto axes=locality_order(output_strides,ndim);
    const auto outer=axes[0],inner=axes[ndim-1],middle=ndim==3?axes[1]:axes[0];
    const auto width=shape[inner],height=shape[middle];
    const auto rows=shape[outer]*(ndim==3?height:1);
    const auto workers=std::min(std::size_t(active_workers(rows*width)),rows);
    const auto* base=static_cast<const char*>(view.data)+offset(view,region.start);
    std::size_t total=0;
    integer_type(view.type,[&](auto tag) {
        using T=decltype(tag);const auto label=static_cast<T>(region.label);
        const auto fill=[&](std::size_t begin,std::size_t end) {
            std::size_t found=0;
            for(auto row=begin;row<end;++row) {
                const auto major=ndim==3?row/height:row,minor=ndim==3?row%height:0;
                const auto* input=base+std::ptrdiff_t(major)*view.strides[outer]
                    +(ndim==3?std::ptrdiff_t(minor)*view.strides[middle]:0);
                auto* target=output+std::ptrdiff_t(major)*output_strides[outer]
                    +(ndim==3?std::ptrdiff_t(minor)*output_strides[middle]:0);
                if(view.strides[inner]==std::ptrdiff_t(sizeof(T))&&output_strides[inner]==1) {
                    // Caller-certified fresh mask, disjoint from label storage.
                    // Independent spans; OpenMP SIMD keeps integer comparison exact.
                    #if defined(_OPENMP) && !defined(_MSC_VER)
                    #pragma omp simd reduction(+:found)
                    #endif
                    for(std::size_t c=0;c<width;++c){const auto hit=word<T>(input+c*sizeof(T))==label;target[c]=std::uint8_t(hit);found+=hit;}
                } else {
                    for(std::size_t c=0;c<width;++c){const auto hit=word<T>(input+std::ptrdiff_t(c)*view.strides[inner])==label;target[std::ptrdiff_t(c)*output_strides[inner]]=std::uint8_t(hit);found+=hit;}
                }
            }
            return found;
        };
        if(workers==1)total=fill(0,rows);
        else {
            std::vector<std::size_t> populations(workers,0);
            partition(rows,int(workers),[&](auto begin,auto end,auto worker){populations[worker]=fill(begin,end);});
            for(auto n:populations)total+=n;
        }
    });
    return total;
}

std::size_t regionprops_mask_population(const std::uint8_t* mask,std::size_t count) noexcept {
    std::size_t found=0;for(std::size_t i=0;i<count;++i)found+=mask[i]!=0;return found;
}

void regionprops_coords_into(const RegionPropsState& state,std::size_t index,const std::uint8_t* mask,std::intptr_t* output) noexcept {
    const auto ndim=state.labels.shape.size();
    regionprops_coords_strided_into(state,index,mask,c_strides(region_shape(state,index),ndim),
        output,std::ptrdiff_t(ndim*sizeof(std::intptr_t)),sizeof(std::intptr_t));
}

void regionprops_coords_strided_into(const RegionPropsState& state,std::size_t index,
    const std::uint8_t* mask,const std::array<std::ptrdiff_t,3>& strides,
    std::intptr_t* output,std::ptrdiff_t point_step,std::ptrdiff_t dimension_step) noexcept {
    const auto& region=state.regions[index];const auto ndim=state.labels.shape.size();
    const auto shape=region_shape(state,index);auto* target=reinterpret_cast<char*>(output);bool first=true;
    const auto append=[&](std::size_t plane,std::size_t row,std::size_t column) {
        // Advance only when another point is emitted: with a negative caller
        // stride, an unused post-last pointer can lie before its allocation.
        if(!first)target+=point_step;
        first=false;
        const std::array<std::size_t,3> p=ndim==2?std::array<std::size_t,3>{row,column,0}
            :std::array<std::size_t,3>{plane,row,column};
        for(std::size_t a=0;a<ndim;++a) {
            const auto coordinate=std::intptr_t(region.start[a]+p[a]);
            std::memcpy(target+std::ptrdiff_t(a)*dimension_step,&coordinate,sizeof(coordinate));
        }
    };
    if(ndim==2) {
        for(std::size_t r=0;r<shape[0];++r) {
            const auto* row=mask+std::ptrdiff_t(r)*strides[0];
            for(std::size_t c=0;c<shape[1];++c)if(row[std::ptrdiff_t(c)*strides[1]])append(0,r,c);
        }
    } else {
        for(std::size_t z=0;z<shape[0];++z)for(std::size_t r=0;r<shape[1];++r) {
            const auto* row=mask+std::ptrdiff_t(z)*strides[0]+std::ptrdiff_t(r)*strides[1];
            for(std::size_t c=0;c<shape[2];++c)if(row[std::ptrdiff_t(c)*strides[2]])append(z,r,c);
        }
    }
}
}
