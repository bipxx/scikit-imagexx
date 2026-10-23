#include "skimagexx/components.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
std::size_t count_of(const std::vector<std::size_t>& shape) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t count=1;
    for(auto length:shape) {
        if(length>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(std::intptr_t)/count)
            throw std::overflow_error("component dimensions overflow");
        count*=length;
    }
    return count;
}
template<class T> T read(const char* data,bool swap) {
    std::array<unsigned char,sizeof(T)> bytes;std::memcpy(bytes.data(),data,sizeof(T));
    if(swap)std::reverse(bytes.begin(),bytes.end());
    T result;std::memcpy(&result,bytes.data(),sizeof(T));return result;
}
template<class T> std::intptr_t index_value(T value,bool& invalid) {
    if constexpr(std::is_floating_point_v<T>) {
        using Bits=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        Bits bits;std::memcpy(&bits,&value,sizeof(T));
        constexpr Bits exponent=sizeof(T)==4 ? Bits(0x7f800000u) : Bits(0x7ff0000000000000ULL);
        if((bits&exponent)==exponent){invalid=true;return std::numeric_limits<std::intptr_t>::min();}
        const double wide=double(value);
        constexpr double limit=sizeof(std::intptr_t)==8 ? 0x1p63 : 0x1p31;
        if(wide < -limit || wide>=limit) {
            invalid=true;return std::numeric_limits<std::intptr_t>::min();
        }
        return static_cast<std::intptr_t>(wide);
    } else {
        using U=std::make_unsigned_t<std::intptr_t>;const U bits=static_cast<U>(value);
        std::intptr_t result;std::memcpy(&result,&bits,sizeof(result));return result;
    }
}
template<class T,bool Half=false> ComponentsConversionInfo prepare(const ComponentsView& view,std::intptr_t* output,std::size_t count) {
    const auto workers=detail::active_workers(count);
    struct alignas(64) Flag {bool invalid=false;};std::vector<Flag> flags(std::size_t(workers),Flag{});
    std::vector<std::size_t> cstrides(view.shape.size());std::size_t stride=1;bool contiguous=true;
    for(auto axis=view.shape.size();axis-->0;) {
        cstrides[axis]=stride;
        if(view.shape[axis]>1 && view.strides[axis]!=std::ptrdiff_t(stride*sizeof(T)))contiguous=false;
        stride*=view.shape[axis];
    }
    // Preallocate odometers outside OpenMP; no allocation/exception in a team.
    std::vector<std::vector<std::size_t>> coordinates(std::size_t(workers),std::vector<std::size_t>(view.shape.size()));
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
        auto& invalid=flags[id].invalid;
#ifdef SKIMAGEXX_AVX2
        if constexpr(!Half)if(contiguous && has_avx2())
            begin+=simd::components_prepare_index(static_cast<const char*>(view.data)+begin*sizeof(T),output+begin,end-begin,view.type,view.swap);
#endif
        auto& position=coordinates[id];std::ptrdiff_t offset=0;
        for(std::size_t axis=0;axis<view.shape.size();++axis) {
            position[axis]=(begin/cstrides[axis])%view.shape[axis];offset+=std::ptrdiff_t(position[axis])*view.strides[axis];
        }
        for(auto i=begin;i<end;++i) {
            const auto word=read<T>(static_cast<const char*>(view.data)+offset,view.swap);
            if constexpr(Half) {
                if((word&0x7c00)==0x7c00) {invalid=true;output[i]=std::numeric_limits<std::intptr_t>::min();}
                else output[i]=index_value(detail::decode_half(word),invalid);
            } else output[i]=index_value(word,invalid);
            for(auto axis=view.shape.size();axis-->0;) {
                if(position[axis]+1<view.shape[axis]){++position[axis];offset+=view.strides[axis];break;}
                offset-=std::ptrdiff_t(view.shape[axis]-1)*view.strides[axis];position[axis]=0;
            }
        }
    });
    ComponentsConversionInfo info;for(const auto& flag:flags)info.invalid_cast|=flag.invalid;return info;
}

struct Mask {std::uint64_t low=0,high=0;};
void set(Mask& mask,std::size_t bit) {if(bit<64)mask.low|=std::uint64_t{1}<<bit;else mask.high|=std::uint64_t{1}<<(bit-64);}
bool contains(Mask available,Mask wanted) {return (available.low&wanted.low)==wanted.low && (available.high&wanted.high)==wanted.high;}
struct Neighbor {std::ptrdiff_t row_delta;Mask required;std::size_t margin;};
struct RowPlan {
    std::vector<std::size_t> axes,strides;
    std::vector<Neighbor> neighbors;
};
RowPlan row_plan(const std::vector<std::size_t>& shape,int connectivity) {
    RowPlan plan;std::size_t stride=1;
    std::vector<std::size_t> full_strides(shape.size()-1);
    for(auto axis=shape.size()-1;axis-->0;){full_strides[axis]=stride;stride*=shape[axis];}
    for(std::size_t axis=0;axis+1<shape.size();++axis)if(shape[axis]>1){plan.axes.push_back(axis);plan.strides.push_back(full_strides[axis]);}
    if(plan.axes.size()>64)throw std::invalid_argument("component rank exceeds native mask capacity");
    std::function<void(std::size_t,int,std::ptrdiff_t,Mask)> visit;
    visit=[&](std::size_t axis,int cost,std::ptrdiff_t delta,Mask required) {
        if(axis==plan.axes.size()) {
            if(delta<0)plan.neighbors.push_back({delta,required,std::size_t(cost<connectivity)});
            return;
        }
        visit(axis+1,cost,delta,required);
        if(cost==connectivity)return;
        auto previous=required;set(previous,2*axis);
        visit(axis+1,cost+1,delta-std::ptrdiff_t(plan.strides[axis]),previous);
        auto next=required;set(next,2*axis+1);
        visit(axis+1,cost+1,delta+std::ptrdiff_t(plan.strides[axis]),next);
    };
    visit(0,0,0,{});return plan;
}
Mask available(std::size_t row,const std::vector<std::size_t>& shape,const RowPlan& plan) {
    Mask result;
    for(std::size_t i=0;i<plan.axes.size();++i) {
        const auto coordinate=(row/plan.strides[i])%shape[plan.axes[i]];
        if(coordinate)set(result,2*i);
        if(coordinate+1<shape[plan.axes[i]])set(result,2*i+1);
    }
    return result;
}
template<class T> std::size_t equal_prefix(const T* input,std::size_t count,T value,bool vectorized) {
    // Multivalued images mostly have single-pixel runs. Test those before
    // paying for an out-of-line SIMD scan; keep vector scans for long runs.
    if(!count || input[0]!=value)return 0;
    if(count==1 || input[1]!=value)return 1;
    std::size_t n=2;
#ifdef SKIMAGEXX_AVX2
    if(vectorized && count>=32/sizeof(T))n=simd::components_equal_prefix(input,count,&value,sizeof(T));
#else
    (void)vectorized;
#endif
    while(n<count && input[n]==value)++n;return n;
}
template<class T> void fill(T* output,std::size_t count,T value,bool vectorized) {
    if(count==1){*output=value;return;}
    std::size_t n=0;
#ifdef SKIMAGEXX_AVX2
    if(vectorized && count>=32/sizeof(T))n=simd::components_fill(output,count,&value,sizeof(T));
#else
    (void)vectorized;
#endif
    std::fill(output+n,output+count,value);
}
template<class Index> Index root(std::vector<Index>& forest,Index node) {
    while(forest[std::size_t(node)]!=node) {
        forest[std::size_t(node)]=forest[std::size_t(forest[std::size_t(node)])];node=forest[std::size_t(node)];
    }
    return node;
}
template<class Index> void unite(std::vector<Index>& forest,Index first,Index second) {
    first=root(forest,first);second=root(forest,second);
    if(first<second)forest[std::size_t(second)]=first;else if(second<first)forest[std::size_t(first)]=second;
}
template<class Index> Index canonical(std::vector<Index>& forest,Index node,std::size_t& count) {
    auto end=node;
    while(forest[std::size_t(end)]>0 && forest[std::size_t(end)]!=end)end=forest[std::size_t(end)];
    Index number;
    if(forest[std::size_t(end)]<0)number=-forest[std::size_t(end)];
    else {number=Index(++count);forest[std::size_t(end)]=-number;}
    while(node!=end && forest[std::size_t(node)]>0) {
        const auto next=forest[std::size_t(node)];forest[std::size_t(node)]=-number;node=next;
    }
    return number;
}
template<class Input,class Index> std::size_t label(const Input* input,Index* output,const std::vector<std::size_t>& shape,
    Input background,int connectivity) {
    if(shape.empty() || connectivity<1 || std::size_t(connectivity)>shape.size())throw std::invalid_argument("invalid component rank or connectivity");
    const auto count=count_of(shape);if(!count)return 0;
    if(!input || !output)throw std::invalid_argument("nonnull component input/output required");
    if(count>=std::size_t(std::numeric_limits<Index>::max()))throw std::overflow_error("component index capacity exceeded");
    const auto width=shape.back(),rows=count/width;const auto plan=row_plan(shape,connectivity);
    const bool vectorized=has_avx2();const auto workers=detail::active_workers(count);
    struct Interval {std::size_t begin=0,end=0;};std::vector<Interval> intervals(std::size_t(workers),Interval{});
    std::vector<Index> forest(count+1);
    auto connect=[&](std::size_t row,std::size_t first,std::size_t last,Input value,Index id,
                     const Neighbor& neighbor) {
        const auto other=std::size_t(std::ptrdiff_t(row)+neighbor.row_delta);
        const auto left=first>neighbor.margin ? first-neighbor.margin : 0;
        const auto right=std::min(width,last+neighbor.margin);
        std::size_t x=left;
        while(x<right) {
            const auto previous=output[other*width+x];
            const auto length=equal_prefix(output+other*width+x,right-x,previous,vectorized);
            if(previous && input[other*width+x]==value)unite(forest,id,previous);
            x+=length;
        }
    };
    const bool line=std::count_if(shape.begin(),shape.end(),[](auto length){return length>1;})<=1;
    if(line) {
        detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t worker) {
            intervals[worker]={begin,end};auto i=begin;
            while(i<end) {
                const auto value=input[i];const auto length=equal_prefix(input+i,end-i,value,vectorized);
                if(value==background)fill(output+i,length,Index{0},vectorized);
                else {const auto id=Index(i+1);forest[std::size_t(id)]=id;fill(output+i,length,id,vectorized);}
                i+=length;
            }
        });
        for(const auto& interval:intervals)
            if(interval.begin && interval.begin<interval.end && input[interval.begin]!=background && input[interval.begin]==input[interval.begin-1])
                unite(forest,output[interval.begin],output[interval.begin-1]);
    } else {
    // Whole rows belong to one worker. Local union operations never touch
    // another worker's roots; all cross-tile equivalences wait for the join.
    detail::partition(rows,workers,[&](std::size_t begin,std::size_t end,std::size_t worker) {
        intervals[worker]={begin,end};
        for(auto row=begin;row<end;++row) {
            const auto mask=available(row,shape,plan);std::size_t x=0;
            while(x<width) {
                const auto value=input[row*width+x];const auto length=equal_prefix(input+row*width+x,width-x,value,vectorized);
                if(value==background)fill(output+row*width+x,length,Index{0},vectorized);
                else {
                    const auto id=Index(row*width+x+1);forest[std::size_t(id)]=id;
                    fill(output+row*width+x,length,id,vectorized);
                    for(const auto& neighbor:plan.neighbors)
                        if(contains(mask,neighbor.required) && std::ptrdiff_t(row)+neighbor.row_delta>=std::ptrdiff_t(begin))
                            connect(row,x,x+length,value,id,neighbor);
                }
                x+=length;
            }
        }
    });
    for(const auto& interval:intervals)if(interval.begin) {
        // Only rows whose backward neighborhood crosses this tile's start
        // participate. Reconciliation is serial and cannot race local unions.
        for(const auto& neighbor:plan.neighbors) {
            const auto end=std::min(interval.end,interval.begin+std::size_t(-neighbor.row_delta));
            for(auto row=interval.begin;row<end;++row) {
                const auto mask=available(row,shape,plan);if(!contains(mask,neighbor.required))continue;
                std::size_t x=0;
                while(x<width) {
                    const auto value=input[row*width+x];const auto length=equal_prefix(input+row*width+x,width-x,value,vectorized);
                    if(value!=background)connect(row,x,x+length,value,output[row*width+x],neighbor);
                    x+=length;
                }
            }
        }
    }
    }
    std::size_t number=0,i=0;
    while(i<count) {
        const auto id=output[i];const auto length=equal_prefix(output+i,count-i,id,vectorized);
        if(id) {const auto final=canonical(forest,id,number);if(final!=id)fill(output+i,length,final,vectorized);}
        i+=length;
    }
    return number;
}
}
ComponentsConversionInfo components_prepare_index_into(const ComponentsView& view,std::intptr_t* output) {
    if(view.shape.size()!=view.strides.size())throw std::invalid_argument("one component stride per dimension required");
    const auto count=count_of(view.shape);if(!count)return {};
    if(!view.data || !output)throw std::invalid_argument("nonnull component preparation storage required");
    if(view.half)return prepare<std::uint16_t,true>(view,output,count);
    switch(view.type) {
        #define PREP(tag,T) case PixelType::tag:return prepare<T>(view,output,count);
        PREP(uint8,std::uint8_t) PREP(int8,std::int8_t) PREP(uint16,std::uint16_t) PREP(int16,std::int16_t)
        PREP(uint32,std::uint32_t) PREP(int32,std::int32_t) PREP(uint64,std::uint64_t) PREP(int64,std::int64_t)
        PREP(float32,float) PREP(float64,double)
        #undef PREP
        default:throw std::invalid_argument("real non-boolean component input required");
    }
}
void components_prepare_bool_into(const ComponentsView& view,std::uint8_t* output) {
    if(view.type!=PixelType::boolean || view.half || view.shape.size()!=view.strides.size())throw std::invalid_argument("boolean component input required");
    const auto count=count_of(view.shape);if(!count)return;
    if(!view.data || !output)throw std::invalid_argument("nonnull component preparation storage required");
    const auto workers=detail::active_workers(count);
    std::vector<std::size_t> cstrides(view.shape.size());std::size_t stride=1;bool contiguous=true;
    for(auto axis=view.shape.size();axis-->0;) {cstrides[axis]=stride;if(view.shape[axis]>1 && view.strides[axis]!=std::ptrdiff_t(stride))contiguous=false;stride*=view.shape[axis];}
    std::vector<std::vector<std::size_t>> coordinates(std::size_t(workers),std::vector<std::size_t>(view.shape.size()));
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
#ifdef SKIMAGEXX_AVX2
        if(contiguous && has_avx2())begin+=simd::components_prepare_bool(static_cast<const std::uint8_t*>(view.data)+begin,output+begin,end-begin);
#endif
        auto& position=coordinates[id];std::ptrdiff_t offset=0;
        for(std::size_t axis=0;axis<view.shape.size();++axis){position[axis]=(begin/cstrides[axis])%view.shape[axis];offset+=std::ptrdiff_t(position[axis])*view.strides[axis];}
        for(auto i=begin;i<end;++i) {
            output[i]=*(static_cast<const std::uint8_t*>(view.data)+offset)!=0;
            for(auto axis=view.shape.size();axis-->0;) {
                if(position[axis]+1<view.shape[axis]){++position[axis];offset+=view.strides[axis];break;}
                offset-=std::ptrdiff_t(view.shape[axis]-1)*view.strides[axis];position[axis]=0;
            }
        }
    });
}
std::size_t components_label_index_into(const std::intptr_t* input,std::intptr_t* output,const std::vector<std::size_t>& shape,std::intptr_t background,int connectivity) {
    if(shape.empty() || shape.size()>3)throw std::invalid_argument("non-boolean component rank must be1-3");
    return label(input,output,shape,background,connectivity);
}
std::size_t components_label_bool_into(const std::uint8_t* input,void* output,bool wide,const std::vector<std::size_t>& shape,std::uint8_t background,int connectivity) {
    if(background>1)throw std::invalid_argument("boolean component background must be0or1");
    if(wide)return label(input,static_cast<std::intptr_t*>(output),shape,background,connectivity);
    return label(input,static_cast<std::int32_t*>(output),shape,background,connectivity);
}
}
