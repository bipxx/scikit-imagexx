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
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

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
        constexpr Bits exponent=[] {
            if constexpr(sizeof(T)==4)return Bits{0x7f800000u};
            else return Bits{0x7ff0000000000000ULL};
        }();
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
    const auto* input=static_cast<const char*>(view.data);
    std::size_t stride=1;
    bool contiguous=true;
    for(auto axis=view.shape.size();axis-->0;) {
        if(view.shape[axis]>1 && view.strides[axis]!=std::ptrdiff_t(stride*sizeof(T)))contiguous=false;
        stride*=view.shape[axis];
    }
    const auto convert=[&](const char* data,bool& invalid) {
        const auto word=read<T>(data,view.swap);
        if constexpr(Half) {
            if((word&0x7c00)==0x7c00) {
                invalid=true;
                return std::numeric_limits<std::intptr_t>::min();
            }
            return index_value(detail::decode_half(word),invalid);
        } else return index_value(word,invalid);
    };
    if(contiguous && workers==1) {
        std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
        if constexpr(!Half)if(count>=4 && has_avx2())
            i=simd::components_prepare_index(input,output,count,view.type,view.swap);
#endif
        ComponentsConversionInfo info;
        for(;i<count;++i)output[i]=convert(input+i*sizeof(T),info.invalid_cast);
        return info;
    }
    struct alignas(64) Flag {
        bool invalid=false;
        unsigned char padding[64-sizeof(bool)]{};
    };
    std::vector<Flag> flags(std::size_t(workers),Flag{});
    std::vector<std::size_t> cstrides(view.shape.size());stride=1;
    for(auto axis=view.shape.size();axis-->0;) {
        cstrides[axis]=stride;
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
            output[i]=convert(input+offset,invalid);
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
    if(count==2 || input[2]!=value)return 2;
    if(count==3 || input[3]!=value)return 3;
    std::size_t n=4;
#ifdef SKIMAGEXX_AVX2
    if(vectorized && count>=8*(32/sizeof(T)))n=simd::components_equal_prefix(input,count,&value,sizeof(T));
#else
    (void)vectorized;
#endif
    while(n<count && input[n]==value)++n;
    return n;
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
template<class Index> Index root(Index* forest,Index node) {
    while(forest[std::size_t(node)]!=node) {
        forest[std::size_t(node)]=forest[std::size_t(forest[std::size_t(node)])];node=forest[std::size_t(node)];
    }
    return node;
}
template<class Index> void unite(Index* forest,Index first,Index second) {
    first=root(forest,first);second=root(forest,second);
    if(first<second)forest[std::size_t(second)]=first;else if(second<first)forest[std::size_t(first)]=second;
}
template<class Index> Index canonical(Index* forest,Index node,std::size_t& count) {
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
template<class Input,class Index> std::size_t label_2d(const Input* input,Index* output,
    std::size_t rows,std::size_t width,Input background,bool has_background,bool diagonal,bool vectorized) {
    const auto count=rows*width;
    std::unique_ptr<std::uint32_t[]> storage(new std::uint32_t[count+1]);
    auto* const forest=storage.get();
    forest[0]=0;
    auto* ids=output;
    auto* keys=input;
    for(std::size_t y=0;y<rows;++y,ids+=width,keys+=width) {
        const auto* previous_keys=y ? keys-width : nullptr;
        const auto* previous_ids=y ? ids-width : nullptr;
        for(std::size_t x=0;x<width;) {
            const auto value=keys[x];
            const auto length=equal_prefix(keys+x,width-x,value,vectorized);
            const auto end=x+length;
            if(has_background && value==background) {
                fill(ids+x,length,Index{0},vectorized);
            } else {
                const auto id=std::uint32_t(ids-output+x+1);
                forest[id]=id;
                fill(ids+x,length,Index(id),vectorized);
                if(previous_keys) {
                    auto left=x-std::size_t(diagonal && x>0);
                    const auto right=end+std::size_t(diagonal && end<width);
                    while(left<right) {
                        if(previous_keys[left]!=value) {
                            ++left;
                            continue;
                        }
                        const auto previous=previous_ids[left];
                        unite(forest,id,std::uint32_t(previous));
                        left+=equal_prefix(previous_ids+left,right-left,previous,vectorized);
                    }
                }
            }
            x=end;
        }
    }
    // Union links always point to an earlier run, whose final ID is ready
    // when reached in C order. Only run starts have initialized parents.
    std::uint32_t number=0;
    for(std::size_t i=0;i<count;) {
        const auto id=std::uint32_t(output[i]);
        if(id)forest[id]=forest[id]==id ? ++number : forest[forest[id]];
        i+=equal_prefix(output+i,count-i,Index(id),vectorized);
    }
    for(std::size_t i=0;i<count;++i)output[i]=Index(forest[std::uint32_t(output[i])]);
    return number;
}
template<class Input,class Index> std::size_t label(const Input* input,Index* output,const std::vector<std::size_t>& shape,
    Input background,int connectivity,bool has_background=true) {
    if(shape.empty() || connectivity<1 || std::size_t(connectivity)>shape.size())throw std::invalid_argument("invalid component rank or connectivity");
    const auto count=count_of(shape);if(!count)return 0;
    if(!input || !output)throw std::invalid_argument("nonnull component input/output required");
    if(count>=std::size_t(std::numeric_limits<Index>::max()))throw std::overflow_error("component index capacity exceeded");
    const bool vectorized=has_avx2();const auto workers=detail::active_workers(count);
    const auto width=shape.back();
    if(shape.size()==2 && workers==1 && count<std::size_t(std::numeric_limits<std::int32_t>::max()))
        return label_2d(input,output,shape[0],width,background,has_background,connectivity==2,vectorized);
    const auto rows=count/width;const auto plan=row_plan(shape,connectivity);
    struct Interval {std::size_t begin=0,end=0;};std::vector<Interval> intervals(std::size_t(workers),Interval{});
    // Only foreground run starts enter the forest. Each is initialized
    // before its first union, so background and interior slots need no fill.
    std::unique_ptr<Index[]> storage(new Index[count+1]);
    auto* const forest=storage.get();
    const auto is_background=[&](Input value){return has_background && value==background;};
    auto connect=[&](std::size_t row,std::size_t first,std::size_t last,Input value,Index id,
                     const Neighbor& neighbor) {
        const auto other=std::size_t(std::ptrdiff_t(row)+neighbor.row_delta);
        const auto left=first>neighbor.margin ? first-neighbor.margin : 0;
        const auto right=std::min(width,last+neighbor.margin);
        const auto* keys=input+other*width;
        const auto* previous_ids=output+other*width;
        for(auto x=left;x<right;) {
            if(keys[x]!=value) {
                ++x;
                continue;
            }
            const auto previous=previous_ids[x];
            if(previous)unite(forest,id,previous);
            x+=equal_prefix(previous_ids+x,right-x,previous,vectorized);
        }
    };
    const bool line=std::count_if(shape.begin(),shape.end(),[](auto length){return length>1;})<=1;
    if(line) {
        detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t worker) {
            intervals[worker]={begin,end};auto i=begin;
            while(i<end) {
                const auto value=input[i];const auto length=equal_prefix(input+i,end-i,value,vectorized);
                if(is_background(value))fill(output+i,length,Index{0},vectorized);
                else {const auto id=Index(i+1);forest[std::size_t(id)]=id;fill(output+i,length,id,vectorized);}
                i+=length;
            }
        });
        for(const auto& interval:intervals)
            if(interval.begin && interval.begin<interval.end && !is_background(input[interval.begin]) && input[interval.begin]==input[interval.begin-1])
                unite(forest,output[interval.begin],output[interval.begin-1]);
    } else {
    // Whole rows belong to one worker. Local union operations never touch
    // another worker's roots; all cross-tile equivalences wait for the join.
    detail::partition(rows,workers,[&](std::size_t begin,std::size_t end,std::size_t worker) {
        intervals[worker]={begin,end};
        for(auto row=begin;row<end;++row) {
            const auto mask=available(row,shape,plan);
            // All runs in this row share the same predecessor rows.
            std::array<const Neighbor*,13> nearby;
            std::size_t nearby_count=0;
            const bool small_plan=plan.neighbors.size()<=nearby.size();
            const auto precedes=[&](const Neighbor& neighbor) {
                return contains(mask,neighbor.required) &&
                    std::ptrdiff_t(row)+neighbor.row_delta>=std::ptrdiff_t(begin);
            };
            if(small_plan)for(const auto& neighbor:plan.neighbors)
                if(precedes(neighbor))nearby[nearby_count++]=&neighbor;
            std::size_t x=0;
            while(x<width) {
                const auto value=input[row*width+x];const auto length=equal_prefix(input+row*width+x,width-x,value,vectorized);
                if(is_background(value))fill(output+row*width+x,length,Index{0},vectorized);
                else {
                    const auto id=Index(row*width+x+1);forest[std::size_t(id)]=id;
                    fill(output+row*width+x,length,id,vectorized);
                    if(small_plan) {
                        for(std::size_t j=0;j<nearby_count;++j)connect(row,x,x+length,value,id,*nearby[j]);
                    } else {
                        for(const auto& neighbor:plan.neighbors)
                            if(precedes(neighbor))connect(row,x,x+length,value,id,neighbor);
                    }
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
                    if(!is_background(value))connect(row,x,x+length,value,output[row*width+x],neighbor);
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
namespace {
template<class T> std::size_t label_dense_integer(const void* input,std::intptr_t* output,
    const std::vector<std::size_t>& shape,std::intptr_t background,int connectivity) {
    const auto key=static_cast<T>(background);
    bool invalid=false;
    // The cast round trip is the same one used by index preparation. For
    // example, UINT64_MAX matches background=-1 on a 64-bit platform.
    const bool has_background=index_value(key,invalid)==background;
    return label(static_cast<const T*>(input),output,shape,key,connectivity,has_background);
}
}
// Binding-only entry point for aligned, native-endian C integer images.
std::size_t components_label_dense_integer_into(const void* input,std::intptr_t* output,
    const std::vector<std::size_t>& shape,std::intptr_t background,int connectivity,PixelType type) {
    if(shape.empty() || shape.size()>3)throw std::invalid_argument("non-boolean component rank must be1-3");
    switch(type) {
        #define LABEL(tag,T) case PixelType::tag: if constexpr(sizeof(T)<=sizeof(std::intptr_t)) \
            return label_dense_integer<T>(input,output,shape,background,connectivity); else break;
        LABEL(int8,std::int8_t) LABEL(uint8,std::uint8_t)
        LABEL(int16,std::int16_t) LABEL(uint16,std::uint16_t)
        LABEL(int32,std::int32_t) LABEL(uint32,std::uint32_t)
        LABEL(int64,std::int64_t) LABEL(uint64,std::uint64_t)
        #undef LABEL
        default:break;
    }
    throw std::invalid_argument("unsupported dense component input dtype");
}
std::size_t components_label_bool_into(const std::uint8_t* input,void* output,bool wide,const std::vector<std::size_t>& shape,std::uint8_t background,int connectivity) {
    if(background>1)throw std::invalid_argument("boolean component background must be0or1");
    if(wide)return label(input,static_cast<std::intptr_t*>(output),shape,background,connectivity);
    return label(input,static_cast<std::int32_t*>(output),shape,background,connectivity);
}

namespace {
template<class F> decltype(auto) integer_width(std::size_t bytes,F visit) {
    switch(bytes) {
        case 1:return visit(std::uint8_t{});
        case 2:return visit(std::uint16_t{});
        case 4:return visit(std::uint32_t{});
        case 8:return visit(std::uint64_t{});
        default:throw std::invalid_argument("unsupported component integer width");
    }
}
}

// Binding-only kernels. The binding admits native C-order arrays and allocates
// disjoint outputs with the public API's dtype and shape.
std::vector<std::uint64_t> components_relabel_sequential_fast(const void* input,std::size_t count,
    std::size_t bytes,bool signed_input,std::unordered_map<std::uint64_t,std::uint64_t>& mapping) {
    std::vector<std::uint64_t> keys;
    const bool admitted=integer_width(bytes,[&](auto tag) {
        using T=decltype(tag);
        const auto* source=static_cast<const char*>(input);
        for(std::size_t i=0;i<count;++i) {
            const auto key=read<T>(source+i*sizeof(T),false);
            if(signed_input && key>std::numeric_limits<std::make_signed_t<T>>::max())return false;
            if(mapping.try_emplace(key,0).second)keys.push_back(key);
        }
        return true;
    });
    if(!admitted)return {};
    std::sort(keys.begin(),keys.end());
    return keys;
}

void components_relabel_write_fast(const void* input,std::size_t count,std::size_t bytes,
    const std::vector<std::uint64_t>& keys,std::uint64_t offset,
    std::unordered_map<std::uint64_t,std::uint64_t>& mapping,void* old_keys,
    void* new_values,void* output,std::size_t output_bytes) {
    integer_width(bytes,[&](auto input_tag) {
        using Input=decltype(input_tag);
        const auto* source=static_cast<const char*>(input);
        auto* old=static_cast<char*>(old_keys);
        integer_width(output_bytes,[&](auto output_tag) {
            using Output=decltype(output_tag);
            auto* values=static_cast<char*>(new_values);
            auto* result=static_cast<char*>(output);
            for(std::size_t k=0;k<keys.size();++k) {
                const auto key=keys[k];
                const auto value=key ? offset++ : 0;
                mapping.at(key)=value;
                const auto old_value=Input(key);
                const auto new_value=Output(value);
                std::memcpy(old+k*sizeof(Input),&old_value,sizeof(Input));
                std::memcpy(values+k*sizeof(Output),&new_value,sizeof(Output));
            }
            const auto& lookup=mapping;
            detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
                for(auto i=begin;i<end;++i) {
                    const auto key=read<Input>(source+i*sizeof(Input),false);
                    const auto found=lookup.find(key);
                    const auto value=found==lookup.end() ? Output{0} : Output(found->second);
                    std::memcpy(result+i*sizeof(Output),&value,sizeof(Output));
                }
            });
        });
    });
}

void components_clear_border_fast(const ComponentsView& view,void* output,
    std::size_t bytes,std::size_t border) {
    const auto count=count_of(view.shape);
    const auto rows=view.shape[0],width=view.shape[1];
    if(border>=rows-border || border>=width-border) {
        std::memset(output,0,count*bytes);
        return;
    }
    std::unique_ptr<std::intptr_t[]> labels(new std::intptr_t[count]);
    std::size_t number;
    if(view.type==PixelType::boolean) {
        std::unique_ptr<std::uint8_t[]> keys(new std::uint8_t[count]);
        components_prepare_bool_into(view,keys.get());
        number=label(keys.get(),labels.get(),view.shape,std::uint8_t{0},2);
    } else {
        number=components_label_dense_integer_into(view.data,labels.get(),view.shape,0,2,view.type);
    }
    std::vector<std::uint8_t> remove(number+1);
    for(std::size_t y=0;y<rows;++y) {
        const auto* ids=labels.get()+y*width;
        if(y<border || rows-y<=border) {
            for(std::size_t x=0;x<width;++x)remove[ids[x]]=1;
        } else {
            for(std::size_t x=0;x<border;++x) {
                remove[ids[x]]=1;
                remove[ids[width-1-x]]=1;
            }
        }
    }
    integer_width(bytes,[&](auto tag) {
        using T=decltype(tag);
        const auto* source=static_cast<const char*>(view.data);
        auto* result=static_cast<char*>(output);
        detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
            for(auto i=begin;i<end;++i) {
                const auto value=remove[labels[i]] ? T{0} : read<T>(source+i*sizeof(T),false);
                std::memcpy(result+i*sizeof(T),&value,sizeof(T));
            }
        });
    });
}

bool components_remove_small_objects_fast(const ComponentsView& view,void* output,
    std::size_t bytes,double threshold,int connectivity) {
    const auto count=count_of(view.shape);
    if(!count)return false;
    const auto* input=static_cast<const char*>(view.data);
    auto* result=static_cast<char*>(output);
    if(threshold==0) {
        std::memcpy(result,input,count*bytes);
        return false;
    }
    const bool boolean=view.type==PixelType::boolean;
    std::unique_ptr<std::intptr_t[]> labels(new std::intptr_t[count]);
    std::size_t maximum=0;
    if(boolean) {
        std::unique_ptr<std::uint8_t[]> keys(new std::uint8_t[count]);
        components_prepare_bool_into(view,keys.get());
        maximum=label(keys.get(),labels.get(),view.shape,std::uint8_t{0},connectivity);
    } else {
        components_prepare_index_into(view,labels.get());
        for(std::size_t i=0;i<count;++i) {
            const auto key=labels[i];
            if(key<0)throw std::invalid_argument("Negative value labels are not supported. Try relabeling the input with `scipy.ndimage.label` or `skimage.morphology.label`.");
            maximum=std::max(maximum,std::size_t(key));
        }
    }
    const auto filter=[&](const auto& size_of) {
        integer_width(bytes,[&](auto tag) {
            using T=decltype(tag);
            detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
                for(auto i=begin;i<end;++i) {
                    const auto value=double(size_of(std::size_t(labels[i])))<=threshold ? T{0} : read<T>(input+i*sizeof(T),false);
                    std::memcpy(result+i*sizeof(T),&value,sizeof(T));
                }
            });
        });
    };
    if(boolean || (maximum<=16*1024*1024 && maximum<=count+1024)) {
        std::vector<std::size_t> sizes(maximum+1);
        for(std::size_t i=0;i<count;++i)++sizes[std::size_t(labels[i])];
        filter([&](std::size_t key){return sizes[key];});
    } else {
        std::unordered_map<std::size_t,std::size_t> sizes;
        for(std::size_t i=0;i<count;++i)++sizes[std::size_t(labels[i])];
        filter([&](std::size_t key){return sizes.at(key);});
    }
    return !boolean && maximum==1;
}
}
