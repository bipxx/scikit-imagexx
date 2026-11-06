#include "skimagexx/find_boundaries.hpp"
#include "execution.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::detail {
namespace {
using Shape=std::vector<std::size_t>;
using Position=std::array<std::size_t,8>;
using Delta=std::array<signed char,8>;
bool mode_valid(FindBoundaryMode mode) {
    return mode==FindBoundaryMode::thick || mode==FindBoundaryMode::inner ||
           mode==FindBoundaryMode::outer || mode==FindBoundaryMode::subpixel;
}
std::size_t item_size(PixelType type) {
    switch(type) {
    case PixelType::boolean:case PixelType::uint8:case PixelType::int8:return 1;
    case PixelType::uint16:case PixelType::int16:return 2;
    case PixelType::uint32:case PixelType::int32:return 4;
    case PixelType::uint64:case PixelType::int64:return 8;
    default:return 0;
    }
}
bool product(const Shape& shape,std::size_t& count) {
    count=1;
    for(auto n:shape) {
        if(!n || n>find_boundaries_max_elements/count)return false;
        count*=n;
    }
    return true;
}
bool addressable(const void* data,const Shape& shape,
                 const std::vector<std::ptrdiff_t>& strides,std::size_t item) {
    if(!data || shape.size()!=strides.size())return false;
    std::size_t low=0,high=item-1;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    for(std::size_t d=0;d<shape.size();++d) {
        auto s=strides[d];
        const auto magnitude=s<0?std::size_t(-(s+1))+1:std::size_t(s);
        auto& span=s<0?low:high;
        if(magnitude && shape[d]-1>(limit-span)/magnitude)return false;
        span+=(shape[d]-1)*magnitude;
    }
    auto address=reinterpret_cast<std::uintptr_t>(data);
    return low<=address && high<=std::numeric_limits<std::uintptr_t>::max()-address;
}
void position_of(std::size_t index,const Shape& shape,Position& position) {
    for(auto d=shape.size();d-->0;) {position[d]=index%shape[d];index/=shape[d];}
}
std::ptrdiff_t offset_of(const Position& position,const std::vector<std::ptrdiff_t>& strides) {
    std::ptrdiff_t result=0;
    for(std::size_t d=0;d<strides.size();++d)result+=std::ptrdiff_t(position[d])*strides[d];
    return result;
}
std::vector<Delta> neighbors(std::size_t rank,unsigned connectivity) {
    std::size_t size=1;for(std::size_t d=0;d<rank;++d)size*=3;
    std::vector<Delta> result;
    for(std::size_t word=0;word<size;++word) {
        auto code=word;unsigned distance=0;Delta delta{};
        for(std::size_t d=0;d<rank;++d) {
            delta[d]=static_cast<signed char>(int(code%3)-1);code/=3;
            distance+=delta[d]!=0;
        }
        if(distance<=connectivity)result.push_back(delta);
    }
    return result;
}
template<class T,bool Boolean=false>T load(const ComponentsView& view,const Position& p) {
    T result;std::memcpy(&result,static_cast<const char*>(view.data)+offset_of(p,view.strides),sizeof(T));
    if constexpr(Boolean)result=result!=0;
    return result;
}
// Independent 2D interior pixels: caller proved disjoint storage; typed
// pointer loads are admitted only for aligned positive C-contiguous input.
// Borders and arbitrary-stride domains stay in the original scalar walker.
#if defined(_MSC_VER)
#define FIND_BOUNDARIES_RESTRICT __restrict
#else
#define FIND_BOUNDARIES_RESTRICT __restrict__
#endif
template<class T,bool Boolean,bool Full,bool Inner>
void interior_row(const T* FIND_BOUNDARIES_RESTRICT input,
    std::uint8_t* FIND_BOUNDARIES_RESTRICT output,std::size_t width,
    std::size_t y,T background,bool background_exists) {
    const auto row=y*width;
    for(std::size_t x=1;x+1<width;++x) {
        auto canonical=[](T value)->T {if constexpr(Boolean)return value!=0;else return value;};
        const T center=canonical(input[row+x]);
        T minimum=center,maximum=center;
#define BOUNDARY_NEIGHBOR(offset) {const T v=canonical(input[offset]);minimum=std::min(minimum,v);maximum=std::max(maximum,v);}
        BOUNDARY_NEIGHBOR(row+x-1) BOUNDARY_NEIGHBOR(row+x+1)
        BOUNDARY_NEIGHBOR(row-width+x) BOUNDARY_NEIGHBOR(row+width+x)
        if constexpr(Full) {
            BOUNDARY_NEIGHBOR(row-width+x-1) BOUNDARY_NEIGHBOR(row-width+x+1)
            BOUNDARY_NEIGHBOR(row+width+x-1) BOUNDARY_NEIGHBOR(row+width+x+1)
        }
#undef BOUNDARY_NEIGHBOR
        output[row+x]=std::uint8_t((maximum!=minimum) &
            (!Inner || !(background_exists && center==background)));
    }
}
#undef FIND_BOUNDARIES_RESTRICT
template<class T,bool Boolean=false>bool run(const ComponentsView& view,
    const FindBoundaryParameters& parameters,std::uint8_t* output,
    const Shape& shape,const std::vector<std::ptrdiff_t>& strides,
    std::size_t input_count,std::size_t output_count,int workers) {
    const auto rank=view.shape.size();
    if constexpr(sizeof(T)==8) {
        if(parameters.mode!=FindBoundaryMode::subpixel) {
            if(parameters.mode==FindBoundaryMode::outer)return false;
            // New value boundary: reject the entire unsupported rounded domain
            // BEFORE writing any output, rather than rounding original labels.
            Position p{};
            for(std::size_t i=0;i<input_count;++i) {
                position_of(i,view.shape,p);auto value=load<T,Boolean>(view,p);
                if constexpr(std::is_signed_v<T>) {
                    if(value<-(std::int64_t{1}<<53) || value>(std::int64_t{1}<<53))return false;
                } else if(value>(std::uint64_t{1}<<53))return false;
            }
        }
    }
    const bool subpixel=parameters.mode==FindBoundaryMode::subpixel;
    const auto near=subpixel?std::vector<Delta>{}:neighbors(rank,parameters.connectivity);
    const auto full=parameters.mode==FindBoundaryMode::outer?neighbors(rank,unsigned(rank)):std::vector<Delta>{};
    auto visits=subpixel?(std::size_t{1}<<rank):near.size()+full.size();
    if(output_count>find_boundaries_max_neighbor_visits/visits)return false;
    const T sentinel=std::numeric_limits<T>::max();
    auto is_background=[&](T value) {return parameters.background_representable &&
        static_cast<std::uint64_t>(value)==parameters.background_bits;};
    const int team=output_count>=parallel_min?workers:1;
    const bool interior=sizeof(T)<=4 && !subpixel && parameters.mode!=FindBoundaryMode::outer &&
        rank==2 && shape[0]>=3 && shape[1]>=3 &&
        view.strides[1]==std::ptrdiff_t(sizeof(T)) && view.strides[0]==std::ptrdiff_t(shape[1]*sizeof(T)) &&
        strides[1]==1 && strides[0]==std::ptrdiff_t(shape[1]) &&
        reinterpret_cast<std::uintptr_t>(view.data)%alignof(T)==0;
    using U=std::make_unsigned_t<T>;const U bits=static_cast<U>(parameters.background_bits);
    T background;std::memcpy(&background,&bits,sizeof(T));
    const bool background_exists=parameters.background_representable && static_cast<std::uint64_t>(background)==parameters.background_bits;
    partition(interior?shape[0]:output_count,team,[&](std::size_t begin,std::size_t end,std::size_t) {
        Position p{},q{};
        if(interior){begin*=shape[1];end*=shape[1];}
        for(auto index=begin;index<end;++index) {
            position_of(index,shape,p);
            if(interior && p[1]==1 && p[0]>0 && p[0]+1<shape[0]) {
                auto* input=static_cast<const T*>(view.data);
                const bool inner=parameters.mode==FindBoundaryMode::inner;
                if(parameters.connectivity==1) {
                    if(inner)interior_row<T,Boolean,false,true>(input,output,shape[1],p[0],background,background_exists);
                    else interior_row<T,Boolean,false,false>(input,output,shape[1],p[0],background,background_exists);
                } else {
                    if(inner)interior_row<T,Boolean,true,true>(input,output,shape[1],p[0],background,background_exists);
                    else interior_row<T,Boolean,true,false>(input,output,shape[1],p[0],background,background_exists);
                }
                index+=shape[1]-3;continue;
            }
            bool boundary=false;
            if(subpixel) {
                std::array<std::size_t,8> odd{};std::size_t odd_count=0;
                for(std::size_t d=0;d<rank;++d) {q[d]=p[d]/2;if(p[d]&1)odd[odd_count++]=d;}
                if(odd_count) {
                    T first{};bool found=false;
                    for(std::size_t corner=0;corner<(std::size_t{1}<<odd_count);++corner) {
                        for(std::size_t bit=0;bit<odd_count;++bit)q[odd[bit]]=p[odd[bit]]/2+((corner>>bit)&1);
                        auto value=load<T,Boolean>(view,q);
                        if(value==sentinel)continue;
                        if(found && value!=first){boundary=true;break;}
                        first=value;found=true;
                    }
                }
            } else {
                T minimum=sentinel,maximum=std::numeric_limits<T>::lowest();
                for(const auto& delta:near) {
                    for(std::size_t d=0;d<rank;++d)q[d]=delta[d]<0?(p[d]?p[d]-1:0):
                        delta[d]>0?std::min(p[d]+1,view.shape[d]-1):p[d];
                    auto value=load<T,Boolean>(view,q);minimum=std::min(minimum,value);maximum=std::max(maximum,value);
                }
                boundary=maximum!=minimum;
                const auto center=load<T,Boolean>(view,p);
                if(parameters.mode==FindBoundaryMode::inner)boundary=boundary && !is_background(center);
                else if(parameters.mode==FindBoundaryMode::outer && boundary && !is_background(center)) {
                    T original_max=std::numeric_limits<T>::lowest(),inverted_min=sentinel;
                    for(const auto& delta:full) {
                        for(std::size_t d=0;d<rank;++d)q[d]=delta[d]<0?(p[d]?p[d]-1:0):
                            delta[d]>0?std::min(p[d]+1,view.shape[d]-1):p[d];
                        auto value=load<T,Boolean>(view,q);original_max=std::max(original_max,value);
                        inverted_min=std::min(inverted_min,is_background(value)?sentinel:value);
                    }
                    boundary=original_max!=inverted_min;
                }
            }
            output[offset_of(p,strides)]=static_cast<std::uint8_t>(boundary);
        }
    });
    return true;
}
}
bool find_boundaries_shape(const ComponentsView& view,FindBoundaryMode mode,Shape& output) {
    if(!mode_valid(mode)||view.half||view.swap||!item_size(view.type)||view.shape.empty()||view.shape.size()>8)return false;
    std::size_t count;if(!product(view.shape,count))return false;
    Shape derived=view.shape;
    if(mode==FindBoundaryMode::subpixel)for(auto& n:derived)n=2*n-1;
    if(!product(derived,count))return false;
    output=std::move(derived);return true;
}
bool find_boundaries_into(const ComponentsView& view,const FindBoundaryParameters& parameters,
    std::uint8_t* output,const std::vector<std::ptrdiff_t>& strides,int workers) {
    Shape shape;if(!find_boundaries_shape(view,parameters.mode,shape))return false;
    if(parameters.connectivity<1 || parameters.connectivity>view.shape.size() || workers<0 || workers>1024)return false;
    if(!addressable(view.data,view.shape,view.strides,item_size(view.type)) || !addressable(output,shape,strides,1))return false;
    for(std::size_t d=0;d<shape.size();++d)if(shape[d]>1 && strides[d]==0)return false;
    // The raw SDK contract requires caller-proven disjoint backing allocations.
    // Exact pointer equality can be refused without inventing allocation bounds.
    if(view.data==output)return false;
    std::size_t input_count,output_count;product(view.shape,input_count);product(shape,output_count);
    const int snapshot=workers?workers:get_num_threads();
    if(snapshot<1 || snapshot>1024)return false;
#define FIND_BOUNDARY_CASE(tag,type) case PixelType::tag:return run<type>(view,parameters,output,shape,strides,input_count,output_count,snapshot)
    switch(view.type) {
    case PixelType::boolean:return run<std::uint8_t,true>(view,parameters,output,shape,strides,input_count,output_count,snapshot);
    FIND_BOUNDARY_CASE(uint8,std::uint8_t);FIND_BOUNDARY_CASE(int8,std::int8_t);
    FIND_BOUNDARY_CASE(uint16,std::uint16_t);FIND_BOUNDARY_CASE(int16,std::int16_t);
    FIND_BOUNDARY_CASE(uint32,std::uint32_t);FIND_BOUNDARY_CASE(int32,std::int32_t);
    FIND_BOUNDARY_CASE(uint64,std::uint64_t);FIND_BOUNDARY_CASE(int64,std::int64_t);
    default:return false;
    }
#undef FIND_BOUNDARY_CASE
}
}
