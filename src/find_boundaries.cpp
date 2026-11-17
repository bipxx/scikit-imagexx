#include "skimagexx/find_boundaries.hpp"
#include "execution.hpp"
#include <array>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <cfenv>
#if defined(__GNUC__) && defined(__x86_64__)
#include <emmintrin.h>
#endif

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
// The source gray morphology stages typed data through binary64, then
// stores back to the label dtype. Full rectangular footprints run one axis
// at a time and store after EVERY axis. A generated dtype-max background can
// therefore change at an intermediate store; integer-only extrema are not an
// equivalent replacement for ordinary wide outer mode.
#if defined(__GNUC__) && defined(__x86_64__)
template<class T>double wide_source_load(T value) {
    static_assert(sizeof(T)==8);
    if constexpr(std::is_signed_v<T>)
        return _mm_cvtsd_f64(_mm_cvtsi64_sd(_mm_setzero_pd(),value));
    else {
        if(value<=std::uint64_t(std::numeric_limits<std::int64_t>::max()))
            return _mm_cvtsd_f64(_mm_cvtsi64_sd(_mm_setzero_pd(),std::int64_t(value)));
        // GCC's ordinary uint64->double conversion: retain the discarded bit
        // before scaling by two. This is a source conversion, not a narrowed
        // label representation; original T words remain separately owned.
        const auto half=std::int64_t((value>>1)|(value&1));
        const auto converted=_mm_cvtsi64_sd(_mm_setzero_pd(),half);
        return _mm_cvtsd_f64(_mm_add_sd(converted,converted));
    }
}
template<class T>T wide_source_store(double value) {
    static_assert(sizeof(T)==8);
    const auto scalar=_mm_set_sd(value);
    if constexpr(std::is_signed_v<T>)
        return _mm_cvttsd_si64(scalar);
    else {
        // Spell the baseline x86_64 cast without an undefined C++ conversion.
        // Out-of-range generated sentinels retain the conversion's invalid
        // flag/result, rather than saturation, truncation to32bit or FP masks.
        constexpr double high=0x1p63;
        if(value<high)return std::uint64_t(_mm_cvttsd_si64(scalar));
        return std::uint64_t(_mm_cvttsd_si64(_mm_sub_sd(scalar,_mm_set_sd(high))))^(std::uint64_t{1}<<63);
    }
}

template<class T>std::vector<T> wide_source_extrema(const std::vector<T>& input,
    const Shape& shape,const std::vector<Delta>& footprint,bool full,bool minimum) {
    const auto strides=[&] {
        std::vector<std::ptrdiff_t> result(shape.size());std::ptrdiff_t step=1;
        for(auto d=shape.size();d-->0;){result[d]=step;step*=std::ptrdiff_t(shape[d]);}
        return result;
    }();
    auto current=input;std::vector<T> next(input.size());
    const auto passes=full?shape.size():std::size_t{1};
    if(shape.size()==2) {
        const auto height=shape[0],width=shape[1];
        for(std::size_t axis=0;axis<passes;++axis) {
            for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x) {
                const auto index=y*width+x;
                double result=0.;bool first=true;
                auto visit=[&](std::size_t source) {
                    const auto value=wide_source_load(current[source]);
                    if(first){result=value;first=false;}
                    else if(minimum?value<result:value>result)result=value;
                };
                if(full) {
                    // Keep the source axis order, three conversions and typed
                    // store per pass; only coordinate decoding is removed.
                    const auto coordinate=axis?x:y,extent=axis?width:height;
                    const auto step=axis?std::size_t{1}:width;
                    visit(index-(coordinate?step:0));
                    visit(index);
                    visit(index+(coordinate+1<extent?step:0));
                }else for(const auto& delta:footprint) {
                    const auto row=delta[0]<0?(y?y-1:0):delta[0]>0?std::min(y+1,height-1):y;
                    const auto column=delta[1]<0?(x?x-1:0):delta[1]>0?std::min(x+1,width-1):x;
                    visit(row*width+column);
                }
                next[index]=wide_source_store<T>(result);
            }
            current.swap(next);
        }
        return current;
    }
    for(std::size_t axis=0;axis<passes;++axis) {
        for(std::size_t i=0;i<input.size();++i) {
            Position center{},point{};position_of(i,shape,center);
            double result=0.;bool first=true;
            auto visit=[&] {
                const auto value=wide_source_load(current[std::size_t(offset_of(point,strides))]);
                if(first){result=value;first=false;}
                else if(minimum?value<result:value>result)result=value;
            };
            if(full) {
                // Radius one, origin zero, reflect endpoint duplicates.
                // Typed current/next interchange is the scipy per-axis store.
                for(int delta=-1;delta<=1;++delta) {
                    point=center;
                    point[axis]=delta<0?(center[axis]?center[axis]-1:0):
                        delta>0?std::min(center[axis]+1,shape[axis]-1):center[axis];
                    visit();
                }
            }else for(const auto& delta:footprint) {
                for(std::size_t d=0;d<shape.size();++d)
                    point[d]=delta[d]<0?(center[d]?center[d]-1:0):
                        delta[d]>0?std::min(center[d]+1,shape[d]-1):center[d];
                visit();
            }
            next[i]=wide_source_store<T>(result);
        }
        current.swap(next);
    }
    return current;
}
#endif

template<class T>bool wide_source_outer(const ComponentsView& view,
    const FindBoundaryParameters& parameters,std::uint8_t* output,
    const std::vector<std::ptrdiff_t>& output_strides,std::size_t size) {
#if defined(__GNUC__) && defined(__x86_64__)
    // The new rounding-sensitive route is source-only pending actual GNUx64
    // cast-ABI/FP validation. Preserve unsupported round/platform refusal;
    // do not modify caller controls or normalize output state.
    // False carries no original scalar value/type. Full-width NumPy scalar
    // comparison promotion cannot be reconstructed from that lost metadata.
    if(!parameters.background_representable)return false;
    const auto control=_mm_getcsr();
    if(std::fegetround()!=FE_TONEAREST || (control&(3u<<13)) ||
       (control&0x1f80u)!=0x1f80u)return false;
    const auto rank=view.shape.size();
    const auto near=neighbors(rank,parameters.connectivity);
    const auto full=neighbors(rank,unsigned(rank));
    // Charge actual source extrema work, including all separable typed axes.
    const auto visits=(parameters.connectivity==rank?6*rank:2*near.size())+6*rank;
    if(size>find_boundaries_max_neighbor_visits/visits)return false;
    std::vector<T> original(size);Position point{};
    if(rank==2) {
        const auto width=view.shape[1];
        for(std::size_t y=0;y<view.shape[0];++y) {
            const auto* row=static_cast<const char*>(view.data)+std::ptrdiff_t(y)*view.strides[0];
            for(std::size_t x=0;x<width;++x)
                std::memcpy(&original[y*width+x],row+std::ptrdiff_t(x)*view.strides[1],sizeof(T));
        }
    }else for(std::size_t i=0;i<size;++i){position_of(i,view.shape,point);original[i]=load<T>(view,point);}
    // Source first computes ordinary dilation!=erosion, then the full
    // footprint foreground/background expression, with original typed labels.
    const auto low=wide_source_extrema(original,view.shape,near,parameters.connectivity==rank,true);
    const auto high=wide_source_extrema(original,view.shape,near,parameters.connectivity==rank,false);
    const auto full_high=wide_source_extrema(original,view.shape,full,true,false);
    auto inverted=original;
    const auto background=[&](T value){return parameters.background_representable &&
        static_cast<std::uint64_t>(value)==parameters.background_bits;};
    for(auto& value:inverted)if(background(value))value=std::numeric_limits<T>::max();
    const auto full_low=wide_source_extrema(inverted,view.shape,full,true,true);
    // Source final != and Boolean ufuncs clear prior arithmetic error status
    // before their integer loops. Preserve inexact and caller controls, and
    // avoid a libc clear when NumPy's error-status check would do nothing.
    constexpr int comparison_errors=FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID;
    if(std::fetestexcept(comparison_errors))std::feclearexcept(comparison_errors);
    // The dependent typed morphology stages stay serial to expose source FP
    // effects on the caller; the existing positive budget was captured once.
    // All writes follow complete input/geometry/capacity validation and stages.
    auto boundary=[&](std::size_t i) {
        const bool bg=background(original[i]);
        const bool adjacent=(full_high[i]!=full_low[i])&&!bg;
        return std::uint8_t((high[i]!=low[i])&&(bg||adjacent));
    };
    if(rank==2) {
        const auto width=view.shape[1];
        for(std::size_t y=0;y<view.shape[0];++y) {
            auto* row=output+std::ptrdiff_t(y)*output_strides[0];
            for(std::size_t x=0;x<width;++x)row[std::ptrdiff_t(x)*output_strides[1]]=boundary(y*width+x);
        }
    }else for(std::size_t i=0;i<size;++i) {
        position_of(i,view.shape,point);output[offset_of(point,output_strides)]=boundary(i);
    }
    return true;
#else
    (void)view;(void)parameters;(void)output;(void)output_strides;(void)size;
    return false; // No unvalidated Windows/nonGNU cast substitution.
#endif
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
// Walk the admitted signed-stride lattice once. Each worker decodes its first
// position only; contiguous/F/reversed layouts then advance by byte strides.
// Border clamping stays literal. Interior neighbors reuse invariant offsets.
template<class T,bool Boolean>
void strided_regular(const ComponentsView& view,const FindBoundaryParameters& parameters,
    std::uint8_t* output,const std::vector<std::ptrdiff_t>& output_strides,
    std::size_t count,int workers,const std::vector<Delta>& near,const std::vector<Delta>& full) {
    const auto rank=view.shape.size();
    const T sentinel=std::numeric_limits<T>::max();
    const bool inner=parameters.mode==FindBoundaryMode::inner;
    const bool outer=parameters.mode==FindBoundaryMode::outer;
    const bool fused=outer && parameters.connectivity==rank;
    auto background=[&](T v){return parameters.background_representable &&
        static_cast<std::uint64_t>(v)==parameters.background_bits;};
    auto read=[&](std::ptrdiff_t offset){T v;std::memcpy(&v,static_cast<const char*>(view.data)+offset,sizeof(T));
        if constexpr(Boolean)v=v!=0;return v;};
    std::array<std::size_t,8> axes{};for(std::size_t d=0;d<rank;++d)axes[d]=d;
    auto magnitude=[](std::ptrdiff_t s){return s<0?std::size_t(-(s+1))+1:std::size_t(s);};
    std::sort(axes.begin(),axes.begin()+rank,[&](auto a,auto b){
        const auto x=magnitude(view.strides[a]),y=magnitude(view.strides[b]);return x<y || (x==y && a<b);});
    const bool has_interior=std::all_of(view.shape.begin(),view.shape.end(),[](auto n){return n>=3;});
    auto offsets=[&](const std::vector<Delta>& neighbors){std::vector<std::ptrdiff_t> result;
        if(has_interior)for(const auto& delta:neighbors){std::ptrdiff_t offset=0;for(std::size_t d=0;d<rank;++d)offset+=delta[d]*view.strides[d];result.push_back(offset);}return result;};
    const auto near_offsets=offsets(near),full_offsets=fused?std::vector<std::ptrdiff_t>{}:offsets(full);
    partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t){
        Position p{};auto word=begin;
        for(std::size_t i=0;i<rank;++i){auto d=axes[i];p[d]=word%view.shape[d];word/=view.shape[d];}
        auto input_offset=offset_of(p,view.strides),output_offset=offset_of(p,output_strides);
        for(auto index=begin;index<end;++index){
            const auto center=read(input_offset);bool boundary=false;
            if(!inner || !background(center)){
                bool interior=has_interior;
                for(std::size_t d=0;d<rank && interior;++d)interior=p[d]>0 && p[d]+1<view.shape[d];
                auto neighbor_offset=[&](const Delta& delta){auto offset=input_offset;
                    for(std::size_t d=0;d<rank;++d)if((delta[d]<0 && p[d])||(delta[d]>0 && p[d]+1<view.shape[d]))offset+=delta[d]*view.strides[d];return offset;};
                T minimum=sentinel,maximum=std::numeric_limits<T>::lowest(),inverted_min=sentinel;
                for(std::size_t k=0;k<near.size();++k){
                    const auto value=read(interior?input_offset+near_offsets[k]:neighbor_offset(near[k]));
                    minimum=std::min(minimum,value);maximum=std::max(maximum,value);
                    if(fused)inverted_min=std::min(inverted_min,background(value)?sentinel:value);
                }
                boundary=maximum!=minimum;
                if(outer && boundary && !background(center)){
                    T original_max=maximum;
                    if(!fused){original_max=std::numeric_limits<T>::lowest();
                        for(std::size_t k=0;k<full.size();++k){
                            const auto value=read(interior?input_offset+full_offsets[k]:neighbor_offset(full[k]));
                            original_max=std::max(original_max,value);inverted_min=std::min(inverted_min,background(value)?sentinel:value);
                        }
                    }
                    boundary=original_max!=inverted_min;
                }
            }
            output[output_offset]=std::uint8_t(boundary);
            // Update only valid positions; no one-past input/output pointer is
            // formed, including a worker ending mid-row or at the final cell.
            for(std::size_t i=0;i<rank;++i){const auto d=axes[i];
                if(++p[d]<view.shape[d]){input_offset+=view.strides[d];output_offset+=output_strides[d];break;}
                p[d]=0;input_offset-=std::ptrdiff_t(view.shape[d]-1)*view.strides[d];output_offset-=std::ptrdiff_t(view.shape[d]-1)*output_strides[d];
            }
        }
    });
}
template<class T,bool Boolean=false>bool run(const ComponentsView& view,
    const FindBoundaryParameters& parameters,std::uint8_t* output,
    const Shape& shape,const std::vector<std::ptrdiff_t>& strides,
    std::size_t input_count,std::size_t output_count,int workers) {
    const auto rank=view.shape.size();
    // Positive aligned F/F regular2D is the same symmetric footprint with axes
    // exchanged. Reuse the existing C-row kernel without moving any pixels.
    // Public intake already proved address arithmetic; the SDK caller retains
    // the existing disjoint-allocation/nonoverlapping-output obligation.
    if constexpr(sizeof(T)<=8) {
        if(rank==2 && (parameters.mode==FindBoundaryMode::thick || parameters.mode==FindBoundaryMode::inner) &&
           shape[0]>=3 && shape[1]>=3 &&
           view.strides[0]==std::ptrdiff_t(sizeof(T)) && view.strides[1]==std::ptrdiff_t(shape[0]*sizeof(T)) &&
           strides[0]==1 && strides[1]==std::ptrdiff_t(shape[0]) &&
           reinterpret_cast<std::uintptr_t>(view.data)%alignof(T)==0) {
            auto transposed=view;auto transposed_shape=shape;auto transposed_strides=strides;
            std::swap(transposed.shape[0],transposed.shape[1]);std::swap(transposed.strides[0],transposed.strides[1]);
            std::swap(transposed_shape[0],transposed_shape[1]);std::swap(transposed_strides[0],transposed_strides[1]);
            // workers is the positive budget already captured by public intake.
            return run<T,Boolean>(transposed,parameters,output,transposed_shape,transposed_strides,input_count,output_count,workers);
        }
    }
    if constexpr(sizeof(T)==8) {
        if(parameters.mode!=FindBoundaryMode::subpixel) {
            if(parameters.mode==FindBoundaryMode::outer)
                return wide_source_outer<T>(view,parameters,output,strides,input_count);
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
    // The wide-label scan above already proved exact integer promotion; the
    // same integer-only interior kernel therefore also covers 64-bit labels.
    const bool interior=!subpixel && parameters.mode!=FindBoundaryMode::outer &&
        rank==2 && shape[0]>=3 && shape[1]>=3 &&
        view.strides[1]==std::ptrdiff_t(sizeof(T)) && view.strides[0]==std::ptrdiff_t(shape[1]*sizeof(T)) &&
        strides[1]==1 && strides[0]==std::ptrdiff_t(shape[1]) &&
        reinterpret_cast<std::uintptr_t>(view.data)%alignof(T)==0;
    using U=std::make_unsigned_t<T>;const U bits=static_cast<U>(parameters.background_bits);
    T background;std::memcpy(&background,&bits,sizeof(T));
    const bool background_exists=parameters.background_representable && static_cast<std::uint64_t>(background)==parameters.background_bits;
    if(!subpixel && !interior){
        strided_regular<T,Boolean>(view,parameters,output,strides,output_count,team,near,full);
        return true;
    }
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
