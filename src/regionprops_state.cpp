#include "skimagexx/regionprops_state.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cfenv>
#include <stdexcept>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
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
template<class Append>
void coords_scaled_walk(const std::uint8_t* mask,std::size_t ndim,
    const std::array<std::size_t,3>& shape,const std::array<std::ptrdiff_t,3>& strides,
    const Append& append) {
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
    coords_scaled_walk(mask,ndim,shape,strides,append);
}

namespace {
bool scalar_frontier(double value,double lower,double upper) noexcept {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    if(bits==0)return true;
    if((bits>>63) || ((bits>>52)&0x7ffu)==0x7ffu)return false;
    return value>=lower && value<=upper;
}
class ScalarEnvironment {
    std::fenv_t original_;
public:
    ScalarEnvironment() {
        if(std::fegetround()!=FE_TONEAREST)
            throw std::invalid_argument("RegionProperties scalar requires nearest rounding");
        #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        if(_mm_getcsr()&((1u<<15)|(1u<<6)|(3u<<13)))
            throw std::invalid_argument("RegionProperties scalar requires gradual underflow");
        #endif
        if(std::feholdexcept(&original_))
            throw std::runtime_error("cannot capture RegionProperties scalar environment");
    }
    ~ScalarEnvironment(){std::fesetenv(&original_);}
    unsigned flags()const noexcept {
        const auto f=std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
        return (f&FE_INVALID?regionprops_fp_invalid:0u)
             | (f&FE_DIVBYZERO?regionprops_fp_divide:0u)
             | (f&FE_OVERFLOW?regionprops_fp_overflow:0u)
             | (f&FE_UNDERFLOW?regionprops_fp_underflow:0u);
    }
};
RegionPropsScalar64 scalar_result(double value,unsigned status)noexcept {
    RegionPropsScalar64 result{0,status};std::memcpy(&result.bits,&value,sizeof(value));return result;
}
}

RegionPropsScalar64 regionprops_scaled_count_f64(std::size_t count,double pixel_area) {
    if(std::uint64_t(count)>UINT64_C(9007199254740992)
       || count>std::size_t(std::numeric_limits<std::intptr_t>::max())
       || !scalar_frontier(pixel_area,0x1p-256,0x1p256))
        throw std::invalid_argument("RegionProperties scaled count outside f64 frontier");
    ScalarEnvironment environment;
    volatile double value=double(count)*pixel_area;
    return scalar_result(value,environment.flags());
}

RegionPropsScalar64 regionprops_extent_f64(double area,double area_bbox) {
    if(!scalar_frontier(area,0x1p-256,0x1p309)
       || !scalar_frontier(area_bbox,0x1p-256,0x1p309))
        throw std::invalid_argument("RegionProperties extent outside f64 frontier");
    ScalarEnvironment environment;
    // NumPy's assessed x86-64 scalar invalid result has this exact negative
    // quiet-NaN word. Return it deterministically without leaking hardware flags.
    if(area==0. && area_bbox==0.)
        return {UINT64_C(0xfff8000000000000),regionprops_fp_invalid};
    volatile double value=area/area_bbox;
    return scalar_result(value,environment.flags());
}

namespace {
bool coords_scaled_layout(std::size_t count,std::size_t ndim,int workers) noexcept {
    return (ndim==2 || ndim==3) && workers>0
        && count<=regionprops_coords_scaled_max_elements
        && count<=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(ndim*sizeof(double));
}
bool coords_scaled_shape(const std::array<std::size_t,3>& shape,std::size_t ndim,
                         std::size_t count,int workers,std::size_t& total) noexcept {
    if(!coords_scaled_layout(count,ndim,workers))return false;
    total=1;
    for(std::size_t a=0;a<ndim;++a) {
        if(shape[a]>regionprops_coords_scaled_max_extent
           || (shape[a] && total>regionprops_coords_scaled_max_elements/shape[a]))return false;
        total*=shape[a];
    }
    return count<=total;
}
bool coords_scaled_overlap(const void* first,std::size_t first_bytes,
                           const void* second,std::size_t second_bytes) noexcept {
    if(!first_bytes || !second_bytes)return false;
    const auto a=reinterpret_cast<std::uintptr_t>(first),b=reinterpret_cast<std::uintptr_t>(second);
    return a<=b ? b-a<first_bytes : a-b<second_bytes;
}
bool coords_scaled_parameters(const std::array<double,3>& values,std::size_t ndim) noexcept {
    for(std::size_t a=0;a<ndim;++a) {
        const auto bits=word<std::uint64_t>(reinterpret_cast<const char*>(&values[a]))&UINT64_C(0x7fffffffffffffff);
        if(bits && (bits<UINT64_C(0x2ff0000000000000) || bits>UINT64_C(0x4ff0000000000000)))return false;
    }
    return true;
}
// Two fixed math consumers share FP context handling; no callbacks/allocations
// or throwing arithmetic run while a worker's temporary context is installed.
template<class Math> bool coords_scaled_math(std::size_t count,std::size_t ndim,int workers,const Math& math) {
    try {
        ScalarEnvironment environment;
        if(!count)return true;
        const int threads=count*ndim>=parallel_min ? int(std::min(count,std::size_t(workers))) : 1;
        if(threads<=1){math(0,count);return true;}
        std::fenv_t computation{};
        if(std::fegetenv(&computation))throw std::runtime_error("cannot capture coords_scaled worker environment");
        #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        const auto computation_csr=_mm_getcsr();
        #endif
        partition(count,threads,[&](auto begin,auto end,auto) {
            // The master already verified fenv support on this runtime.
            std::fenv_t previous{};std::fegetenv(&previous);
            #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            const auto previous_csr=_mm_getcsr();
            #endif
            std::fesetenv(&computation);
            #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            _mm_setcsr(computation_csr);
            #endif
            math(begin,end);
            std::fesetenv(&previous);
            #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
            _mm_setcsr(previous_csr);
            #endif
        });
        return true;
    }catch(const std::invalid_argument&){return false;}
}
}

bool regionprops_coords_scaled_local_into(const std::uint8_t* mask,std::size_t ndim,
    const std::array<std::size_t,3>& shape,const std::array<std::ptrdiff_t,3>& strides,
    std::size_t count,std::intptr_t* output,int workers) {
    std::size_t total=0;
    if(!coords_scaled_shape(shape,ndim,count,workers,total) || (total && !mask) || (count && !output))return false;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    std::size_t positive=0,negative=0;
    for(std::size_t a=0;a<ndim;++a) {
        if(strides[a]==std::numeric_limits<std::ptrdiff_t>::min())return false;
        const auto step=std::size_t(strides[a]<0?-strides[a]:strides[a]);
        auto& span=strides[a]<0?negative:positive;
        if(shape[a] && step && shape[a]-1>(limit-span)/step)return false;
        if(shape[a])span+=(shape[a]-1)*step;
    }
    if(!total)return true;
    const auto address=reinterpret_cast<std::uintptr_t>(mask);
    if(address<negative || address>std::numeric_limits<std::uintptr_t>::max()-positive-1)return false;
    if(coords_scaled_overlap(reinterpret_cast<const void*>(address-negative),positive+negative+1,
                             output,count*ndim*sizeof(std::intptr_t)))return false;
    std::size_t point=0;bool mismatch=false;
    const auto append=[&](std::size_t plane,std::size_t row,std::size_t column) {
        // count/fill stability is a caller invariant; still do not overrun a
        // supplied span if an invalid caller changes the mask between them.
        if(point>=count){mismatch=true;return;}
        if(ndim==2){output[point]=std::intptr_t(row);output[count+point]=std::intptr_t(column);}
        else {output[point]=std::intptr_t(plane);output[count+point]=std::intptr_t(row);output[2*count+point]=std::intptr_t(column);}
        ++point;
    };
    coords_scaled_walk(mask,ndim,shape,strides,append);
    return !mismatch && point==count;
}

bool regionprops_coords_scaled_origin_into(std::intptr_t* coordinates,std::size_t count,
    std::size_t ndim,const std::array<std::size_t,3>& shape,
    const std::array<std::intptr_t,3>& origins,int workers) {
    std::size_t total=0;
    if(!coords_scaled_shape(shape,ndim,count,workers,total) || (count && !coordinates))return false;
    for(std::size_t a=0;a<ndim;++a) {
        if(origins[a]<-regionprops_coords_scaled_max_origin || origins[a]>regionprops_coords_scaled_max_origin)return false;
        for(std::size_t p=0;p<count;++p) {
            const auto value=coordinates[a*count+p];
            if(value<0 || std::size_t(value)>=shape[a])return false;
        }
    }
    if(!count)return true;
    const int threads=count*ndim>=parallel_min ? int(std::min(count,std::size_t(workers))) : 1;
    partition(count,threads,[&](auto begin,auto end,auto) {
        for(std::size_t a=0;a<ndim;++a) {
            auto* column=coordinates+a*count;const auto origin=origins[a];
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(std::size_t p=begin;p<end;++p)column[p]+=origin;
        }
    });
    return true;
}

bool regionprops_coords_scaled_multiply_f64_into(const std::intptr_t* coordinates,
    std::size_t count,std::size_t ndim,const std::array<double,3>& spacing,double* output,int workers) {
    if(!coords_scaled_layout(count,ndim,workers) || !coords_scaled_parameters(spacing,ndim)
       || (count && (!coordinates || !output)))return false;
    const auto size=count*ndim;
    if(coords_scaled_overlap(coordinates,size*sizeof(std::intptr_t),output,size*sizeof(double)))return false;
    constexpr auto bound=2*regionprops_coords_scaled_max_origin;
    for(std::size_t p=0;p<size;++p)if(coordinates[p]<-bound || coordinates[p]>bound)return false;
    return coords_scaled_math(count,ndim,workers,[&](std::size_t begin,std::size_t end) noexcept {
        for(std::size_t a=0;a<ndim;++a) {
            const auto* input=coordinates+a*count;auto* column=output+a*count;const auto scale=spacing[a];
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(std::size_t p=begin;p<end;++p)column[p]=double(input[p])*scale;
        }
    });
}

bool regionprops_coords_scaled_add_f64_into(double* values,std::size_t count,std::size_t ndim,
    const std::array<double,3>& offset,int workers) {
    if(!coords_scaled_layout(count,ndim,workers) || !coords_scaled_parameters(offset,ndim) || (count && !values))return false;
    const auto size=count*ndim;
    for(std::size_t p=0;p<size;++p) {
        const auto bits=word<std::uint64_t>(reinterpret_cast<const char*>(values+p))&UINT64_C(0x7fffffffffffffff);
        if(bits && (bits<UINT64_C(0x2cb0000000000000) || bits>UINT64_C(0x5140000000000000)))return false;
    }
    return coords_scaled_math(count,ndim,workers,[&](std::size_t begin,std::size_t end) noexcept {
        for(std::size_t a=0;a<ndim;++a) {
            auto* column=values+a*count;const auto shift=offset[a];
            #if defined(_OPENMP) && !defined(_MSC_VER)
            #pragma omp simd
            #endif
            for(std::size_t p=begin;p<end;++p)column[p]+=shift;
        }
    });
}
}
