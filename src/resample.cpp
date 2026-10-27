// Grid coordinates and accumulation follow SciPy v1.18.1 ni_interpolation.c.
// Copyright (C) 2003-2005 Peter J. Verveer, SciPy Developers; BSD-3-Clause.
// See third_party/scipy-ndimage-LICENSE.txt. Axis plans, bounded per-worker
// scratch and SIMD traversal are implemented in this repository.
#include "resample_detail.hpp"
#include "execution.hpp"
#include "resize_cast.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx::detail {
namespace {
constexpr auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
std::size_t size(const std::vector<std::size_t>& shape) {
    std::size_t result=1;
    for(auto length:shape) {
        if(length>limit || (length && result>limit/length))
            throw std::overflow_error("zoom dimensions overflow");
        result*=length;
    }
    return result;
}
std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> result(shape.size());std::size_t step=1;
    for(auto axis=shape.size();axis-->0;) {result[axis]=step;step*=shape[axis];}
    return result;
}
void validate_border(GaussianBorder border) {
    switch(border) {
        case GaussianBorder::reflect:case GaussianBorder::constant:
        case GaussianBorder::nearest:case GaussianBorder::mirror:
        case GaussianBorder::wrap:return;
    }
    throw std::invalid_argument("unsupported zoom boundary mode");
}
// Fractional-coordinate mapping preserves ndimage's reflection convention.
// Grid-constant coordinates stay unmodified; outside taps become cval.
double map_coordinate(double coordinate,std::ptrdiff_t length,GaussianBorder border) {
    if(border==GaussianBorder::constant)return coordinate;
    if(length<=1)return 0.;
    if(coordinate<0) {
        switch(border) {
            case GaussianBorder::mirror: {
                const auto period=2*length-2;
                coordinate=period*std::ptrdiff_t(-coordinate/period)+coordinate;
                return coordinate<=1-length ? coordinate+period : -coordinate;
            }
            case GaussianBorder::reflect: {
                const auto period=2*length;
                if(coordinate < -period)coordinate=period*std::ptrdiff_t(-coordinate/period)+coordinate;
                return coordinate < -length ? coordinate+period : (coordinate > -1e-15 ? 1e-15 : -coordinate)-1;
            }
            case GaussianBorder::wrap:
                return coordinate+length*(std::ptrdiff_t((-1-coordinate)/length)+1);
            case GaussianBorder::nearest:return 0.;
            default:break;
        }
    } else if(coordinate>length-1) {
        switch(border) {
            case GaussianBorder::mirror: {
                const auto period=2*length-2;
                coordinate-=period*std::ptrdiff_t(coordinate/period);
                return coordinate>=length ? period-coordinate : coordinate;
            }
            case GaussianBorder::reflect: {
                const auto period=2*length;
                coordinate-=period*std::ptrdiff_t(coordinate/period);
                return coordinate>=length ? period-coordinate-1 : coordinate;
            }
            case GaussianBorder::wrap:return coordinate-length*std::ptrdiff_t(coordinate/length);
            case GaussianBorder::nearest:return double(length-1);
            default:break;
        }
    }
    return coordinate;
}
ZoomPlan make_plan(const std::vector<std::size_t>& input,
                   const std::vector<std::size_t>& output,std::size_t padding,
                   const ZoomOptions& options) {
    ZoomPlan plan;plan.count=size(output);plan.output_strides=strides(output);
    plan.order=options.order;plan.cval=options.cval;plan.axes.resize(input.size());
    const auto source_strides=strides(input);const auto taps=std::size_t(options.order+1);
    for(std::size_t axis=0;axis<input.size();++axis) {
        if(plan.corners>limit/taps)throw std::overflow_error("zoom footprint overflow");
        plan.corners*=taps;
        auto& points=plan.axes[axis];points.resize(output[axis]);
        const double scale=double(input[axis]-2*padding)/double(output[axis]);
        for(std::size_t i=0;i<points.size();++i) {
            double coordinate=(double(i)+.5)*scale-.5+double(padding);
            if(options.border!=GaussianBorder::constant && options.border!=GaussianBorder::nearest)
                coordinate=map_coordinate(coordinate,std::ptrdiff_t(input[axis]),options.border);
            auto& point=points[i];point.weights=spline_weights(coordinate,options.order);
            const auto start=std::ptrdiff_t(std::floor(coordinate+(options.order&1 ? 0. : .5)))-options.order/2;
            for(std::size_t tap=0;tap<taps;++tap) {
                auto index=start+std::ptrdiff_t(tap);
                if(options.border==GaussianBorder::constant && (index<0 || std::size_t(index)>=input[axis]))
                    point.offsets[tap]=-1;
                else {
                    index=std::ptrdiff_t(map_coordinate(double(index),std::ptrdiff_t(input[axis]),options.border));
                    point.offsets[tap]=index*std::ptrdiff_t(source_strides[axis]);
                }
            }
        }
    }
    if(plan.corners>limit/input.size())throw std::overflow_error("zoom footprint overflow");
    plan.footprint.resize(plan.corners*input.size());
    for(std::size_t corner=0;corner<plan.corners;++corner) {
        auto remainder=corner;
        for(auto axis=input.size();axis-->0;) {
            plan.footprint[corner*input.size()+axis]=std::uint8_t(remainder%taps);remainder/=taps;
        }
    }
    return plan;
}
template<class Input> double sample(const Input* input,const ZoomPlan& plan,
              const std::vector<const ZoomPoint*>& points) {
    double total=0.;const auto rank=points.size();
    for(std::size_t corner=0;corner<plan.corners;++corner) {
        const auto* taps=plan.footprint.data()+corner*rank;
        std::ptrdiff_t index=0;bool outside=false;
        for(std::size_t axis=0;axis<rank;++axis) {
            const auto offset=points[axis]->offsets[taps[axis]];
            outside|=offset<0;
            // Never include a sentinel in a pointer offset or pointer sum.
            if(offset>=0)index+=offset;
        }
        double coefficient=outside ? plan.cval : double(input[index]);
        if(plan.order)for(std::size_t axis=0;axis<rank;++axis)
            coefficient*=points[axis]->weights[taps[axis]];
        total+=coefficient;
    }
    return total;
}
template<class Input> void pad_into(const Input* input,double* output,const std::vector<std::size_t>& shape,
              const std::vector<std::size_t>& padded,std::size_t padding,
              GaussianBorder border,double cval) {
    const auto source_strides=strides(shape),target_strides=strides(padded);
    const auto count=size(padded);
    partition(count,active_workers(count),[&](std::size_t begin,std::size_t end,std::size_t) {
        for(auto i=begin;i<end;++i) {
            std::size_t index=0;bool outside=false;
            for(std::size_t axis=0;axis<shape.size();++axis) {
                const auto coordinate=std::ptrdiff_t((i/target_strides[axis])%padded[axis])-std::ptrdiff_t(padding);
                outside|=coordinate<0 || coordinate>=std::ptrdiff_t(shape[axis]);
                const auto clamped=std::size_t(std::clamp(coordinate,std::ptrdiff_t{0},std::ptrdiff_t(shape[axis]-1)));
                index+=clamped*source_strides[axis];
            }
            output[i]=outside && border==GaussianBorder::constant ? double(Input(cval)) : double(input[index]);
        }
    });
}
template<class Input,class Output,bool Boolean=false> void sample_into(const Input* input,Output* output,
                        PixelType source,PixelType target,const ZoomPlan& plan,const ZoomOptions& options) {
    const auto threads=active_workers(plan.count);
    std::vector<std::vector<const ZoomPoint*>> scratch(std::size_t(threads),
        std::vector<const ZoomPoint*>(plan.axes.size()));
    (void)source;(void)target;
    partition(plan.count,threads,[&](std::size_t begin,std::size_t end,std::size_t worker) {
        #ifdef SKIMAGEXX_AVX2
        if(options.allow_simd && has_avx2())begin=simd::zoom_sample_typed(input,output,plan,begin,end,source,target);
        #endif
        auto& points=scratch[worker];
        for(auto i=begin;i<end;++i) {
            for(std::size_t axis=0;axis<points.size();++axis)
                points[axis]=&plan.axes[axis][(i/plan.output_strides[axis])%plan.axes[axis].size()];
            output[i]=resize_sample_cast<Output,Boolean>(sample(input,plan,points));
        }
    });
}
template<class T,bool Boolean=false> void zoom(const T* input,T* output,PixelType type,
               const std::vector<std::size_t>& input_shape,
               const std::vector<std::size_t>& output_shape,const ZoomOptions& options) {
    if(input_shape.empty() || input_shape.size()!=output_shape.size())
        throw std::invalid_argument("matching nonzero zoom ranks required");
    if(options.order<0 || options.order>5)throw std::invalid_argument("spline order must be 0 through 5");
    if constexpr(!std::is_floating_point_v<T>) {
        if(options.order!=0)throw std::invalid_argument("integer zoom storage requires nearest interpolation");
    }
    validate_border(options.border);
    const auto input_count=size(input_shape),output_count=size(output_shape);
    if(options.identity_copy && input_shape!=output_shape)
        throw std::invalid_argument("identity zoom requires equal shapes");
    if(!output_count)return;
    if(!input_count || !input || !output)throw std::invalid_argument("nonnull nonempty zoom input/output required");
    if(options.identity_copy) {
        if(input!=output)std::memcpy(output,input,input_count*sizeof(T));
        return;
    }
    if(input==output)throw std::invalid_argument("zoom output must not alias input");
    auto source_shape=input_shape;std::vector<double> coefficients;
    std::size_t padding=0;
    if(options.order>1) {
        if(options.border==GaussianBorder::nearest || options.border==GaussianBorder::constant)padding=12;
        for(auto& length:source_shape) {
            if(length>limit-2*padding)throw std::overflow_error("zoom padding overflow");
            length+=2*padding;
        }
        coefficients.resize(size(source_shape));
        if(padding) {
            pad_into(input,coefficients.data(),input_shape,source_shape,padding,options.border,options.cval);
            spline_filter_into(coefficients.data(),coefficients.data(),source_shape,options.order,
                               options.border,options.legacy_reflect,options.allow_simd);
        } else {
            // Promotion happens before the spline recurrence. The native
            // float32 pad constant above rounds in its original storage type.
            partition(input_count,active_workers(input_count),[&](std::size_t begin,std::size_t end,std::size_t) {
                for(auto i=begin;i<end;++i)coefficients[i]=double(input[i]);
            });
            spline_filter_into(coefficients.data(),coefficients.data(),source_shape,options.order,
                               options.border,options.legacy_reflect,options.allow_simd);
        }
    }
    const auto plan=make_plan(source_shape,output_shape,padding,options);
    if(options.order>1)sample_into<double,T,Boolean>(coefficients.data(),output,PixelType::float64,type,plan,options);
    else sample_into<T,T,Boolean>(input,output,type,type,plan,options);
}
} // namespace
void zoom_typed_into(const void* input,void* output,PixelType type,
                     const std::vector<std::size_t>& shape,const std::vector<std::size_t>& target,
                     const ZoomOptions& options) {
    switch(type) {
        case PixelType::boolean:zoom<std::uint8_t,true>(static_cast<const std::uint8_t*>(input),static_cast<std::uint8_t*>(output),type,shape,target,options);break;
        #define ZOOM(tag,T) case PixelType::tag:zoom(static_cast<const T*>(input),static_cast<T*>(output),type,shape,target,options);break;
        ZOOM(uint8,std::uint8_t) ZOOM(int8,std::int8_t) ZOOM(uint16,std::uint16_t) ZOOM(int16,std::int16_t)
        ZOOM(uint32,std::uint32_t) ZOOM(int32,std::int32_t) ZOOM(uint64,std::uint64_t) ZOOM(int64,std::int64_t)
        ZOOM(float32,float) ZOOM(float64,double)
        #undef ZOOM
        default:throw std::invalid_argument("unsupported zoom storage type");
    }
}
void zoom_into(const double* input,double* output,const std::vector<std::size_t>& shape,
               const std::vector<std::size_t>& target,const ZoomOptions& options) {
    zoom_typed_into(input,output,PixelType::float64,shape,target,options);
}
} // namespace skimagexx::detail

#include <cfenv>
#include <type_traits>

namespace skimagexx::detail {
namespace {
template<class T>bool profile_finite(const T* input,std::size_t count){
    if constexpr(std::is_floating_point_v<T>){
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())return simd::profile_all_finite(input,count,std::is_same_v<T,float>);
#endif
        using Word=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        constexpr Word mask=sizeof(T)==4?Word(0x7f800000u):Word(UINT64_C(0x7ff0000000000000));
        for(std::size_t i=0;i<count;++i){Word bits;std::memcpy(&bits,input+i,sizeof(T));if((bits&mask)==mask)return false;}
    }
    return true;
}
bool profile_point(ZoomPoint& point,double coordinate,std::size_t length,
                   std::size_t stride,int order,GaussianBorder border){
    // Bound before floor/integer conversion; huge/unrepresentable source
    // coordinates are outside the assessed ordinary profile domain.
    const auto bound=double(std::numeric_limits<std::ptrdiff_t>::max()/4);
    if(!std::isfinite(coordinate)||coordinate<=-bound||coordinate>=bound)return false;
    if(border!=GaussianBorder::constant&&border!=GaussianBorder::nearest)
        coordinate=map_coordinate(coordinate,std::ptrdiff_t(length),border);
    const auto start=std::ptrdiff_t(std::floor(coordinate+(order?0.:.5)));
    if(order)point.weights=spline_weights(coordinate,order);
    for(int tap=0;tap<=order;++tap){
        auto index=start+tap;
        if(border==GaussianBorder::constant&&(index<0||std::size_t(index)>=length))point.offsets[tap]=-1;
        else{
            index=std::ptrdiff_t(map_coordinate(double(index),std::ptrdiff_t(length),border));
            point.offsets[tap]=index*std::ptrdiff_t(stride);
        }
    }
    return true;
}
template<class T,bool Boolean=false>bool profile(const ProfileInput& input,const double* coordinates,
                std::size_t count,T* output,int order,GaussianBorder border,double cval,int workers){
    const auto* data=static_cast<const T*>(input.data);
    if(!profile_finite(data,input.height*input.width)||!std::isfinite(cval)||std::fegetround()!=FE_TONEAREST)return false;
    const int team=count>=parallel_min?workers:1;const bool vectorized=has_avx2();
    std::vector<ZoomPlan> plans(static_cast<std::size_t>(team));
    std::vector<std::vector<const ZoomPoint*>> points(static_cast<std::size_t>(team),std::vector<const ZoomPoint*>(2));
    std::vector<unsigned char> refused(static_cast<std::size_t>(team),0);
    for(auto& plan:plans){
        plan.axes.assign(2,std::vector<ZoomPoint>(4));plan.output_strides={1,1};
        plan.count=4;plan.corners=order?4:1;plan.order=order;plan.cval=cval;
        plan.footprint=order?std::vector<std::uint8_t>{0,0,0,1,1,0,1,1}:std::vector<std::uint8_t>{0,0};
    }
    partition(count,team,[&](auto begin,auto end,auto worker){
        std::fenv_t saved;if(std::feholdexcept(&saved)){refused[worker]=1;return;}
        auto& plan=plans[worker];auto& current=points[worker];bool accepted=std::fegetround()==FE_TONEAREST;
        for(auto i=begin;i<end&&accepted;){
            const auto lanes=std::min(std::size_t(4),end-i);
            for(std::size_t lane=0;lane<lanes;++lane){
                accepted&=profile_point(plan.axes[0][lane],coordinates[i+lane],input.height,input.width,order,border);
                accepted&=profile_point(plan.axes[1][lane],coordinates[count+i+lane],input.width,1,order,border);
            }
            if(!accepted)break;
            std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)done=simd::profile_sample_typed(data,output+i,plan,lanes,input.type);
#else
            (void)vectorized;
#endif
            for(;done<lanes;++done){
                current[0]=&plan.axes[0][done];current[1]=&plan.axes[1][done];
                output[i+done]=resize_sample_cast<T,Boolean>(sample(data,plan,current));
            }
            i+=lanes;
        }
        refused[worker]=!accepted||std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
        std::fesetenv(&saved);
    });
    for(auto value:refused)if(value)return false;
    return true;
}
}
bool profile_sample(const ProfileInput& input,const double* coordinates,std::size_t count,
                    void* output,int order,GaussianBorder border,double cval,int workers){
    if(order!=0&&order!=1)throw std::invalid_argument("profile sampling supports order0 or1");
    if(workers<1||workers>1024)throw std::invalid_argument("profile worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    validate_border(border);
    if(!input.height||!input.width)throw std::invalid_argument("profile spatial axes must be nonempty");
    size({input.height,input.width});
    if(count>limit/2)throw std::overflow_error("profile coordinate extent overflow");
    if(!count)return true;
    if(!input.data||!coordinates||!output)throw std::invalid_argument("nonnull prepared profile buffers required");
    switch(input.type){
        case PixelType::boolean:return profile<std::uint8_t,true>(input,coordinates,count,static_cast<std::uint8_t*>(output),order,border,cval,workers);
#define PROFILE(tag,T) case PixelType::tag:return profile<T>(input,coordinates,count,static_cast<T*>(output),order,border,cval,workers);
        PROFILE(uint8,std::uint8_t) PROFILE(float32,float) PROFILE(float64,double)
#undef PROFILE
        default:throw std::invalid_argument("unsupported prepared profile storage type");
    }
}
}
