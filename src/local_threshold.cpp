#include "skimagexx/local_threshold.hpp"
#include "skimagexx/image.hpp"
#include "local_threshold_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
std::size_t multiply(std::size_t a,std::size_t b) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    if(b && a>limit/b)throw std::overflow_error("local threshold shape is too large");
    return a*b;
}
std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> result(shape.size(),1);
    for(std::size_t i=shape.size();i-->1;)result[i-1]=multiply(result[i],shape[i]);
    return result;
}
std::size_t reflection(std::ptrdiff_t index,std::size_t length) {
    if(length==1)return 0;
    const auto period=std::ptrdiff_t(2*(length-1));
    index%=period;if(index<0)index+=period;
    return std::size_t(index<std::ptrdiff_t(length) ? index : period-index);
}
void prefix_pair(double* sums,double* squares,const std::vector<std::size_t>& shape,std::size_t count) {
    const auto pitch=strides(shape);
    const auto workers=detail::active_workers(count);
    const bool vectorized=has_avx2();
    for(std::size_t axis=0;axis<shape.size();++axis) {
        const auto length=shape[axis],inner=pitch[axis],outer=count/(length*inner);
        const auto chunks=(inner+7)/8;
        detail::partition(outer*chunks,workers,[&](auto begin,auto end,auto) {
            for(auto block=begin;block<end;++block) {
                const auto column=(block%chunks)*8;
                const auto offset=(block/chunks)*length*inner+column;
                const auto lanes=std::min(std::size_t(8),inner-column);
                std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(vectorized && lanes>=4)done=simd::local_prefix_columns(sums+offset,squares+offset,length,inner,lanes);
#else
                (void)vectorized;
#endif
                if(done<lanes)for(std::size_t row=1;row<length;++row) {
                    const auto current=offset+row*inner,previous=current-inner;
                    for(auto lane=done;lane<lanes;++lane) {
                        sums[current+lane]+=sums[previous+lane];
                        squares[current+lane]+=squares[previous+lane];
                    }
                }
            }
        });
    }
}

template<class F,class Output> void finish(const double* sums,const double* squares,Output* output,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& padded_shape,
    const std::vector<std::size_t>& window,std::size_t count,std::size_t area,
    LocalThreshold method,const LocalThresholdParameters& parameters) {
    const auto dims=shape.size(),corners=std::size_t(1)<<dims;
    const auto pitch=strides(padded_shape);
    std::vector<std::size_t> offsets(corners);std::vector<int> signs(corners);
    for(std::size_t mask=0;mask<corners;++mask) {
        std::size_t bits=0;
        for(std::size_t axis=0;axis<dims;++axis) {
            if(mask & (std::size_t(1)<<(dims-axis-1))) {offsets[mask]+=window[axis]*pitch[axis];++bits;}
        }
        signs[mask]=dims%2==bits%2 ? 1 : -1;
    }
    const auto width=shape.back(),rows=count/width;
    detail::partition(rows,detail::active_workers(count),[&](auto begin,auto end,auto) {
        std::vector<const double*> mean_views(corners),square_views(corners);
        for(auto row=begin;row<end;++row) {
            auto remainder=row;std::size_t base=0;
            for(std::size_t axis=dims-1;axis-->0;) {
                base+=(remainder%shape[axis])*pitch[axis];remainder/=shape[axis];
            }
            for(std::size_t corner=0;corner<corners;++corner) {
                mean_views[corner]=sums+base+offsets[corner];
                square_views[corner]=squares+base+offsets[corner];
            }
            std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=simd::local_threshold_rows(mean_views.data(),square_views.data(),signs.data(),corners,
                output+row*width,width,area,std::is_same_v<F,float>,std::is_same_v<Output,double>,method,parameters);
#endif
            for(;x<width;++x) {
                double sum=signs[0]>0 ? mean_views[0][x] : -mean_views[0][x];
                double square=signs[0]>0 ? square_views[0][x] : -square_views[0][x];
                for(std::size_t corner=1;corner<corners;++corner) {
                    sum+=signs[corner]>0 ? mean_views[corner][x] : -mean_views[corner][x];
                    square+=signs[corner]>0 ? square_views[corner][x] : -square_views[corner][x];
                }
                const F mean=detail::local_divide<F>(sum,area,parameters.divide_double);
                const F mean_square=detail::local_divide<F>(square,area,parameters.divide_double);
                const F variance=F(mean_square-F(mean*mean));
                const F sd=std::sqrt(std::isnan(variance) ? variance : std::max(F(0),variance));
                output[row*width+x]=Output(detail::local_formula(mean,sd,method,parameters));
            }
        }
    });
}

template<class Input,bool Boolean=false> void execute(const Input* input,void* output,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& window,
    LocalThreshold method,const LocalThresholdParameters& parameters) {
    using F=std::conditional_t<std::is_same_v<Input,float>,float,double>;
    const auto dims=shape.size();auto padded=shape;
    std::size_t count=1,padded_count=1,area=1;
    const auto source_pitch=strides(shape);
    std::vector<std::vector<std::size_t>> maps(dims);
    for(std::size_t axis=0;axis<dims;++axis) {
        if(window[axis]>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-shape[axis])throw std::overflow_error("padded shape overflow");
        padded[axis]+=window[axis];
        count=multiply(count,shape[axis]);padded_count=multiply(padded_count,padded[axis]);area=multiply(area,window[axis]);
        maps[axis].resize(padded[axis]);
        for(std::size_t i=0;i<padded[axis];++i)
            maps[axis][i]=reflection(std::ptrdiff_t(i)-std::ptrdiff_t(window[axis]/2+1),shape[axis])*source_pitch[axis];
    }
    std::unique_ptr<double[]> sums(new double[padded_count]),squares(new double[padded_count]);
    const auto width=padded.back(),rows=padded_count/width;
    detail::partition(rows,detail::active_workers(padded_count),[&](auto begin,auto end,auto) {
        for(auto row=begin;row<end;++row) {
            auto remainder=row;std::size_t source=0;
            for(std::size_t axis=dims-1;axis-->0;) {source+=maps[axis][remainder%padded[axis]];remainder/=padded[axis];}
            for(std::size_t x=0;x<width;++x) {
                const auto raw=input[source+maps.back()[x]];
                const F value=Boolean ? F(raw!=0) : F(raw);
                sums[row*width+x]=double(value);squares[row*width+x]=double(F(value*value));
            }
        }
    });
    prefix_pair(sums.get(),squares.get(),padded,padded_count);
    if constexpr(std::is_same_v<F,float>) {
        const bool wide_output=parameters.k_double || (method==LocalThreshold::sauvola && parameters.r_double);
        if(wide_output)finish<F>(sums.get(),squares.get(),static_cast<double*>(output),shape,padded,window,count,area,method,parameters);
        else finish<F>(sums.get(),squares.get(),static_cast<float*>(output),shape,padded,window,count,area,method,parameters);
    } else finish<F>(sums.get(),squares.get(),static_cast<double*>(output),shape,padded,window,count,area,method,parameters);
}
}
void local_threshold_into(const void* input,void* output,PixelType type,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& window,LocalThreshold method,const LocalThresholdParameters& parameters) {
    if(!input || !output || shape.empty() || shape.size()!=window.size() || shape.size()>=sizeof(std::size_t)*8)
        throw std::invalid_argument("nonempty matching shape/window and nonnull buffers required");
    if(method!=LocalThreshold::niblack && method!=LocalThreshold::sauvola)throw std::invalid_argument("invalid local threshold method");
    auto p=parameters;
    if(method==LocalThreshold::sauvola && std::isnan(p.r)) {
        p.r=(type==PixelType::float32 || type==PixelType::float64) ? 1. :
            type==PixelType::boolean ? .5 : std::ldexp(1.,int(pixel_size(type)*8)-1)-.5;
    }
    if(!std::isfinite(p.k) || (method==LocalThreshold::sauvola && (!std::isfinite(p.r) || p.r==0)))
        throw std::invalid_argument("finite scalar parameters and nonzero r required");
    for(std::size_t axis=0;axis<shape.size();++axis)
        if(!shape[axis] || !window[axis] || window[axis]%2==0)throw std::invalid_argument("positive shape and odd windows required");
    if(type==PixelType::boolean) {execute<std::uint8_t,true>(static_cast<const std::uint8_t*>(input),output,shape,window,method,p);return;}
#define LOCAL_CASE(tag,ctype) case PixelType::tag:execute(static_cast<const ctype*>(input),output,shape,window,method,p);return;
    switch(type) {
    LOCAL_CASE(uint8,std::uint8_t) LOCAL_CASE(int8,std::int8_t)
    LOCAL_CASE(uint16,std::uint16_t) LOCAL_CASE(int16,std::int16_t)
    LOCAL_CASE(uint32,std::uint32_t) LOCAL_CASE(int32,std::int32_t)
    LOCAL_CASE(uint64,std::uint64_t) LOCAL_CASE(int64,std::int64_t)
    LOCAL_CASE(float32,float) LOCAL_CASE(float64,double)
    default:throw std::invalid_argument("unsupported local threshold input type");
    }
#undef LOCAL_CASE
}
}
