#include "skimagexx/gabor.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "gabor_math_detail.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {
void kernel_into(void* output,bool single,std::size_t height,std::size_t width,
                       const GaborParameters& p,bool accurate) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(2*sizeof(double));
    if(!output || !height || !width || height%2!=1 || width%2!=1 || width>limit || height>limit/width)
        throw std::invalid_argument("nonempty odd Gabor dimensions and a valid output buffer required");
    if(!std::isfinite(p.cosine) || !std::isfinite(p.sine) ||
       !std::isfinite(p.sigma_x_squared) || !std::isfinite(p.sigma_y_squared) ||
       p.sigma_x_squared<=0 || p.sigma_y_squared<=0 ||
       !std::isfinite(p.angular_frequency) || !std::isfinite(p.offset) || !std::isfinite(p.normalization))
        throw std::invalid_argument("finite Gabor parameters and positive squared sigmas required");
    const auto workers=detail::active_workers(height*width);
    detail::partition(height,workers,[&](auto begin,auto end,auto){
        for(auto row=begin;row<end;++row){
            const auto y=double(row)-double(height/2);std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())x=accurate ? simd::gabor_kernel_accurate_row(output,single,row*width,width,y,p)
                                   : simd::gabor_kernel_row(output,single,row*width,width,y,p);
#endif
            for(;x<width;++x){
                const auto coordinate=double(x)-double(width/2);
                const auto rx=coordinate*p.cosine+y*p.sine;
                const auto ry=-coordinate*p.sine+y*p.cosine;
                const auto magnitude=std::exp(-0.5*(rx*rx/p.sigma_x_squared+ry*ry/p.sigma_y_squared));
                const auto phase=p.angular_frequency*rx+p.offset;
                const auto i=2*(row*width+x);
                if(single){auto* out=static_cast<float*>(output);
                    out[i]=float(magnitude*std::cos(phase))*float(p.normalization);
                    out[i+1]=float(magnitude*std::sin(phase))*float(p.normalization);
                }else{auto* out=static_cast<double*>(output);
                    out[i]=magnitude*std::cos(phase)*p.normalization;
                    out[i+1]=magnitude*std::sin(phase)*p.normalization;
                }
            }
        }
    });
}
}
void gabor_kernel_into(void* output,bool single,std::size_t height,std::size_t width,const GaborParameters& p) {
    kernel_into(output,single,height,width,p,false);
}
void gabor_kernel_accurate_into(void* output,bool single,std::size_t height,std::size_t width,const GaborParameters& p) {
    kernel_into(output,single,height,width,p,true);
}
}
