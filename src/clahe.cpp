#include "skimagexx/clahe.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {
std::size_t product(const std::vector<std::size_t>& dimensions) {
    std::size_t result=1;
    for(const auto length:dimensions) {
        if(!length || result>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/length)
            throw std::invalid_argument("positive, addressable CLAHE dimensions required");
        result*=length;
    }
    return result;
}
std::vector<std::size_t> strides(const std::vector<std::size_t>& dimensions) {
    std::vector<std::size_t> result(dimensions.size(),1);
    for(std::size_t d=dimensions.size()-1;d>0;--d)result[d-1]=result[d]*dimensions[d];
    return result;
}
std::size_t reflect(std::size_t index,std::size_t length) {
    if(length==1)return 0;
    const auto period=2*(length-1);index%=period;
    return index<length ? index : period-index;
}
// Redistribution follows the BSD-licensed scikit-image 0.26 implementation,
// originally adapted from Karel Zuiderveld, Graphics Gems IV. In particular,
// its final strided update may distribute more than the remaining excess.
void clip_histogram(std::vector<std::int64_t>& histogram,std::int64_t limit) {
    std::int64_t excess=0;
    for(auto& value:histogram)if(value>limit) {excess+=value-limit;value=limit;}
    const auto increment=excess/std::int64_t(histogram.size()),upper=limit-increment;
    for(auto& value:histogram)if(value<upper) {value+=increment;excess-=increment;}
    for(auto& value:histogram)if(value>=upper && value<limit) {excess+=value-limit;value=limit;}
    while(excess>0) {
        const auto previous=excess;
        for(std::size_t start=0;start<histogram.size();++start) {
            const auto under=std::count_if(histogram.begin(),histogram.end(),[=](auto v){return v<limit;});
            if(!under)return; // No later strided update could change this histogram.
            const auto step=std::max(std::int64_t(1),std::int64_t(under)/excess);
            for(std::size_t j=start;j<histogram.size();j+=std::size_t(step))
                if(histogram[j]<limit) {++histogram[j];--excess;}
            if(excess<=0)break;
        }
        if(excess==previous)break;
    }
}
}
void clahe_into(const std::uint16_t* input,void* output,bool single,
        const std::vector<std::size_t>& shape,const std::vector<std::size_t>& kernel,
        std::size_t clip_count,std::size_t bins) {
    const auto ndim=shape.size();
    if(!input || !output || !ndim || ndim>8 || kernel.size()!=ndim || !bins || bins>65536 || !clip_count)
        throw std::invalid_argument("CLAHE requires 1..8 axes, positive kernels, clipping and 1..65536 bins");
    const auto count=product(shape),tile_elements=product(kernel);
    const auto source_strides=strides(shape),kernel_strides=strides(kernel);
    std::vector<std::size_t> tile_shape(ndim);
    for(std::size_t d=0;d<ndim;++d) {
        if(kernel[d]>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-shape[d])
            throw std::invalid_argument("CLAHE padded dimension exceeds addressable storage");
        tile_shape[d]=(shape[d]-1)/kernel[d]+1;
    }
    const auto tiles=product(tile_shape);
    const auto tile_strides=strides(tile_shape);
    if(tiles>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bins)
        throw std::invalid_argument("CLAHE mapping exceeds addressable storage");
    const auto bounds=std::minmax_element(input,input+count);
    const auto low=*bounds.first,high=*bounds.second;
    const std::size_t bin_width=1+16384/bins;
    const int workers=detail::active_workers(count);
    std::vector<std::uint16_t> quantized(count),mapped(count);
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        if(low==high) {
            std::fill(quantized.begin()+begin,quantized.begin()+end,std::uint16_t(std::min(unsigned(low),16383u)/bin_width));
            return;
        }
        std::size_t i=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())i+=simd::clahe_quantize(input+i,quantized.data()+i,end-i,low,high,bin_width);
#endif
        for(;i<end;++i) {
            const double value=(double(input[i])-double(low))/double(high-low)*16383.;
            const double floor=std::floor(value),fraction=value-floor;
            const auto rounded=std::size_t(floor)+(fraction>.5 || (fraction==.5 && (std::size_t(floor)&1)));
            quantized[i]=std::uint16_t(rounded/bin_width);
        }
    });
    std::vector<std::int32_t> mappings(tiles*bins);
    const auto limit=std::int64_t(std::min(clip_count,tile_elements));
    const double scale=16383./double(tile_elements);
    detail::partition(tiles,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        std::vector<std::int64_t> histogram(bins);
        std::vector<std::size_t> origin(ndim);
        for(std::size_t tile=begin;tile<end;++tile) {
            for(std::size_t d=0;d<ndim;++d)origin[d]=(tile/tile_strides[d]%tile_shape[d])*kernel[d];
            std::fill(histogram.begin(),histogram.end(),0);
            for(std::size_t row=0;row<tile_elements/kernel.back();++row) {
                std::size_t base=0;
                for(std::size_t d=0;d+1<ndim;++d)
                    base+=reflect(origin[d]+row*kernel.back()/kernel_strides[d]%kernel[d],shape[d])*source_strides[d];
                if(origin.back()+kernel.back()<=shape.back()) {
                    const auto* values=quantized.data()+base+origin.back();
                    for(std::size_t x=0;x<kernel.back();++x)++histogram[values[x]];
                } else {
                    for(std::size_t x=0;x<kernel.back();++x)
                        ++histogram[quantized[base+reflect(origin.back()+x,shape.back())]];
                }
            }
            clip_histogram(histogram,limit);
            std::int64_t total=0;
            auto* mapping=mappings.data()+tile*bins;
            for(std::size_t b=0;b<bins;++b) {
                total+=histogram[b];mapping[b]=std::int32_t(std::min(double(total)*scale,16383.));
            }
        }
    });
    const auto width=shape.back(),rows=count/width,edges=std::size_t(1)<<ndim;
    detail::partition(rows,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        std::vector<const std::int32_t*> maps(edges);
        std::vector<double> fractions(ndim),factors(edges*(ndim-1));
        std::vector<std::size_t> block(ndim);
        for(std::size_t row=begin;row<end;++row) {
            for(std::size_t d=0;d+1<ndim;++d) {
                const auto padded=(row*width/source_strides[d]%shape[d])+kernel[d]/2;
                block[d]=padded/kernel[d];fractions[d]=double(padded%kernel[d])/double(kernel[d]);
            }
            for(std::size_t x=0;x<width;) {
                const auto offset=(x+kernel.back()/2)%kernel.back();
                block.back()=(x+kernel.back()/2)/kernel.back();
                const auto length=std::min(width-x,kernel.back()-offset);
                for(std::size_t edge=0;edge<edges;++edge) {
                    std::size_t tile=0;
                    for(std::size_t d=0;d<ndim;++d) {
                        const auto upper=(edge>>(ndim-1-d))&1;
                        const auto coordinate=block[d]+upper;
                        tile+=std::min(coordinate ? coordinate-1 : 0,tile_shape[d]-1)*tile_strides[d];
                    }
                    maps[edge]=mappings.data()+tile*bins;
                    for(std::size_t k=0;k+1<ndim;++k) {
                        const auto d=ndim-2-k,upper=(edge>>(k+1))&1;
                        factors[edge*(ndim-1)+k]=upper ? fractions[d] : 1.-fractions[d];
                    }
                }
                const auto base=row*width+x;std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())done=simd::clahe_line(quantized.data()+base,mapped.data()+base,length,
                    maps.data(),factors.data(),ndim,offset,kernel.back());
#endif
                for(std::size_t i=done;i<length;++i) {
                    const auto value=quantized[base+i];
                    const double fraction=double(offset+i)/double(kernel.back());
                    float total=0;
                    for(std::size_t edge=0;edge<edges;++edge) {
                        double coefficient=edge&1 ? fraction : 1.-fraction;
                        for(std::size_t k=0;k+1<ndim;++k)coefficient*=factors[edge*(ndim-1)+k];
                        total+=float(double(maps[edge][value])*coefficient);
                    }
                    mapped[base+i]=std::uint16_t(total);
                }
                x+=length;
            }
        }
    });
    const auto result_bounds=std::minmax_element(mapped.begin(),mapped.end());
    const auto minimum=*result_bounds.first,maximum=*result_bounds.second;
    detail::partition(count,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        if(single) {
            auto* p=static_cast<float*>(output);
            for(auto i=begin;i<end;++i)p[i]=minimum==maximum ? std::min(float(mapped[i]),1.f) :
                float(mapped[i]-minimum)/float(maximum-minimum);
        } else {
            auto* p=static_cast<double*>(output);
            for(auto i=begin;i<end;++i)p[i]=minimum==maximum ? std::min(double(mapped[i]),1.) :
                double(mapped[i]-minimum)/double(maximum-minimum);
        }
    });
}
}
