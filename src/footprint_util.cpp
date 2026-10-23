#include "skimagexx/footprint_util.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
template<std::size_t Bytes> void strided(unsigned char* out,const unsigned char* in,std::ptrdiff_t step,std::size_t count) {
    for(std::size_t i=0;i<count;++i)std::memcpy(out+i*Bytes,in+std::ptrdiff_t(i)*step,Bytes);
}
void copy(unsigned char* out,const unsigned char* in,std::ptrdiff_t step,std::size_t count,std::size_t bytes) {
    if(step==std::ptrdiff_t(bytes)){std::memcpy(out,in,count*bytes);return;}
    switch(bytes) {
        case 1:strided<1>(out,in,step,count);break;
        case 2:strided<2>(out,in,step,count);break;
        case 4:strided<4>(out,in,step,count);break;
        case 8:strided<8>(out,in,step,count);break;
        case 16:strided<16>(out,in,step,count);break;
        default:for(std::size_t i=0;i<count;++i)std::memcpy(out+i*bytes,in+std::ptrdiff_t(i)*step,bytes);
    }
}
}
void footprint_pad_into(const FootprintPadView& view) {
    const auto rank=view.shape.size();
    if(!rank || view.before.size()!=rank || view.input_strides.size()!=rank || view.output_strides.size()!=rank)
        throw std::invalid_argument("one footprint padding and stride per axis required");
    if(!view.itemsize)return;
    std::size_t input_count=1,output_count=1;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    for(std::size_t axis=0;axis<rank;++axis) {
        const auto size=view.shape[axis],padding=std::size_t(size%2==0);
        if(view.before[axis]>padding)throw std::invalid_argument("invalid footprint before padding");
        if(size>limit-padding || size+padding>limit/output_count)throw std::overflow_error("footprint dimensions overflow");
        output_count*=size+padding;
        if(size && input_count>limit/size)throw std::overflow_error("footprint dimensions overflow");
        input_count*=size;
    }
    if(view.itemsize && output_count>limit/view.itemsize)throw std::overflow_error("footprint bytes overflow");
    if(!view.output || !view.zero || (input_count && !view.input))throw std::invalid_argument("nonnull footprint buffers required");
    const auto* atom=static_cast<const unsigned char*>(view.zero);
    auto* output=static_cast<unsigned char*>(view.output);
    const bool zero=std::all_of(atom,atom+view.itemsize,[](auto value){return value==0;});
    // Assign every destination byte once: initialize only the border, and
    // copy interior chunks directly. Large images avoid a redundant full
    // zero pass and a separate parallel-team barrier.
    const auto fill=[&](unsigned char* target,std::size_t count) {
        const auto bytes=count*view.itemsize;
        if(zero)std::memset(target,0,bytes);
        else if(bytes) {
            std::memcpy(target,atom,view.itemsize);
            for(auto done=view.itemsize;done<bytes;) {
                const auto next=std::min(done,bytes-done);std::memcpy(target+done,target,next);done+=next;
            }
        }
    };
    const auto inner=view.fortran ? std::size_t{0}:rank-1,width=view.shape[inner],out_width=width+(width%2==0);
    const auto chunk=std::max(std::size_t{1},std::size_t{65536}/view.itemsize);
    const auto chunks=1+(out_width-1)/chunk,rows=output_count/out_width,tasks=rows*chunks;
    detail::partition(tasks,detail::active_workers(output_count),[&](auto begin,auto end,auto){
        for(auto task=begin;task<end;++task) {
            auto row=task/chunks;const auto first=(task%chunks)*chunk,last=std::min(first+chunk,out_width);
            std::ptrdiff_t source=0,dest=std::ptrdiff_t(first)*view.output_strides[inner];bool interior=true;
            for(std::size_t k=0;k<rank;++k) {
                const auto axis=view.fortran ? k:rank-1-k;if(axis==inner)continue;
                const auto out_size=view.shape[axis]+(view.shape[axis]%2==0);
                const auto coordinate=row%out_size;row/=out_size;dest+=std::ptrdiff_t(coordinate)*view.output_strides[axis];
                if(coordinate<view.before[axis] || coordinate-view.before[axis]>=view.shape[axis])interior=false;
                else source+=std::ptrdiff_t(coordinate-view.before[axis])*view.input_strides[axis];
            }
            auto* target=output+dest;
            if(!interior){fill(target,last-first);continue;}
            const auto low=std::max(first,view.before[inner]),high=std::min(last,view.before[inner]+width);
            if(low>=high){fill(target,last-first);continue;}
            fill(target,low-first);
            source+=std::ptrdiff_t(low-view.before[inner])*view.input_strides[inner];
            copy(target+(low-first)*view.itemsize,static_cast<const unsigned char*>(view.input)+source,view.input_strides[inner],high-low,view.itemsize);
            fill(target+(high-first)*view.itemsize,last-high);
        }
    });
}
}
