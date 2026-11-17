#include "skimagexx/crop.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
std::ptrdiff_t add(std::ptrdiff_t a,std::ptrdiff_t b) {
    const auto lo=std::numeric_limits<std::ptrdiff_t>::min(),hi=std::numeric_limits<std::ptrdiff_t>::max();
    if((b>0 && a>hi-b)||(b<0 && a<lo-b))throw std::overflow_error("crop offset overflow");
    return a+b;
}
std::ptrdiff_t product(std::ptrdiff_t index,std::ptrdiff_t stride) {
    if(index==0)return 0;
    const auto lo=std::numeric_limits<std::ptrdiff_t>::min(),hi=std::numeric_limits<std::ptrdiff_t>::max();
    if((stride>0 && index>hi/stride)||(stride<0 && stride<lo/index))throw std::overflow_error("crop stride overflow");
    return index*stride;
}
// Constant word sizes let the compiler inline POD loads/stores without
// alignment/type-punning assumptions. Only line endpoints need checked offsets.
template<std::size_t Bytes>
void copy_line(char* target,const char* source,std::size_t count,
               std::ptrdiff_t target_step,std::ptrdiff_t source_step) {
    for(std::size_t i=0;i<count;++i) {
        std::memcpy(target,source,Bytes);
        if(i+1<count){target+=target_step;source+=source_step;}
    }
}
void copy_line(char* target,const char* source,std::size_t count,
               std::ptrdiff_t target_step,std::ptrdiff_t source_step,std::size_t bytes) {
    switch(bytes) {
    case 1:return copy_line<1>(target,source,count,target_step,source_step);
    case 2:return copy_line<2>(target,source,count,target_step,source_step);
    case 4:return copy_line<4>(target,source,count,target_step,source_step);
    case 8:return copy_line<8>(target,source,count,target_step,source_step);
    case 16:return copy_line<16>(target,source,count,target_step,source_step);
    case 32:return copy_line<32>(target,source,count,target_step,source_step);
    default:
        for(std::size_t i=0;i<count;++i) {
            std::memcpy(target,source,bytes);
            if(i+1<count){target+=target_step;source+=source_step;}
        }
    }
}
std::ptrdiff_t clip(std::ptrdiff_t value,std::ptrdiff_t size) {
    if(value<0)value=add(value,size);
    return std::max(std::ptrdiff_t(0),std::min(value,size));
}
}
CropPlan crop_view(const CropView& input,const std::vector<CropSlice>& slices) {
    if(input.shape.size()!=input.strides.size()||slices.size()!=input.shape.size()||input.itemsize==0)throw std::invalid_argument("crop geometry mismatch");
    CropPlan result;result.view=input;result.axes.resize(slices.size());std::ptrdiff_t offset=0;bool empty=false;
    for(std::size_t i=0;i<slices.size();++i) {
        if(input.shape[i]<0)throw std::invalid_argument("negative crop shape");
        auto start=clip(slices[i].start,input.shape[i]),stop=clip(slices[i].stop,input.shape[i]);
        stop=std::max(start,stop);result.axes[i]={start,stop};result.view.shape[i]=stop-start;
        empty|=stop==start;offset=add(offset,product(start,input.strides[i]));
    }
    if(!empty && offset)result.view.data=static_cast<const char*>(input.data)+offset;
    return result;
}
void crop_copy_into(const CropView& input,const std::vector<CropSlice>& slices,void* output,const std::vector<std::ptrdiff_t>& strides) {
    const auto plan=crop_view(input,slices);const auto& view=plan.view;
    if(strides.size()!=view.shape.size())throw std::invalid_argument("crop output rank mismatch");
    for(auto n:view.shape)if(n==0)return;
    if(!input.data||!output)throw std::invalid_argument("null crop payload");
    if(view.shape.empty()){std::memcpy(output,view.data,view.itemsize);return;}
    const auto rank=view.shape.size();std::vector<std::ptrdiff_t> index(rank,0);
    // Preserve bulk C/F runs. Otherwise traverse a whole destination-axis
    // line, including reversed/Fortran source, without a rank-wide checked
    // offset walk and out-of-line variable-size memcpy for every element.
    auto unit=[&](std::size_t axis){return view.shape[axis]>1 && view.strides[axis]==std::ptrdiff_t(view.itemsize) && strides[axis]==std::ptrdiff_t(view.itemsize);};
    std::size_t run_axis=rank-1;
    if(!unit(run_axis))for(std::size_t i=0;i<rank-1;++i)if(unit(i)){run_axis=i;break;}
    const bool contiguous=unit(run_axis);
    if(!contiguous && strides[run_axis]!=std::ptrdiff_t(view.itemsize))
        for(std::size_t i=0;i<rank-1;++i)if(strides[i]==std::ptrdiff_t(view.itemsize)){run_axis=i;break;}
    const auto width=std::size_t(view.shape[run_axis]);
    if(width>std::numeric_limits<std::size_t>::max()/view.itemsize)throw std::overflow_error("crop run overflow");
    const auto source_last=product(std::ptrdiff_t(width-1),view.strides[run_axis]);
    const auto target_last=product(std::ptrdiff_t(width-1),strides[run_axis]);
    for(;;) {
        std::ptrdiff_t from=0,to=0;
        for(std::size_t i=0;i<rank;++i){from=add(from,product(index[i],view.strides[i]));to=add(to,product(index[i],strides[i]));}
        // Validate endpoints before entering a monotone pointer-stride line.
        // Intermediate offsets lie between the checked first/last endpoints.
        add(from,source_last);add(to,target_last);
        auto* target=static_cast<char*>(output)+to;
        const auto* source=static_cast<const char*>(view.data)+from;
        if(contiguous)std::memcpy(target,source,width*view.itemsize);
        else copy_line(target,source,width,strides[run_axis],view.strides[run_axis],view.itemsize);
        std::size_t axis=rank;
        while(axis){--axis;index[axis]+=axis==run_axis?std::ptrdiff_t(width):1;if(index[axis]<view.shape[axis])break;index[axis]=0;}
        if(axis==0 && index[0]==0)break;
    }
}
}
