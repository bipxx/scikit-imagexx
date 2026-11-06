#include "skimagexx/array_views.hpp"
#include "skimagexx/slice_along_axes.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace skimagexx {
namespace {
using I=std::ptrdiff_t;
void geometry(const CropView& a) {
    if(a.shape.size()!=a.strides.size())throw std::invalid_argument("invalid view geometry");
    for(auto n:a.shape)if(n<0)throw std::invalid_argument("negative view shape");
}
I scale(I stride,I positive) {
    const auto lo=std::numeric_limits<I>::min(),hi=std::numeric_limits<I>::max();
    if(positive<1)throw std::invalid_argument("view step must be positive");
    if((stride>0&&stride>hi/positive)||(stride<0&&stride<lo/positive))throw std::overflow_error("view stride overflow");
    return stride*positive;
}
}
CropView view_as_blocks(const CropView& a,const std::vector<I>& block) {
    geometry(a);
    if(block.size()!=a.shape.size())throw std::invalid_argument("block rank mismatch");
    CropView out=a;out.shape.clear();out.strides.clear();
    for(std::size_t i=0;i<block.size();++i) {
        if(block[i]<1||a.shape[i]%block[i])throw std::invalid_argument("invalid block shape");
        out.shape.push_back(a.shape[i]/block[i]);out.strides.push_back(scale(a.strides[i],block[i]));
    }
    out.shape.insert(out.shape.end(),block.begin(),block.end());
    out.strides.insert(out.strides.end(),a.strides.begin(),a.strides.end());return out;
}
CropView view_as_windows(const CropView& a,const std::vector<I>& window,const std::vector<I>& step) {
    geometry(a);
    if(window.size()!=a.shape.size()||step.size()!=a.shape.size())throw std::invalid_argument("window rank mismatch");
    CropView out=a;out.shape.clear();out.strides.clear();
    for(std::size_t i=0;i<window.size();++i) {
        if(window[i]<1||window[i]>a.shape[i]||step[i]<1)throw std::invalid_argument("invalid window shape or step");
        out.shape.push_back((a.shape[i]-window[i])/step[i]+1);out.strides.push_back(scale(a.strides[i],step[i]));
    }
    out.shape.insert(out.shape.end(),window.begin(),window.end());
    out.strides.insert(out.strides.end(),a.strides.begin(),a.strides.end());return out;
}
SliceAlongAxesResult slice_along_axes(const CropView& a,const std::vector<SliceAlongAxesPair>& bounds,const std::vector<I>* supplied,bool copy) {
    SliceAlongAxesResult out;out.plan.view=a;
    if(bounds.empty())return out; // Identity shortcut intentionally ignores other arguments.
    geometry(a);std::vector<I> axes;
    if(supplied) {axes=*supplied;if(axes.size()!=bounds.size())throw std::invalid_argument("axes and slices length mismatch");}
    else {if(bounds.size()>a.shape.size())throw std::invalid_argument("too many slices");for(std::size_t i=0;i<a.shape.size();++i)axes.push_back(I(i));}
    auto sorted=axes;std::sort(sorted.begin(),sorted.end());
    if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end())throw std::invalid_argument("duplicate slice axis");
    for(auto ax:axes)if(ax<0||std::size_t(ax)>=a.shape.size())throw std::invalid_argument("slice axis out of range");
    std::vector<CropSlice> slices;for(auto n:a.shape)slices.push_back({0,n});
    for(std::size_t i=0;i<bounds.size();++i) {
        auto first=bounds[i].first,last=bounds[i].last,n=a.shape[std::size_t(axes[i])];
        auto modulo=[n](I v){if(v>=0)return v;if(!n)throw std::domain_error("negative slice bound on zero axis");auto r=v%n;return r<0?r+n:r;};
        first=modulo(first);last=modulo(last);
        if(first>last||last>n)throw std::invalid_argument("invalid slice bounds");
        slices[std::size_t(axes[i])]={first,last};
    }
    out.plan=crop_view(a,slices);out.kind=SliceAlongAxesResult::Kind::borrowed;
    if(!copy)return out;
    auto& v=out.plan.view;
    if(a.itemsize>std::size_t(std::numeric_limits<I>::max()))throw std::overflow_error("slice itemsize overflow");
    I stride=I(a.itemsize);
    std::vector<I> dense(v.shape.size());
    bool empty=std::find(v.shape.begin(),v.shape.end(),0)!=v.shape.end();
    for(std::size_t i=v.shape.size();i-->0;) {dense[i]=stride;if(v.shape[i]&&stride>std::numeric_limits<I>::max()/v.shape[i])throw std::overflow_error("slice copy size overflow");stride*=v.shape[i];}
    out.payload.resize(empty?0:std::size_t(stride));
    if(!empty)crop_copy_into(a,slices,out.payload.data(),dense);
    v.strides=std::move(dense);out.kind=SliceAlongAxesResult::Kind::owned;return out;
}
}
