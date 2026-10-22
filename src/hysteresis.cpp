#include "skimagexx/hysteresis.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx { namespace {
std::size_t extent(const std::vector<std::size_t>& shape) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t n=1;for(auto d:shape){if(d>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(std::size_t)/n)throw std::overflow_error("hysteresis dimensions overflow");n*=d;}return n;
}
std::vector<std::size_t> pitches(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> out(shape.size(),1);for(auto d=shape.size();d-->1;)out[d-1]=out[d]*shape[d];return out;
}
std::ptrdiff_t linear(const HysteresisView& v,const std::vector<std::size_t>& shape) {
    if(std::all_of(v.strides.begin(),v.strides.end(),[](auto s){return s==0;}))return 0;
    std::ptrdiff_t expected=1;for(auto d=shape.size();d-->0;){if(shape[d]>1 && v.strides[d]!=expected)return -1;expected*=std::ptrdiff_t(shape[d]);}return 1;
}
std::ptrdiff_t offset(std::size_t i,const HysteresisView& v,const std::vector<std::size_t>& shape,std::ptrdiff_t step) {
    if(step>=0)return std::ptrdiff_t(i)*step;
    std::ptrdiff_t out=0;for(auto d=shape.size();d-->0;){out+=std::ptrdiff_t(i%shape[d])*v.strides[d];i/=shape[d];}return out;
}
template<class T> T value(const HysteresisView& view,std::ptrdiff_t i) {
    const auto x=static_cast<const T*>(view.data)[i];
    if constexpr(std::is_same_v<T,std::uint8_t>)return view.type==PixelType::boolean ? T(x!=0) : x;
    else return x;
}
template<class T> void map(const HysteresisView& a,const HysteresisView& b,void* output,const std::vector<std::size_t>& shape,bool minimum) {
    const auto n=extent(shape);if(!n)return;
    const auto ap=linear(a,shape),bp=linear(b,shape);
    const bool flat=ap>=0 && bp>=0;
    std::size_t inner=shape.size();while(inner && shape[inner-1]==1)--inner;
    const auto width=inner ? shape[inner-1] : n;
    const auto as=flat ? ap : a.strides[inner-1],bs=flat ? bp : b.strides[inner-1];
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        for(auto position=begin;position<end;) {
            const auto length=flat ? end-position : std::min(end-position,width-position%width);
            const auto ai=offset(position,a,shape,ap),bi=offset(position,b,shape,bp);std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(as>=0 && as<=1 && bs>=0 && bs<=1 && has_avx2())done=simd::hysteresis_map(static_cast<const T*>(a.data)+ai,
                static_cast<const T*>(b.data)+bi,static_cast<unsigned char*>(output)+position*(minimum ? sizeof(T) : 1),length,a.type,as,bs,minimum);
#endif
        for(auto k=done;k<length;++k) {
            const auto i=position+k;
            const auto x=value<T>(a,ai+std::ptrdiff_t(k)*as),y=value<T>(b,bi+std::ptrdiff_t(k)*bs);
            if(minimum) {
                T result;
                if constexpr(std::is_floating_point_v<T>)result=std::isnan(x) ? x : std::isnan(y) ? y : x<y ? x : y;
                else result=x<y ? x : y;
                static_cast<T*>(output)[i]=result;
            } else static_cast<std::uint8_t*>(output)[i]=x>y;
        }
        position+=length;
        }
    });
}
void dispatch(const HysteresisView& a,const HysteresisView& b,void* out,const std::vector<std::size_t>& shape,bool minimum) {
    if(a.strides.size()!=shape.size() || b.strides.size()!=shape.size())throw std::invalid_argument("hysteresis view rank mismatch");
    if(extent(shape) && (!a.data || !b.data || !out))throw std::invalid_argument("nonnull hysteresis buffers required");
    if(a.type!=b.type) {
        if(minimum || !((a.type==PixelType::int64 && b.type==PixelType::uint64) || (a.type==PixelType::uint64 && b.type==PixelType::int64)))throw std::invalid_argument("unsupported mixed hysteresis comparison");
        const auto n=extent(shape);const auto ap=linear(a,shape),bp=linear(b,shape);
        detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){
            const auto ai=offset(i,a,shape,ap),bi=offset(i,b,shape,bp);bool result;
            if(a.type==PixelType::int64){const auto x=value<std::int64_t>(a,ai);const auto y=value<std::uint64_t>(b,bi);result=x>=0 && std::uint64_t(x)>y;}
            else {const auto x=value<std::uint64_t>(a,ai);const auto y=value<std::int64_t>(b,bi);result=y<0 || x>std::uint64_t(y);}
            static_cast<std::uint8_t*>(out)[i]=result;
        }});return;
    }
    switch(a.type) {
#define MAP(tag,T) case PixelType::tag:map<T>(a,b,out,shape,minimum);return;
    MAP(boolean,std::uint8_t) MAP(uint8,std::uint8_t) MAP(int8,std::int8_t) MAP(uint16,std::uint16_t) MAP(int16,std::int16_t)
    MAP(uint32,std::uint32_t) MAP(int32,std::int32_t) MAP(uint64,std::uint64_t) MAP(int64,std::int64_t) MAP(float32,float) MAP(float64,double)
#undef MAP
    }
    throw std::invalid_argument("unsupported hysteresis dtype");
}
std::size_t root(std::vector<std::size_t>& parents,std::size_t i) {
    while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;
}
void unite(std::vector<std::size_t>& parents,std::size_t a,std::size_t b) {
    a=root(parents,a);b=root(parents,b);if(a>b)std::swap(a,b);parents[b]=a;
}
}
void hysteresis_minimum_into(const HysteresisView& a,const HysteresisView& b,void* out,const std::vector<std::size_t>& shape){dispatch(a,b,out,shape,true);}
void hysteresis_greater_into(const HysteresisView& a,const HysteresisView& b,std::uint8_t* out,const std::vector<std::size_t>& shape){dispatch(a,b,out,shape,false);}
void hysteresis_connect_into(const std::uint8_t* weak,const std::uint8_t* strong,std::uint8_t* out,const std::vector<std::size_t>& shape) {
    const auto n=extent(shape);if(!n)return;
    if(!weak || !strong || !out)throw std::invalid_argument("nonnull hysteresis masks required");
    const auto workers=detail::active_workers(n);struct Counts{std::size_t weak=0,seeds=0;bool background=false;};
    std::vector<Counts> totals(std::size_t(workers),Counts{});
    detail::partition(n,workers,[&](auto begin,auto end,auto worker){Counts c;for(auto i=begin;i<end;++i){c.weak+=weak[i]!=0;c.seeds+=weak[i] && strong[i];c.background|=!weak[i] && strong[i];}totals[worker]=c;});
    Counts counts;for(auto c:totals){counts.weak+=c.weak;counts.seeds+=c.seeds;counts.background|=c.background;}
    if(!counts.seeds || counts.seeds==counts.weak || counts.weak==n) {
        detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)out[i]=weak[i] ? counts.seeds && (counts.seeds==counts.weak || counts.weak==n) : counts.background;});return;
    }
    const auto pitch=pitches(shape);
    if(workers==1) {
        std::vector<std::size_t> queue;queue.reserve(counts.weak);
        for(std::size_t i=0;i<n;++i){out[i]=weak[i] ? strong[i]!=0 : counts.background;if(weak[i] && strong[i])queue.push_back(i);}
        for(std::size_t head=0;head<queue.size();++head){const auto i=queue[head];
            auto visit=[&](auto j){if(weak[j] && !out[j]){out[j]=1;queue.push_back(j);}};
            if(shape.size()==2){const auto width=shape[1],column=i%width;
                if(i>=width)visit(i-width);if(i<n-width)visit(i+width);
                if(column)visit(i-1);if(column+1<width)visit(i+1);
            } else if(shape.size()==1){if(i)visit(i-1);if(i+1<n)visit(i+1);}
            else for(std::size_t d=0;d<shape.size();++d){const auto coordinate=i/pitch[d]%shape[d];
                if(coordinate){const auto j=i-pitch[d];if(weak[j] && !out[j]){out[j]=1;queue.push_back(j);}}
                if(coordinate+1<shape[d]){const auto j=i+pitch[d];if(weak[j] && !out[j]){out[j]=1;queue.push_back(j);}}
            }
        }return;
    }
    std::vector<std::size_t> parents(n);std::vector<std::pair<std::size_t,std::size_t>> segments(std::size_t(workers),{0,0});
    detail::partition(n,workers,[&](auto begin,auto end,auto worker){segments[worker]={begin,end};
        for(auto i=begin;i<end;++i)parents[i]=i;
        if(shape.size()==2){const auto width=shape[1];for(auto i=begin;i<end;++i)if(weak[i]){
            if(i>=width && i-width>=begin && weak[i-width])unite(parents,i,i-width);
            if(i%width && i>begin && weak[i-1])unite(parents,i,i-1);
        }} else if(shape.size()==1){for(auto i=begin;i<end;++i)if(weak[i] && i>begin && weak[i-1])unite(parents,i,i-1);}
        else for(auto i=begin;i<end;++i)if(weak[i])for(std::size_t d=0;d<shape.size();++d)if((i/pitch[d])%shape[d] && i-pitch[d]>=begin && weak[i-pitch[d]])unite(parents,i,i-pitch[d]);
    });
    for(auto segment:segments)for(std::size_t d=0;d<shape.size();++d)if(shape[d]>1){const auto end=segment.first+std::min(pitch[d],segment.second-segment.first);
        for(auto i=segment.first;i<end;++i)if((i/pitch[d])%shape[d] && weak[i] && weak[i-pitch[d]])unite(parents,i,i-pitch[d]);
    }
    std::vector<std::uint8_t> selected(n,0);
    for(std::size_t i=0;i<n;++i)if(weak[i]){parents[i]=root(parents,i);if(strong[i])selected[parents[i]]=1;}
    detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)out[i]=weak[i] ? selected[parents[i]] : counts.background;});
}
void hysteresis_threshold_into(const void* image,const void* low,const void* high,std::uint8_t* out,PixelType type,
    const std::vector<std::size_t>& shape,bool low_scalar,bool high_scalar) {
    const auto n=extent(shape);if(!n)return;
    const auto p=pitches(shape);std::vector<std::ptrdiff_t> stride(p.begin(),p.end()),zero(shape.size(),0);
    HysteresisView a{image,type,stride},l{low,type,low_scalar ? zero : stride},h{high,type,high_scalar ? zero : stride};
    auto run=[&](auto sample){using T=decltype(sample);std::vector<T> clipped(n);std::vector<std::uint8_t> weak(n),strong(n);
        hysteresis_minimum_into(l,h,clipped.data(),shape);HysteresisView c{clipped.data(),type,stride};
        hysteresis_greater_into(a,c,weak.data(),shape);hysteresis_greater_into(a,h,strong.data(),shape);hysteresis_connect_into(weak.data(),strong.data(),out,shape);
    };
    switch(type) {
#define RUN(tag,T) case PixelType::tag:run(T{});return;
    RUN(boolean,std::uint8_t) RUN(uint8,std::uint8_t) RUN(int8,std::int8_t) RUN(uint16,std::uint16_t) RUN(int16,std::int16_t)
    RUN(uint32,std::uint32_t) RUN(int32,std::int32_t) RUN(uint64,std::uint64_t) RUN(int64,std::int64_t) RUN(float32,float) RUN(float64,double)
#undef RUN
    }
    throw std::invalid_argument("unsupported hysteresis dtype");
}
}
