#include "skimagexx/adaptive.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx { namespace {
std::size_t checked_product(std::size_t a,std::size_t b) {
    if(b && a>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)/b)throw std::overflow_error("adaptive dimensions overflow");
    return a*b;
}
std::ptrdiff_t mapped(std::ptrdiff_t x,std::size_t size,GaussianBorder mode) {
    const auto n=std::ptrdiff_t(size);
    if(x>=0 && x<n)return x;
    if(mode==GaussianBorder::constant)return -1;
    if(mode==GaussianBorder::nearest)return std::clamp(x,std::ptrdiff_t(0),n-1);
    if(mode==GaussianBorder::wrap){x%=n;return x<0 ? x+n : x;}
    if(mode==GaussianBorder::mirror && n==1)return 0;
    const auto period=mode==GaussianBorder::reflect ? 2*n : 2*(n-1);
    x%=period;if(x<0)x+=period;
    return x<n ? x : mode==GaussianBorder::reflect ? period-1-x : period-x;
}
std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> out(shape.size(),1);
    for(auto i=shape.size();i-->1;)out[i-1]=checked_product(out[i],shape[i]);
    return out;
}
template<class T> void mean_axis(const T* input,T* output,std::size_t count,const std::vector<std::size_t>& shape,
    std::size_t axis,std::size_t window,GaussianBorder border,double cval,int workers) {
    const auto pitch=strides(shape);const auto length=shape[axis],step=pitch[axis];
    if(window>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-length)throw std::overflow_error("adaptive window overflow");
    const auto padded=length+window-1,lines=count/length,groups=(lines+3)/4;
    std::vector<std::ptrdiff_t> mapping(padded);
    for(std::size_t x=0;x<padded;++x)mapping[x]=mapped(std::ptrdiff_t(x)-std::ptrdiff_t(window/2),length,border);
    const auto team=groups<std::size_t(workers) ? int(groups) : workers;
    const auto capacity=std::min(std::size_t(4),lines);
    detail::partition(groups,team,[&](auto begin,auto end,auto) {
        std::vector<double> buffer(checked_product(padded,capacity)),result(checked_product(length,capacity));
        for(auto group=begin;group<end;++group) {
            const auto first=group*4,lanes=std::min(std::size_t(4),lines-first);
            std::size_t base[4]{};
            for(std::size_t lane=0;lane<lanes;++lane){const auto line=first+lane;base[lane]=(line/step)*length*step+line%step;}
            for(std::size_t x=0;x<padded;++x)for(std::size_t lane=0;lane<lanes;++lane)
                buffer[x*lanes+lane]=mapping[x]<0 ? cval : double(input[base[lane]+std::size_t(mapping[x])*step]);
            bool done=false;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2() && lanes==4){simd::adaptive_mean4(buffer.data(),result.data(),length,window);done=true;}
#endif
            if(!done)for(std::size_t lane=0;lane<lanes;++lane) {
                double total=0;for(std::size_t x=0;x<window;++x)total+=buffer[x*lanes+lane];
                result[lane]=total/double(window);
                for(std::size_t x=1;x<length;++x){total+=buffer[(x+window-1)*lanes+lane]-buffer[(x-1)*lanes+lane];result[x*lanes+lane]=total/double(window);}
            }
            for(std::size_t x=0;x<length;++x)for(std::size_t lane=0;lane<lanes;++lane)output[base[lane]+x*step]=T(result[x*lanes+lane]);
        }
    });
}
template<class T> void median(const T* input,T* output,std::size_t count,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& window,GaussianBorder border,double cval,int workers,bool zero_strides) {
    std::size_t area=1;std::vector<std::size_t> padded(shape.size());
    for(std::size_t d=0;d<shape.size();++d){area=checked_product(area,window[d]);if(window[d]>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-shape[d])throw std::overflow_error("adaptive padding overflow");padded[d]=shape[d]+window[d]-1;}
    if(area==1){std::copy_n(input,count,output);return;}
    if(count==1){output[0]=border==GaussianBorder::constant ? T(cval) : input[0];return;}
    if(zero_strides) {
        // SciPy's zero-stride offset table has sentinel=max_size*0+1.
        // Mapped coordinate 1 consequently selects cval even in nonconstant
        // modes. Interior coordinates share the table at the window center.
        const auto ip=strides(shape);std::vector<std::vector<std::size_t>> valid(shape.size());
        for(std::size_t d=0;d<shape.size();++d){valid[d].resize(shape[d]);for(std::size_t x=0;x<shape[d];++x){const auto radius=window[d]/2;const auto representative=window[d]>=shape[d] || x<radius || x>=shape[d]-radius ? x : radius;for(std::size_t k=0;k<window[d];++k){const auto coordinate=mapped(std::ptrdiff_t(representative)+std::ptrdiff_t(k)-std::ptrdiff_t(radius),shape[d],border);valid[d][x]+=coordinate>=0 && coordinate!=1;}}}
        const double value=input[0];
        detail::partition(count,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){auto remainder=i;std::size_t good=1;for(std::size_t d=0;d<shape.size();++d){good*=valid[d][remainder/ip[d]];remainder%=ip[d];}const bool select_value=value<cval ? area/2<good : area/2>=area-good;output[i]=T(select_value ? value : cval);}});
        return;
    }
    const auto pp=strides(padded),ip=strides(shape);
    std::size_t padded_count=1;for(auto n:padded)padded_count=checked_product(padded_count,n);
    std::vector<T> storage(padded_count);
    detail::partition(padded_count,workers,[&](auto begin,auto end,auto) {
        for(auto i=begin;i<end;++i) {
            auto remainder=i;std::size_t position=0;bool constant=false;
            for(std::size_t d=0;d<shape.size();++d){const auto coordinate=remainder/pp[d];remainder%=pp[d];const auto x=mapped(std::ptrdiff_t(coordinate)-std::ptrdiff_t(window[d]/2),shape[d],border);if(x<0){constant=true;break;}position+=std::size_t(x)*ip[d];}
            storage[i]=constant ? T(cval) : input[position];
        }
    });
    std::vector<std::size_t> offsets(area);
    for(std::size_t tap=0;tap<area;++tap){auto remainder=tap;for(auto d=shape.size();d-->0;){offsets[tap]+=(remainder%window[d])*pp[d];remainder/=window[d];}}
    if(shape.size()==2 && window[0]==3 && window[1]==3) {
        detail::partition(shape[0],workers,[&](auto begin,auto end,auto) {
            for(auto row=begin;row<end;++row){std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())done=simd::adaptive_median9(storage.data()+row*pp[0],output+row*shape[1],shape[1],pp[0],std::is_same_v<T,float>);
#endif
                for(auto x=done;x<shape[1];++x){T values[9];for(std::size_t k=0;k<9;++k)values[k]=storage[row*pp[0]+x+offsets[k]];std::nth_element(values,values+4,values+9);output[row*shape[1]+x]=values[4];}
            }
        });return;
    }
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
        std::vector<T> values(area);
        for(auto i=begin;i<end;++i){auto remainder=i;std::size_t base=0;for(std::size_t d=0;d<shape.size();++d){base+=(remainder/ip[d])*pp[d];remainder%=ip[d];}for(std::size_t tap=0;tap<area;++tap)values[tap]=storage[base+offsets[tap]];std::nth_element(values.begin(),values.begin()+std::ptrdiff_t(area/2),values.end());output[i]=values[area/2];}
    });
}
template<class T> void run(const T* input,T* output,std::size_t count,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& window,const std::vector<GaussianBorder>& modes,AdaptiveMethod method,double cval,bool zero_strides) {
    const auto workers=detail::active_workers(count);
    if(method==AdaptiveMethod::median){median(input,output,count,shape,window,modes.empty()?GaussianBorder::reflect:modes[0],cval,workers,zero_strides);return;}
    std::vector<T> scratch;const T* current=input;
    const auto active=std::count_if(window.begin(),window.end(),[](auto w){return w>1;});std::size_t pass=0;
    if(active>1)scratch.resize(count);
    for(std::size_t axis=0;axis<shape.size();++axis)if(window[axis]>1){auto* destination=(std::size_t(active)-pass)%2==1 ? output : scratch.data();mean_axis(current,destination,count,shape,axis,window[axis],modes[axis],cval,workers);current=destination;++pass;}
    if(!active)std::copy_n(input,count,output);
}
template<class T,class U> void offset(const T* input,U* output,std::size_t count,double value) {
    const U scalar=U(value);
    detail::partition(count,detail::active_workers(count),[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())begin+=simd::adaptive_offset(input+begin,output+begin,end-begin,std::is_same_v<T,float>,std::is_same_v<U,float>,value);
#endif
        for(auto i=begin;i<end;++i)output[i]=U(U(input[i])-scalar);
    });
}
}}
namespace skimagexx {
void adaptive_filter_into(const void* input,void* output,PixelType type,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& window,const std::vector<GaussianBorder>& modes,AdaptiveMethod method,double cval,bool zero_strides) {
    if(shape.size()!=window.size() || shape.size()!=modes.size())throw std::invalid_argument("adaptive rank mismatch");
    if(type!=PixelType::float32 && type!=PixelType::float64)throw std::invalid_argument("adaptive floating input required");
    if(method!=AdaptiveMethod::mean && method!=AdaptiveMethod::median)throw std::invalid_argument("unknown adaptive method");
    for(std::size_t d=0;d<shape.size();++d)if(!window[d] || unsigned(modes[d])>4)throw std::invalid_argument("invalid adaptive window or mode");
    if(method==AdaptiveMethod::median)for(std::size_t d=0;d<shape.size();++d)if(window[d]%2==0 || (std::any_of(window.begin(),window.end(),[](auto w){return w>1;}) && modes[d]!=modes[0]))throw std::invalid_argument("median requires odd windows and one border mode");
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return;
    std::size_t count=1;for(auto n:shape)count=checked_product(count,n);
    if(!count)return;if(!input || !output)throw std::invalid_argument("nonnull adaptive buffers required");
    if(type==PixelType::float32)run(static_cast<const float*>(input),static_cast<float*>(output),count,shape,window,modes,method,cval,zero_strides);
    else run(static_cast<const double*>(input),static_cast<double*>(output),count,shape,window,modes,method,cval,zero_strides);
}
void adaptive_offset_into(const void* input,void* output,std::size_t count,PixelType source,PixelType target,double value) {
    if((source!=PixelType::float32 && source!=PixelType::float64)||(target!=PixelType::float32 && target!=PixelType::float64))throw std::invalid_argument("adaptive floating offset required");
    if(!count)return;if(!input || !output)throw std::invalid_argument("nonnull adaptive offset buffers required");
    checked_product(count,1);
    if(source==PixelType::float32){if(target==PixelType::float32)offset(static_cast<const float*>(input),static_cast<float*>(output),count,value);else offset(static_cast<const float*>(input),static_cast<double*>(output),count,value);}
    else {if(target==PixelType::float32)offset(static_cast<const double*>(input),static_cast<float*>(output),count,value);else offset(static_cast<const double*>(input),static_cast<double*>(output),count,value);}
}
}
