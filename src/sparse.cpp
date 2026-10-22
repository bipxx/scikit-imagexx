#include "skimagexx/sparse.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace skimagexx { namespace {
std::size_t product(const std::vector<std::size_t>& shape) {
    std::size_t n=1;const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
    for(auto d:shape){if(d>limit || (d && n>limit/d))throw std::overflow_error("sparse dimensions overflow");n*=d;}
    return n;
}
std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> p(shape.size(),1);for(auto d=shape.size();d-->1;)p[d-1]=p[d]*shape[d];return p;
}
std::ptrdiff_t mapped(std::ptrdiff_t x,std::size_t length,GaussianBorder border) {
    const auto n=std::ptrdiff_t(length);if(x>=0 && x<n)return x;
    if(border==GaussianBorder::constant)return -1;
    if(!n)throw std::invalid_argument("cannot pad empty sparse axis");
    if(border==GaussianBorder::nearest)return std::clamp(x,std::ptrdiff_t(0),n-1);
    if(border==GaussianBorder::wrap){x%=n;return x<0 ? x+n : x;}
    if(n==1)return 0;
    const auto period=border==GaussianBorder::reflect ? 2*n : 2*(n-1);
    x%=period;if(x<0)x+=period;
    return x<n ? x : border==GaussianBorder::reflect ? period-1-x : period-x;
}
unsigned flags(){const auto f=std::fetestexcept(FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);return unsigned(bool(f&FE_OVERFLOW))|unsigned(bool(f&FE_UNDERFLOW))*2|unsigned(bool(f&FE_INVALID))*4;}
template<class T> std::vector<unsigned> run(const T* input,T* output,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& kernel,const std::vector<std::vector<std::size_t>>& coordinates,
    const T* weights,GaussianBorder border,bool valid) {
    auto padded_shape=shape,out_shape=shape;
    for(std::size_t d=0;d<shape.size();++d){
        if(!valid){const auto padding=2*(kernel[d]/2);if(padding>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-shape[d])throw std::overflow_error("sparse padding overflow");padded_shape[d]+=padding;}
        if(kernel[d]>padded_shape[d]+1)throw std::invalid_argument("sparse kernel exceeds valid input");
        out_shape[d]=padded_shape[d]+1-kernel[d];
    }
    const auto n=product(out_shape),padded_count=product(padded_shape);
    std::vector<unsigned> combined(coordinates.size()*2,0);if(!n)return combined;
    if(!input || !output || !weights)throw std::invalid_argument("nonnull sparse buffers required");
    const auto workers=detail::active_workers(n);const auto pp=strides(padded_shape),ip=strides(shape),op=strides(out_shape);
    std::vector<T> storage;const T* padded=input;
    if(!valid && padded_shape!=shape){
        storage.resize(padded_count);padded=storage.data();
        const auto width=padded_shape.back(),radius=kernel.back()/2;
        std::vector<std::ptrdiff_t> left(radius),right(radius);
        for(std::size_t x=0;x<radius;++x){left[x]=mapped(std::ptrdiff_t(x)-std::ptrdiff_t(radius),shape.back(),border);right[x]=mapped(std::ptrdiff_t(shape.back()+x),shape.back(),border);}
        detail::partition(padded_count/width,workers,[&](auto begin,auto end,auto){
            for(auto row=begin;row<end;++row){auto remainder=row*width;std::size_t source=0;bool constant=false;
                for(std::size_t d=0;d+1<shape.size();++d){const auto c=remainder/pp[d];remainder%=pp[d];const auto x=mapped(std::ptrdiff_t(c)-std::ptrdiff_t(kernel[d]/2),shape[d],border);if(x<0){constant=true;break;}source+=std::size_t(x)*ip[d];}
                auto* destination=storage.data()+row*width;
                if(constant){std::fill_n(destination,width,T(0));continue;}
                for(std::size_t x=0;x<radius;++x)destination[x]=left[x]<0 ? T(0) : input[source+std::size_t(left[x])];
                std::copy_n(input+source,shape.back(),destination+radius);
                for(std::size_t x=0;x<radius;++x)destination[radius+shape.back()+x]=right[x]<0 ? T(0) : input[source+std::size_t(right[x])];
            }
        });
    }
    std::vector<std::size_t> offsets(coordinates.size(),0);
    for(std::size_t t=0;t<coordinates.size();++t)for(std::size_t d=0;d<shape.size();++d)offsets[t]+=coordinates[t][d]*pp[d];
    std::vector<std::vector<unsigned>> statuses(std::size_t(workers),combined);
    detail::partition(n,workers,[&](auto begin,auto end,auto worker){
        const auto capacity=std::min(std::size_t(16384),end-begin);std::vector<T> temporary(capacity);auto& status=statuses[worker];
        struct Span {std::size_t destination,source,length;};std::vector<Span> spans;
        spans.reserve(capacity/out_shape.back()+2);
        for(auto block=begin;block<end;){const auto stop=block+std::min(capacity,end-block),length=stop-block;
            // Row addresses depend on the output tile, not on the kernel tap.
            spans.clear();
            for(auto position=block;position<stop;){const auto span=std::min(stop-position,out_shape.back()-position%out_shape.back());
                auto remainder=position;std::size_t source=0;for(std::size_t d=0;d<shape.size();++d){source+=(remainder/op[d])*pp[d];remainder%=op[d];}
                spans.push_back({position-block,source,span});position+=span;
            }
            for(std::size_t t=0;t<coordinates.size();++t){
                const T weight=weights[t];const bool identity=weight==T(1);
                std::feclearexcept(FE_ALL_EXCEPT);
                for(const auto span:spans){
                    auto* destination=temporary.data()+span.destination;const auto* values=padded+span.source+offsets[t];std::size_t done=0;
                    if(identity)std::copy_n(values,span.length,destination);
                    else {
#ifdef SKIMAGEXX_AVX2
                        if(has_avx2())done=simd::sparse_product(values,destination,span.length,&weight,std::is_same_v<T,float>);
#endif
                        for(auto k=done;k<span.length;++k)destination[k]=T(weight*values[k]);
                    }
                }
                status[2*t]|=flags();
                if(!t)std::copy_n(temporary.data(),length,output+block);
                else {
                    std::feclearexcept(FE_ALL_EXCEPT);std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                    if(has_avx2())done=simd::sparse_add(temporary.data(),output+block,length,std::is_same_v<T,float>);
#endif
                    for(auto k=done;k<length;++k)output[block+k]=T(output[block+k]+temporary[k]);
                    status[2*t+1]|=flags();
                }
            }
            block=stop;
        }
    });
    for(const auto& status:statuses)for(std::size_t i=0;i<combined.size();++i)combined[i]|=status[i];
    return combined;
}
}}
namespace skimagexx {
std::vector<unsigned> correlate_sparse_into(const void* input,void* output,PixelType type,
    const std::vector<std::size_t>& shape,const std::vector<std::size_t>& kernel,
    const std::vector<std::vector<std::size_t>>& coordinates,const void* weights,GaussianBorder border,bool valid) {
    if(shape.empty() || shape.size()!=kernel.size() || coordinates.empty() || unsigned(border)>4)throw std::invalid_argument("invalid sparse dimensions or mode");
    product(shape);product(kernel);
    for(auto k:kernel)if(!k || (!valid && k%2==0))throw std::invalid_argument("invalid sparse kernel shape");
    for(const auto& c:coordinates){if(c.size()!=shape.size())throw std::invalid_argument("sparse coordinate rank mismatch");for(std::size_t d=0;d<c.size();++d)if(c[d]>=kernel[d])throw std::invalid_argument("sparse coordinate outside kernel");}
    if(std::any_of(coordinates.front().begin(),coordinates.front().end(),[](auto c){return c!=0;}))throw std::invalid_argument("sparse corner must be first");
    if(type==PixelType::float32)return run(static_cast<const float*>(input),static_cast<float*>(output),shape,kernel,coordinates,static_cast<const float*>(weights),border,valid);
    if(type==PixelType::float64)return run(static_cast<const double*>(input),static_cast<double*>(output),shape,kernel,coordinates,static_cast<const double*>(weights),border,valid);
    throw std::invalid_argument("sparse floating input required");
}
}
