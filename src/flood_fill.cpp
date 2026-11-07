#include "skimagexx/flood_fill.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
namespace skimagexx { namespace {
using I=std::ptrdiff_t;
I multiply(I a,I b) {
    if(a<0||b<0||(b&&a>std::numeric_limits<I>::max()/b))
        throw std::overflow_error("flood shape overflow");
    return a*b;
}
I product(const std::vector<I>& s){I n=1;for(auto v:s)n=multiply(n,v);return n;}
std::vector<I> coordinates(I k,const std::vector<I>& s,bool f=false){
    std::vector<I> c(s.size());for(std::size_t j=0;j<s.size();++j){
        auto a=f?j:s.size()-1-j;c[a]=k%s[a];k/=s[a];}return c;
}
I offset(const std::vector<I>& c,const std::vector<I>& strides){I v=0;
    for(std::size_t a=0;a<c.size();++a)v+=c[a]*strides[a];return v;}
template<class T>T load(const T* p,I bytes){T v;std::memcpy(&v,reinterpret_cast<const char*>(p)+bytes,sizeof(T));return v;}
// The existing crop traversal uses the same C-order odometer. Keep one index
// per traversal and update the two offsets only on axis advance/carry.
void advance(std::vector<I>& index,const std::vector<I>& shape,
             const std::vector<I>& input_strides,const std::vector<I>& mask_strides,
             I& input_offset,I& mask_offset){
    for(std::size_t a=shape.size();a-->0;){
        if(++index[a]<shape[a]){input_offset+=input_strides[a];mask_offset+=mask_strides[a];return;}
        index[a]=0;input_offset-=(shape[a]-1)*input_strides[a];mask_offset-=(shape[a]-1)*mask_strides[a];
    }
}
void validate(const FloodGeometry& g){
    auto n=g.shape.size();if(!n||g.strides.size()!=n||g.seed.size()!=n||g.footprint_shape.size()!=n)
        throw std::invalid_argument("flood rank mismatch");
    product(g.shape);
    if(product(g.footprint_shape)!=I(g.footprint.size()))throw std::invalid_argument("flood footprint size mismatch");
    for(std::size_t a=0;a<n;++a){if(g.footprint_shape[a]%2!=1)throw std::invalid_argument("flood footprint must be odd");
        if(g.shape[a]&& (g.seed[a]<0||g.seed[a]>=g.shape[a]))throw std::out_of_range("flood seed out of range");}
}
}
template<class T>FloodMask flood(const T* image,const FloodGeometry& g,bool tolerance,T low,T high){
    validate(g);FloodMask out;out.shape=g.shape;out.padding.assign(g.shape.size(),0);
    if(!product(g.shape))return out;
    std::vector<std::vector<I>> neighbors;
    for(I k=0;k<I(g.footprint.size());++k)if(g.footprint[k]){
        auto c=coordinates(k,g.footprint_shape);
        for(std::size_t a=0;a<c.size();++a){c[a]-=g.footprint_shape[a]/2;out.padding[a]=std::max(out.padding[a],c[a]<0?-c[a]:c[a]);}
        neighbors.push_back(std::move(c));
    }
    if(neighbors.empty())throw std::invalid_argument("flood empty footprint");
    auto padded=g.shape;for(std::size_t a=0;a<padded.size();++a){
        if(out.padding[a]>(std::numeric_limits<I>::max()-padded[a])/2)throw std::overflow_error("flood shape overflow");
        padded[a]+=2*out.padding[a];}
    out.strides.resize(padded.size());I count=1;
    for(std::size_t j=0;j<padded.size();++j){auto a=g.fortran?j:padded.size()-1-j;out.strides[a]=count;count=multiply(count,padded[a]);}
    out.origin=offset(out.padding,out.strides);out.storage.assign(std::size_t(count),2);
    auto logical=product(g.shape);
    // One padded typed copy permits constant-address neighbor reads. Border
    // payload is never read: flags2 is checked before any working-image load.
    std::vector<T> working(static_cast<std::size_t>(count));
    std::vector<I> index(g.shape.size(),0);I input_offset=0,mask_offset=out.origin;
    for(I k=0;k<logical;++k){
        out.storage[mask_offset]=0;
        std::memcpy(working.data()+mask_offset,reinterpret_cast<const char*>(image)+input_offset,sizeof(T));
        advance(index,g.shape,g.strides,out.strides,input_offset,mask_offset);
    }
    // Upstream stable squared-distance order, then unconditionally drops first
    // offset (also when an explicit footprint omits its logical center).
    std::stable_sort(neighbors.begin(),neighbors.end(),[](const auto& a,const auto& b){
        I x=0,y=0;for(auto v:a)x+=v*v;for(auto v:b)y+=v*v;return x<y;});
    if(!neighbors.empty())neighbors.erase(neighbors.begin());
    std::vector<I> steps;for(const auto& c:neighbors)steps.push_back(offset(c,out.strides));
    auto seed=out.origin+offset(g.seed,out.strides);T value=load(image,offset(g.seed,g.strides));
    std::vector<I> queue;queue.reserve(std::size_t(logical));queue.push_back(seed);out.storage[seed]=1;
    for(std::size_t head=0;head<queue.size();++head)for(auto step:steps){
        I next=queue[head]+step;if(next<0||next>=count||out.storage[next])continue;
        T v=working[next];
        if(tolerance?(v>=low&&v<=high):(v==value)){out.storage[next]=1;queue.push_back(next);}
    }
    return out;
}
template<class T>std::vector<T> flood_fill(const T* image,const FloodGeometry& g,T value,bool tol,T low,T high){
    auto mask=flood(image,g,tol,low,high);auto n=product(g.shape);std::vector<T> out(std::size_t(n),T{});
    std::vector<I> index(g.shape.size(),0);I input_offset=0,mask_offset=mask.origin;
    for(I k=0;k<n;++k){
        out[k]=mask.storage[mask_offset]==1?value:load(image,input_offset);
        advance(index,g.shape,g.strides,mask.strides,input_offset,mask_offset);
    }
    return out;
}
template<class T>void flood_fill_into(T* image,const FloodGeometry& g,T value,bool tol,T low,T high){
    auto mask=flood(image,g,tol,low,high);auto n=product(g.shape);
    std::vector<I> index(g.shape.size(),0);I input_offset=0,mask_offset=mask.origin;
    for(I k=0;k<n;++k){
        if(mask.storage[mask_offset]==1)std::memcpy(reinterpret_cast<char*>(image)+input_offset,&value,sizeof(T));
        advance(index,g.shape,g.strides,mask.strides,input_offset,mask_offset);
    }
}
#define INSTANTIATE(T) template FloodMask flood(const T*,const FloodGeometry&,bool,T,T); \
 template std::vector<T> flood_fill(const T*,const FloodGeometry&,T,bool,T,T); \
 template void flood_fill_into(T*,const FloodGeometry&,T,bool,T,T);
INSTANTIATE(std::uint8_t) INSTANTIATE(std::uint16_t) INSTANTIATE(std::uint32_t) INSTANTIATE(std::uint64_t)
INSTANTIATE(std::int8_t) INSTANTIATE(std::int16_t) INSTANTIATE(std::int32_t) INSTANTIATE(std::int64_t)
INSTANTIATE(float) INSTANTIATE(double)
#undef INSTANTIATE
}
