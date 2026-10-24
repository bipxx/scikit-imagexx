#include "skimagexx/tv_chambolle.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
struct Layout {
    std::size_t rank=0,n=0,bytes=0;
    std::array<std::size_t,4> shape{},steps{};
    std::array<std::ptrdiff_t,4> strides{};
    std::uintptr_t low=0,high=0;
    bool contiguous=true;
};
std::size_t width(TvScalarType t) {
    if(t==TvScalarType::float32)return sizeof(float);
    if(t==TvScalarType::float64)return sizeof(double);
    throw std::invalid_argument("TV requires native float32 or float64");
}
std::size_t magnitude(std::ptrdiff_t x) {
    return x<0 ? std::size_t(-(x+1))+1:std::size_t(x);
}
Layout layout(const void* data,TvScalarType type,const std::vector<std::size_t>& shape,
              const std::vector<std::ptrdiff_t>* strides,bool writable,bool planes) {
    Layout r;r.rank=shape.size();r.bytes=width(type);
    if(!r.rank || r.rank>4)throw std::invalid_argument("TV spatial rank must be in [1,4]");
    if(strides && strides->size()!=r.rank)throw std::invalid_argument("one TV byte stride per spatial axis required");
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    r.n=1;
    for(std::size_t a=r.rank;a-->0;) {
        r.shape[a]=shape[a];r.steps[a]=r.n;
        if(shape[a] && r.n>limit/r.bytes/shape[a])throw std::overflow_error("TV spatial extent overflow");
        r.strides[a]=strides ? (*strides)[a]:std::ptrdiff_t(r.n*r.bytes);
        if(shape[a]>1 && (r.n>limit/r.bytes || r.strides[a]!=std::ptrdiff_t(r.n*r.bytes)))r.contiguous=false;
        r.n*=shape[a];
    }
    if(planes && r.n>limit/r.bytes/r.rank)throw std::overflow_error("TV plane extent overflow");
    if(!r.n)return r;
    if(!data)throw std::invalid_argument("nonnull TV data required for nonempty shape");
    std::size_t below=0,above=0;
    if(planes)above=r.n*r.rank*r.bytes-r.bytes;
    else for(std::size_t a=0;a<r.rank;++a) {
        const auto stride=magnitude(r.strides[a]);
        if(r.shape[a]>1 && stride>limit/(r.shape[a]-1))throw std::overflow_error("TV byte stride extent overflow");
        const auto distance=stride*(r.shape[a]-1);auto& side=r.strides[a]<0 ? below:above;
        if(distance>limit-side)throw std::overflow_error("TV byte stride sum overflow");side+=distance;
    }
    const auto address=reinterpret_cast<std::uintptr_t>(data);
    if(address<below || above>std::numeric_limits<std::uintptr_t>::max()-address || r.bytes>std::numeric_limits<std::uintptr_t>::max()-address-above)
        throw std::overflow_error("TV address interval overflow");
    r.low=address-below;r.high=address+above+r.bytes;
    if(writable && !planes) {
        std::array<std::pair<std::size_t,std::size_t>,4> axes{};std::size_t used=0;
        for(std::size_t a=0;a<r.rank;++a)if(r.shape[a]>1)axes[used++]={magnitude(r.strides[a]),r.shape[a]};
        std::sort(axes.begin(),axes.begin()+used);std::size_t occupied=r.bytes;
        for(std::size_t i=0;i<used;++i) {
            if(axes[i].first<occupied)throw std::invalid_argument("TV output must not overlap itself");
            const auto extra=axes[i].first*(axes[i].second-1);
            if(extra>limit-occupied)throw std::overflow_error("TV output extent overflow");occupied+=extra;
        }
    }
    return r;
}
void compatible(const Layout& a,const Layout& b,TvScalarType t0,TvScalarType t1) {
    if(a.rank!=b.rank || a.shape!=b.shape || t0!=t1)throw std::invalid_argument("TV stage shape and dtype must match");
}
void disjoint(const Layout& a,const Layout& b) {
    if(a.n && b.n && a.low<b.high && b.low<a.high)throw std::invalid_argument("TV stage buffers must be disjoint");
}
std::ptrdiff_t offset(const Layout& r,std::size_t i) {
    if(r.contiguous)return std::ptrdiff_t(i*r.bytes);
    std::ptrdiff_t result=0;
    for(std::size_t a=0;a<r.rank;++a)result+=std::ptrdiff_t((i/r.steps[a])%r.shape[a])*r.strides[a];
    return result;
}
// Logical address transport only; arithmetic and the axis/element visitation
// order below remain unchanged. Carry never forms a one-past stride product.
struct InputCursor {
    const Layout& layout;
    std::array<std::size_t,4> coordinate{};
    std::ptrdiff_t byte;
    InputCursor(const Layout& value,std::size_t index):layout(value),byte(offset(value,index)) {
        for(std::size_t a=0;a<layout.rank;++a)
            coordinate[a]=(index/layout.steps[a])%layout.shape[a];
    }
    void advance() {
        for(std::size_t a=layout.rank;a-->0;) {
            if(coordinate[a]+1<layout.shape[a]) {
                ++coordinate[a];byte+=layout.strides[a];return;
            }
            byte-=std::ptrdiff_t(coordinate[a])*layout.strides[a];coordinate[a]=0;
        }
    }
};
template<class T> T read(const void* p,std::ptrdiff_t byte) {T x;std::memcpy(&x,static_cast<const unsigned char*>(p)+byte,sizeof(T));return x;}
template<class T> void write(void* p,std::ptrdiff_t byte,T x) {std::memcpy(static_cast<unsigned char*>(p)+byte,&x,sizeof(T));}
template<class T> void divergence(const TvConstPlanes& p,const TvView& d,const Layout& r,const Layout& out) {
    detail::partition(r.n,detail::active_workers(r.n),[&](std::size_t begin,std::size_t end,std::size_t){
        std::size_t i=begin;
#ifdef SKIMAGEXX_AVX2
        if(out.contiguous && has_avx2())i+=simd::tv_divergence(p.data,static_cast<unsigned char*>(d.data)+begin*sizeof(T),r.n,r.rank,begin,end-begin,r.shape.data(),r.steps.data(),std::is_same_v<T,float>);
#endif
        for(;i<end;++i) {
            T value=T(0);
            for(std::size_t a=0;a<r.rank;++a)value=T(value+read<T>(p.data,std::ptrdiff_t((a*r.n+i)*sizeof(T))));
            value=T(-value);
            for(std::size_t a=0;a<r.rank;++a)if((i/r.steps[a])%r.shape[a])
                value=T(value+read<T>(p.data,std::ptrdiff_t((a*r.n+i-r.steps[a])*sizeof(T))));
            write<T>(d.data,offset(out,i),value);
        }
    });
}
template<class T> void gradient(const TvConstView& out,const TvPlanes& g,const Layout& r,const Layout& input) {
    detail::partition(r.n,detail::active_workers(r.n),[&](std::size_t begin,std::size_t end,std::size_t){
        for(std::size_t a=0;a<r.rank;++a) {
            std::size_t i=begin;
            while(i<end) {
                const auto position=(i/r.steps[a])%r.shape[a];
                if(position+1==r.shape[a]) {
                    const auto stop=std::min(end,i+r.steps[a]-i%r.steps[a]);
                    for(;i<stop;++i)write<T>(g.data,std::ptrdiff_t((a*r.n+i)*sizeof(T)),T(0));
                    continue;
                }
                const auto stop=std::min(end,i+(r.shape[a]-position-1)*r.steps[a]-i%r.steps[a]);
#ifdef SKIMAGEXX_AVX2
                if(input.contiguous && has_avx2()) {
                    const auto* first=static_cast<const unsigned char*>(out.data)+i*sizeof(T);
                    auto* destination=static_cast<unsigned char*>(g.data)+(a*r.n+i)*sizeof(T);
                    i+=simd::tv_gradient(first,destination,r.steps[a],stop-i,std::is_same_v<T,float>);
                }
#endif
                if(!input.contiguous && i<stop) {
                    InputCursor left_cursor(input,i),right_cursor(input,i+r.steps[a]);
                    for(;i<stop;++i) {
                        const T left=read<T>(out.data,left_cursor.byte),right=read<T>(out.data,right_cursor.byte);
                        write<T>(g.data,std::ptrdiff_t((a*r.n+i)*sizeof(T)),T(right-left));
                        if(i+1<stop){left_cursor.advance();right_cursor.advance();}
                    }
                } else {
                    for(;i<stop;++i) {
                        const T left=read<T>(out.data,offset(input,i)),right=read<T>(out.data,offset(input,i+r.steps[a]));
                        write<T>(g.data,std::ptrdiff_t((a*r.n+i)*sizeof(T)),T(right-left));
                    }
                }
            }
        }
    });
}
template<class T> void norm(const TvConstPlanes& g,const TvView& output,const Layout& r,const Layout& out) {
    detail::partition(r.n,detail::active_workers(r.n),[&](std::size_t begin,std::size_t end,std::size_t){
        std::size_t i=begin;
#ifdef SKIMAGEXX_AVX2
        if(out.contiguous && has_avx2())i+=simd::tv_norm(static_cast<const unsigned char*>(g.data)+begin*sizeof(T),static_cast<unsigned char*>(output.data)+begin*sizeof(T),r.n,r.rank,end-begin,std::is_same_v<T,float>);
#endif
        for(;i<end;++i) {
            T value=T(0);
            for(std::size_t a=0;a<r.rank;++a) {const T x=read<T>(g.data,std::ptrdiff_t((a*r.n+i)*sizeof(T)));const T squared=T(x*x);value=T(value+squared);}
            write<T>(output.data,offset(out,i),T(std::sqrt(value)));
        }
    });
}
template<class T> void dual(const TvPlanes& p,const TvConstPlanes& g,const TvConstView& denominator,double tau_work,const Layout& r,const Layout& denom) {
    const T tau=T(tau_work);
    detail::partition(r.n,detail::active_workers(r.n),[&](std::size_t begin,std::size_t end,std::size_t){
        std::size_t i=begin;
#ifdef SKIMAGEXX_AVX2
        if(denom.contiguous && has_avx2())i+=simd::tv_dual(p.data,g.data,static_cast<const unsigned char*>(denominator.data)+begin*sizeof(T),r.n,r.rank,begin,end-begin,double(tau),std::is_same_v<T,float>);
#endif
        for(;i<end;++i) {
            const T divisor=read<T>(denominator.data,offset(denom,i));
            for(std::size_t a=0;a<r.rank;++a) {
                const auto byte=std::ptrdiff_t((a*r.n+i)*sizeof(T));const T product=T(tau*read<T>(g.data,byte));
                const T difference=T(read<T>(p.data,byte)-product);write<T>(p.data,byte,T(difference/divisor));
            }
        }
    });
}
}
std::uint32_t tv_chambolle_capabilities() noexcept {return 0x3f;}
void tv_chambolle_divergence_into(const TvConstPlanes& p,const TvView& d) {
    const auto a=layout(p.data,p.type,p.shape,nullptr,false,true),b=layout(d.data,d.type,d.shape,&d.strides,true,false);compatible(a,b,p.type,d.type);disjoint(a,b);
    if(p.type==TvScalarType::float32)divergence<float>(p,d,a,b);else divergence<double>(p,d,a,b);
}
void tv_chambolle_gradient_into(const TvConstView& out,const TvPlanes& g) {
    const auto a=layout(out.data,out.type,out.shape,&out.strides,false,false),b=layout(g.data,g.type,g.shape,nullptr,true,true);compatible(a,b,out.type,g.type);disjoint(a,b);
    if(out.type==TvScalarType::float32)gradient<float>(out,g,b,a);else gradient<double>(out,g,b,a);
}
void tv_chambolle_norm_into(const TvConstPlanes& g,const TvView& norm_out) {
    const auto a=layout(g.data,g.type,g.shape,nullptr,false,true),b=layout(norm_out.data,norm_out.type,norm_out.shape,&norm_out.strides,true,false);compatible(a,b,g.type,norm_out.type);disjoint(a,b);
    if(g.type==TvScalarType::float32)norm<float>(g,norm_out,a,b);else norm<double>(g,norm_out,a,b);
}
void tv_chambolle_dual_into(const TvPlanes& p,const TvConstPlanes& g,const TvConstView& denominator,double tau_work) {
    const auto a=layout(p.data,p.type,p.shape,nullptr,true,true),b=layout(g.data,g.type,g.shape,nullptr,false,true),c=layout(denominator.data,denominator.type,denominator.shape,&denominator.strides,false,false);
    compatible(a,b,p.type,g.type);compatible(a,c,p.type,denominator.type);disjoint(a,b);disjoint(a,c);disjoint(b,c);
    if(p.type==TvScalarType::float32)dual<float>(p,g,denominator,tau_work,a,c);else dual<double>(p,g,denominator,tau_work,a,c);
}
}
