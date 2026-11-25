#include "skimagexx/tv_chambolle.hpp"
#include "skimagexx/tv_denoise.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace detail {
// Finite reductions retain the profile's exact ordering and discard FP status.
std::uint64_t tv_finite_square_sum(TvReductionInput input, void* output, TvReductionPlan plan);
std::uint64_t tv_finite_sum(TvReductionInput input, TvReductionPlan plan);
template<class T> void tv_finite_add(const void* input, const T* d, T* out, std::size_t n);
template<class T> void tv_finite_scale_norm(T* norm, T coefficient, std::size_t n);
}
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
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=r.n>1 && input.contiguous && has_avx2();
#endif
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
                if(vectorized) {
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

template<class T> using ScalarBits = std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
template<class T> std::uint64_t bits(T value) { return std::bit_cast<ScalarBits<T>>(value); }
template<class T> T scalar(std::uint64_t raw) { return std::bit_cast<T>(ScalarBits<T>(raw)); }

template<class T,class Energy>
TvDenoiseResult run_finite(const TvConstView& input,const TvDenoiseWorkspace& w,
    const TvDenoiseParameters& parameters,TvDenoiseTrace* trace,
    const Layout& source,const Layout& contiguous) {
    const auto type=sizeof(T)==4?TvReductionType::float32:TvReductionType::float64;
    const auto n=source.n;
    const auto* d=static_cast<const T*>(w.d.data);
    auto* out=static_cast<T*>(w.out.data);
    auto* norm_data=static_cast<T*>(w.norm.data);
    const double tau=1.0/(2.0*double(source.rank));
    const T norm_coefficient=T(tau/parameters.weight);
    const Energy weight=Energy(parameters.weight),eps=Energy(parameters.eps),size=Energy(double(n));
    Energy initial=0,previous=0;
    for(std::size_t iteration=0;iteration<parameters.max_iterations;++iteration) {
        if(iteration) {
            divergence<T>(w.p_read,w.d,contiguous,contiguous);
            if(source.contiguous)detail::tv_finite_add(input.data,d,out,n);
            else detail::partition(n,detail::active_workers(n),[&](std::size_t begin,std::size_t end,std::size_t){
                if(begin==end)return;
                InputCursor cursor(source,begin);
                for(std::size_t i=begin;i<end;++i) {
                    out[i]=T(read<T>(input.data,cursor.byte)+d[i]);
                    if(i+1<end)cursor.advance();
                }
            });
        }
        const auto squared_sum=detail::tv_finite_square_sum({d,n,type},w.squared.data,parameters.reduction);
        gradient<T>(iteration?w.out_read:input,w.g,contiguous,iteration?contiguous:source);
        norm<T>(w.g_read,w.norm,contiguous,contiguous);
        const auto norm_sum=detail::tv_finite_sum({norm_data,n,type},parameters.reduction);
        const Energy weighted=Energy(weight*Energy(scalar<T>(norm_sum)));
        Energy energy=Energy(Energy(scalar<T>(squared_sum))+weighted);
        detail::tv_finite_scale_norm(norm_data,norm_coefficient,n);
        dual<T>(w.p,w.g_read,w.norm_read,tau,contiguous,contiguous);
        energy=Energy(energy/size);
        Energy difference=0,threshold=0;bool stopped=false;
        if(!iteration)initial=previous=energy;
        else {
            difference=std::abs(Energy(previous-energy));
            threshold=Energy(eps*initial);
            stopped=difference<threshold;
            if(!stopped)previous=energy;
        }
        if(trace)trace[iteration]={squared_sum,norm_sum,bits(energy),bits(difference),bits(threshold),sizeof(Energy)==4,stopped};
        if(stopped)return {iteration+1,false};
    }
    return {parameters.max_iterations,parameters.max_iterations==1};
}
}

namespace detail {
TvDenoiseResult tv_chambolle_run_validated(const TvConstView& input,
    const TvDenoiseWorkspace& workspace,const TvDenoiseParameters& parameters,
    TvDenoiseTrace* trace,bool reset) {
    // The finite guard already certified the shared shape, dtype, C layout,
    // aliases and disjointness. Derive the two traversal layouts once.
    const auto source=layout(input.data,input.type,input.shape,&input.strides,false,false);
    const auto contiguous=layout(workspace.p.data,input.type,input.shape,nullptr,true,true);
    if(reset) {
        const auto spatial=source.n*source.bytes;
        std::memset(workspace.p.data,0,spatial*source.rank);
        std::memset(workspace.g.data,0,spatial*source.rank);
        std::memset(workspace.d.data,0,spatial);
    }
    if(input.type==TvScalarType::float64)
        return run_finite<double,double>(input,workspace,parameters,trace,source,contiguous);
    if(parameters.reduction.profile==TvReductionProfile::numpy_1244_buffered)
        return run_finite<float,double>(input,workspace,parameters,trace,source,contiguous);
    return run_finite<float,float>(input,workspace,parameters,trace,source,contiguous);
}
} // namespace detail

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
