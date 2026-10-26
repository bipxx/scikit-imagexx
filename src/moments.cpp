#include "moments_internal.hpp"
#include "execution.hpp"
#include <atomic>
#include <cfenv>
#include <limits>
#include <stdexcept>

namespace skimagexx::detail {
namespace {
// Same save/capture/restore convention and status bits as colocalization.
class Capture {
    std::fenv_t saved_;bool valid_;
public:
    Capture():valid_(std::feholdexcept(&saved_)==0){}
    bool valid()const{return valid_;}
    ~Capture(){if(valid_)std::fesetenv(&saved_);}
    unsigned flags()const {
        const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
        return ((f&FE_DIVBYZERO)?1u:0u)|((f&FE_OVERFLOW)?2u:0u)|
               ((f&FE_UNDERFLOW)?4u:0u)|((f&FE_INVALID)?8u:0u);
    }
};
std::size_t multiply(std::size_t a,std::size_t b,std::size_t limit) {
    if(b && a>limit/b)throw std::invalid_argument("moments projection extent overflow");
    return a*b;
}
template<class T> unsigned project(const T* calc,const std::vector<std::size_t>&shape,
                                   const T*powers,std::size_t K,std::size_t axis,T*out,int workers) {
    if(axis>=shape.size())throw std::invalid_argument("moments projection axis out of range");
    if(workers<1 || workers>1024)throw std::invalid_argument("moments worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T);
    std::size_t outer=1,inner=1;const auto n=shape[axis];
    for(std::size_t d=0;d<shape.size();++d) {
        if(shape[d]>limit)throw std::invalid_argument("moments projection extent overflow");
        if(d<axis)outer=multiply(outer,shape[d],limit);
        if(d>axis)inner=multiply(inner,shape[d],limit);
    }
    const auto count=multiply(multiply(outer,K,limit),inner,limit);
    multiply(multiply(outer,n,limit),inner,limit);multiply(n,K,limit);
    if(!count)return 0;
    if(!out || (n && (!calc || !powers)))throw std::invalid_argument("moments projection null buffer");
    const auto work=n && count>parallel_min/n ? parallel_min : count*n;
    const auto team=work>=parallel_min ? workers:1;
    const bool vectorized=has_avx2();std::atomic<unsigned> flags{0};
    partition(count,team,[&](std::size_t begin,std::size_t end,std::size_t) {
        Capture capture;
        if(!capture.valid()){flags.fetch_or(16u,std::memory_order_relaxed);return;}
#ifdef SKIMAGEXX_AVX2
        if(vectorized)simd::moments_project(calc,powers,n,K,inner,out,begin,end);
        else
#else
        (void)vectorized;
#endif
        for(auto i=begin;i<end;++i)out[i]=moments_one(calc,powers,n,K,inner,i);
        flags.fetch_or(capture.flags(),std::memory_order_relaxed);
    });
    const auto result=flags.load(std::memory_order_relaxed);
    if(result&16u)throw std::runtime_error("cannot capture moments floating environment");
    return result;
}
}
#define SKIMAGEXX_MOMENTS_DEFINE(T) \
unsigned moments_project(const T*a,const std::vector<std::size_t>&s,const T*p, \
                         std::size_t K,std::size_t axis,T*out,int workers) { \
    return project(a,s,p,K,axis,out,workers); }
SKIMAGEXX_MOMENTS_DEFINE(float)
SKIMAGEXX_MOMENTS_DEFINE(double)
SKIMAGEXX_MOMENTS_DEFINE(std::complex<float>)
SKIMAGEXX_MOMENTS_DEFINE(std::complex<double>)
#undef SKIMAGEXX_MOMENTS_DEFINE
}
