#include "coords_internal.hpp"
#include "execution.hpp"
#include <atomic>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace skimagexx::detail {
namespace {
class Capture {
    std::fenv_t saved_;bool valid_;
public:
    Capture():valid_(std::feholdexcept(&saved_)==0){}
    ~Capture(){if(valid_)std::fesetenv(&saved_);}
    bool valid()const{return valid_;}
    unsigned flags()const {
        const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
        return ((f&FE_DIVBYZERO)?1u:0u)|((f&FE_OVERFLOW)?2u:0u)|((f&FE_UNDERFLOW)?4u:0u)|((f&FE_INVALID)?8u:0u);
    }
};
std::size_t multiply(std::size_t a,std::size_t b,std::size_t limit) {
    if(b&&a>limit/b)throw std::invalid_argument("coordinate moments extent overflow");
    return a*b;
}
template<class T>unsigned reduce(const T* powers,std::size_t N,std::size_t D,std::size_t K,T* output,int workers) {
    if(D<1||D>3||K<2)throw std::invalid_argument("coordinate moments require D1..3 and K>1");
    if(workers<1||workers>1024)throw std::invalid_argument("coordinate moments worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T);
    auto count=std::size_t(1);for(std::size_t axis=0;axis<D;++axis)count=multiply(count,K,limit);
    multiply(multiply(N,D,limit),K,limit);
    if(!output||(N&&!powers))throw std::invalid_argument("coordinate moments require nonnull buffers");
    const auto factor=N>parallel_min/D?parallel_min:N*D;
    const auto work=factor&&count>parallel_min/factor?parallel_min:factor*count;
    const auto team=work>=parallel_min?workers:1;std::atomic<unsigned> flags{0};
    const bool vectorized=has_avx2();
    partition(count,team,[&](std::size_t begin,std::size_t end,std::size_t){
        Capture capture;if(!capture.valid()){flags.fetch_or(32u,std::memory_order_relaxed);return;}
#ifdef SKIMAGEXX_AVX2
        if(vectorized)simd::moments_coords_reduce(powers,N,D,K,output,begin,end);
        else
#else
        (void)vectorized;
#endif
        for(auto cell=begin;cell<end;++cell)output[cell]=coords_cell(powers,N,D,K,cell);
        unsigned status=capture.flags();
        for(auto cell=begin;cell<end;++cell)if(!std::isfinite(output[cell]))status|=16u;
        flags.fetch_or(status,std::memory_order_relaxed);
    });
    const auto status=flags.load(std::memory_order_relaxed);
    if(status&32u)throw std::runtime_error("cannot capture coordinate moments floating environment");
    return status;
}
}
unsigned moments_coords_reduce(const float*p,std::size_t N,std::size_t D,std::size_t K,float*out,int workers){return reduce(p,N,D,K,out,workers);}
unsigned moments_coords_reduce(const double*p,std::size_t N,std::size_t D,std::size_t K,double*out,int workers){return reduce(p,N,D,K,out,workers);}
}
