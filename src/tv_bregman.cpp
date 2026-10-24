#include "skimagexx/tv_bregman.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
void tv_bregman_copy4(const float*,float*,float*,float*,float*,std::size_t);
void tv_bregman_copy4(const double*,double*,double*,double*,double*,std::size_t);
}
#endif
namespace {
struct Span { std::uintptr_t begin,end; };
struct GuardEnvironment {
    std::fenv_t original{},held{};
    bool saved=false,valid=false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    unsigned csr=_mm_getcsr();
#endif
    GuardEnvironment() noexcept {
        saved=std::fegetenv(&original)==0;
        valid=saved && std::feholdexcept(&held)==0;
    }
    ~GuardEnvironment() noexcept {
        if(saved)std::fesetenv(&original);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
        _mm_setcsr(csr);
#endif
    }
};
bool multiply(std::size_t a,std::size_t b,std::size_t limit,std::size_t& value) {
    if(a && b>limit/a)return false;value=a*b;return true;
}
bool span(const void* data,std::size_t n,std::size_t width,Span& value,std::size_t alignment=0) {
    if(!data || n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/width)return false;
    const auto start=reinterpret_cast<std::uintptr_t>(data),bytes=n*width;
    if(start%(alignment?alignment:width) || bytes>std::numeric_limits<std::uintptr_t>::max()-start)return false;
    value={start,start+bytes};return true;
}
bool overlap(Span a,Span b) {return a.begin<b.end && b.begin<a.end;}
bool fp_profile() {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr() & ((1u<<15)|(1u<<6)|(3u<<13)))return false;
#endif
    return true;
}
template<class T> bool finite(const T* data,std::size_t n) {
    for(std::size_t i=0;i<n;++i)if(!std::isfinite(data[i]))return false;return true;
}
template<class T> bool coefficients(double value,T& weight,T& lam,T& norm) {
    if(value>double(std::numeric_limits<T>::max()))return false;
    weight=T(value);const double wide_lam=2.0*double(weight);
    if(!std::isfinite(wide_lam) || wide_lam>double(std::numeric_limits<T>::max()))return false;
    lam=T(wide_lam);const double wide_norm=double(weight)+4.0*double(lam);
    if(!std::isfinite(wide_norm) || wide_norm>double(std::numeric_limits<T>::max()))return false;
    norm=T(wide_norm);
    return weight>T(0) && std::isfinite(weight) && lam>T(0) && std::isfinite(lam) && norm>T(0) && std::isfinite(norm);
}
template<class T> const char* typed_guard(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& p) {
    T w,lam,norm;if(!coefficients(p.weight,w,lam,norm))return "TV Bregman typed weight/lambda/norm must be finite positive";
    if(!finite(static_cast<const T*>(image.data),image.elements))return "TV Bregman input must be finite";
    if(!finite(static_cast<const T*>(workspace.out.data),workspace.out.elements))return "TV Bregman initial out must be finite";
    return nullptr;
}
template<class T> int initialize(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,int requested) {
    auto* out=static_cast<T*>(workspace.out.data);auto* dx=static_cast<T*>(workspace.dx.data);auto* dy=static_cast<T*>(workspace.dy.data);auto* bx=static_cast<T*>(workspace.bx.data);auto* by=static_cast<T*>(workspace.by.data);
    const auto n=workspace.out.elements;const int threads=n>=detail::parallel_min?requested:1;std::atomic<int> actual{1};const bool vectorized=has_avx2();
    detail::partition(n,threads,[&](std::size_t begin,std::size_t end,std::size_t id) {
#ifdef _OPENMP
        if(id==0)actual.store(omp_get_num_threads());
#else
        (void)id;
#endif
#ifdef SKIMAGEXX_AVX2
        if(vectorized){simd::tv_bregman_copy4(out+begin,dx+begin,dy+begin,bx+begin,by+begin,end-begin);return;}
#else
        (void)vectorized;
#endif
        for(auto i=begin;i<end;++i){const T value=out[i];dx[i]=value;dy[i]=value;bx[i]=value;by[i]=value;}
    });
    const auto* input=static_cast<const T*>(image.data);const auto r=image.rows,c=image.cols,d=image.channels,stride=(c+2)*d;
    for(std::size_t y=0;y<r;++y)std::memcpy(out+(y+1)*stride+d,input+y*c*d,c*d*sizeof(T));
    // Source order: interior, top, left, bottom, right. Corners untouched.
    std::memcpy(out+d,input+c*d,c*d*sizeof(T));
    for(std::size_t y=0;y<r;++y)std::memcpy(out+(y+1)*stride,input+(y*c+1)*d,d*sizeof(T));
    std::memcpy(out+(r+1)*stride+d,input+(r-1)*c*d,c*d*sizeof(T));
    for(std::size_t y=0;y<r;++y)std::memcpy(out+(y+1)*stride+(c+1)*d,input+(y*c+c-1)*d,d*sizeof(T));
    return actual.load();
}
template<class T> TvBregmanResult process(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& parameters,TvBregmanTrace* trace,int workers) {
    T weight,lam,norm;coefficients(parameters.weight,weight,lam,norm);
    const int actual=initialize<T>(image,workspace,workers);
    const auto* input=static_cast<const T*>(image.data);auto* out=static_cast<T*>(workspace.out.data);auto* dx=static_cast<T*>(workspace.dx.data);auto* dy=static_cast<T*>(workspace.dy.data);auto* bx=static_cast<T*>(workspace.bx.data);auto* by=static_cast<T*>(workspace.by.data);
    const auto r=image.rows,c=image.cols,d=image.channels,stride=(c+2)*d;
    double rmse=std::numeric_limits<double>::max();std::size_t iteration=0;
    while(iteration<std::size_t(parameters.max_iterations>0?parameters.max_iterations:0) && rmse>parameters.eps) {
        rmse=0.;
        for(std::size_t y=1;y<=r;++y)for(std::size_t x=1;x<=c;++x)for(std::size_t k=0;k<d;++k) {
            const auto i=y*stride+x*d+k;const T previous=out[i];const T ux=T(out[i+d]-previous),uy=T(out[i+stride]-previous);
            T total=T(out[i+stride]+out[i-stride]);total=T(total+out[i+d]);total=T(total+out[i-d]);
            total=T(total+dx[i-d]);total=T(total-dx[i]);total=T(total+dy[i-stride]);total=T(total-dy[i]);total=T(total-bx[i-d]);total=T(total+bx[i]);total=T(total-by[i-stride]);total=T(total+by[i]);
            const T unew=T(T(T(lam*total)+T(weight*input[((y-1)*c+x-1)*d+k]))/norm);out[i]=unew;
            T tx=T(unew-previous);const T squared=T(tx*tx);rmse+=double(squared);
            const T bxx=bx[i],byy=by[i];T dxx,dyy;
            if(parameters.isotropic) {
                tx=T(ux+bxx);const T ty=T(uy+byy);const T length=T(std::sqrt(double(T(T(tx*tx)+T(ty*ty)))));
                const T factor=T(length*lam);const double denominator=double(factor)+1.0;
                dxx=T(double(T(factor*tx))/denominator);dyy=T(double(T(factor*ty))/denominator);
            }else {
                const double threshold=1.0/double(lam);
                T s=T(ux+bxx);dxx=double(s)>threshold?T(double(s)-threshold):double(s)<-threshold?T(double(s)+threshold):T(0);
                s=T(uy+byy);dyy=double(s)>threshold?T(double(s)-threshold):double(s)<-threshold?T(double(s)+threshold):T(0);
            }
            dx[i]=dxx;dy[i]=dyy;bx[i]=T(bxx+T(ux-dxx));by[i]=T(byy+T(uy-dyy));
        }
        rmse=std::sqrt(rmse/double(image.elements));
        if(trace){std::uint64_t bits;std::memcpy(&bits,&rmse,sizeof(bits));trace[iteration]={bits,!(rmse>parameters.eps)};}
        ++iteration;
    }
    return {iteration,rmse,actual};
}
} // namespace

const char* tv_bregman_ineligibility(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& p,TvBregmanTrace* trace,std::size_t capacity) noexcept {
    if(!fp_profile())return "TV Bregman requires nearest/gradual floating profile";
    GuardEnvironment environment;if(!environment.valid)return "TV Bregman floating environment snapshot unavailable";
    std::size_t width;
    if(image.type==TvBregmanType::float32)width=4;else if(image.type==TvBregmanType::float64)width=8;else return "TV Bregman requires native F32/F64";
    if(image.rows<2 || image.cols<2 || !image.channels)return "TV Bregman requires rows/cols>=2 and positive channels";
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/width;std::size_t count,padded;
    if(image.rows>std::numeric_limits<std::size_t>::max()-2 || image.cols>std::numeric_limits<std::size_t>::max()-2 || !multiply(image.rows,image.cols,limit,count) || !multiply(count,image.channels,limit,count) || !multiply(image.rows+2,image.cols+2,limit,padded) || !multiply(padded,image.channels,limit,padded))return "TV Bregman dimensions overflow";
    if(image.elements!=count)return "TV Bregman input element count mismatch";
    std::array<Span,7> spans{};if(!span(image.data,count,width,spans[0]))return "TV Bregman input extent/alignment invalid";
    const std::array<TvBregmanBuffer,5> buffers={workspace.out,workspace.dx,workspace.dy,workspace.bx,workspace.by};
    for(std::size_t i=0;i<buffers.size();++i)if(buffers[i].elements!=padded || !span(buffers[i].data,padded,width,spans[i+1]))return "TV Bregman workspace count/extent/alignment invalid";
    std::size_t used=6;
    if(trace) {
        const auto required=p.max_iterations>0?std::size_t(p.max_iterations):0;
        if(capacity<required || !span(trace,capacity,sizeof(TvBregmanTrace),spans[6],alignof(TvBregmanTrace)))return "TV Bregman trace capacity/extent/alignment invalid";
        used=7;
    }else if(capacity)return "TV Bregman trace pointer missing";
    for(std::size_t i=0;i<used;++i)for(std::size_t j=0;j<i;++j)if(overlap(spans[i],spans[j]))return "TV Bregman buffers/trace must be disjoint";
    if(!std::isfinite(p.weight) || p.weight<=0 || !std::isfinite(p.eps) || p.eps<0)return "TV Bregman finite positive weight/nonnegative eps required";
    return image.type==TvBregmanType::float32?typed_guard<float>(image,workspace,p):typed_guard<double>(image,workspace,p);
}
TvBregmanResult tv_bregman_into(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& p,TvBregmanTrace* trace,std::size_t capacity) {
    const int workers=get_num_threads();
    if(const char* reason=tv_bregman_ineligibility(image,workspace,p,trace,capacity))throw std::invalid_argument(reason);
    return image.type==TvBregmanType::float32?process<float>(image,workspace,p,trace,workers):process<double>(image,workspace,p,trace,workers);
}
} // namespace skimagexx
