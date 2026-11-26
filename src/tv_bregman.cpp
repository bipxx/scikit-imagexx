#include "skimagexx/tv_bregman.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#if defined(SKIMAGEXX_AVX2) && (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define SKIMAGEXX_BREGMAN_AVX2_GUARD
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4752)
#define BREGMAN_AVX_ENTRY __declspec(noinline)
#else
#define BREGMAN_AVX_ENTRY __attribute__((target("avx2"), noinline))
#endif
#endif

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace simd {
void tv_bregman_copy4(const float*,float*,float*,float*,float*,std::size_t);
void tv_bregman_copy4(const double*,double*,double*,double*,double*,std::size_t);
void tv_bregman_wavefront(const float*,float*,float*,float*,float*,float*,float*,
                          std::size_t,std::size_t,float,float,float,bool);
void tv_bregman_wavefront(const double*,double*,double*,double*,double*,double*,double*,
                          std::size_t,std::size_t,double,double,double,bool);
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
    if(a && b>limit/a)return false;
    value=a*b;return true;
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
#ifdef SKIMAGEXX_BREGMAN_AVX2_GUARD
template<class T>
BREGMAN_AVX_ENTRY bool finite_blocks(const T* data,std::size_t n,std::size_t& i) {
    constexpr std::size_t lanes=32/sizeof(T);
    for(;i+lanes<=n;i+=lanes) {
        // A finite IEEE value cannot have an all-ones exponent, including
        // when the sign or mantissa bits are set.
        const auto bits=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+i));
        __m256i nonfinite;
        if constexpr(sizeof(T)==sizeof(float)) {
            const auto exponent=_mm256_set1_epi32(0x7f800000);
            nonfinite=_mm256_cmpeq_epi32(_mm256_and_si256(bits,exponent),exponent);
        } else {
            const auto exponent=_mm256_set1_epi64x(0x7ff0000000000000LL);
            nonfinite=_mm256_cmpeq_epi64(_mm256_and_si256(bits,exponent),exponent);
        }
        if(_mm256_movemask_epi8(nonfinite))return false;
    }
    return true;
}
#undef BREGMAN_AVX_ENTRY
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

template<class T> bool finite(const T* data,std::size_t n) {
    std::size_t i=0;
#ifdef SKIMAGEXX_BREGMAN_AVX2_GUARD
    if(n>=32/sizeof(T) && has_avx2() && !finite_blocks(data,n,i))return false;
#endif
    for(;i<n;++i)if(!std::isfinite(data[i]))return false;
    return true;
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
    const auto row_width=c*d, row_bytes=row_width*sizeof(T), pixel_bytes=d*sizeof(T);
    // Populate each interior row and its side borders together after copying
    // the original padded workspace to the auxiliary planes. Corners stay put.
    for(std::size_t y=0;y<r;++y) {
        auto* destination=out+(y+1)*stride;
        const auto* source=input+y*row_width;
        std::memcpy(destination+d,source,row_bytes);
        std::memcpy(destination,source+d,pixel_bytes);
        std::memcpy(destination+row_width+d,source+row_width-d,pixel_bytes);
    }
    std::memcpy(out+d,input+row_width,row_bytes);
    std::memcpy(out+(r+1)*stride+d,input+(r-1)*row_width,row_bytes);
    return actual.load();
}
template<class T, bool Isotropic>
inline T sweep_pixel(T source,T* out,T* dx,T* dy,T* bx,T* by,std::ptrdiff_t i,
                     std::ptrdiff_t step,std::ptrdiff_t stride,T weight,T lam,T norm,
                     T previous,T left,T right) {
    const T below=out[i+stride];
    const T ux=T(right-previous),uy=T(below-previous);
    T total=T(below+out[i-stride]);total=T(total+right);total=T(total+left);
    total=T(total+dx[i-step]);total=T(total-dx[i]);total=T(total+dy[i-stride]);total=T(total-dy[i]);total=T(total-bx[i-step]);total=T(total+bx[i]);total=T(total-by[i-stride]);total=T(total+by[i]);
    const T unew=T(T(T(lam*total)+T(weight*source))/norm);out[i]=unew;
    T tx=T(unew-previous);const T squared=T(tx*tx);
    const T bxx=bx[i],byy=by[i];T dxx,dyy;
    if constexpr(Isotropic) {
        tx=T(ux+bxx);const T ty=T(uy+byy);const T length=T(std::sqrt(double(T(T(tx*tx)+T(ty*ty)))));
        const T factor=T(length*lam);const double denominator=double(factor)+1.0;
        dxx=T(double(T(factor*tx))/denominator);dyy=T(double(T(factor*ty))/denominator);
    }else {
        const double threshold=1.0/double(lam);
        T s=T(ux+bxx);dxx=double(s)>threshold?T(double(s)-threshold):double(s)<-threshold?T(double(s)+threshold):T(0);
        s=T(uy+byy);dyy=double(s)>threshold?T(double(s)-threshold):double(s)<-threshold?T(double(s)+threshold):T(0);
    }
    dx[i]=dxx;dy[i]=dyy;bx[i]=T(bxx+T(ux-dxx));by[i]=T(byy+T(uy-dyy));
    return squared;
}

template<class T, bool Grayscale, bool Isotropic>
TvBregmanResult sweep(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& parameters,TvBregmanTrace* trace,int actual,T weight,T lam,T norm,[[maybe_unused]] T* squared=nullptr) {
    const auto* input=static_cast<const T*>(image.data);auto* out=static_cast<T*>(workspace.out.data);auto* dx=static_cast<T*>(workspace.dx.data);auto* dy=static_cast<T*>(workspace.dy.data);auto* bx=static_cast<T*>(workspace.bx.data);auto* by=static_cast<T*>(workspace.by.data);
    const std::ptrdiff_t step=Grayscale?1:std::ptrdiff_t(image.channels);
    const std::ptrdiff_t row_width=std::ptrdiff_t(image.cols)*step;
    const std::ptrdiff_t stride=row_width+2*step;
    const auto max_iterations=std::size_t(parameters.max_iterations>0?parameters.max_iterations:0);
    double rmse=std::numeric_limits<double>::max();std::size_t iteration=0;
    while(iteration<max_iterations && rmse>parameters.eps) {
        rmse=0.;
        std::size_t y=0;
#ifdef SKIMAGEXX_AVX2
        if constexpr(Grayscale) {
            if(squared) {
                constexpr std::size_t lanes=32/sizeof(T);
                const auto cols=image.cols;
                for(;y+4<=image.rows;) {
                    const std::size_t height=cols>=lanes && image.rows-y>=lanes ? lanes : 4;
                    const auto first=std::ptrdiff_t(y+1)*stride+1;
                    auto edge=[&](std::size_t lane,std::size_t begin,std::size_t end) {
                        for(auto x=begin;x<end;++x) {
                            const auto i=first+std::ptrdiff_t(lane)*stride+std::ptrdiff_t(x);
                            squared[(x+lane)*lanes+lane]=sweep_pixel<T,Isotropic>(
                                input[(y+lane)*cols+x],out,dx,dy,bx,by,i,1,stride,weight,lam,norm,
                                out[i],out[i-1],out[i+1]);
                        }
                    };
                    // Ramp up the diagonal, then finish the right-hand triangle.
                    for(std::size_t lane=0;lane<height-1;++lane)edge(lane,0,height-1-lane);
                    simd::tv_bregman_wavefront(input+y*cols,out+first,dx+first,dy+first,
                                               bx+first,by+first,squared,cols,height,weight,lam,norm,Isotropic);
                    for(std::size_t lane=1;lane<height;++lane)edge(lane,cols-lane,cols);
                    // The convergence test consumes each term in source row order.
                    for(std::size_t lane=0;lane<height;++lane)
                        for(std::size_t x=0;x<cols;++x)rmse+=double(squared[(x+lane)*lanes+lane]);
                    y+=height;
                }
            }
        }
#endif
        const T* source=input+y*row_width;
        std::ptrdiff_t row=std::ptrdiff_t(y+1)*stride+step;
        for(;y<image.rows;++y,row+=stride) {
            // Carry adjacent grayscale values through the ordered sweep.
            // The left neighbor has been updated and the right one has not.
            if constexpr(Grayscale) {
                T left=out[row-1],previous=out[row];
                for(std::ptrdiff_t i=row;i<row+row_width;++i,++source) {
                    const T right=out[i+1];
                    rmse+=double(sweep_pixel<T,Isotropic>(*source,out,dx,dy,bx,by,i,1,stride,
                        weight,lam,norm,previous,left,right));
                    left=out[i];
                    previous=right;
                }
            } else {
                for(std::ptrdiff_t i=row;i<row+row_width;++i,++source)
                    rmse+=double(sweep_pixel<T,Isotropic>(*source,out,dx,dy,bx,by,i,step,stride,
                        weight,lam,norm,out[i],out[i-step],out[i+step]));
            }
        }
        rmse=std::sqrt(rmse/double(image.elements));
        if(trace){std::uint64_t bits;std::memcpy(&bits,&rmse,sizeof(bits));trace[iteration]={bits,!(rmse>parameters.eps)};}
        ++iteration;
    }
    return {iteration,rmse,actual};
}
template<class T> TvBregmanResult process(const TvBregmanImage& image,const TvBregmanWorkspace& workspace,const TvBregmanParameters& parameters,TvBregmanTrace* trace,int workers) {
    T weight{},lam{},norm{};
    if(!coefficients(parameters.weight,weight,lam,norm))throw std::invalid_argument("TV Bregman coefficient prewrite check failed");
    if(image.channels==1) {
        T* squared=nullptr;
#ifdef SKIMAGEXX_AVX2
        constexpr std::size_t lanes=32/sizeof(T);
        constexpr std::size_t local_cols=520;
        alignas(32) T local_squared[(local_cols+lanes-1)*lanes];
        std::unique_ptr<T[]> large_squared;
        // Gather offsets hold the distance from the first row's pixel to
        // each diagonal lane. Fewer than four rows or columns stay scalar.
        if(parameters.max_iterations>0 && image.rows>=4 && image.cols>=4 &&
           image.cols<std::size_t(std::numeric_limits<int>::max())/lanes && has_avx2()) {
            if(image.cols<=local_cols) squared=local_squared;
            else {
                large_squared.reset(new T[(image.cols+lanes-1)*lanes]);
                squared=large_squared.get();
            }
        }
#endif
        const int actual=initialize<T>(image,workspace,workers);
        if(parameters.isotropic)return sweep<T,true,true>(image,workspace,parameters,trace,actual,weight,lam,norm,squared);
        return sweep<T,true,false>(image,workspace,parameters,trace,actual,weight,lam,norm,squared);
    }
    const int actual=initialize<T>(image,workspace,workers);
    if(parameters.isotropic)return sweep<T,false,true>(image,workspace,parameters,trace,actual,weight,lam,norm);
    return sweep<T,false,false>(image,workspace,parameters,trace,actual,weight,lam,norm);
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
