#include "skimagexx/threshold.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "threshold_detail.hpp"
#if defined(SKIMAGEXX_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {
// NumPy's float32 add.reduce uses eight accumulators and 128-element blocks.
float pairwise_sum(const float* a,std::size_t n) {
    if(n<8) {float s=-0.0f;for(std::size_t i=0;i<n;++i)s+=a[i];return s;}
    if(n<=128) {
        float r[8];std::copy_n(a,8,r);std::size_t i=8;
        for(;i+8<=n;i+=8)for(int j=0;j<8;++j)r[j]+=a[i+j];
        float s=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
        for(;i<n;++i)s+=a[i];return s;
    }
    auto half=n/2;half-=half%8;
    return pairwise_sum(a,half)+pairwise_sum(a+half,n-half);
}
template<class T> std::vector<std::size_t> isodata(const float* c,const double* centers,std::size_t n,unsigned& warnings,double bin_width) {
    std::vector<float> counts(n);std::vector<T> sums(n);
    float weight=0;T sum=0;
    for(std::size_t i=0;i<n;++i) {
        weight+=c[i];sum+=T(T(c[i])*T(centers[i]));counts[i]=weight;sums[i]=sum;
    }
    std::vector<std::size_t> result;
    const T width=std::isnan(bin_width) ? T(T(centers[1])-T(centers[0])) : T(bin_width);
    for(std::size_t i=0;i+1<n;++i) {
        const T lower=T(sums[i]/T(counts[i]));
        const T numerator=T(sums.back()-sums[i]);
        const T denominator=T(counts.back()-counts[i]);
        const T higher=T(numerator/denominator);
        if(counts[i]==0)warnings|=8;
        if(denominator==0)warnings|=numerator==0 ? 16u : 32u;
        const T distance=T(T(T(lower+higher)/T(2))-T(centers[i]));
        if(distance>=T(0) && distance<width)result.push_back(i);
    }
    return result;
}
}
// Range reduction and rational coefficients follow NumPy's BSD-licensed
// float32 log approximation. Scalar fma preserves its rounding without an ISA
// requirement; the Python wrapper selects the installed NumPy's provider.
float threshold_log(float value,bool numpy_log) {
    if(!numpy_log || !(value>0) || !std::isfinite(value))return std::log(value);
    int exponent=0;float x=std::frexp(value,&exponent);
    if(x<=detail::threshold_log_sqrt_half) {x+=x;--exponent;}
    x-=1.0f;
    const auto& p=detail::threshold_log_p;const auto& q=detail::threshold_log_q;
    float numerator=p[5],denominator=q[5];
    for(int i=4;i>=0;--i) {numerator=std::fma(numerator,x,p[i]);denominator=std::fma(denominator,x,q[i]);}
    return std::fma(float(exponent),detail::threshold_log_ln2,numerator/denominator);
}
namespace {
#ifdef SKIMAGEXX_AVX2
bool threshold_fma_available() {
    if(!has_avx2())return false;
    static const bool available=[] {
#if defined(_MSC_VER)
        int regs[4];__cpuidex(regs,1,0);return (regs[2]&(1<<12))!=0;
#elif defined(__GNUC__) || defined(__clang__)
        return bool(__builtin_cpu_supports("fma"));
#else
        return false;
#endif
    }();
    return available;
}
#endif
}
void threshold_logs(const float* input,float* output,std::size_t count,bool numpy_log) {
    if(count && (!input || !output))throw std::invalid_argument("nonnull log storage required");
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if(numpy_log && threshold_fma_available())i=simd::threshold_logs(input,output,count);
#endif
    for(;i<count;++i)output[i]=threshold_log(input[i],numpy_log);
}
std::vector<std::size_t> threshold_indices(const double* raw,const double* centers,std::size_t n,
                                          HistogramThreshold method,bool float_centers,unsigned* warning_output,bool numpy_log,double bin_width) {
    unsigned local_warnings=0;auto& warnings=warning_output ? *warning_output : local_warnings;warnings=0;
    if(!n || !raw || !centers)throw std::invalid_argument("nonempty histogram required");
    if(n==1)return {0};
    if(method==HistogramThreshold::triangle) {
        const auto* c=raw;
        const auto peak=std::size_t(std::max_element(c,c+n)-c);
        std::size_t low=0,high=n-1;
        while(low<n && c[low]==0)++low;
        while(high>0 && c[high]==0)--high;
        if(low>=n)throw std::invalid_argument("nonzero histogram required");
        if(low==high)return {low};
        const bool flip=peak-low<high-peak;
        const auto start=flip ? n-high-1 : low;
        const auto top=flip ? n-peak-1 : peak;
        const auto span=top-start;
        if(!span)throw std::invalid_argument("empty triangle span");
        const double height=c[peak],norm=std::sqrt(height*height+double(span)*double(span));
        const double h=height/norm,w=double(span)/norm;
        double best=-std::numeric_limits<double>::infinity();std::size_t arg=start;
        for(std::size_t x=0;x<span;++x) {
            const auto index=flip ? n-(x+start)-1 : x+start;
            const double value=h*double(x)-w*double(c[index]);
            if(value>best) {best=value;arg=x+start;}
        }
        return {flip ? n-arg-1 : arg};
    }
    std::vector<float> converted(raw,raw+n);const auto* c=converted.data();
    if(method==HistogramThreshold::isodata)
        return float_centers ? isodata<float>(c,centers,n,warnings,bin_width) : isodata<double>(c,centers,n,warnings,bin_width);
    if(method!=HistogramThreshold::yen)throw std::invalid_argument("unknown threshold method");
    const float total=pairwise_sum(c,n);
    if(total==0)warnings|=64;
    std::vector<float> p(n),left(n),left_sq(n),right_sq(n);
    float sum=0,squared=0;
    for(std::size_t i=0;i<n;++i) {
        p[i]=c[i]/total;sum+=p[i];squared+=p[i]*p[i];left[i]=sum;left_sq[i]=squared;
    }
    squared=0;for(std::size_t i=n;i-->0;) {squared+=p[i]*p[i];right_sq[i]=squared;}
    std::vector<float> scores(n-1);
    for(std::size_t i=0;i+1<n;++i) {
        const float product=left[i]*(1.0f-left[i]);
        const float denominator=left_sq[i]*right_sq[i+1];
        const float inverse=1.0f/denominator;
        const float operand=inverse*(product*product);
        if(denominator==0)warnings|=1;
        if(std::isinf(inverse) && product*product==0)warnings|=2;
        if(operand==0)warnings|=4;
        scores[i]=operand;
    }
    threshold_logs(scores.data(),scores.data(),scores.size(),numpy_log);
    float best=-std::numeric_limits<float>::infinity();std::size_t arg=0;
    for(std::size_t i=0;i<scores.size();++i) {
        const float value=scores[i];
        if(!std::isnan(best) && (std::isnan(value) || value>best)) {best=value;arg=i;}
    }
    return {arg};
}

template<class T> std::vector<std::uint64_t> histogram_edges(const T* image,std::size_t count,
                                                            const T* edges,std::size_t bins) {
    if(!bins || !edges || (count && !image))throw std::invalid_argument("invalid histogram storage");
    for(std::size_t i=0;i<bins;++i)
        if(!std::isfinite(edges[i]) || !(edges[i]<edges[i+1]))throw std::invalid_argument("finite increasing edges required");
    if(!std::isfinite(edges[bins]))throw std::invalid_argument("finite increasing edges required");
    const auto workers=detail::active_workers(count);
    std::vector<std::vector<std::uint64_t>> partials(static_cast<std::size_t>(workers),std::vector<std::uint64_t>(bins));
    const double low=edges[0],scale=double(bins)/(double(edges[bins])-low);
    detail::partition(count,workers,[&](auto begin,auto end,auto id) {
        auto* output=partials[id].data();auto i=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2() && std::isfinite(scale) && scale>0 && bins<std::size_t(std::numeric_limits<int>::max()))
            i+=simd::histogram_edges(image+begin,end-begin,edges,bins,output);
#endif
        for(;i<end;++i) {
            const auto v=image[i];
            if(!(v>=edges[0] && v<=edges[bins]))continue;
            std::size_t index;
            const double estimate=(double(v)-low)*scale;
            if(!std::isfinite(estimate) || estimate<0 || estimate>double(bins))
                index=std::size_t(std::upper_bound(edges,edges+bins+1,v)-edges-1);
            else index=std::min(std::size_t(estimate),bins-1);
            if(index==bins)index=bins-1;
            while(index && v<edges[index])--index;
            while(index+1<bins && v>=edges[index+1])++index;
            ++output[index];
        }
    });
    auto result=std::move(partials[0]);
    for(std::size_t id=1;id<partials.size();++id)
        for(std::size_t b=0;b<bins;++b)result[b]+=partials[id][b];
    return result;
}
template std::vector<std::uint64_t> histogram_edges(const float*,std::size_t,const float*,std::size_t);
template std::vector<std::uint64_t> histogram_edges(const double*,std::size_t,const double*,std::size_t);
}
