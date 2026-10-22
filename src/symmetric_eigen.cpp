#include "skimagexx/symmetric_eigen.hpp"
#include "symmetric_eigen_detail.hpp"
#include "execution.hpp"
#include <atomic>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace skimagexx { namespace {
template<class T> bool execute(const void* const* fields,T* output,std::size_t count,std::size_t n,
    bool pair,PairPrecision precision,bool interleaved) {
    const auto limit=detail::eigen_limit<T>(n,pair);std::atomic<bool> supported{true};
    const auto factor=std::max(std::size_t(4),n*n*n);
    const auto work=count>std::numeric_limits<std::size_t>::max()/factor ? std::numeric_limits<std::size_t>::max() : count*factor;
    const auto blocks=(count+7)/8;
    const auto workers=std::min(detail::active_workers(work),int(std::min(blocks,std::size_t(std::numeric_limits<int>::max()))));
    detail::partition(blocks,workers,[&](auto block_begin,auto block_end,auto){
        const auto begin=block_begin*8,end=std::min(count,block_end*8);
        bool valid=true;std::size_t i=begin;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2() && precision==PairPrecision::native)i=simd::symmetric_eigenvalues(fields,output,count,n,begin,end,std::is_same_v<T,float>,pair,interleaved,valid);
#endif
        if(!valid){supported.store(false,std::memory_order_relaxed);return;}
        std::vector<double> matrix(n*n),values(n);
        for(;i<end;++i){
            double scale=0.;
            for(std::size_t field=0;field<n*(n+1)/2;++field){const auto x=double(static_cast<const T*>(fields[field])[i]);
                if(!std::isfinite(x) || std::abs(x)>limit){supported.store(false,std::memory_order_relaxed);return;}
                scale=std::max(scale,std::abs(x));
            }
            if(n==1){output[i]=static_cast<const T*>(fields[0])[i];continue;}
            if(pair && n==2){
                const auto upper=interleaved ? 2*i+1 : i,lower=interleaved ? 2*i : count+i;
                const auto a=static_cast<const T*>(fields[0])[i],b=static_cast<const T*>(fields[1])[i],c=static_cast<const T*>(fields[2])[i];
                const T mean=T(T(a+c)/T(2));
                if constexpr(std::is_same_v<T,float>){
                    if(precision!=PairPrecision::native){
                        const auto diff=double(T(a-c))/2;
                        const auto square=precision==PairPrecision::wide_scalar ? double(b)*double(b) : double(T(b*b));
                        const auto root=std::sqrt(square+diff*diff);
                        output[upper]=T(double(mean)+root);output[lower]=T(double(mean)-root);continue;
                    }
                }
                const T half=T(T(a-c)/T(2));const T root=std::sqrt(T(T(b*b)+T(half*half)));
                output[upper]=T(mean+root);output[lower]=T(mean-root);continue;
            }
            if(!scale)scale=1;
            std::size_t field=0;
            for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){const auto x=double(static_cast<const T*>(fields[field++])[i])/scale;matrix[r*n+c]=matrix[c*n+r]=x;}
            if(!detail::eigen_jacobi(matrix.data(),values.data(),n)){supported.store(false,std::memory_order_relaxed);return;}
            for(std::size_t d=0;d<n;++d)output[interleaved ? i*n+n-1-d : d*count+i]=T(values[d]*scale);
        }
    });
    return supported.load(std::memory_order_relaxed);
}
}}
namespace skimagexx {
bool symmetric_eigenvalues_into(const void* const* fields,void* output,std::size_t count,std::size_t dimension,
    bool single,bool pair,PairPrecision precision,bool interleaved) {
    if(!dimension || dimension>64 || unsigned(precision)>unsigned(PairPrecision::wide_scalar_narrow_square))throw std::invalid_argument("invalid symmetric eigenvalue dimensions or precision");
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/(dimension*(single ? sizeof(float) : sizeof(double))))throw std::overflow_error("symmetric eigenvalue dimensions overflow");
    if(!count)return true;
    if(!fields || !output)throw std::invalid_argument("nonnull eigenvalue buffers required");
    for(std::size_t i=0;i<dimension*(dimension+1)/2;++i)if(!fields[i])throw std::invalid_argument("nonnull eigenvalue component required");
    return single ? execute(fields,static_cast<float*>(output),count,dimension,pair,precision,interleaved)
                  : execute(fields,static_cast<double*>(output),count,dimension,pair,precision,interleaved);
}
}
