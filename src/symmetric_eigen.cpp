#include "skimagexx/symmetric_eigen.hpp"
#include "symmetric_eigen_detail.hpp"
#include "execution.hpp"
#include <array>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace skimagexx::detail {
namespace {
template<class T> struct RidgeScratch {
    std::unique_ptr<T[]> storage;
    std::size_t capacity=0;
    bool in_use=false;
};

template<class T> auto& ridge_scratch() {
    static thread_local RidgeScratch<T> scratch;
    return scratch;
}

template<class T> void release_ridge_scratch(void* pointer) {
    auto& scratch=ridge_scratch<T>();
    if(pointer==scratch.storage.get())scratch.in_use=false;
    else delete[] static_cast<T*>(pointer);
}

template<class T> std::unique_ptr<void,void(*)(void*)> lease_ridge_scratch(std::size_t elements) {
    // Reuse workspace across scale calls on the caller's thread. A provider
    // can reenter a ridge kernel; give nested calls independent storage.
    constexpr std::size_t max_cached=16*1024*1024/sizeof(T);
    auto& scratch=ridge_scratch<T>();
    if(scratch.in_use || elements>max_cached)
        return {new T[elements],release_ridge_scratch<T>};
    if(elements>scratch.capacity){
        scratch.storage.reset(new T[elements]);
        scratch.capacity=elements;
    }
    scratch.in_use=true;
    return {scratch.storage.get(),release_ridge_scratch<T>};
}
}

std::unique_ptr<void,void(*)(void*)> acquire_ridge_scratch(std::size_t elements,bool single) {
    return single ? lease_ridge_scratch<float>(elements) : lease_ridge_scratch<double>(elements);
}
}

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
        if(i==end)return;
        std::array<double,25> small_matrix;
        std::array<double,5> small_values;
        std::vector<double> large_matrix(n>5 ? n*n : 0),large_values(n>5 ? n : 0);
        auto* matrix=n>5 ? large_matrix.data() : small_matrix.data();
        auto* values=n>5 ? large_values.data() : small_values.data();
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
            if(!detail::eigen_jacobi(matrix,values,n)){supported.store(false,std::memory_order_relaxed);return;}
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
