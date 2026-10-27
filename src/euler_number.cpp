#include "euler_number.hpp"
#include "perimeter_crofton.hpp"
#include "execution.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx::detail {
namespace {
void check_workers(int workers){
    if(workers<1||workers>1024)throw std::invalid_argument("Euler worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
}
template<bool Volume>void counts(const EulerInput& in,std::ptrdiff_t* out,int workers){
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(in.height<2||in.width<2||(Volume?in.depth<2:in.depth!=1))
        throw std::invalid_argument("Euler input requires zero-padded extents");
    if(in.height>limit/in.width||in.depth>limit/(in.height*in.width))
        throw std::overflow_error("Euler storage extent overflow");
    if(!in.data||!out)throw std::invalid_argument("nonnull Euler buffers required");
    const auto plane=in.height*in.width,total=in.depth*plane;
    const auto team=total>=parallel_min?workers:1;
    const bool vectorized=has_avx2();
#ifndef SKIMAGEXX_AVX2
    (void)vectorized;
#endif
    using Histogram=std::array<std::ptrdiff_t,Volume?256:16>;
    std::vector<Histogram> local(static_cast<std::size_t>(team));
    partition(in.depth*in.height,team,[&](auto begin,auto end,auto worker){
        auto* histogram=local[worker].data();
        for(auto row=begin;row<end;++row){
            const auto z=row/in.height,y=row%in.height;
            if(!y||(Volume&&!z)){histogram[0]+=std::ptrdiff_t(in.width);continue;}
            const auto* current=in.data+row*in.width;const auto* above=current-in.width;
            ++histogram[0];std::size_t x=1; // The complete first column is zero padding.
            if constexpr(Volume){
                const auto* previous=current-plane;const auto* previous_above=previous-in.width;
#ifdef SKIMAGEXX_AVX2
                if(vectorized)x=simd::euler_row(current,above,previous,previous_above,in.width,histogram);
#endif
                for(;x<in.width;++x)++histogram[current[x]+4*current[x-1]+2*above[x]+8*above[x-1]
                    +16*previous[x]+64*previous[x-1]+32*previous_above[x]+128*previous_above[x-1]];
            }else{
#ifdef SKIMAGEXX_AVX2
                if(vectorized)x=simd::crofton_row(current,above,in.width,histogram);
#endif
                for(;x<in.width;++x)++histogram[current[x]+4*current[x-1]+2*above[x]+8*above[x-1]];
            }
        }
    });
    Histogram sum{};
    for(const auto& histogram:local)for(std::size_t i=0;i<sum.size();++i)sum[i]+=histogram[i];
    std::memcpy(out,sum.data(),sizeof(sum));
}
}
void euler_counts(const EulerInput& input,bool volume,std::ptrdiff_t* histogram,int workers){
    check_workers(workers);
    if(volume)counts<true>(input,histogram,workers);else counts<false>(input,histogram,workers);
}
bool euler_weighted(const std::ptrdiff_t* histogram,const std::int8_t* coefficients,
                    bool volume,std::int64_t& result,int workers){
    check_workers(workers);
    if(!histogram||!coefficients)throw std::invalid_argument("nonnull Euler weighted buffers required");
    const auto bins=volume?256u:16u;std::uint64_t budget=UINT64_C(1)<<53;
    for(std::size_t i=0;i<bins;++i){
        if(histogram[i]<0)throw std::invalid_argument("Euler counts must be nonnegative");
        const auto magnitude=unsigned(coefficients[i]<0?-int(coefficients[i]):int(coefficients[i]));
        if(magnitude){
            const auto count=std::uint64_t(histogram[i]);
            if(count>budget/magnitude)return false;
            budget-=count*magnitude;
        }
    }
    std::int64_t numerator=0;
    for(std::size_t i=0;i<bins;++i)numerator+=std::int64_t(coefficients[i])*std::int64_t(histogram[i]);
    result=volume?numerator/8:numerator;
    return true;
}
}
