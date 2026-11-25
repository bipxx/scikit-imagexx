#include "shannon_entropy.hpp"
#include "execution.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace skimagexx::detail {
namespace {
// NumPy DOUBLE_pairwise_sum (128 block), same association as colocalization.
// Primary BSD notice: third_party/numpy-LICENSE.txt.
template<class Get>double pairwise(std::size_t start,std::size_t n,const Get& get) {
    if(n<8){double sum=-0.;for(std::size_t i=0;i<n;++i)sum+=get(start+i);return sum;}
    if(n<=128){
        double v[8];for(std::size_t i=0;i<8;++i)v[i]=get(start+i);
        std::size_t i=8;for(;i<n-n%8;i+=8)for(std::size_t j=0;j<8;++j)v[j]+=get(start+i+j);
        double sum=((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7]));
        for(;i<n;++i)sum+=get(start+i);return sum;
    }
    auto left=n/2;left-=left%8;
    const double a=pairwise(start,left,get),b=pairwise(start+left,n-left,get);return a+b;
}
class Capture {
    std::fenv_t saved_;
public:
    Capture(){if(std::feholdexcept(&saved_))throw std::runtime_error("cannot capture entropy floating environment");}
    ~Capture(){std::fesetenv(&saved_);}
};
}
double shannon_entropy_counts(const std::ptrdiff_t* counts,std::size_t size,int workers){
    if(workers<1||workers>1024)throw std::invalid_argument("entropy worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
    if(size>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double)||(size&&!counts))
        throw std::invalid_argument("entropy count extent overflow or null storage");
    std::size_t integer_total=0;
    auto min_c=std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t max_c=0;
    for(std::size_t i=0;i<size;++i){
        const auto count=counts[i];
        if(count<=0||std::size_t(count)>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-integer_total)
            throw std::invalid_argument("entropy requires positive addressable counts");
        integer_total+=std::size_t(count);
        min_c=std::min(min_c,count);
        max_c=std::max(max_c,count);
    }
    Capture capture;
    if(!size)return 0.;
    // Every partial sum is exact when the integer total fits in binary64.
    const double total=integer_total<=9007199254740992ULL ? double(integer_total) :
        0.+pairwise(0,size,[&](auto i){return double(counts[i]);});
    const auto term=[&](std::ptrdiff_t count) {
        const double p=double(count)/total;
        return -p*std::log(p);
    };
    if(min_c==max_c) {
        const double uniform_term=term(min_c);
        return 0.+pairwise(0,size,[&](auto){return uniform_term;});
    }
    constexpr std::ptrdiff_t max_cached_count=256;
    using TermCache=std::array<double,max_cached_count+1>;
    const auto cached_term=[&](TermCache& term_cache,std::ptrdiff_t count) {
        if(count>max_cached_count)return term(count);
        auto& cached=term_cache[std::size_t(count)];
        // A nonuniform distribution has at least two symbols, so a cached
        // count has 0 < p < 1 and contributes a strictly positive term.
        if(cached==0.)cached=term(count);
        return cached;
    };
    if(workers==1 || size<parallel_min || max_c<=max_cached_count) {
        if(min_c>max_cached_count)return 0.+pairwise(0,size,[&](auto i){return term(counts[i]);});
        TermCache term_cache{};
        return 0.+pairwise(0,size,[&](auto i){return cached_term(term_cache,counts[i]);});
    }

    std::vector<double> terms(size);
    partition(size,workers,[&](auto begin,auto end,auto){
        Capture worker_capture;
        if(min_c>max_cached_count) {
            for(auto i=begin;i<end;++i)terms[i]=term(counts[i]);
        } else {
            TermCache term_cache{};
            for(auto i=begin;i<end;++i)terms[i]=cached_term(term_cache,counts[i]);
        }
    });
    return 0.+pairwise(0,size,[&](auto i){return terms[i];});
}
}
