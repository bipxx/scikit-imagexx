#include "shannon_entropy.hpp"
#include "execution.hpp"
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
    for(std::size_t i=0;i<size;++i){
        if(counts[i]<=0||std::size_t(counts[i])>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-integer_total)
            throw std::invalid_argument("entropy requires positive addressable counts");
        integer_total+=std::size_t(counts[i]);
    }
    Capture capture;
    if(!size)return 0.;
    const double total=0.+pairwise(0,size,[&](auto i){return double(counts[i]);});
    std::vector<double> terms(size);
    partition(size,size>=parallel_min?workers:1,[&](auto begin,auto end,auto){
        Capture worker_capture;
        for(auto i=begin;i<end;++i){const double p=double(counts[i])/total;terms[i]=-p*std::log(p);}
    });
    return 0.+pairwise(0,size,[&](auto i){return terms[i];});
}
}
