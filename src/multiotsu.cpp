#include "skimagexx/threshold.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace skimagexx {
// Float32 cumulative moments, objective ordering and first-maximum tie policy
// follow scikit-image 0.26.0's BSD-licensed _multiotsu.pyx. In particular, its
// first moment starts at prob[0], and the [0,0] score is deliberately zero.
std::vector<std::size_t> threshold_multiotsu(const float* prob,std::size_t n,std::size_t classes) {
    if(!prob || n<classes || classes<2 || classes>16 || n>4096)
        throw std::invalid_argument("multi-Otsu requires 2..16 classes and at most 4096 bins");
    // Two classes need only prefix and suffix scores, rather than the whole
    // triangular table. Keep the same prefix rounding and zero [0,0] score.
    std::vector<float> mass(n),moment(n),first(n,0);
    mass[0]=moment[0]=prob[0];
    for(std::size_t i=1;i<n;++i) {
        mass[i]=mass[i-1]+prob[i];
        moment[i]=moment[i-1]+float(i)*prob[i];
        if(mass[i]>0)first[i]=(moment[i]*moment[i])/mass[i];
    }
    if(classes==2) {
        std::vector<float> scores(n-1);std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())done=simd::multiotsu_two_scores(mass.data(),moment.data(),first.data(),
            scores.data(),n-1,mass.back(),moment.back());
#endif
        for(std::size_t j=done;j+1<n;++j) {
            const float w=mass.back()-mass[j],s=moment.back()-moment[j];
            scores[j]=first[j]+(w>0 ? (s*s)/w : 0);
        }
        float best=0;std::size_t index=0;
        for(std::size_t j=0;j+1<n;++j)if(scores[j]>best) {best=scores[j];index=j;}
        if(!(best>0))return {};
        return {index};
    }
    std::vector<float> lut(n*(n+1)/2,0),tail(n);
    std::copy(first.begin(),first.end(),lut.begin());
    const auto row=[n](std::size_t i){return i*(2*n-i+1)/2;};
    const int workers=detail::active_workers(n*n);
    detail::partition(n-1,workers,[&](std::size_t begin,std::size_t end,std::size_t) {
        for(std::size_t i=begin+1;i<=end;++i) {
            auto* out=lut.data()+row(i);
            for(std::size_t j=i;j<n;++j) {
                const float w=mass[j]-mass[i-1],s=moment[j]-moment[i-1];
                out[j-i]=w>0 ? (s*s)/w : 0;
            }
        }
    });
    for(std::size_t i=0;i<n;++i)tail[i]=lut[row(i)+n-1-i];
    struct Result {float score=0;std::vector<std::size_t> indices;};
    std::vector<Result> results(std::size_t(workers),Result{0,std::vector<std::size_t>(classes-1,0)});
    detail::partition(n-classes+1,workers,[&](std::size_t begin,std::size_t end,std::size_t id) {
        auto& best=results[id];std::vector<std::size_t> current(classes-1);
        std::vector<float> scores(n),fixed(classes-2);
        const auto consider=[&](std::size_t j,float score) {
            if(score>best.score) {best.score=score;current.back()=j;best.indices=current;}
        };
        const auto visit=[&](auto&& self,std::size_t depth,std::size_t start)->void {
            if(depth==classes-2) {
                const float first=lut[current[0]];
                const auto prev=current[depth-1];
                for(std::size_t k=0;k+1<depth;++k)
                    fixed[k]=lut[row(current[k]+1)+current[k+1]-current[k]-1];
                const auto* middle=lut.data()+row(prev+1);
                const auto count=n-1-start;
                std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())done=simd::multiotsu_scores(first,tail.data()+start+1,middle,
                    fixed.data(),depth-1,scores.data(),count);
#endif
                for(std::size_t k=done;k<count;++k) {
                    float score=first+tail[start+k+1];
                    for(std::size_t a=0;a+1<depth;++a)score+=fixed[a];
                    scores[k]=score+middle[k];
                }
                for(std::size_t k=0;k<count;++k)consider(start+k,scores[k]);
                return;
            }
            const auto stop=n-classes+depth+1;
            for(std::size_t j=start;j<stop;++j) {current[depth]=j;self(self,depth+1,j+1);}
        };
        for(std::size_t i=begin;i<end;++i) {current[0]=i;visit(visit,1,i+1);}
    });
    // Partitions and local traversal are lexicographic: strict comparison
    // preserves upstream's first representative even when workers vary.
    Result best=results[0];
    for(std::size_t i=1;i<results.size();++i)if(results[i].score>best.score)best=results[i];
    if(!(best.score>0))return {};
    return best.indices;
}
}
