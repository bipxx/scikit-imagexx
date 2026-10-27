#include "perimeter.hpp"
#include "execution.hpp"
#include "skimagexx/morphology.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx::detail {
namespace {
void check_workers(int workers){
    if(workers<1||workers>1024)throw std::invalid_argument("Perimeter worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
}
}
std::size_t perimeter_counts(const std::uint8_t* image,std::size_t height,std::size_t width,
    bool cross,std::ptrdiff_t* histogram,int workers){
    check_workers(workers);
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(height>limit-2||width>limit-2||(height+2)>limit/(width+2))
        throw std::overflow_error("Perimeter padded extent overflow");
    const auto n=height*width,cols=width+2;
    if(!histogram||(n&&!image))throw std::invalid_argument("nonnull Perimeter buffers required");
    std::vector<std::uint8_t> eroded(n),border((height+2)*cols,0);
    MorphologyFootprint footprint;
    for(std::ptrdiff_t y=-1;y<=1;++y)for(std::ptrdiff_t x=-1;x<=1;++x)
        if(!cross||!y||!x){footprint.offsets.push_back(y);footprint.offsets.push_back(x);}
    footprint.scipy_separable=!cross;
    morphology_into(image,eroded.data(),PixelType::uint8,PixelType::uint8,{height,width},
        {footprint},MorphologyOp::binary_erosion,MorphologyMode::minimum);
    const auto team=n>=parallel_min?workers:1;
    const bool vectorized=has_avx2();
    partition(height,team,[&](auto begin,auto end,auto){
        for(auto y=begin;y<end;++y){
            if(!width)continue;
            const auto* source=image+y*width;const auto* inner=eroded.data()+y*width;
            auto* target=border.data()+(y+1)*cols+1;std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)x=simd::perimeter_border(source,inner,target,width);
#else
            (void)vectorized;
#endif
            for(;x<width;++x)target[x]=std::uint8_t(source[x]-inner[x]);
        }
    });
    using Histogram=std::array<std::ptrdiff_t,256>;
    std::vector<Histogram> local(static_cast<std::size_t>(team));
    partition(height,team,[&](auto begin,auto end,auto worker){
        auto* counts=local[worker].data();
        for(auto y=begin;y<end;++y){
            const auto* current=border.data()+(y+1)*cols+1;
            const auto* previous=current-cols;const auto* next=current+cols;std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)x=simd::perimeter_row(previous,current,next,width,counts);
#endif
            for(;x<width;++x){
                const auto j=std::ptrdiff_t(x);
                const auto value=current[j]+2u*(previous[j]+next[j]+current[j-1]+current[j+1])
                    +10u*(previous[j-1]+previous[j+1]+next[j-1]+next[j+1]);
                ++counts[value&255u];
            }
        }
    });
    Histogram total{};
    for(const auto& counts:local)for(std::size_t i=0;i<256;++i)total[i]+=counts[i];
    std::size_t length=256;while(length>50&&!total[length-1])--length;
    std::memcpy(histogram,total.data(),sizeof(total));return length;
}
double perimeter_weighted(const std::ptrdiff_t* histogram,const double* weights,int workers){
    check_workers(workers);
    if(!histogram||!weights)throw std::invalid_argument("nonnull Perimeter weighted buffers required");
    struct Capture {
        std::fenv_t saved;
        Capture(){if(std::feholdexcept(&saved))throw std::runtime_error("cannot capture Perimeter floating environment");}
        ~Capture(){std::fesetenv(&saved);}
    } capture;
    double result=0.;
    for(std::size_t i=0;i<50;++i)result+=double(histogram[i])*weights[i];
    return result;
}
}
