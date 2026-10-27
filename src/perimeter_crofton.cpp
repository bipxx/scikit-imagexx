#include "perimeter_crofton.hpp"
#include "execution.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx::detail {
namespace {
void check_workers(int workers){
    if(workers<1||workers>1024)throw std::invalid_argument("Crofton worker snapshot outside [1,1024]");
#ifndef _OPENMP
    if(workers!=1)throw std::invalid_argument("this build has no OpenMP support");
#endif
}
template<class T>bool positive(T value){
    if constexpr(std::is_floating_point_v<T>){
        using Word=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
        Word word;std::memcpy(&word,&value,sizeof(word));
        constexpr Word infinity=sizeof(T)==4?Word(0x7f800000u):Word(UINT64_C(0x7ff0000000000000));
        // Exact ordered >0 truth values, with no floating exception side effects.
        return word>0&&word<=infinity;
    }else return value>T(0);
}
template<class T>void counts(const CroftonInput& in,std::ptrdiff_t* out,int workers){
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(in.height>limit-2||in.width>limit-2)throw std::overflow_error("Crofton padded extent overflow");
    const auto rows=in.height+2,cols=in.width+2;
    if(rows>limit/cols||(in.height&&(in.width>limit/sizeof(T)/in.height)))
        throw std::overflow_error("Crofton storage extent overflow");
    if(!out||(in.height&&in.width&&!in.data))throw std::invalid_argument("nonnull Crofton buffers required");
    std::vector<std::uint8_t> binary(rows*cols,0);
    const auto* data=static_cast<const T*>(in.data);
    const auto team=in.height*in.width>=parallel_min?workers:1;
    const bool vectorized=has_avx2();
    partition(in.height,team,[&](auto begin,auto end,auto){
        for(auto y=begin;y<end;++y){
            if(!in.width)continue;
            const auto* source=data+y*in.width;auto* target=binary.data()+(y+1)*cols+1;
            std::size_t x=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)x=simd::crofton_threshold(source,in.type,target,in.width);
#else
            (void)vectorized;
#endif
            for(;x<in.width;++x)target[x]=positive(source[x]);
        }
    });
    using Histogram=std::array<std::ptrdiff_t,16>;
    std::vector<Histogram> local(static_cast<std::size_t>(team));
    partition(rows,team,[&](auto begin,auto end,auto worker){
        auto* histogram=local[worker].data();
        for(auto y=begin;y<end;++y){
            if(!y){histogram[0]+=std::ptrdiff_t(cols);continue;}
            const auto* current=binary.data()+y*cols;const auto* previous=current-cols;
            ++histogram[0];std::size_t x=1; // Both left padding bytes are zero.
#ifdef SKIMAGEXX_AVX2
            if(vectorized)x=simd::crofton_row(current,previous,cols,histogram);
#endif
            for(;x<cols;++x)++histogram[current[x]+4*current[x-1]+2*previous[x]+8*previous[x-1]];
        }
    });
    Histogram total{};
    for(const auto& histogram:local)for(std::size_t i=0;i<16;++i)total[i]+=histogram[i];
    std::memcpy(out,total.data(),sizeof(total));
}
}
void crofton_counts(const CroftonInput& input,std::ptrdiff_t* histogram,int workers){
    check_workers(workers);
    switch(input.type){
    case PixelType::boolean:return counts<std::uint8_t>(input,histogram,workers);
#define CROFTON(tag,T) case PixelType::tag:return counts<T>(input,histogram,workers);
    CROFTON(uint8,std::uint8_t) CROFTON(int8,std::int8_t) CROFTON(uint16,std::uint16_t) CROFTON(int16,std::int16_t)
    CROFTON(uint32,std::uint32_t) CROFTON(int32,std::int32_t) CROFTON(uint64,std::uint64_t) CROFTON(int64,std::int64_t)
    CROFTON(float32,float) CROFTON(float64,double)
#undef CROFTON
    }
    throw std::invalid_argument("unsupported Crofton pixel type");
}
double crofton_perimeter(const std::ptrdiff_t* histogram,const double* coefficients,int workers){
    check_workers(workers);
    if(!histogram||!coefficients)throw std::invalid_argument("nonnull Crofton weighted buffers required");
    struct Capture {
        std::fenv_t saved;
        Capture(){if(std::feholdexcept(&saved))throw std::runtime_error("cannot capture Crofton floating environment");}
        ~Capture(){std::fesetenv(&saved);}
    } capture;
    double result=0.;
    for(std::size_t i=0;i<16;++i)result+=coefficients[i]*double(histogram[i]);
    return result;
}
}
