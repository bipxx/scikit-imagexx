#include "rolling_ball_kernels.hpp"
#include "rolling_ball_kernels_simd.hpp"
#include "execution.hpp"
#include <array>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
namespace skimagexx::detail { namespace {
class Capture {
    std::fenv_t saved_; bool valid_;
public:
    Capture():valid_(std::feholdexcept(&saved_)==0){}
    ~Capture(){if(valid_)std::fesetenv(&saved_);}
    bool valid()const{return valid_;}
    bool exceptional()const{return std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW)!=0;}
};
bool supported_environment(){
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)))return false;
#endif
    return true;
}
// NumPy FLOAT/DOUBLE_pairwise_sum, PW_BLOCKSIZE=128; for rank <=32 only
// the eight-lane leaf is needed. BSD notice: third_party/numpy-LICENSE.txt.
// Cache ONLY constant nodes of this exact tree, never reassociate fractions.
RollingRowPlan row_plan(const double* terms,std::size_t n,bool exact_integer){
    RollingRowPlan p;
    if(exact_integer||n<8){
        p.prefix=-0.;for(std::size_t i=0;i+1<n;++i)p.prefix=p.prefix+terms[i];
        return p;
    }
    double lane[8];
    for(std::size_t j=0;j<8;++j)lane[j]=(j+1==n)?0.:terms[j];
    const auto full=n-n%8;
    for(std::size_t i=8;i<full;i+=8)
        for(std::size_t j=0;j<8;++j)if(i+j+1<n)lane[j]=lane[j]+terms[i+j];
    if(n%8){
        p.prefix=((lane[0]+lane[1])+(lane[2]+lane[3]))+((lane[4]+lane[5])+(lane[6]+lane[7]));
        for(std::size_t i=full;i+1<n;++i)p.prefix=p.prefix+terms[i];
    }else{
        p.spine=true;p.add_lane7=n>8;p.prefix=lane[7];
        p.left=(lane[0]+lane[1])+(lane[2]+lane[3]);p.pair45=lane[4]+lane[5];p.lane6=lane[6];
    }
    return p;
}
double sum_row(const RollingRowPlan& p,double last,double zero){
    if(!p.spine)return zero+(p.prefix+last);
    const double lane7=p.add_lane7?p.prefix+last:last;
    return zero+(p.left+(p.pair45+(p.lane6+lane7)));
}
double finish(double sum,bool ball,double coefficient,double radius){
    const double scaling=(ball?coefficient:1.)-sum;
    const double height=std::sqrt(scaling<0.?0.:scaling);
    // Both source sqrt operations and the late multiply/mask are observable.
    if(ball){const double distance=std::sqrt(sum);return distance>radius?std::numeric_limits<double>::infinity():height;}
    const double value=coefficient*height;
    return scaling<0.?std::numeric_limits<double>::infinity():value;
}
bool run(const RollingGrid& grid,bool ball,double coefficient,double radius,
         double* output,int workers){
    Capture keep;if(!keep.valid()||!supported_environment())return false;
    if(!std::isfinite(coefficient)||(ball&&!std::isfinite(radius)))return false;
    if(!grid.count)return true;
    std::size_t total=0;for(std::size_t a=0;a<grid.ndim;++a)total+=grid.axes[a].count;
    std::vector<double> cache(total);std::array<const double*,32> table{};
    bool exact_integer=ball;std::uint64_t sum_bound=0;std::size_t cursor=0;
    constexpr std::uint64_t exact_limit=std::uint64_t(1)<<53;
    for(std::size_t a=0;a<grid.ndim;++a){
        const auto& axis=grid.axes[a];table[a]=cache.data()+cursor;std::uint64_t maximum=0;
        for(std::size_t i=0;i<axis.count;++i){
            const double x=axis.coordinates[i];if(!std::isfinite(x))return false;
            const double quotient=ball?x:x/axis.semi_axis;
            const double squared=quotient*quotient;cache[cursor++]=squared;
            if(!std::isfinite(squared))return false;
            if(exact_integer){
                if(std::abs(x)>94906265.||std::trunc(x)!=x)exact_integer=false;
                else{const auto magnitude=std::uint64_t(std::abs(x));const auto square=magnitude*magnitude;if(square>maximum)maximum=square;}
            }
        }
        if(exact_integer){if(maximum>exact_limit-sum_bound)exact_integer=false;else sum_bound+=maximum;}
    }
    if(keep.exceptional())return false;
    const auto columns=grid.axes[grid.ndim-1].count,rows=grid.count/columns;
    const bool vectorized=has_avx2();(void)vectorized;std::atomic<bool> failed{false};
    partition(rows,grid.count>=parallel_min?workers:1,[&](auto begin,auto end,auto){
        Capture worker;if(!worker.valid()||!supported_environment()){failed.store(true,std::memory_order_relaxed);return;}
        std::array<std::size_t,32> indices{};auto remainder=begin;
        for(std::size_t a=grid.ndim-1;a-->0;){indices[a]=remainder%grid.axes[a].count;remainder/=grid.axes[a].count;}
        const auto advance=[&](){for(std::size_t a=grid.ndim-1;a-->0;){if(++indices[a]<grid.axes[a].count)break;indices[a]=0;}};
        const auto plan=[&](){std::array<double,32> terms{};for(std::size_t a=0;a+1<grid.ndim;++a)terms[a]=table[a][indices[a]];return row_plan(terms.data(),grid.ndim,exact_integer);};
        volatile double identity=0.;const double zero=identity;
        for(auto row=begin;row<end;){
#ifdef SKIMAGEXX_AVX2
            if(vectorized&&columns<4&&end-row>=4){
                RollingRowPlan plans[4];for(auto& p:plans){p=plan();advance();}
                simd::rolling_kernel_four_rows(table[grid.ndim-1],output+row*columns,columns,plans,ball,coefficient,radius,zero);row+=4;continue;
            }
#endif
            const auto p=plan();std::size_t column=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)column=simd::rolling_kernel_row(table[grid.ndim-1],output+row*columns,columns,p,ball,coefficient,radius,zero);
#endif
            for(;column<columns;++column)output[row*columns+column]=finish(sum_row(p,table[grid.ndim-1][column],zero),ball,coefficient,radius);
            advance();++row;
        }
        if(worker.exceptional())failed.store(true,std::memory_order_relaxed);
    });
    return !failed.load(std::memory_order_relaxed)&&!keep.exceptional();
}
}
bool ball_kernel_into(const RollingGrid& grid,double radius,double radius_squared,double* output,int workers){return run(grid,true,radius_squared,radius,output,workers);}
bool ellipsoid_kernel_into(const RollingGrid& grid,double intensity,double* output,int workers){return run(grid,false,intensity,0.,output,workers);}
}
