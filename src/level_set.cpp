#include "level_set.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <vector>

namespace skimagexx::detail { namespace {
std::uint64_t magnitude(std::int64_t value) {
    return value < 0 ? std::uint64_t(0)-std::uint64_t(value) : std::uint64_t(value);
}
unsigned parity(std::size_t coordinate, std::uint64_t divisor, bool negative) {
    const auto q=coordinate/divisor;
    return unsigned((q+(negative && coordinate%divisor!=0))&1);
}
unsigned row_parity(LevelSetShape shape, std::size_t row,
                    std::uint64_t divisor, bool negative) {
    unsigned result=0;
    for(std::size_t d=shape.ndim-1;d-->0;) {
        const auto coordinate=row%shape.dimensions[d];row/=shape.dimensions[d];
        result^=parity(coordinate,divisor,negative);
    }
    return result;
}
class Capture {
    std::fenv_t saved_;
    bool active_;
public:
    Capture():active_(std::feholdexcept(&saved_)==0) {
        if(active_)std::fesetround(FE_TONEAREST);
    }
    bool active()const{return active_;}
    ~Capture(){if(active_)std::fesetenv(&saved_);}
};
}
void checkerboard_level_set_into(LevelSetShape shape,std::int64_t square_size,
                                 std::int8_t* output,int workers) {
    if(!shape.count)return;
    if(!shape.ndim){*output=0;return;}
    const auto columns=shape.dimensions[shape.ndim-1];
    const auto divisor=magnitude(square_size);const bool negative=square_size<0;
    const bool vectorized=has_avx2();
    std::vector<std::int8_t> pattern(columns);
    for(std::size_t start=0;start<columns;) {
        const auto quotient=start/divisor,remainder=start%divisor;
        const auto run=negative?(remainder?divisor-remainder+1:std::uint64_t(1))
                               :divisor-remainder;
        const auto n=std::min<std::uint64_t>(columns-start,run);
        const auto value=std::int8_t((quotient+(negative&&remainder!=0))&1);
        std::memset(pattern.data()+start,value,std::size_t(n));start+=std::size_t(n);
    }
    partition(shape.count,shape.count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
        while(begin<end) {
            const auto row=begin/columns,column=begin%columns;
            const auto n=std::min(end-begin,columns-column);
            const bool invert=row_parity(shape,row,divisor,negative)!=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)simd::level_set_checker_row(pattern.data()+column,output+begin,n,invert);
            else
#endif
                for(std::size_t i=0;i<n;++i)output[begin+i]=std::int8_t(pattern[column+i]^invert);
            begin+=n;
        }
    });
}
bool disk_level_set_geometry_admissible(LevelSetShape shape,const std::int64_t* center,double radius) {
    if(!std::isfinite(radius))return false;
    if(!shape.count)return true;
    constexpr std::uint64_t sum_limit=std::uint64_t(1)<<53;
    constexpr std::uint64_t distance_limit=94906265;
    std::uint64_t sum=0;
    for(std::size_t d=0;d<shape.ndim;++d) {
        const auto last=shape.dimensions[d]-1;
        const auto c=magnitude(center[d]);
        if(c>distance_limit||last>2*distance_limit)return false;
        const auto far=center[d]<0?last+c:std::max<std::uint64_t>(c,last>c?last-c:c-last);
        if(far>distance_limit)return false;
        const auto square=far*far;
        if(square>sum_limit-sum)return false;
        sum+=square;
    }
    return true;
}
bool disk_level_set_into(LevelSetShape shape,const std::int64_t* center,double radius,
                         std::int8_t* output,int workers) {
    if(!shape.count)return true;
    if(std::fegetround()!=FE_TONEAREST)return false;
    const auto columns=shape.ndim?shape.dimensions[shape.ndim-1]:std::size_t(1);
    std::vector<double> last_axis(columns);
    for(std::size_t i=0;i<columns;++i) {
        const auto delta=shape.ndim?std::int64_t(i)-center[shape.ndim-1]:std::int64_t(0);
        last_axis[i]=double(delta*delta);
    }
    const bool vectorized=has_avx2();std::atomic<bool> capture_failed{false};
    partition(shape.count,shape.count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
        Capture capture;
        if(!capture.active()){capture_failed.store(true,std::memory_order_relaxed);return;}
        while(begin<end) {
            auto row=begin/columns;const auto column=begin%columns;
            const auto n=std::min(end-begin,columns-column);
            std::uint64_t prefix=0;
            if(shape.ndim)for(std::size_t d=shape.ndim-1;d-->0;) {
                const auto coordinate=row%shape.dimensions[d];row/=shape.dimensions[d];
                const auto delta=std::int64_t(coordinate)-center[d];prefix+=std::uint64_t(delta*delta);
            }
            std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)done=simd::level_set_disk_row(last_axis.data()+column,output+begin,n,double(prefix),radius);
#endif
            for(;done<n;++done) {
                const double distance=std::sqrt(double(prefix)+last_axis[column+done]);
                const double phi=radius-distance;output[begin+done]=std::int8_t(phi>0.0);
            }
            begin+=n;
        }
    });
    return !capture_failed.load(std::memory_order_relaxed);
}
}
