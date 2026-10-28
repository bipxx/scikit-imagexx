#include "subdivide_polygon_simd.hpp"
#include "execution.hpp"
#include <atomic>
#include <cfenv>
#include <cstdint>
#include <cstring>

namespace skimagexx::detail { namespace {
constexpr int bad_flags=FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW;
bool finite(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,8);
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
class Capture {
    std::fenv_t saved_;bool active_=false;
public:
    Capture(){if(std::fegetround()==FE_TONEAREST)active_=std::feholdexcept(&saved_)==0;}
    bool active()const{return active_;}
    ~Capture(){if(active_)std::fesetenv(&saved_);}
};
}
bool subdivide_polygon_into(const SubdividePolygonInput& input,unsigned degree,
    bool circular,bool preserve_ends,double* output,int workers) {
    const auto* weights=subdivide_polygon_weights(degree);
    if(!weights||!input.count||(!circular&&input.count<weights->count))return false;
    Capture capture;if(!capture.active())return false;
    for(std::size_t i=0;i<input.count;++i) {
        const auto* point=input.data+std::ptrdiff_t(i)*input.row_stride;
        if(!finite(point[0])||!finite(point[input.column_stride]))return false;
    }
    const auto count=circular?input.count:input.count-weights->count+1;
    auto* middle=output+(!circular&&preserve_ends?2:0);
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=has_avx2();
#endif
    std::atomic<bool> refused{false};
    partition(count,count>=parallel_min?workers:1,[&](auto begin,auto end,auto) {
        Capture local;if(!local.active()){refused.store(true,std::memory_order_relaxed);return;}
        auto point=begin;
#ifdef SKIMAGEXX_AVX2
        if(vectorized)point+=simd::subdivide_polygon_points(input,begin,end-begin,*weights,circular,middle);
#endif
        for(;point<end;++point)subdivide_polygon_point(input,point,*weights,circular,middle);
        if(std::fetestexcept(bad_flags))refused.store(true,std::memory_order_relaxed);
    });
    if(refused.load(std::memory_order_relaxed))return false;
    if(circular)std::memcpy(output+count*4,output,2*sizeof(double));
    else if(preserve_ends) {
        std::memcpy(output,input.data,sizeof(double));std::memcpy(output+1,input.data+input.column_stride,sizeof(double));
        const auto* last=input.data+std::ptrdiff_t(input.count-1)*input.row_stride;
        std::memcpy(output+count*4+2,last,sizeof(double));std::memcpy(output+count*4+3,last+input.column_stride,sizeof(double));
    }
    return true;
}
}
