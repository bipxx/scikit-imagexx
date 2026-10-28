#include "approximate_polygon_simd.hpp"
#include "execution.hpp"
#include <atomic>
#include <cfenv>
#include <cstring>
#include <utility>
#include <vector>

namespace skimagexx::detail { namespace {
constexpr int bad_flags=FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW;
bool finite(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,8);
    return (bits&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}
class Capture {
    std::fenv_t saved_;
    bool active_=false;
public:
    Capture(){if(std::fegetround()==FE_TONEAREST)active_=std::feholdexcept(&saved_)==0;}
    bool active()const{return active_;}
    ~Capture(){if(active_)std::fesetenv(&saved_);}
};
}
bool approximate_polygon_select(const ApproximatePolygonInput& input,double tolerance,
    std::uint8_t* keep,double* distances,int workers,
    ApproximatePolygonTrigProvider trig,void* context) {
    Capture capture;if(!capture.active())return false;
    if(!finite(tolerance)||!(tolerance>0))return false;
    for(std::size_t i=0;i<input.count;++i) {
        const auto* point=input.data+std::ptrdiff_t(i)*input.row_stride;
        if(!finite(point[0])||!finite(point[input.column_stride]))return false;
    }
    std::fill(keep,keep+input.count,std::uint8_t(0));
    std::fill(distances,distances+input.count,0.);
    keep[0]=keep[input.count-1]=1;
#ifdef SKIMAGEXX_AVX2
    const bool vectorized=has_avx2();
#endif
    std::vector<std::pair<std::size_t,std::size_t>> stack{{0,input.count-1}};
    while(!stack.empty()) {
        const auto interval=stack.back();stack.pop_back();
        const auto start=interval.first,end=interval.second;
        const auto* first=input.data+std::ptrdiff_t(start)*input.row_stride;
        const auto* last=input.data+std::ptrdiff_t(end)*input.row_stride;
        ApproximatePolygonSegment segment{first[0],first[input.column_stride],last[0],last[input.column_stride],0,0,0,0,0};
        segment.dr=segment.r1-segment.r0;segment.dc=segment.c1-segment.c0;
        if(std::fetestexcept(bad_flags))return false;
        if(trig) {
            const auto value=trig(context,segment.dr,segment.dc);
            segment.sine=value.sin_angle;segment.cosine=value.cos_angle;
        }else {
            const auto angle=-std::atan2(segment.dr,segment.dc);
            segment.sine=std::sin(angle);segment.cosine=std::cos(angle);
        }
        if(!finite(segment.sine)||!finite(segment.cosine)||std::fetestexcept(bad_flags))return false;
        segment.offset=segment.c0*segment.sine+segment.r0*segment.cosine;
        if(!finite(segment.offset)||std::fetestexcept(bad_flags))return false;
        if(end-start<2)continue;
        const auto begin=start+1,count=end-start-1;
        std::atomic<bool> refused{false};
        partition(count,count>=parallel_min?workers:1,[&](auto low,auto high,auto) {
            Capture local;if(!local.active()){refused.store(true,std::memory_order_relaxed);return;}
            auto index=begin+low;const auto stop=begin+high;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)index+=simd::approximate_polygon_distances(input,index,stop-index,distances,segment);
#endif
            for(;index<stop;++index)distances[index]=approximate_polygon_distance(input,index,segment);
            if(std::fetestexcept(bad_flags))refused.store(true,std::memory_order_relaxed);
        });
        if(refused.load(std::memory_order_relaxed))return false;
        auto split=begin;auto maximum=distances[begin];
        for(auto index=begin;index<end;++index) {
            const auto value=distances[index];if(!finite(value))return false;
            if(value>maximum){maximum=value;split=index;}
        }
        if(maximum>tolerance) {
            keep[split]=1;stack.emplace_back(split,end);stack.emplace_back(start,split);
        }
    }
    return true;
}
}
