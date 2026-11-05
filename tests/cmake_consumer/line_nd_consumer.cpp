// Installed SDK89 assembled component client; no private/source-tree includes.
#include <skimagexx/line_nd.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
using Int = std::intptr_t;
std::size_t checks = 0;
void check(bool ok) { ++checks; if (!ok) throw std::runtime_error("installed line_nd check="+std::to_string(checks)); }
template<class T> LineNDVector view(const std::vector<T>& values, LineNDType type) {
    return {values.data(), values.size(), std::ptrdiff_t(sizeof(T)), type};
}
template<class T> void words(const LineNDPlan& plan, const std::vector<T>& expected, int workers) {
    check(expected.size() == plan.points*plan.dimensions);
    // Negative point stride, positive gapped dimension stride, byte-unaligned.
    const auto point_step = std::ptrdiff_t(3*plan.dimensions*sizeof(T));
    const auto offset = std::size_t(plan.points ? 1+(plan.points-1)*point_step : 1);
    std::vector<unsigned char> actual(3*expected.size()*sizeof(T)+2, 0xa7), wanted(actual);
    const auto flags = std::fetestexcept(FE_ALL_EXCEPT), rounding = std::fegetround();
    line_nd_into(plan, plan.points ? actual.data()+offset : nullptr, -point_step, 3*sizeof(T), workers);
    for (std::size_t i=0;i<plan.points;++i) for (std::size_t d=0;d<plan.dimensions;++d) {
        const auto at=std::ptrdiff_t(offset)-std::ptrdiff_t(i)*point_step+std::ptrdiff_t(d*3*sizeof(T));
        std::memcpy(wanted.data()+at, &expected[i*plan.dimensions+d], sizeof(T));
        check(std::memcmp(actual.data()+at,&expected[i*plan.dimensions+d],sizeof(T))==0);
    }
    check(actual == wanted);  // Every gap and leading/trailing byte remains a sentinel.
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags && std::fegetround()==rounding);
}
void controls(int workers) {
    const auto global=get_num_threads(), flags=std::fetestexcept(FE_ALL_EXCEPT), rounding=std::fegetround();
    check(global==workers);
    LineNDPlan plan;
    // Exact frozen original half-positive/descending and broadcast-start rows.
    std::vector<double> start{.5}, stop{4.5};
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),true,true,plan));
    check(plan.points==5 && plan.dimensions==1); words<Int>(plan,{0,1,2,3,4},workers);
    std::swap(start,stop);
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),true,true,plan));
    words<Int>(plan,{4,4,2,2,0},workers);
    start={-.5};stop={2.5,1.5};
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),false,true,plan));
    check(plan.rounded_dimensions==1);words<Int>(plan,{-1,0,0,0,1,0},workers);
    // Saved profile-specific modulo holdout: standalone default is NumPy1.
    const std::uint32_t near_half_bits=0xbeffffff;
    float near_half;std::memcpy(&near_half,&near_half_bits,sizeof(near_half));
    std::vector<float> promotion_start{near_half},promotion_stop{4.5f};
    check(line_nd_prepare(view(promotion_start,LineNDType::float32),view(promotion_stop,LineNDType::float32),true,true,plan));
    check(plan.scalar_promotion==LineNDScalarPromotion::legacy_float64);
    words<Int>(plan,{0,0,2,2,4,4},workers);
    check(line_nd_prepare(view(promotion_start,LineNDType::float32),view(promotion_stop,LineNDType::float32),true,true,plan,LineNDScalarPromotion::legacy_float64));
    words<Int>(plan,{0,0,2,2,4,4},workers);
    check(line_nd_prepare(view(promotion_start,LineNDType::float32),view(promotion_stop,LineNDType::float32),true,true,plan,LineNDScalarPromotion::weak_input));
    check(plan.scalar_promotion==LineNDScalarPromotion::weak_input);
    words<Int>(plan,{-1,0,1,2,3,4},workers);
    // Frozen source float16 row words, interleaved by point for the SDK buffer.
    std::vector<std::uint16_t> hs{0x3e00,0xc100}, ht{0x4480,0x4300};
    check(line_nd_prepare(view(hs,LineNDType::float16),view(ht,LineNDType::float16),false,false,plan));
    words<std::uint16_t>(plan,{15872,49408,16384,48640,16640,47104,16896,14336,17152,15872,17408,16640},workers);
    std::vector<float> fs{1.5f,-2.5f},ft{4.5f,3.5f};
    check(line_nd_prepare(view(fs,LineNDType::float32),view(ft,LineNDType::float32),false,false,plan));
    words<float>(plan,{1.5f,-2.5f,2.f,-1.5f,2.5f,-.5f,3.f,.5f,3.5f,1.5f,4.f,2.5f},workers);
    start={1.5,-2.5};stop={4.5,3.5};
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),false,false,plan));
    words<double>(plan,{1.5,-2.5,2,-1.5,2.5,-.5,3,.5,3.5,1.5,4,2.5},workers);
    std::vector<std::int8_t> is{120},it{-120};
    check(line_nd_prepare(view(is,LineNDType::int8),view(it,LineNDType::int8),false,true,plan));
    check(plan.points==16);
    std::vector<Int> wrapped{120,105,90,75,60,45,30,15,0,-15,-30,-45,-60,-75,-90,-105};
    words<Int>(plan,wrapped,workers);
    std::vector<std::uint8_t> us{1,2},ut{5,6};
    check(line_nd_prepare(view(us,LineNDType::uint8),view(ut,LineNDType::uint8),false,true,plan));
    words<Int>(plan,{1,2,2,3,3,4,4,5},workers);
    std::vector<std::int64_t> ls{1,2},lt{5,6};
    check(line_nd_prepare(view(ls,LineNDType::int64),view(lt,LineNDType::int64),false,true,plan));
    words<Int>(plan,{1,2,2,3,3,4,4,5},workers);
    std::vector<std::uint64_t> qs{1,2},qt{5,6};
    check(line_nd_prepare(view(qs,LineNDType::uint64),view(qt,LineNDType::uint64),false,true,plan));
    words<Int>(plan,{1,2,2,3,3,4,4,5},workers);
    // Signed, gapped, byte-unaligned input spans; prepare snapshots operands.
    std::vector<unsigned char> a(80,0xcd),b(80,0xcd);
    for (std::size_t d=0;d<3;++d) {
        const double av=double(d),bv=double(d+4);
        std::memcpy(a.data()+49-d*24,&av,8);std::memcpy(b.data()+1+d*24,&bv,8);
    }
    check(line_nd_prepare({a.data()+49,3,-24,LineNDType::float64},{b.data()+1,3,24,LineNDType::float64},false,false,plan));
    std::fill(a.begin(),a.end(),0);std::fill(b.begin(),b.end(),0);
    words<double>(plan,{0,1,2,1,2,3,2,3,4,3,4,5},workers);
    // Coincident endpoint controls, empty nullptr span and eight dimensions.
    start={1.5,-2.5};stop=start;
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),false,true,plan));
    check(plan.points==0);words<Int>(plan,{},workers);
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),true,true,plan));
    words<Int>(plan,{2,-2},workers);
    start.assign(8,0);stop.assign(8,2);
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),true,true,plan));
    words<Int>(plan,{0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2},workers);
    // Source underflow stage facts with caller sticky flags preserved exactly.
    start={0,0};stop={4.2,std::numeric_limits<double>::denorm_min()};
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),false,false,plan));
    check(plan.zero_step && plan.events.divide_underflow);
    std::vector<double> tiny(2*plan.points);
    const auto events=line_nd_into(plan,tiny.data(),16,8,workers);
    check(events.divide_underflow && events.multiply_underflow);
    // A genuine threshold-crossing independent holdout, exact +/-unit steps.
    start={0,0};stop={131073,-131073};
    check(line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),true,false,plan));
    check(plan.points==131074 && plan.points*plan.dimensions>262144);
    std::vector<double> large(2*plan.points+2,777777.0);
    line_nd_into(plan,large.data()+1,16,8,workers);
    for(std::size_t i=0;i<plan.points;++i) {
        const double positive=double(i),negative=i ? -positive : 0.0;
        check(std::memcmp(&large[1+2*i],&positive,8)==0);
        check(std::memcmp(&large[2+2*i],&negative,8)==0);
    }
    check(large.front()==777777.0 && large.back()==777777.0);
    // Ordinary refusal leaves plan byte-for-byte unchanged.
    std::vector<unsigned char> unchanged(sizeof(plan));
    std::memcpy(unchanged.data(),&plan,sizeof(plan));
    check(!line_nd_prepare({nullptr,0,8,LineNDType::float64},view(stop,LineNDType::float64),false,false,plan));
    check(std::memcmp(&plan,unchanged.data(),sizeof(plan))==0);
    check(!line_nd_prepare(view(start,LineNDType::float64),view(fs,LineNDType::float32),false,false,plan));
    check(std::memcmp(&plan,unchanged.data(),sizeof(plan))==0);
    start={std::numeric_limits<double>::infinity()};stop={1};
    check(!line_nd_prepare(view(start,LineNDType::float64),view(stop,LineNDType::float64),false,true,plan));
    check(std::memcmp(&plan,unchanged.data(),sizeof(plan))==0);
    check(get_num_threads()==global && std::fetestexcept(FE_ALL_EXCEPT)==flags && std::fegetround()==rounding);
}
}
int main() {
    check(get_num_threads()==1);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
    check(rounding==FE_TONEAREST);
    std::feraiseexcept(FE_DIVBYZERO);
    for(int workers:{1,4}) {set_num_threads(workers);controls(get_num_threads());}
    set_num_threads(1);check(get_num_threads()==1);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(flags);
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags && std::fegetround()==rounding);
    std::cout<<"installed line_nd SDK component controls="<<checks<<"; workers=1,4; default=1; avx2="<<has_avx2()<<"; large_elements=262148\n";
}
