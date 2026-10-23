#include "../src/resize_range.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
using namespace skimagexx;
namespace {
void require(bool ok){if(!ok)throw std::runtime_error("resize range check failed");}
template<class T> T value(std::uint64_t word) {
    using U=std::conditional_t<sizeof(T)==1,std::uint8_t,std::conditional_t<sizeof(T)==2,std::uint16_t,std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>>>;
    const U bits=U(word);T result;std::memcpy(&result,&bits,sizeof(T));return result;
}
template<class T> void typed(PixelType type,const std::vector<std::uint64_t>& words,std::uint64_t low,std::uint64_t high,
                             bool nan,bool valid,bool positive,bool negative) {
    std::vector<T> input;for(auto word:words)input.push_back(value<T>(word));const auto snapshot=input;
    T lower{},upper{};const auto info=detail::resize_range_into(input.data(),input.size(),type,&lower,&upper);
    require(info.has_nan==nan && info.has_value==valid && info.positive_zero==positive && info.negative_zero==negative);
    for(auto pair:{std::pair<T,T>{lower,value<T>(low)},std::pair<T,T>{upper,value<T>(high)}}) {
        if constexpr(std::is_floating_point_v<T>) {
            if(!valid){require(std::isnan(pair.first));continue;}
            if(positive && negative && pair.second==0){require(pair.first==0);continue;}
        }
        if(std::memcmp(&pair.first,&pair.second,sizeof(T))) {std::cerr<<"type="<<int(type)<<" size="<<input.size()<<"\n";require(false);}
    }
    require(!std::memcmp(input.data(),snapshot.data(),input.size()*sizeof(T)));
}
void check(PixelType type,const std::vector<std::uint64_t>& words,std::uint64_t low,std::uint64_t high,
           bool nan,bool valid,bool positive,bool negative) {
    switch(type) {
        case PixelType::boolean:return typed<std::uint8_t>(type,words,low,high,nan,valid,positive,negative);
        #define RANGE(tag,T) case PixelType::tag:return typed<T>(type,words,low,high,nan,valid,positive,negative);
        RANGE(uint8,std::uint8_t) RANGE(int8,std::int8_t) RANGE(uint16,std::uint16_t) RANGE(int16,std::int16_t)
        RANGE(uint32,std::uint32_t) RANGE(int32,std::int32_t) RANGE(uint64,std::uint64_t) RANGE(int64,std::int64_t)
        RANGE(float32,float) RANGE(float64,double)
        #undef RANGE
        default:require(false);
    }
}
template<class T> void workers(PixelType type) {
    std::vector<T> input(257*1025);
    for(std::size_t i=0;i<input.size();++i)input[i]=T(i%107);
    T low1{},high1{},low4{},high4{};
    set_num_threads(1);const auto one=detail::resize_range_into(input.data(),input.size(),type,&low1,&high1);
    #ifdef _OPENMP
    set_num_threads(4);
    #endif
    const auto four=detail::resize_range_into(input.data(),input.size(),type,&low4,&high4);
    require(low1==T(0) && high1==(type==PixelType::boolean ? T(1) : T(106)));
    require(!std::memcmp(&low1,&low4,sizeof(T)) && !std::memcmp(&high1,&high4,sizeof(T)));
    require(one.has_value==four.has_value && one.has_nan==four.has_nan && one.positive_zero==four.positive_zero && one.negative_zero==four.negative_zero);
    set_num_threads(1);
}
}
int main() {
    require(get_num_threads()==1);
    require(!detail::resize_range_into(nullptr,0,PixelType::uint64,nullptr,nullptr).has_value);
    #include "data/resize_range.inc"
    workers<std::uint8_t>(PixelType::boolean);workers<std::uint8_t>(PixelType::uint8);workers<std::int8_t>(PixelType::int8);
    workers<std::uint16_t>(PixelType::uint16);workers<std::int16_t>(PixelType::int16);
    workers<std::uint32_t>(PixelType::uint32);workers<std::int32_t>(PixelType::int32);
    workers<std::uint64_t>(PixelType::uint64);workers<std::int64_t>(PixelType::int64);
    workers<float>(PixelType::float32);workers<double>(PixelType::float64);
    std::cout<<"133 NumPy range fixtures and exact worker checks passed\n";
}
