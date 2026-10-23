#include "../src/resize_clip.hpp"
#include "skimagexx/image.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace skimagexx;
namespace {
void require(bool value){if(!value)throw std::runtime_error("resize clip check failed");}
template<class T> T value(std::uint64_t word) {
    using U=std::conditional_t<sizeof(T)==1,std::uint8_t,std::conditional_t<sizeof(T)==2,std::uint16_t,
            std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>>>;
    const U bits=U(word);T result;std::memcpy(&result,&bits,sizeof(T));return result;
}
template<class T> void typed(PixelType type,const std::vector<std::uint64_t>& words,std::uint64_t lower,std::uint64_t upper,
                             int policy,const std::vector<std::uint64_t>& expected) {
    require(words.size()==expected.size());std::vector<T> data;for(auto word:words)data.push_back(value<T>(word));
    const auto low=value<T>(lower),high=value<T>(upper);
    detail::resize_clip_into(data.data(),data.size(),type,&low,&high,static_cast<detail::ResizeClipPolicy>(policy));
    for(std::size_t i=0;i<data.size();++i) {
        const auto wanted=value<T>(expected[i]);
        if(std::memcmp(&data[i],&wanted,sizeof(T))){std::cerr<<"type="<<int(type)<<" legacy="<<policy<<" index="<<i<<"\n";require(false);}
    }
}
void check(PixelType type,const std::vector<std::uint64_t>& words,std::uint64_t lower,std::uint64_t upper,
           int policy,const std::vector<std::uint64_t>& expected) {
    switch(type) {
        case PixelType::boolean:return typed<std::uint8_t>(type,words,lower,upper,policy,expected);
        #define CLIP(tag,T) case PixelType::tag:return typed<T>(type,words,lower,upper,policy,expected);
        CLIP(uint8,std::uint8_t) CLIP(int8,std::int8_t) CLIP(uint16,std::uint16_t) CLIP(int16,std::int16_t)
        CLIP(uint32,std::uint32_t) CLIP(int32,std::int32_t) CLIP(uint64,std::uint64_t) CLIP(int64,std::int64_t)
        CLIP(float32,float) CLIP(float64,double)
        #undef CLIP
        default:require(false);
    }
}
template<class T> void workers(PixelType type) {
    std::vector<T> input(257*1025);
    for(std::size_t i=0;i<input.size();++i)input[i]=value<T>(std::uint64_t{0x9e3779b97f4a7c15ULL}*(i+1));
    const T low=T(1),high=T(7);
    for(int policy:{0,1,2}) {
        auto one=input,four=input;set_num_threads(1);
        detail::resize_clip_into(one.data(),one.size(),type,&low,&high,static_cast<detail::ResizeClipPolicy>(policy));
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        detail::resize_clip_into(four.data(),four.size(),type,&low,&high,static_cast<detail::ResizeClipPolicy>(policy));
        require(!std::memcmp(one.data(),four.data(),one.size()*sizeof(T)));
    }
    set_num_threads(1);
}
template<class E,class F> void rejects(F action){bool caught=false;try{action();}catch(const E&){caught=true;}require(caught);}
}
int main() {
    require(get_num_threads()==1);
    detail::resize_clip_into(nullptr,0,PixelType::uint64,nullptr,nullptr,detail::ResizeClipPolicy::modern);
    rejects<std::invalid_argument>([]{detail::resize_clip_into(nullptr,1,PixelType::uint64,nullptr,nullptr,detail::ResizeClipPolicy::modern);});
    rejects<std::overflow_error>([]{detail::resize_clip_into(nullptr,std::numeric_limits<std::size_t>::max(),PixelType::uint64,nullptr,nullptr,detail::ResizeClipPolicy::modern);});
    #include "data/resize_clip.inc"
    #include "data/resize_clip_legacy.inc"
    #include "data/resize_clip_propagating.inc"
    workers<std::uint8_t>(PixelType::boolean);workers<std::uint8_t>(PixelType::uint8);workers<std::int8_t>(PixelType::int8);
    workers<std::uint16_t>(PixelType::uint16);workers<std::int16_t>(PixelType::int16);
    workers<std::uint32_t>(PixelType::uint32);workers<std::int32_t>(PixelType::int32);
    workers<std::uint64_t>(PixelType::uint64);workers<std::int64_t>(PixelType::int64);
    workers<float>(PixelType::float32);workers<double>(PixelType::float64);
    std::cout<<"249 NumPy clip fixtures, guards and exact worker checks passed\n";
}
