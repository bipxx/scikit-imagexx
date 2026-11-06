#include "cmake_consumer/finite_radon_consumer.hpp"
#ifdef FRT_WRAP_GETTER
int getter_calls=0;
extern "C" int __real__ZN9skimagexx15get_num_threadsEv();
extern "C" int __wrap__ZN9skimagexx15get_num_threadsEv(){++getter_calls;return __real__ZN9skimagexx15get_num_threadsEv();}
#endif
int main(){finite_radon_consumer();
#ifdef FRT_WRAP_GETTER
    std::uint32_t x[1]={1};getter_calls=0;auto a=skimagexx::frt2({{x,{1,1},{4,4},4},skimagexx::PixelType::uint32,false});assert(getter_calls==1);getter_calls=0;(void)skimagexx::frt2({{x,{1,1},{4,4},4},skimagexx::PixelType::uint32,false},4);assert(getter_calls==0);
    std::uint32_t y[2]={1,1};getter_calls=0;(void)skimagexx::ifrt2({{y,{2,1},{4,4},4},skimagexx::PixelType::uint32,false});assert(getter_calls==1);
#endif
}
