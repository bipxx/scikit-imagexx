#include <skimagexx/crop.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
extern "C" void* __real_memcpy(void*,const void*,std::size_t);
namespace {std::uintptr_t low=0,high=0;std::size_t blocks=0,bytes=0;}
extern "C" void* __wrap_memcpy(void* out,const void* in,std::size_t count) {
    const auto address=reinterpret_cast<std::uintptr_t>(out);
    if(address>=low && address<high){++blocks;bytes+=count;}
    return __real_memcpy(out,in,count);
}
int main(int argc,char**) {
    using namespace skimagexx;std::int16_t input[20],output[20];for(int i=0;i<20;++i)input[i]=std::int16_t(i);
    low=reinterpret_cast<std::uintptr_t>(output);high=low+sizeof(output);
    crop_copy_into({input,{4,5},{10,2},2},{{0,4},{0,5}},output,{10,2});
    assert(blocks==4 && bytes==40);for(int i=0;i<20;++i)assert(output[i]==input[i]);std::cout<<"C blocks="<<blocks<<" bytes="<<bytes<<"\n";
    blocks=bytes=0;crop_copy_into({input,{4,5},{2,8},2},{{0,4},{0,5}},output,{2,8});
    const auto expected=argc>1?20u:5u;assert(blocks==expected && bytes==40);for(int i=0;i<20;++i)assert(output[i]==input[i]);std::cout<<"F blocks="<<blocks<<" bytes="<<bytes<<"\n";
    blocks=bytes=0;crop_copy_into({input+19,{4,5},{-10,-2},2},{{0,4},{0,5}},output,{10,2});
    assert(blocks==20 && bytes==40);for(int i=0;i<20;++i)assert(output[i]==input[19-i]);std::cout<<"reverse blocks="<<blocks<<" bytes="<<bytes<<"\n";
}
