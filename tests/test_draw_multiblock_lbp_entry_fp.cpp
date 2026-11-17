// Pending GNU/Linux x86-64 glibc regression; link retained full32 helper.
// No stored instruction/data addresses are replayed. Seed current empty stack.
#include "skimagexx/draw_multiblock_lbp.hpp"
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#if !defined(__linux__) || !defined(__x86_64__) || !defined(__GLIBC__)
#error "requires admitted Linux x86-64 glibc full32 helper"
#endif
extern "C" int reference_fenv_restore(const void*,std::size_t);
using Bytes=std::array<unsigned char,32>;
void require(bool x){if(!x)throw std::runtime_error("LBP entry FP regression");}
Bytes save(){static_assert(sizeof(std::fenv_t)==32);std::fenv_t f;require(std::fegetenv(&f)==0);Bytes b{};std::memcpy(b.data(),&f,32);return b;}
void restore(const Bytes& b){require(reference_fenv_restore(b.data(),b.size())==0);require(save()==b);}
template<class T>T word(const Bytes& b,std::size_t k){T x;std::memcpy(&x,b.data()+k,sizeof x);return x;}
template<class T>void set(Bytes& b,std::size_t k,T x){std::memcpy(b.data()+k,&x,sizeof x);}
int main(){
 const auto process=save();
 try{
  require(word<std::uint16_t>(process,8)==0xffff);
  for(unsigned profile=0;profile<2;++profile)for(bool active:{false,true})for(bool plus:{false,true}){
   const std::array<std::uint64_t,2> aw{{0x3ff0000000000000ULL,0x4000000000000000ULL}},bw{{0x3fe0000000000000ULL,0x4010000000000000ULL}};
   const std::array<std::uint64_t,2> expected=plus?std::array<std::uint64_t,2>{{0x3ff8000000000000ULL,0x4018000000000000ULL}}:std::array<std::uint64_t,2>{{0x3fe0000000000000ULL,0x4020000000000000ULL}};
   std::array<double,2>a{},b{},out{};std::memcpy(a.data(),aw.data(),16);std::memcpy(b.data(),bw.data(),16);
   skimagexx::MultiblockLbpFloatView av{{a.data(),{2},{8},8},false},bv{{b.data(),{2},{8},8},false},ov{{out.data(),{2},{8},8},false};
   const auto caller=save();auto seed=caller;
   set<std::uint16_t>(seed,0,895);set<std::uint16_t>(seed,4,active?(profile?63:1087):1058);set<std::uint32_t>(seed,28,active?8127:8098);
   restore(seed);const auto before=save();
   const auto status=skimagexx::multiblock_lbp_binary(av,bv,ov,plus);
   const auto after=save();restore(caller);
   require(word<std::uint16_t>(before,4)==(active?(profile?63:1087):1058));
   require(word<std::uint16_t>(after,0)==895&&word<std::uint16_t>(after,4)==(active?32:1058)&&word<std::uint32_t>(after,28)==8098);
   require(status==0&&std::memcmp(out.data(),expected.data(),16)==0);
   require(std::memcmp(a.data(),aw.data(),16)==0&&std::memcmp(b.data(),bw.data(),16)==0);
  }
  restore(process);std::cout<<"8 binary calls; exact result/input; semantic target FP; full32 caller/process\n";
 }catch(...){restore(process);throw;}
}
