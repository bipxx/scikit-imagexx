#include <skimagexx/rolling_ball.hpp>
#include <skimagexx/image.hpp>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "fixtures/rolling_ball_sdk_dtype_fp_v1.inc"
extern "C" int reference_fenv_restore(const void*,std::size_t);
namespace {
void check(bool v){if(!v)throw std::runtime_error("SDK FP control failed");}
template<class T>T read(std::ifstream& f){T v;f.read(reinterpret_cast<char*>(&v),sizeof(v));check(bool(f));return v;}
FPState state(const std::fenv_t& f){FPState s;const auto* p=reinterpret_cast<const unsigned char*>(&f);std::memcpy(&s.control,p,2);std::memcpy(&s.status,p+4,2);std::memcpy(&s.mxcsr,p+28,4);return s;}
bool equal(FPState a,FPState b){return a.control==b.control&&a.status==b.status&&a.mxcsr==b.mxcsr;}
void set(std::fenv_t& f,FPState s){auto* p=reinterpret_cast<unsigned char*>(&f);std::memcpy(p,&s.control,2);std::memcpy(p+4,&s.status,2);std::memcpy(p+28,&s.mxcsr,4);}
}
int main(int argc,char** argv){
 static_assert(sizeof(std::fenv_t)==32,"Retained Linux glibc32 ABI required");
 try{
 check(argc==2&&skimagexx::get_num_threads()==1);std::ifstream f(argv[1],std::ios::binary);char magic[8];f.read(magic,8);check(std::string(magic,8)==std::string("ROLLDT1\0",8));check(read<std::uint64_t>(f)==11);
 unsigned checked=0;
 for(std::size_t c=0;c<11;++c){
  const auto length=read<std::uint64_t>(f),code=read<std::uint64_t>(f),half=read<std::uint64_t>(f),swap=read<std::uint64_t>(f),rank=read<std::uint64_t>(f),backing=read<std::uint64_t>(f),origin=read<std::uint64_t>(f);const double radius=read<double>(f);const bool safe=read<std::uint64_t>(f)!=0;
  std::string name(length,' ');f.read(name.data(),std::streamsize(length));skimagexx::RollingBallGeometry g;g.shape.resize(rank);g.strides.resize(rank);std::size_t n=1;
  for(auto& x:g.shape){x=std::ptrdiff_t(read<std::int64_t>(f));n*=std::size_t(x);}for(auto& x:g.strides)x=std::ptrdiff_t(read<std::int64_t>(f));const std::size_t widths[]={1,1,1,2,2,4,4,8,8,4,8};
  std::vector<unsigned char> bytes(backing),expected(n*widths[code]);f.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(backing));f.read(reinterpret_cast<char*>(expected.data()),std::streamsize(expected.size()));check(bool(f));
  for(unsigned profile=0;profile<2;++profile)for(int workers:{1,4}){
   std::fenv_t caller,incoming,entry,after,restored;check(std::fegetenv(&caller)==0);incoming=caller;set(incoming,original_fp[c][profile].before);check(reference_fenv_restore(&incoming,32)==0);check(std::fegetenv(&entry)==0);
   skimagexx::RollingBallResult actual;
   try{actual=skimagexx::rolling_ball({bytes.data()+origin,skimagexx::PixelType(code),bool(half),bool(swap)},g,{},radius,safe,workers);check(std::fegetenv(&after)==0);}
   catch(...){reference_fenv_restore(&caller,32);throw;}
   check(reference_fenv_restore(&caller,32)==0);check(std::fegetenv(&restored)==0);check(std::memcmp(&caller,&restored,32)==0);
   if(actual.bytes!=expected||!equal(state(entry),original_fp[c][profile].before)||!equal(state(after),original_fp[c][profile].after)){
    const auto a=state(after),e=original_fp[c][profile].after;std::cerr<<"FIRST_STRICT_FAILURE "<<name<<" profile="<<profile<<" workers="<<workers<<" actual="<<a.control<<','<<a.status<<','<<a.mxcsr<<" expected="<<e.control<<','<<e.status<<','<<e.mxcsr<<" words_equal="<<(actual.bytes==expected)<<'\n';return 1;
   }
   ++checked;
  }
 }
 std::cout<<"strict_target_FP_words="<<checked<<" profile_templates=2 workers1/4 exactcaller32 PASS originals=0 timing=0\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
