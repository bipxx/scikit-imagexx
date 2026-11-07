// Deferred native-only whole-find controls; authentic masks from root-admitted8refs.
// No execution authority here. Reuses exact admitted stage-client full32 envelope.
#include "skimagexx/find_boundaries.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using State=std::array<unsigned char,32>;
using namespace skimagexx;
using namespace skimagexx::detail;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
State state(){static_assert(sizeof(std::fenv_t)==32);std::fenv_t actual;require(!std::fegetenv(&actual),"fegetenv");State s{};std::memcpy(s.data(),&actual,32);return s;}
std::uint16_t word16(const State& s,std::size_t at){std::uint16_t v;std::memcpy(&v,s.data()+at,2);return v;}
std::uint32_t word32(const State& s,std::size_t at){std::uint32_t v;std::memcpy(&v,s.data()+at,4);return v;}
bool semantic(const State& a,const State& b){return word16(a,0)==word16(b,0)&&word16(a,4)==word16(b,4)&&word32(a,28)==word32(b,28);}
void raw32(const State& s){auto flags=std::cout.flags();auto fill=std::cout.fill();std::cout<<'"'<<std::hex<<std::setfill('0');for(auto b:s)std::cout<<std::setw(2)<<unsigned(b);std::cout<<'"';std::cout.flags(flags);std::cout.fill(fill);}
struct Envelope {
    State process;void* library=nullptr;using Restore=int(*)(const void*,std::size_t);Restore reset=nullptr;
    explicit Envelope(const char* path):process(state()){
        require(word16(process,8)==0xffff,"nonempty x87 stack");
        require((word16(process,0)&0x3f)==0x3f&&!(word16(process,0)&0xc00),"x87 control");
        require((word32(process,28)&0x1f80)==0x1f80&&!(word32(process,28)&0x6000),"SSE control");
        library=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);require(library,"admitted helper dlopen");
        auto layout=reinterpret_cast<int(*)(std::uint32_t*,std::size_t)>(dlsym(library,"reference_fenv_layout"));
        reset=reinterpret_cast<Restore>(dlsym(library,"reference_fenv_restore"));require(layout&&reset,"helper exports");
        std::uint32_t actual[8],expected[]={32,0,4,8,12,20,28,8};require(!layout(actual,8)&&!std::memcmp(actual,expected,sizeof(actual)),"helper ABI layout");restore(process);
    }
    void restore(const State& s){require(!reset(s.data(),32),"full32 restore refused");require(state()==s,"full32 caller mismatch");}
    void finish(){restore(process);}
    ~Envelope(){if(library)dlclose(library);if(reset)reset(process.data(),32);}
};

void set16(State& s,std::size_t at,std::uint16_t value){std::memcpy(s.data()+at,&value,2);}
void set32(State& s,std::size_t at,std::uint32_t value){std::memcpy(s.data()+at,&value,4);}
template<class T>void check(Envelope& env,const T* data,PixelType type,const std::uint8_t* expected,
    std::uint16_t sw,std::uint32_t before_mxcsr,std::uint32_t after_mxcsr,int budget){
    const auto caller=state();auto incoming=caller;
    set16(incoming,0,895);set16(incoming,4,sw);set32(incoming,28,before_mxcsr);
    auto expected_after=incoming;set32(expected_after,28,after_mxcsr);
    ComponentsView view;view.data=data;view.type=type;view.shape={3,4};view.strides={32,8};
    std::array<std::uint8_t,14> output{};output.fill(199);std::array<T,12> saved_words{};std::memcpy(saved_words.data(),data,96);
    FindBoundaryParameters p;p.mode=FindBoundaryMode::outer;p.connectivity=1;p.background_bits=0;p.background_representable=true;
    env.restore(incoming);bool accepted=false;State actual_after;
    try{accepted=find_boundaries_into(view,p,output.data()+1,{4,1},budget);actual_after=state();}
    catch(...){env.restore(caller);throw;}
    env.restore(caller);
    require(accepted,"whole find refusal");require(!std::memcmp(output.data()+1,expected,12),"authentic mask mismatch");
    require(output.front()==199&&output.back()==199,"output canary");require(!std::memcmp(data,saved_words.data(),96),"input mutation");
    require(semantic(actual_after,expected_after),"captured semantic FP mismatch");
    std::cout<<"{\"budget\":"<<budget<<",\"after32\":";raw32(actual_after);std::cout<<",\"caller32_exact\":true}\n";
    env.restore(caller);
}
int main(int argc,char** argv){
    require(argc==2,"admitted helper path required");Envelope env(argv[1]);
    const std::int64_t input0[]={INT64_C(0),INT64_C(0),INT64_C(1),INT64_C(1),INT64_C(2),INT64_C(2),INT64_C(3),INT64_C(3),INT64_C(0),INT64_C(0),INT64_C(1),INT64_C(1)};
    const std::uint8_t mask0[]={1,1,1,1,0,1,1,1,1,1,1,1};
    const std::uint64_t input1[]={UINT64_C(0),UINT64_C(18446744073709551615),UINT64_C(18446744073709551614),UINT64_C(1),UINT64_C(9223372036854775807),UINT64_C(9223372036854775808),UINT64_C(9223372036854775809),UINT64_C(0),UINT64_C(9007199254740992),UINT64_C(9007199254740993),UINT64_C(9007199254740994),UINT64_C(2)};
    const std::uint8_t mask1[]={0,0,0,0,1,1,0,1,1,1,1,1};
    for(int budget:{1,4}){
        check(env,input0,PixelType::int64,mask0,32,8099,8098,budget); // minimum
        check(env,input0,PixelType::int64,mask0,0,8097,8096,budget); // modern
        check(env,input1,PixelType::uint64,mask1,32,8099,8098,budget); // minimum
        check(env,input1,PixelType::uint64,mask1,0,8097,8096,budget); // modern
    }
    env.finish();return 0;
}
