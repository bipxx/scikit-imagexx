#include "skimagexx/h_extrema.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error H-extrema real native guards require enabled assertions
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

using namespace skimagexx;
namespace {
std::size_t assertions=0;
void check(bool value){++assertions;assert(value);}
template<class F> void rejected(F&& function) {
    bool caught=false;try{function();}catch(const std::exception&){caught=true;}check(caught);
}
void put(unsigned char* pointer,std::uint64_t bits,std::size_t bytes,bool swap) {
    if(bytes==1)*pointer=static_cast<unsigned char>(bits);
    else if(bytes==2){const auto value=std::uint16_t(bits);std::memcpy(pointer,&value,2);}
    else if(bytes==4){const auto value=std::uint32_t(bits);std::memcpy(pointer,&value,4);}
    else std::memcpy(pointer,&bits,8);
    if(swap)std::reverse(pointer,pointer+bytes);
}
template<class T> std::uint64_t bits(T value) {
    if constexpr(sizeof(T)==4){std::uint32_t raw;std::memcpy(&raw,&value,4);return raw;}
    else {std::uint64_t raw;std::memcpy(&raw,&value,8);return raw;}
}
void admitted(const HExtremaRealInfo& info) {
    check(info.admitted && info.execution.workers==1 && !info.execution.simd);
}

// Fixed exact pairs 6-5 and 0-1 (bool uses 1-0). All lengths include every
// remainder mod8, with independent endian, negative-stride and unaligned roots.
void layouts(ReconstructionType type,std::size_t bytes,std::uint64_t six,bool single,
             bool swapped,bool reverse,std::size_t n) {
    const auto rec_bytes=single ? 4u : 8u;
    const bool boolean=type==ReconstructionType::boolean;
    std::vector<unsigned char> image((2*n-1)*bytes+3,0xa7),rec((2*n-1)*rec_bytes+3,0xb7),out(n+2,0xcc);
    for(std::size_t i=0;i<n;++i) {
        put(image.data()+1+2*i*bytes,i%2 ? 0 : six,bytes,swapped);
        const double value=i%2 ? 1. : boolean ? 0. : 5.;
        put(rec.data()+1+2*i*rec_bytes,single ? bits(float(value)) : bits(value),rec_bytes,!swapped);
    }
    const std::size_t shape[]={n};
    const std::ptrdiff_t image_stride[]={std::ptrdiff_t(2*bytes)*(reverse ? -1 : 1)};
    const std::ptrdiff_t rec_stride[]={std::ptrdiff_t(2*rec_bytes)*(reverse ? -1 : 1)};
    const ReconstructionView a{image.data()+1+(reverse ? 2*(n-1)*bytes : 0),image.data(),image.size(),
        {shape,1},{image_stride,1},type,swapped};
    const ReconstructionView b{rec.data()+1+(reverse ? 2*(n-1)*rec_bytes : 0),rec.data(),rec.size(),
        {shape,1},{rec_stride,1},single ? ReconstructionType::float32 : ReconstructionType::float64,!swapped};
    check(h_extrema_validate_real_view(a)==n && h_extrema_validate_real_view(b)==n);
    for(auto method:{ReconstructionMethod::dilation,ReconstructionMethod::erosion}) {
        admitted(h_extrema_real_mask_into(a,b,1.,method,{out.data()+1,n}));
        for(std::size_t i=0;i<n;++i) {
            const auto physical=reverse ? n-1-i : i;
            check(out[i+1]==static_cast<unsigned char>(method==ReconstructionMethod::dilation ? physical%2==0 : physical%2!=0));
        }
        check(out.front()==0xcc && out.back()==0xcc);
    }
}
template<class T> void rounding(ReconstructionType image_type) {
    // At 4, the lower adjacent value is 2^-22 away for F32, 2^-51 for F64.
    // Subtract quarter/half/three-quarter of that gap: ties-to-even gives
    // [4,4,previous(4)]. A widened subtraction would fail the first two >=4.
    const T four=4;
    const int power=sizeof(T)==4 ? -24 : -53;
    const T q=std::ldexp(T(1),power),values[]={q,T(2)*q,T(3)*q};
    std::uint16_t half=0x4400;
    const void* pointer=image_type==ReconstructionType::float16 ? static_cast<const void*>(&half) : static_cast<const void*>(&four);
    const std::size_t bytes=image_type==ReconstructionType::float16 ? 2 : sizeof(T),shape[]={3};
    const std::ptrdiff_t zero[]={0},stride[]={sizeof(T)};
    const ReconstructionView a{pointer,pointer,bytes,{shape,1},{zero,1},image_type,false};
    const ReconstructionView b{values,values,sizeof(values),{shape,1},{stride,1},sizeof(T)==4 ? ReconstructionType::float32 : ReconstructionType::float64,false};
    unsigned char out[]={0xcc,0xcc,0xcc};
    admitted(h_extrema_real_mask_into(a,b,4.,ReconstructionMethod::dilation,{out,3}));
    check(out[0]==1 && out[1]==1 && out[2]==0);
    if constexpr(sizeof(T)==4) {
        std::fill(std::begin(out),std::end(out),0xcc);
        for(double height:{4.+std::ldexp(1.,-23),4.-std::ldexp(1.,-23)}) {
            check(!h_extrema_real_mask_into(a,b,height,ReconstructionMethod::dilation,{out,3}).admitted);
            check(out[0]==0xcc && out[1]==0xcc && out[2]==0xcc);
        }
    }
}
template<class T> void subnormal_and_state() {
    const T smallest=std::numeric_limits<T>::denorm_min();
    const T values[]={smallest,T(2)*smallest,0.,-0.},zero=0.;
    const std::size_t shape[]={4};const std::ptrdiff_t stride[]={sizeof(T)},broadcast[]={0};
    const auto type=sizeof(T)==4 ? ReconstructionType::float32 : ReconstructionType::float64;
    const ReconstructionView a{values,values,sizeof(values),{shape,1},{stride,1},type,false};
    const ReconstructionView b{&zero,&zero,sizeof(zero),{shape,1},{broadcast,1},type,false};
    unsigned char out[4]={};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INVALID);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT),round=std::fegetround();
    admitted(h_extrema_real_mask_into(a,b,double(smallest),ReconstructionMethod::dilation,{out,4}));
    check(out[0]==1 && out[1]==1 && out[2]==0 && out[3]==0);
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags && std::fegetround()==round);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::fill(std::begin(out),std::end(out),0xcc);
    for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        check(std::fesetround(mode)==0);
        check(!h_extrema_real_mask_into(a,b,0.,ReconstructionMethod::dilation,{out,4}).admitted);
        check(std::fegetround()==mode);
        check(std::all_of(std::begin(out),std::end(out),[](auto x){return x==0xcc;}));
    }
    check(std::fesetround(round)==0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const auto csr=_mm_getcsr();
    for(unsigned mode:{1u<<6,1u<<15,(1u<<6)|(1u<<15)}) {
        _mm_setcsr(csr|mode);
        check(!h_extrema_real_mask_into(a,b,0.,ReconstructionMethod::dilation,{out,4}).admitted);
        check(_mm_getcsr()==(csr|mode));
        check(std::all_of(std::begin(out),std::end(out),[](auto x){return x==0xcc;}));
    }
    _mm_setcsr(csr);
#endif
}
void wide_integer_rounding() {
    const std::uint64_t image[]={9007199254740993ULL,9007199254740995ULL};
    const double rec[]={9007199254740992.,9007199254740994.};
    const std::size_t shape[]={2};const std::ptrdiff_t stride[]={8};
    const ReconstructionView a{image,image,sizeof(image),{shape,1},{stride,1},ReconstructionType::uint64,false};
    const ReconstructionView b{rec,rec,sizeof(rec),{shape,1},{stride,1},ReconstructionType::float64,false};
    unsigned char out[2]={};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    admitted(h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{out,2}));
    check(out[0]==0 && out[1]==1);check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    std::feclearexcept(FE_ALL_EXCEPT);
}
void refusal_and_metadata() {
    double value=6.,rec=5.;const std::size_t shape[]={1};const std::ptrdiff_t stride[]={8};
    ReconstructionView a{&value,&value,sizeof(value),{shape,1},{stride,1},ReconstructionType::float64,false};
    ReconstructionView b{&rec,&rec,sizeof(rec),{shape,1},{stride,1},ReconstructionType::float64,false};
    unsigned char out=0xcc;
    for(double height:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        check(!h_extrema_real_mask_into(a,b,height,ReconstructionMethod::dilation,{&out,1}).admitted);check(out==0xcc);
    }
    for(double bad:{std::numeric_limits<double>::max(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        value=bad;
        check(!h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{&out,1}).admitted);check(out==0xcc);
        value=6.;rec=bad;
        check(!h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{&out,1}).admitted);check(out==0xcc);rec=5.;
    }
    value=std::numeric_limits<double>::max()/2.;rec=-value;
    admitted(h_extrema_real_mask_into(a,b,std::numeric_limits<double>::max(),ReconstructionMethod::dilation,{&out,1}));
    check(out==1);value=6.;rec=5.;out=0xcc;
    auto bad=a;bad.allocation_bytes=1;
    rejected([&]{h_extrema_validate_real_view(bad);}); // Metadata-only, no pixel read.
    const std::size_t empty[]={0};bad=a;bad.shape={empty,1};
    rejected([&]{h_extrema_validate_real_view(bad);});
    bad=a;bad.type=ReconstructionType::float32;
    rejected([&]{h_extrema_validate_view(bad);}); // Integer-v1 admission unchanged.
    bad=b;bad.type=ReconstructionType::float32;
    rejected([&]{h_extrema_real_mask_into(a,bad,1.,ReconstructionMethod::dilation,{&out,1});});
    rejected([&]{h_extrema_real_mask_into(a,b,1.,static_cast<ReconstructionMethod>(99),{&out,1});});
    rejected([&]{h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{&out,0});});
    check(out==0xcc);
    // Live one-sample descriptors only. Reject before a write can alter them;
    // never construct an unsafe multi-read or freed-memory test fixture.
    std::array<unsigned char,sizeof(a)> before_a{},before_b{};
    std::memcpy(before_a.data(),&a,sizeof(a));std::memcpy(before_b.data(),&b,sizeof(b));
    rejected([&]{h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{reinterpret_cast<unsigned char*>(&a),1});});
    rejected([&]{h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{reinterpret_cast<unsigned char*>(&b),1});});
    check(std::memcmp(before_a.data(),&a,sizeof(a))==0 && std::memcmp(before_b.data(),&b,sizeof(b))==0);
    const auto saved_value=value,saved_rec=rec;
    rejected([&]{h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{reinterpret_cast<unsigned char*>(&value),1});});
    rejected([&]{h_extrema_real_mask_into(a,b,1.,ReconstructionMethod::dilation,{reinterpret_cast<unsigned char*>(&rec),1});});
    check(value==saved_value && rec==saved_rec);
}
}
int main() {
    check(get_num_threads()==1);
    const bool portable=std::getenv("SKIMAGEXX_EXPECT_OPENMP_OFF")!=nullptr;
    struct Item {ReconstructionType type;std::size_t bytes;std::uint64_t six;bool single;};
    const Item items[]={
        {ReconstructionType::boolean,1,1,false},
        {ReconstructionType::uint8,1,6,false},{ReconstructionType::int8,1,6,false},
        {ReconstructionType::uint16,2,6,false},{ReconstructionType::int16,2,6,false},
        {ReconstructionType::uint32,4,6,false},{ReconstructionType::int32,4,6,false},
        {ReconstructionType::uint64,8,6,false},{ReconstructionType::int64,8,6,false},
        {ReconstructionType::float16,2,0x4600,true},{ReconstructionType::float32,4,0x40c00000,true},
        {ReconstructionType::float64,8,0x4018000000000000ULL,false}};
    for(int workers:{1,4}) {
        if(portable && workers==4){rejected([]{set_num_threads(4);});check(get_num_threads()==1);continue;}
        set_num_threads(workers);check(get_num_threads()==workers);
        for(const auto& item:items)for(bool swap:{false,true})for(bool reverse:{false,true})
            for(std::size_t n=1;n<=33;++n)layouts(item.type,item.bytes,item.six,item.single,swap,reverse,n);
        rounding<float>(ReconstructionType::float16);rounding<float>(ReconstructionType::float32);rounding<double>(ReconstructionType::float64);
        subnormal_and_state<float>();subnormal_and_state<double>();wide_integer_rounding();refusal_and_metadata();
    }
    set_num_threads(1);
    std::cout<<"H-extrema real native assertions "<<assertions<<" PASS; exact rounded thresholds, all lengths1..33, endian/unaligned/reverse/broadcast, subnormals, refusal before writes, descriptor/extent and preserved FEnv; serial mask, default1; AVX2="<<has_avx2()<<'\n';
}
