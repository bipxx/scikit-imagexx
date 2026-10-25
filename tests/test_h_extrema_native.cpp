#include "skimagexx/h_extrema.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cassert>
#include <cfenv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error H-extrema native guards require enabled assertions
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

using namespace skimagexx;
namespace {
std::size_t assertions=0;
void check(bool value){++assertions;assert(value);}
template<class F> void rejected(F&& function) {
    bool failed=false;try{function();}catch(const std::exception&){failed=true;}check(failed);
}
std::uint64_t word(const unsigned char* pointer,std::size_t bytes) {
    if(bytes==1)return *pointer;
    if(bytes==2){std::uint16_t value;std::memcpy(&value,pointer,2);return value;}
    if(bytes==4){std::uint32_t value;std::memcpy(&value,pointer,4);return value;}
    std::uint64_t value;std::memcpy(&value,pointer,8);return value;
}
void put(unsigned char* pointer,std::uint64_t value,std::size_t bytes,bool swap) {
    if(bytes==1)*pointer=static_cast<unsigned char>(value);
    else if(bytes==2){const auto x=std::uint16_t(value);std::memcpy(pointer,&x,2);}
    else if(bytes==4){const auto x=std::uint32_t(value);std::memcpy(pointer,&x,4);}
    else std::memcpy(pointer,&value,8);
    if(swap)std::reverse(pointer,pointer+bytes);
}
void boundary(ReconstructionType type,std::size_t bytes,bool signed_value,bool swapped,bool reverse) {
    const auto mask=bytes==8 ? ~std::uint64_t(0) : (std::uint64_t(1)<<(8*bytes))-1;
    const auto sign=std::uint64_t(1)<<(8*bytes-1);
    const std::vector<std::uint64_t> input=signed_value
        ? std::vector<std::uint64_t>{sign,sign+1,mask,0,sign-1}
        : std::vector<std::uint64_t>{0,1,mask-1,mask};
    std::vector<std::uint64_t> lower=signed_value
        ? std::vector<std::uint64_t>{sign,sign,mask-1,mask,sign-2}
        : std::vector<std::uint64_t>{0,0,mask-2,mask-1};
    std::vector<std::uint64_t> upper=signed_value
        ? std::vector<std::uint64_t>{sign+1,sign+2,0,1,sign-1}
        : std::vector<std::uint64_t>{1,2,mask,mask};
    const std::size_t n=input.size(),shape[]={n};
    const std::ptrdiff_t strides[]={std::ptrdiff_t(2*bytes)*(reverse ? -1 : 1)};
    std::vector<unsigned char> allocation((2*n-1)*bytes+3,0xa7);
    for(std::size_t i=0;i<n;++i)put(allocation.data()+1+2*i*bytes,input[i],bytes,swapped);
    ReconstructionView view{allocation.data()+1+(reverse ? 2*(n-1)*bytes : 0),allocation.data(),allocation.size(),
                            {shape,1},{strides,1},type,swapped};
    check(h_extrema_validate_view(view)==n);
    check(h_extrema_integer_range(view).bits==mask);
    if(reverse){std::reverse(lower.begin(),lower.end());std::reverse(upper.begin(),upper.end());}
    std::vector<unsigned char> output(n*bytes+2,0xcc);
    for(const auto method:{ReconstructionMethod::dilation,ReconstructionMethod::erosion}) {
        const auto info=h_extrema_integer_seed_into(view,1,method,{output.data()+1,n*bytes});
        check(info.workers==1 && !info.simd);
        const auto& expected=method==ReconstructionMethod::dilation ? lower : upper;
        for(std::size_t i=0;i<n;++i)check(word(output.data()+1+i*bytes,bytes)==expected[i]);
        check(output.front()==0xcc && output.back()==0xcc);
    }
    const auto before=allocation;
    rejected([&]{h_extrema_integer_seed_into(view,1,ReconstructionMethod::dilation,{allocation.data()+1,n*bytes});});
    check(allocation==before);
    rejected([&]{h_extrema_integer_seed_into(view,1,ReconstructionMethod::dilation,{output.data(),n*bytes-1});});
    // Metadata-only invalid extent: every real pointer still lies inside live
    // storage, and this function cannot read any image pixels.
    auto short_allocation=view;short_allocation.allocation_bytes=1;
    rejected([&]{h_extrema_validate_view(short_allocation);});
}
void byte_exhaustion() {
    std::vector<unsigned char> values(256),output(256);
    for(unsigned i=0;i<256;++i)values[i]=static_cast<unsigned char>(i);
    const std::size_t shape[]={256};const std::ptrdiff_t stride[]={1};
    const ReconstructionView view{values.data(),values.data(),values.size(),{shape,1},{stride,1},ReconstructionType::uint8,false};
    for(unsigned height=0;height<256;++height) {
        h_extrema_integer_seed_into(view,height,ReconstructionMethod::dilation,{output.data(),output.size()});
        for(unsigned i=0;i<256;++i)check(output[i]==unsigned(std::max(0,int(i)-int(height))));
        h_extrema_integer_seed_into(view,height,ReconstructionMethod::erosion,{output.data(),output.size()});
        for(unsigned i=0;i<256;++i)check(output[i]==std::min(255u,i+height));
    }
    std::fill(output.begin(),output.end(),0xcc);
    rejected([&]{h_extrema_integer_seed_into(view,256,ReconstructionMethod::dilation,{output.data(),output.size()});});
    check(std::all_of(output.begin(),output.end(),[](auto x){return x==0xcc;}));
}
void residue() {
    const std::uint64_t image[]={0,3,6,9007199254740993ULL,9007199254740995ULL,9223372036854775809ULL};
    double reconstructed[]={0,0,3,9007199254740992.,9007199254740994.,9223372036854775808.};
    const std::size_t shape[]={6};const std::ptrdiff_t stride[]={8};
    const ReconstructionView a{image,image,sizeof(image),{shape,1},{stride,1},ReconstructionType::uint64,false};
    const ReconstructionView b{reconstructed,reconstructed,sizeof(reconstructed),{shape,1},{stride,1},ReconstructionType::float64,false};
    unsigned char result[6]={};const unsigned char expected[]={0,1,1,0,1,0};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto saved_flags=std::fetestexcept(FE_ALL_EXCEPT);
    h_extrema_integer_mask_into(a,b,1.,ReconstructionMethod::dilation,{result,6});
    check(std::memcmp(result,expected,6)==0);check(std::fetestexcept(FE_ALL_EXCEPT)==saved_flags);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::fill(std::begin(result),std::end(result),0xcc);
    reconstructed[5]=std::numeric_limits<double>::infinity();
    rejected([&]{h_extrema_integer_mask_into(a,b,1.,ReconstructionMethod::dilation,{result,6});});
    check(std::all_of(std::begin(result),std::end(result),[](auto x){return x==0xcc;}));
    reconstructed[5]=9223372036854775808.;
    const auto old_round=std::fegetround();check(std::fesetround(FE_UPWARD)==0);
    rejected([&]{h_extrema_integer_mask_into(a,b,1.,ReconstructionMethod::dilation,{result,6});});
    check(std::fegetround()==FE_UPWARD);check(std::fesetround(old_round)==0);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    const auto old_csr=_mm_getcsr();_mm_setcsr(old_csr|(1u<<15)|(1u<<6));
    rejected([&]{h_extrema_integer_mask_into(a,b,1.,ReconstructionMethod::dilation,{result,6});});
    check(_mm_getcsr()==(old_csr|(1u<<15)|(1u<<6)));_mm_setcsr(old_csr);
#endif
    check(std::all_of(std::begin(result),std::end(result),[](auto x){return x==0xcc;}));
}
void descriptor_aliases() {
    // One live sample only: even an unfixed first write cannot trigger a later
    // pixel dereference. The repaired implementation must reject before writes.
    const unsigned char pixel=5;const double rec=4.;
    const std::size_t shape[]={1};const std::ptrdiff_t byte_stride[]={1},double_stride[]={8};
    ReconstructionView image{&pixel,&pixel,sizeof(pixel),{shape,1},{byte_stride,1},ReconstructionType::uint8,false};
    ReconstructionView reconstructed{&rec,&rec,sizeof(rec),{shape,1},{double_stride,1},ReconstructionType::float64,false};
    unsigned char image_before[sizeof(image)],reconstructed_before[sizeof(reconstructed)];
    std::memcpy(image_before,&image,sizeof(image));std::memcpy(reconstructed_before,&reconstructed,sizeof(reconstructed));
    rejected([&]{h_extrema_integer_seed_into(image,1,ReconstructionMethod::dilation,
        {reinterpret_cast<unsigned char*>(&image),1});});
    check(std::memcmp(image_before,&image,sizeof(image))==0);
    rejected([&]{h_extrema_integer_mask_into(image,reconstructed,1.,ReconstructionMethod::dilation,
        {reinterpret_cast<unsigned char*>(&image),1});});
    check(std::memcmp(image_before,&image,sizeof(image))==0);
    rejected([&]{h_extrema_integer_mask_into(image,reconstructed,1.,ReconstructionMethod::dilation,
        {reinterpret_cast<unsigned char*>(&reconstructed),1});});
    check(std::memcmp(reconstructed_before,&reconstructed,sizeof(reconstructed))==0);
}
void parallel_seed(int workers) {
    const std::size_t n=65537,shape[]={n};const std::ptrdiff_t stride[]={2};
    std::vector<std::uint16_t> image(n),output(n);
    for(std::size_t i=0;i<n;++i)image[i]=std::uint16_t(i);
    const ReconstructionView view{image.data(),image.data(),n*2,{shape,1},{stride,1},ReconstructionType::uint16,false};
    const auto info=h_extrema_integer_seed_into(view,4,ReconstructionMethod::erosion,
        {reinterpret_cast<unsigned char*>(output.data()),n*2});
    check(info.workers==workers && !info.simd);
    for(std::size_t i=0;i<n;++i)check(output[i]==std::min(65535u,unsigned(image[i])+4u));
}
}
int main() {
    check(get_num_threads()==1);
    const bool portable=std::getenv("SKIMAGEXX_EXPECT_OPENMP_OFF")!=nullptr;
    const ReconstructionType types[]={ReconstructionType::uint8,ReconstructionType::int8,ReconstructionType::uint16,ReconstructionType::int16,
        ReconstructionType::uint32,ReconstructionType::int32,ReconstructionType::uint64,ReconstructionType::int64};
    for(int workers:{1,4}) {
        if(portable && workers==4){rejected([]{set_num_threads(4);});check(get_num_threads()==1);continue;}
        set_num_threads(workers);
        for(std::size_t i=0;i<8;++i)for(bool swapped:{false,true})for(bool reverse:{false,true})
            boundary(types[i],std::size_t(1)<<(i/2),bool(i%2),swapped,reverse);
        byte_exhaustion();residue();descriptor_aliases();parallel_seed(workers);
    }
    set_num_threads(1);
    std::cout<<"H-extrema integer native assertions "<<assertions<<" PASS; default1, widths8/16/32/64, saturation, strided/endian/unaligned, raw wide rounding, allocation and FP guards; AVX2="<<has_avx2()<<'\n';
}
