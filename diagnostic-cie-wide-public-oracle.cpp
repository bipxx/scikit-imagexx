#include "skimagexx/image.hpp"
#include "skimagexx/perceptual.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

using namespace skimagexx;
using Op = PerceptualTransform;

template<class T> void inspect(PixelType type, Op op, std::size_t count, int workers,
    std::size_t channels, std::ptrdiff_t pixels, std::ptrdiff_t channel_stride,
    std::array<double,3> white, int kind, int round) {
    using Out = std::conditional_t<std::is_same_v<T,float>,float,double>;
    const auto pixel_span = static_cast<std::size_t>(std::abs(pixels))*(count-1);
    const auto channel_span = static_cast<std::size_t>(std::abs(channel_stride))*(channels-1);
    std::vector<T> src(pixel_span+channel_span+1);
    std::vector<Out> dst(count*channels);
    std::uint32_t state = 527337461;
    const std::array<double,20> edges = {
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::quiet_NaN(),
        -0.0, 0.0, -0.1, 0.008856, 0.04045, 0.0089, 0.0405, 0.012,
        0.001, 1e8, 1e8+10, 0.9, 0.1, 0.25, 1e20, -1e20
    };
    for (std::size_t i=0;i<src.size();++i) {
        state = state*1664525u+1013904223u;
        if constexpr (std::is_same_v<T,std::uint8_t>) src[i]=kind==0 ? static_cast<T>(i) : static_cast<T>(state>>24);
        else src[i]=kind==0 ? T((state>>8)*0x1p-24) : T(edges[(state>>12)%edges.size()]);
    }
    const auto start = src.data()+(pixels<0 ? pixel_span : 0)+(channel_stride<0 ? channel_span : 0);
    set_num_threads(workers);
    if (std::fesetround(round)!=0) std::abort();
    std::feclearexcept(FE_ALL_EXCEPT);
    const auto status=perceptual_into(start,dst.data(),count,type,op,white,channels,pixels,channel_stride);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    if (std::fegetround()!=round) std::abort();
    std::cout.put(static_cast<char>(flags));
    std::cout.put(static_cast<char>(status.unsupported));
    std::cout.write(reinterpret_cast<const char*>(&status.negative_z),sizeof(status.negative_z));
    std::cout.write(reinterpret_cast<const char*>(dst.data()),static_cast<std::streamsize>(dst.size()*sizeof(Out)));
}

template<class T> void compare_type(PixelType type) {
    constexpr std::array<Op,4> ops={Op::xyz2lab,Op::rgb2lab,Op::lab2lch,Op::rgb2luv};
    constexpr std::array<double,3> normal={0.95047,1.0,1.08883}, alternate={0.96422,1.0,0.82521};
    for (const auto op:ops) {
        for (const auto round:{FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
            for (const int mode:{0,1}) {
                for (const std::size_t n:{1,4,7,8,9,17,1023,1024,1025,4096}) {
                    inspect<T>(type,op,n,1,3,3,1,normal,mode,round);
                }
                inspect<T>(type,op,4099,1,3,3,1,alternate,mode,round);
                inspect<T>(type,op,1025,1,3,-3,-1,normal,mode,round);
                inspect<T>(type,op,1025,1,3,0,1,normal,mode,round);
                inspect<T>(type,op,1025,1,3,1,1025,normal,mode,round);
                if (op==Op::lab2lch) inspect<T>(type,op,1025,1,4,4,1,normal,mode,round);
            }
        }
        for (const auto workers:{1,2,3,4})
            inspect<T>(type,op,262157,workers,3,3,1,normal,0,FE_TONEAREST);
    }
}

void guard_final_packet(Op op, std::size_t n) {
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto bytes = 3*n;
    const auto accessible = ((bytes+page-1)/page)*page;
    auto* base = static_cast<std::uint8_t*>(mmap(nullptr,accessible+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if (base==MAP_FAILED || mprotect(base+accessible,page,PROT_NONE)!=0) std::abort();
    auto* src = base+accessible-bytes;
    for (std::size_t i=0;i<bytes;++i) src[i]=static_cast<std::uint8_t>(i*127+2);
    std::vector<double> dst(bytes);
    const auto status=perceptual_into(src,dst.data(),n,PixelType::uint8,op);
    if (status.unsupported || status.negative_z || munmap(base,accessible+page)!=0) std::abort();
    std::cout.write(reinterpret_cast<const char*>(dst.data()),static_cast<std::streamsize>(dst.size()*sizeof(double)));
}

template<class T, class U> void guard_float_span(Op op, std::size_t n, U tail, bool mixed_halves=false, bool unity_white=false) {
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    const auto elements=3*n, bytes=elements*sizeof(T);
    const auto accessible=((bytes+page-1)/page)*page;
    auto* base=static_cast<char*>(mmap(nullptr,accessible+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if (base==MAP_FAILED || mprotect(base+accessible,page,PROT_NONE)!=0) std::abort();
    auto* src=reinterpret_cast<T*>(base+accessible-bytes);
    constexpr T fixture[] = {0.0,0.001,0.004,0.008855,0.008856,0.008857,0.04044,0.04045,0.04046,0.5,1.0};
    for (std::size_t i=0;i<elements;++i) src[i]=unity_white ? T(1) : mixed_halves ? ((i/12)%2 ? T(0.5) : T(0.001)) : fixture[i%11];
    std::memcpy(src+elements-1,&tail,sizeof(tail));
    std::vector<T> dst(elements);
    std::feclearexcept(FE_ALL_EXCEPT);
    const auto white=unity_white ? std::array<double,3>{1,1,1} : std::array<double,3>{0.95047,1.0,1.08883};
    const auto status=perceptual_into(src,dst.data(),n,std::is_same_v<T,float> ? PixelType::float32 : PixelType::float64,op,white);
    std::cout.put(static_cast<char>(std::fetestexcept(FE_ALL_EXCEPT)));
    if (status.unsupported || status.negative_z || munmap(base,accessible+page)!=0) std::abort();
    std::cout.write(reinterpret_cast<const char*>(dst.data()),static_cast<std::streamsize>(bytes));
}

int main() {
    const bool disabled = std::getenv("SKIMAGEXX_DISABLE_SIMD") != nullptr;
    if (has_avx2()==disabled) return 2;
    compare_type<std::uint8_t>(PixelType::uint8);
    compare_type<float>(PixelType::float32);
    compare_type<double>(PixelType::float64);
    set_num_threads(1);
    guard_final_packet(Op::lab2lch,4);
    guard_final_packet(Op::xyz2lab,4096);
    for (auto op: {Op::xyz2lab, Op::rgb2lab})
        for (auto n: {63u,64u,65u})
            for (auto bits: {0u,0x80000000u,1u,0x3f800001u,0x7f800000u,0x7f800001u,0xffc00000u})
                guard_float_span<float>(op,n,bits);
    for (auto op: {Op::xyz2lab, Op::rgb2lab}) {
        for (auto n: {63u,64u,65u,68u,72u})
            for (auto bits: {0ULL,0x8000000000000000ULL,1ULL,0x3ff0000000000001ULL,
                 0x7ff0000000000000ULL,0x7ff0000000000001ULL,0xfff8000000000000ULL})
                guard_float_span<double>(op,n,bits);
        for (auto rounding: {FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            if (std::fesetround(rounding)!=0) std::abort();
            guard_float_span<double>(op,64,0ULL,true);
        }
    }
    if (std::fesetround(FE_TONEAREST)!=0) std::abort();
    guard_float_span<double>(Op::xyz2lab,64,0x3ff0000000000000ULL,false,true);
    return std::cout ? 0 : 3;
}
