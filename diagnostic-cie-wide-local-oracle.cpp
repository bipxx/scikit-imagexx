// Compile this in place of the candidate leaf object for the hosted route
// witness only. Include the exact candidate leaf directory via -I.
#include "simd_color_avx2.cpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cstdio>

int main() {
    using namespace skimagexx;
    using namespace skimagexx::simd;
    constexpr auto full_mask = _MM_MASK_MASK;
    const unsigned initial = _mm_getcsr();
    if (!__builtin_cpu_supports("x86-64-v4") || !has_avx2() ||
        (initial & full_mask) != full_mask || std::fegetround() != FE_TONEAREST ||
        !ordinary_lab_wide_available()) return 1;
    constexpr std::array<double,8> inputs = {1.0,0.008856,0.008857,0.04045,0.04046,0.5,1.4142135623730951,2.0};
    constexpr std::array<double,2> scales = {2.4*0.69314718055994530942,(1.0/3.0)*0.69314718055994530942};
    for (const auto scale : scales) {
        auto wide = inputs;
        LabPair paired;
        std::memcpy(&paired,inputs.data(),sizeof(paired));
        std::feclearexcept(FE_ALL_EXCEPT);
        ordinary_lab_power<true>(paired,scale);
        const int flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);
        ordinary_lab_power_wide(wide.data(),scale);
        if (std::fetestexcept(FE_ALL_EXCEPT) != flags || std::memcmp(&paired,wide.data(),sizeof(paired)) != 0) return 2;
    }

    constexpr std::size_t pixels = 64;
    std::array<double,pixels*3> span{}, wide{}, fallback{};
    for (std::size_t i=0;i<span.size();++i) span[i] = i%2 ? 0.5 : 0.001;
    const std::array<double,3> white = {0.95047,1.0,1.08883};
    const auto no_sticky = initial & ~_MM_EXCEPT_MASK;
    for (auto op : {PerceptualTransform::xyz2lab,PerceptualTransform::rgb2lab}) {
        std::feclearexcept(FE_ALL_EXCEPT);
        _mm_setcsr(no_sticky);
        if (!ordinary_lab_wide_available()) return 3;
        const auto first = perceptual_into(span.data(),wide.data(),pixels,PixelType::float64,op,white);
        const auto flags = std::fetestexcept(FE_ALL_EXCEPT);
        std::feclearexcept(FE_ALL_EXCEPT);
        // Invalid-only unmask, with finite positive operands that cannot raise
        // invalid. Leave inexact and all other exceptions masked.
        _mm_setcsr(no_sticky & ~_MM_MASK_INVALID);
        if (ordinary_lab_wide_available()) return 4;
        const auto second = perceptual_into(span.data(),fallback.data(),pixels,PixelType::float64,op,white);
        const auto fallback_flags = std::fetestexcept(FE_ALL_EXCEPT);
        const unsigned observed_mask = _mm_getcsr() & full_mask;
        _mm_setcsr(no_sticky);
        if (observed_mask != (full_mask & ~_MM_MASK_INVALID) || first.unsupported || second.unsupported ||
            first.negative_z != second.negative_z || flags != fallback_flags ||
            std::memcmp(wide.data(),fallback.data(),sizeof(wide)) != 0) return 5;
    }
    _mm_setcsr(initial);
    std::puts("v4 profile, two direct ZMM helper calls, two invalid-unmasked fallback comparisons: pass");
}
