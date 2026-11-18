// Compile in place of the candidate leaf object with the exact source directory
// on the include path. Execute only on a real OS-effective x86-64-v4 host.
#include "simd_color_avx2.cpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cstdio>

namespace {
// The predecessor scalar scale occurs at runtime even for a constant caller.
__attribute__((noipa))
__m256 paired_f32_power(__m256 input, double exponent) {
    return skimagexx::simd::ordinary_lab_power<false>(input,exponent);
}
}

int main() {
    using namespace skimagexx;
    using namespace skimagexx::simd;
    constexpr unsigned full_mask = _MM_MASK_MASK;
    const unsigned initial = _mm_getcsr();
    if (!has_avx2() || (initial & full_mask) != full_mask ||
        std::fegetround() != FE_TONEAREST || !ordinary_lab_wide_available()) return 1;
    constexpr std::array<float,8> mixed = {1.0f,0.008857f,0.04045f,0.04046f,0.5f,1.25f,1.41421354f,2.0f};
    for (const auto exponent : {double(2.4f),1.0/3.0}) {
        for (const bool neutral : {true,false}) {
            const auto x = neutral ? _mm256_set1_ps(1.0f) : _mm256_loadu_ps(mixed.data());
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto paired = paired_f32_power(x,exponent);
            const int old_flags = std::fetestexcept(FE_ALL_EXCEPT);
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto wide = ordinary_lab_power<true>(x,exponent);
            const int wide_flags = std::fetestexcept(FE_ALL_EXCEPT);
            if (old_flags != wide_flags || std::memcmp(&paired,&wide,sizeof(wide)) != 0) return 2;
            if (neutral && wide_flags != FE_INEXACT) return 3;
        }
    }

    constexpr std::size_t pixels = 64;
    std::array<float,pixels*3> span{}, paired{}, public_result{};
    for (std::size_t i=0;i<span.size();++i) span[i] = i%2 ? 0.5f : 0.001f;
    const std::array<double,3> white = {0.95047,1.0,1.08883};
    for (const int direction : {FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO}) {
        if (std::fesetround(direction) != 0) return 4;
        // At nearest, run both the enabled and invalid-unmasked path. The
        // selected positive inputs cannot raise unmasked FE_INVALID.
        for (const bool unmask_invalid : {false,true}) {
            if (unmask_invalid && direction != FE_TONEAREST) continue;
            std::feclearexcept(FE_ALL_EXCEPT);
            const auto mask = unmask_invalid ? (full_mask & ~_MM_MASK_INVALID) : full_mask;
            _mm_setcsr((_mm_getcsr() & ~full_mask) | mask);
            if (ordinary_lab_wide_available() == unmask_invalid) return 5;
            for (auto op : {PerceptualTransform::xyz2lab,PerceptualTransform::rgb2lab}) {
                std::feclearexcept(FE_ALL_EXCEPT);
                const auto completed = op==PerceptualTransform::xyz2lab ?
                    ordinary_lab_transform<false,false>(span.data(),paired.data(),pixels,white) :
                    ordinary_lab_transform<true,false>(span.data(),paired.data(),pixels,white);
                const auto old_flags = std::fetestexcept(FE_ALL_EXCEPT);
                std::feclearexcept(FE_ALL_EXCEPT);
                const auto status = perceptual_into(span.data(),public_result.data(),pixels,PixelType::float32,op,white);
                if (completed != pixels || status.unsupported || status.negative_z ||
                    old_flags != std::fetestexcept(FE_ALL_EXCEPT) ||
                    std::memcmp(paired.data(),public_result.data(),sizeof(paired)) != 0 ||
                    (_mm_getcsr() & full_mask) != mask || std::fegetround() != direction) return 6;
            }
        }
    }
    if (std::fesetround(FE_TONEAREST) != 0) return 7;
    _mm_setcsr(initial);
    std::puts("f32 v4 direct power raw, neutral FE_INEXACT, both transforms directed and invalid-unmasked: pass");
}
