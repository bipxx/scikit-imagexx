#pragma once
#include <immintrin.h>
#include <cmath>

namespace skimagexx::simd::math {
inline __m256 exp8(__m256 x) {
    const auto regular = _mm256_and_ps(_mm256_cmp_ps(x,_mm256_set1_ps(-80),_CMP_GE_OQ),
                                       _mm256_cmp_ps(x,_mm256_set1_ps(80),_CMP_LE_OQ));
    if (_mm256_movemask_ps(regular) != 255) {
        float values[8]; _mm256_storeu_ps(values,x);
        for (auto& value : values) value = std::exp(value);
        return _mm256_loadu_ps(values);
    }
    const auto k = _mm256_round_ps(_mm256_mul_ps(x,_mm256_set1_ps(1.4426950408889634f)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto r = _mm256_sub_ps(x,_mm256_mul_ps(k,_mm256_set1_ps(0.693359375f)));
    r = _mm256_sub_ps(r,_mm256_mul_ps(k,_mm256_set1_ps(-0.00021219444005469058f)));
    constexpr float coefficients[] = {1,1,0.5f,1.0f/6,1.0f/24,1.0f/120,1.0f/720,1.0f/5040};
    auto polynomial = _mm256_set1_ps(coefficients[7]);
    for (int degree = 6; degree >= 0; --degree)
        polynomial = _mm256_add_ps(_mm256_set1_ps(coefficients[degree]),_mm256_mul_ps(r,polynomial));
    const auto power = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(k),_mm256_set1_epi32(127)),23);
    return _mm256_mul_ps(polynomial,_mm256_castsi256_ps(power));
}
// IEEE-754 range reduction. Unusual inputs use libm rather than approximate
// handling of subnormals, infinities, overflow, or NaNs.
inline __m256d exp4(__m256d x) {
    const auto regular = _mm256_and_pd(_mm256_cmp_pd(x,_mm256_set1_pd(-700),_CMP_GE_OQ),
                                       _mm256_cmp_pd(x,_mm256_set1_pd(700),_CMP_LE_OQ));
    if (_mm256_movemask_pd(regular) != 15) {
        double values[4]; _mm256_storeu_pd(values,x);
        for (auto& value : values) value = std::exp(value);
        return _mm256_loadu_pd(values);
    }
    const auto k = _mm256_round_pd(_mm256_mul_pd(x,_mm256_set1_pd(1.4426950408889634074)),
                                   _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    auto r = _mm256_sub_pd(x,_mm256_mul_pd(k,_mm256_set1_pd(0.69314718055994528623)));
    r = _mm256_sub_pd(r,_mm256_mul_pd(k,_mm256_set1_pd(2.3190468138462995584e-17)));
    // exp(r), |r| <= log(2)/2. Degree 13 bounds the truncation error
    // below double rounding noise in this reduced interval.
    constexpr double coefficients[] = {1,1,1.0/2,1.0/6,1.0/24,1.0/120,1.0/720,1.0/5040,
        1.0/40320,1.0/362880,1.0/3628800,1.0/39916800,1.0/479001600,1.0/6227020800};
    auto polynomial = _mm256_set1_pd(coefficients[13]);
    for (int degree = 12; degree >= 0; --degree)
        polynomial = _mm256_add_pd(_mm256_set1_pd(coefficients[degree]),_mm256_mul_pd(r,polynomial));
    const auto integers = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k));
    const auto power = _mm256_slli_epi64(_mm256_add_epi64(integers,_mm256_set1_epi64x(1023)),52);
    return _mm256_mul_pd(polynomial,_mm256_castsi256_pd(power));
}

inline __m256d log2_4(__m256d x) {
    const auto regular = _mm256_and_pd(
        _mm256_cmp_pd(x,_mm256_set1_pd(2.2250738585072014e-308),_CMP_GE_OQ),
        _mm256_cmp_pd(x,_mm256_set1_pd(INFINITY),_CMP_LT_OQ));
    if (_mm256_movemask_pd(regular) != 15) {
        double values[4]; _mm256_storeu_pd(values,x);
        for (auto& value : values) value = std::log2(value);
        return _mm256_loadu_pd(values);
    }
    const auto bits = _mm256_castpd_si256(x);
    const auto exponents = _mm256_srli_epi64(bits,52);
    auto e = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(exponents,
        _mm256_set1_epi64x(0x4330000000000000LL))),_mm256_set1_pd(4503599627371519.0));
    auto m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(bits,
        _mm256_set1_epi64x(0x000fffffffffffffLL)),_mm256_set1_epi64x(0x3ff0000000000000LL)));
    const auto upper = _mm256_cmp_pd(m,_mm256_set1_pd(1.4142135623730951),_CMP_GT_OQ);
    m = _mm256_blendv_pd(m,_mm256_mul_pd(m,_mm256_set1_pd(0.5)),upper);
    e = _mm256_add_pd(e,_mm256_and_pd(upper,_mm256_set1_pd(1)));
    const auto t = _mm256_div_pd(_mm256_sub_pd(m,_mm256_set1_pd(1)),_mm256_add_pd(m,_mm256_set1_pd(1)));
    const auto square = _mm256_mul_pd(t,t);
    constexpr double coefficients[] = {1,1.0/3,1.0/5,1.0/7,1.0/9,1.0/11,1.0/13,1.0/15,1.0/17,1.0/19};
    auto polynomial = _mm256_set1_pd(coefficients[9]);
    for (int degree = 8; degree >= 0; --degree)
        polynomial = _mm256_add_pd(_mm256_set1_pd(coefficients[degree]),_mm256_mul_pd(square,polynomial));
    return _mm256_add_pd(e,_mm256_mul_pd(_mm256_mul_pd(t,polynomial),_mm256_set1_pd(2.8853900817779268147)));
}
} // namespace skimagexx::simd::math
