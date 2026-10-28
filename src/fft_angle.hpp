#pragma once
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace skimagexx::detail {
// Exact nearest-even reconstruction of double(quarter_pi_x87 / n).
// quarter_pi_x87 = 0xc90fdaa22168c235 * 2^-64. n must be positive.
inline double fft_angle_nearest(std::uint64_t n) {
    constexpr std::uint64_t m = UINT64_C(0xc90fdaa22168c235);
    unsigned shift = 0;
    for (auto v = n; v >>= 1;) ++shift;
    std::uint64_t remainder, quotient;
    for (;;) {
        const auto high = shift == 64 ? m : shift ? m >> (64-shift) : 0;
        const auto low = shift == 64 ? 0 : m << shift;
#if defined(_MSC_VER) && defined(_M_X64)
        quotient = _udiv128(high, low, n, &remainder);
#else
        const auto numerator = (static_cast<unsigned __int128>(high) << 64) | low;
        quotient = static_cast<std::uint64_t>(numerator / n);
        remainder = static_cast<std::uint64_t>(numerator % n);
#endif
        if (quotient >= (UINT64_C(1) << 63)) break;
        ++shift;
    }
    if (remainder > n-remainder || (remainder == n-remainder && (quotient & 1))) {
        if (++quotient == 0) { quotient = UINT64_C(1) << 63; --shift; }
    }
    auto significand = quotient >> 11;
    const auto tail = quotient & 2047;
    if (tail > 1024 || (tail == 1024 && (significand & 1))) ++significand;
    auto exponent = 1022-shift;
    if (significand == (UINT64_C(1) << 53)) { significand >>= 1; ++exponent; }
    const std::uint64_t bits = (std::uint64_t(exponent) << 52) | (significand & ((UINT64_C(1) << 52)-1));
    double result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
}
