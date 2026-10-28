#include "fft_angle.hpp"
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    if (!input) return 3;
    std::uint64_t n, expected;
    std::size_t count = 0;
    while (input >> std::dec >> n >> std::hex >> expected) {
        for (int mode : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
            if (std::fesetround(mode)) return 4;
            std::feclearexcept(FE_ALL_EXCEPT);
            std::feraiseexcept(FE_INVALID | FE_INEXACT);
            const int flags = std::fetestexcept(FE_ALL_EXCEPT);
            const double actual = skimagexx::detail::fft_angle_nearest(n);
            std::uint64_t bits;
            std::memcpy(&bits, &actual, sizeof(bits));
            if (bits != expected || flags != std::fetestexcept(FE_ALL_EXCEPT) || std::fegetround() != mode) {
                std::cerr << "denominator " << n << " mode " << mode << " actual " << std::hex << bits << " expected " << expected << '\n';
                return 1;
            }
        }
        ++count;
    }
    std::fesetround(FE_TONEAREST);
    if (!input.eof() || count == 0) return 5;
    std::cout << count << " frozen denominators x4 rounding modes exact; flags unchanged\n";
}
