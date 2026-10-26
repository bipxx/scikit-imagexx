// Independent expected words captured from genuine scikit-image 0.26.0
// BEFORE this leaf existed. See build/hu-native-v1/oracle-provenance.json.
#include "moments_hu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks = 0;
std::size_t calls = 0;

void require(bool condition, const std::string& context) {
    ++checks;
    if (!condition) throw std::runtime_error(context + " check " + std::to_string(checks));
}

template<class T>
bool same(T actual, T expected) {
    if (std::isnan(expected)) return std::isnan(actual);
    return std::memcmp(&actual, &expected, sizeof(T)) == 0;
}

template<class T>
void captured(const char* name, const std::array<T, 16>& input,
              const std::array<T, 7>& expected) {
    // C, Fortran, stepped, each reversed axis, and both axes reversed.
    const std::array<std::array<std::ptrdiff_t, 2>, 6> layouts{{
        {{4, 1}}, {{1, 4}}, {{9, 2}}, {{-9, 2}}, {{9, -2}}, {{-9, -2}}
    }};
    for (const auto& layout : layouts) {
        const auto rs = layout[0], cs = layout[1];
        const auto low = std::min(std::ptrdiff_t(0), 3 * rs) + std::min(std::ptrdiff_t(0), 3 * cs);
        const auto high = std::max(std::ptrdiff_t(0), 3 * rs) + std::max(std::ptrdiff_t(0), 3 * cs);
        std::vector<T> storage(static_cast<std::size_t>(high - low + 5), T(-17.25));
        T* data = storage.data() + 2 - low;
        for (std::ptrdiff_t row = 0; row < 4; ++row)
            for (std::ptrdiff_t col = 0; col < 4; ++col)
                data[row * rs + col * cs] = input[static_cast<std::size_t>(row * 4 + col)];
        const auto before = storage;
        std::array<T, 9> output;
        output.fill(T(-33.5));
        skimagexx::detail::moments_hu(data, rs, cs, output.data() + 1);
        ++calls;
        require(std::memcmp(storage.data(), before.data(), storage.size() * sizeof(T)) == 0,
                std::string(name) + " input changed");
        require(output.front() == T(-33.5), std::string(name) + " output prefix");
        require(output.back() == T(-33.5), std::string(name) + " output suffix");
        for (std::size_t i = 0; i < 7; ++i)
            require(same(output[i + 1], expected[i]), std::string(name) + " result " + std::to_string(i));
    }
}

template<class T>
void broadcast() {
    // The independently captured hu-dtype-bool case is an all-one matrix.
    const std::array<T, 3> input{{T(-2), T(1), T(-3)}};
    const auto before = input;
    const std::array<T, 7> expected{{T(2), T(4), T(8), T(8), T(64), T(16), T(0)}};
    std::array<T, 9> output;
    output.fill(T(-33.5));
    skimagexx::detail::moments_hu(input.data() + 1, 0, 0, output.data() + 1);
    ++calls;
    require(input == before, "broadcast input changed");
    require(output.front() == T(-33.5) && output.back() == T(-33.5), "broadcast output guard");
    for (std::size_t i = 0; i < 7; ++i)
        require(same(output[i + 1], expected[i]), "broadcast result " + std::to_string(i));
}

void original_cases() {
    captured<double>("hu-dtype-float16",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<float>("hu-dtype-float32",
        {{0x1.0000000000000p-4f, 0x1.0000000000000p-3f, 0x1.8000000000000p-3f, 0x1.0000000000000p-2f, 0x1.4000000000000p-2f, 0x1.8000000000000p-2f, 0x1.c000000000000p-2f, 0x1.0000000000000p-4f, 0x1.0000000000000p-3f, 0x1.8000000000000p-3f, 0x1.0000000000000p-2f, 0x1.4000000000000p-2f, 0x1.8000000000000p-2f, 0x1.c000000000000p-2f, 0x1.0000000000000p-4f, 0x1.0000000000000p-3f}},
        {{0x1.4000000000000p-2f, 0x1.2200000000000p-1f, 0x1.f400000000000p-1f, 0x1.b400000000000p-1f, 0x1.6ee0000000000p-3f, 0x1.0200000000000p-1f, 0x1.82e0000000000p-1f}});
    captured<double>("hu-dtype-float64",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-dtype-int32",
        {{0x1.0000000000000p+0, 0x1.0000000000000p+1, 0x1.8000000000000p+1, 0x1.0000000000000p+2, 0x1.4000000000000p+2, 0x1.8000000000000p+2, 0x1.c000000000000p+2, 0x1.0000000000000p+0, 0x1.0000000000000p+1, 0x1.8000000000000p+1, 0x1.0000000000000p+2, 0x1.4000000000000p+2, 0x1.8000000000000p+2, 0x1.c000000000000p+2, 0x1.0000000000000p+0, 0x1.0000000000000p+1}},
        {{0x1.4000000000000p+2, 0x1.2200000000000p+7, 0x1.f400000000000p+7, 0x1.b400000000000p+7, 0x1.6ee0000000000p+13, 0x1.0200000000000p+11, 0x1.82e0000000000p+15}});
    captured<double>("hu-dtype-bool",
        {{0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0}},
        {{0x1.0000000000000p+1, 0x1.0000000000000p+2, 0x1.0000000000000p+3, 0x1.0000000000000p+3, 0x1.0000000000000p+6, 0x1.0000000000000p+4, 0x0.0p+0}});
    captured<double>("hu-dtype-complex64",
        {{0x1.0000000000000p+0, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-dtype-complex128",
        {{0x1.0000000000000p+0, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-layout-F-float64",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-layout-step-float64",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-layout-reverse-float64",
        {{0x1.0000000000000p-3, 0x1.0000000000000p-4, 0x1.c000000000000p-2, 0x1.8000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p-2, 0x1.8000000000000p-3, 0x1.0000000000000p-3, 0x1.0000000000000p-4, 0x1.c000000000000p-2, 0x1.8000000000000p-2, 0x1.4000000000000p-2, 0x1.0000000000000p-2, 0x1.8000000000000p-3, 0x1.0000000000000p-3, 0x1.0000000000000p-4}},
        {{0x1.0000000000000p-1, 0x1.9000000000000p-2, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.1000000000000p-1, -0x1.82e0000000000p-1}});
    captured<double>("hu-layout-endian-float32",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<float>("hu-layout-metadata-float32",
        {{0x1.0000000000000p-4f, 0x1.0000000000000p-3f, 0x1.8000000000000p-3f, 0x1.0000000000000p-2f, 0x1.4000000000000p-2f, 0x1.8000000000000p-2f, 0x1.c000000000000p-2f, 0x1.0000000000000p-4f, 0x1.0000000000000p-3f, 0x1.8000000000000p-3f, 0x1.0000000000000p-2f, 0x1.4000000000000p-2f, 0x1.8000000000000p-2f, 0x1.c000000000000p-2f, 0x1.0000000000000p-4f, 0x1.0000000000000p-3f}},
        {{0x1.4000000000000p-2f, 0x1.2200000000000p-1f, 0x1.f400000000000p-1f, 0x1.b400000000000p-1f, 0x1.6ee0000000000p-3f, 0x1.0200000000000p-1f, 0x1.82e0000000000p-1f}});
    captured<double>("hu-layout-readonly-float16",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-layout-unaligned-float64",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3}},
        {{0x1.4000000000000p-2, 0x1.2200000000000p-1, 0x1.f400000000000p-1, 0x1.b400000000000p-1, 0x1.6ee0000000000p-3, 0x1.0200000000000p-1, 0x1.82e0000000000p-1}});
    captured<double>("hu-points",
        {{0x1.0000000000000p+0, 0x0.0p+0, 0x1.b000000000000p-2, 0x1.e000000000000p-5, 0x0.0p+0, 0x1.8000000000000p-4, 0x1.e000000000000p-4, 0x1.da00000000000p-4, 0x1.2000000000000p-1, 0x0.0p+0, 0x1.e600000000000p-3, 0x1.0e00000000000p-5, 0x0.0p+0, 0x1.b000000000000p-5, 0x1.0e00000000000p-4, 0x1.0aa0000000000p-4}},
        {{0x1.f800000000000p-1, 0x1.c200000000000p-5, 0x1.0428000000000p-3, 0x1.1940000000000p-6, -0x1.1c45700000000p-12, 0x1.07ac000000000p-8, 0x1.8b82000000000p-11}});
    captured<double>("hu-reflected",
        {{0x1.0000000000000p+0, 0x0.0p+0, 0x1.b000000000000p-2, -0x1.e000000000000p-5, 0x0.0p+0, -0x1.8000000000000p-4, 0x1.e000000000000p-4, -0x1.da00000000000p-4, 0x1.2000000000000p-1, 0x0.0p+0, 0x1.e600000000000p-3, -0x1.0e00000000000p-5, 0x0.0p+0, -0x1.b000000000000p-5, 0x1.0e00000000000p-4, -0x1.0aa0000000000p-4}},
        {{0x1.f800000000000p-1, 0x1.c200000000000p-5, 0x1.0428000000000p-3, 0x1.1940000000000p-6, -0x1.1c45700000000p-12, 0x1.07ac000000000p-8, -0x1.8b82000000000p-11}});
    captured<double>("hu-rotated",
        {{0x1.0000000000000p+0, 0x0.0p+0, 0x1.2000000000000p-1, 0x0.0p+0, 0x0.0p+0, -0x1.8000000000000p-4, 0x0.0p+0, -0x1.b000000000000p-5, 0x1.b000000000000p-2, 0x1.e000000000000p-4, 0x1.e600000000000p-3, 0x1.0e00000000000p-4, -0x1.e000000000000p-5, -0x1.da00000000000p-4, -0x1.0e00000000000p-5, -0x1.0aa0000000000p-4}},
        {{0x1.f800000000000p-1, 0x1.c200000000000p-5, 0x1.0428000000000p-3, 0x1.1940000000000p-6, -0x1.1c45700000000p-12, 0x1.07ac000000000p-8, 0x1.8b82000000000p-11}});
    captured<double>("hu-zero",
        {{0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}},
        {{0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}});
    captured<double>("hu-diagonal",
        {{0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+1, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}},
        {{0x1.8000000000000p+1, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}});
    captured<double>("hu-near-cancel",
        {{0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000001p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0}},
        {{0x0.0p+0, 0x0.0p+0, 0x1.0000000000002p+3, 0x1.0000000000000p+3, 0x1.0000000000001p+6, 0x0.0p+0, 0x1.0000000000000p-46}});
    captured<double>("hu-nan",
        {{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}},
        {{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}});
    captured<double>("hu-inf",
        {{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}},
        {{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}});
    captured<double>("hu-overflow",
        {{0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022}},
        {{0x1.fffffffffffffp+1023, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}});
    captured<double>("hu-oversized",
        {{0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.0000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.8000000000000p-3, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4, 0x1.0000000000000p-3, 0x1.4000000000000p-2, 0x1.8000000000000p-2, 0x1.c000000000000p-2, 0x1.0000000000000p-4}},
        {{0x1.2000000000000p-1, 0x1.a000000000000p-5, 0x1.2200000000000p+0, 0x1.5400000000000p-1, 0x1.b940000000000p-4, 0x1.7000000000000p-6, -0x1.21b0000000000p-1}});
    captured<double>("hu-overflow-raise",
        {{0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022, 0x1.fffffffffffffp+1022}},
        {{0x1.fffffffffffffp+1023, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}});
}
}  // namespace

int main() {
    try {
        original_cases();
        broadcast<float>();
        broadcast<double>();
        std::cout << "PASS " << checks << " controls, " << calls << " Hu calls\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
