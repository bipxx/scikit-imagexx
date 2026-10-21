#pragma once
#include <cstddef>
#include <cstdint>
namespace skimagexx::simd {
#ifdef SKIMAGEXX_AVX2
void gray(const float*, float*, std::size_t);
void gray(const double*, double*, std::size_t);
void gray(const std::uint8_t*, double*, std::size_t);
void convolve(const float* const*, float*, std::size_t, const double*, std::size_t, double);
void convolve(const double* const*, double*, std::size_t, const double*, std::size_t, double);
void convolve(const std::uint8_t* const*, double*, std::size_t, const double*, std::size_t, double);
std::size_t edges(const float*, const float*, const float*, float*, std::size_t);
std::size_t edges(const double*, const double*, const double*, double*, std::size_t);
std::size_t edges(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*, double*, std::size_t);
void otsu_scores(const float*, const float*, const double*, const double*, double*, std::size_t);
bool uniform(const std::uint8_t*, std::size_t);
#endif
}
