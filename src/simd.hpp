#pragma once
#include <cstddef>
#include <cstdint>
#include "skimagexx/convert.hpp"
namespace skimagexx::simd {
#ifdef SKIMAGEXX_AVX2
void gray(const float*, float*, std::size_t);
void gray(const double*, double*, std::size_t);
void gray(const std::uint8_t*, double*, std::size_t);
void convolve(const float* const*, float*, std::size_t, const double*, std::size_t, double);
void convolve(const double* const*, double*, std::size_t, const double*, std::size_t, double);
void convolve(const std::uint8_t* const*, double*, std::size_t, const double*, std::size_t, double);
std::size_t gradient(const float*, const float*, const float*, float*, std::size_t, double, double, int);
std::size_t gradient(const double*, const double*, const double*, double*, std::size_t, double, double, int);
std::size_t gradient(const std::uint8_t*, const std::uint8_t*, const std::uint8_t*, double*, std::size_t, double, double, int);
void gray2rgb(const float*, float*, std::size_t);
void gray2rgb(const double*, double*, std::size_t);
void gray2rgb(const std::uint8_t*, std::uint8_t*, std::size_t);
void invert(const float*, float*, std::size_t, bool);
void invert(const double*, double*, std::size_t, bool);
void invert(const std::uint8_t*, std::uint8_t*, std::size_t, bool);
double sum(const float*, std::size_t);
double sum(const double*, std::size_t);
double sum(const std::uint8_t*, std::size_t);
void otsu_scores(const float*, const float*, const double*, const double*, double*, std::size_t);
bool uniform(const std::uint8_t*, std::size_t);
void lookup(const std::uint8_t*, std::uint8_t*, std::size_t, const std::int32_t*);
void lookup(const std::uint8_t*, double*, std::size_t, const double*);
void bounds(const float*, std::size_t, double&, double&);
void bounds(const double*, std::size_t, double&, double&);
void bounds(const std::uint8_t*, std::size_t, double&, double&);
void bounds(const std::uint16_t*, std::size_t, double&, double&);
void bounds(const std::int16_t*, std::size_t, double&, double&);
std::uint32_t maximum(const std::uint32_t*, std::size_t);
std::int32_t maximum(const std::int32_t*, std::size_t);
std::uint64_t maximum(const std::uint64_t*, std::size_t);
std::int64_t maximum(const std::int64_t*, std::size_t);
void rescale(const float*, float*, std::size_t, double, double, double, double);
void rescale(const double*, double*, std::size_t, double, double, double, double);
void rescale(const std::uint8_t*, double*, std::size_t, double, double, double, double);
bool gamma_simple(const float*, float*, std::size_t, double, double);
bool gamma_simple(const double*, double*, std::size_t, double, double);
bool exposure(const float*, float*, std::size_t, int, double, double, bool);
bool exposure(const double*, double*, std::size_t, int, double, double, bool);
std::size_t convert(const void*, void*, std::size_t, PixelType, PixelType, bool, ConversionInfo&);
#endif
}
