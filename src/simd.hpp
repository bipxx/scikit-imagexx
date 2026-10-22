#pragma once
#include <cstddef>
#include <cstdint>
#include "skimagexx/convert.hpp"
#include "skimagexx/distance.hpp"
#include "skimagexx/local_threshold.hpp"
namespace skimagexx::detail { struct ColorSpec; }
namespace skimagexx { struct PerceptualInfo; }
namespace skimagexx::detail { struct PerceptualSpec; }
namespace skimagexx::detail { struct StainSpec; }
namespace skimagexx::simd {
bool contrast_finite(const void*,std::size_t,bool);
unsigned char contrast_truth(const unsigned char*,std::size_t);
std::size_t contrast_widen16(const void*,std::int32_t*,std::size_t,bool,bool);
bool contrast_select(void*,void*,std::size_t,PixelType,const std::size_t*,std::size_t);
// A contiguous span with positive or negative element-size byte stride.
double downscale_sum(const void*,std::size_t,std::ptrdiff_t,PixelType,bool,bool,double,bool&);
std::size_t downscale_linear(const void*,void*,std::size_t,PixelType,bool,bool,double,bool&);
std::size_t downscale_row(const void*,void*,std::size_t,std::size_t,std::ptrdiff_t,std::ptrdiff_t,
    std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,PixelType,bool,bool,double,double,bool&);
std::size_t integrate_add(const double*,double*,std::size_t);
std::size_t gaussian_difference(const void*,const void*,void*,std::size_t,bool);
std::size_t gaussian_unsharp(const void*,const void*,void*,std::size_t,bool,bool,double,bool,bool,double);
std::size_t integral_prefix_u64(const std::uint64_t*,std::uint64_t*,std::size_t);
std::size_t integral_strided_prefix_u64(const void*,std::uint64_t*,std::size_t,std::ptrdiff_t,bool);
std::size_t integral_offset_u64(std::uint64_t*,std::size_t,std::uint64_t);
std::size_t integral_cast_u64(const void*,std::uint64_t*,std::size_t,PixelType);
std::size_t integral_add(const void*,void*,std::size_t,PixelType);
std::size_t gaussian_prepare(const void*,double*,std::size_t,PixelType,bool);
std::size_t gaussian_rows(const void* const*,const void* const*,const void*,void*,std::size_t,std::size_t,const double*,double,double,bool,bool);
std::size_t farid_row(const void* const*,const std::ptrdiff_t*,const double*,std::size_t,void*,PixelType,std::size_t,std::size_t,std::size_t,double,bool,bool);
std::size_t farid_finish(float*,std::size_t,double);
std::size_t farid_finish(double*,std::size_t,double);
std::size_t laplace_nd(const float* const*,const float* const*,const float*,float*,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t);
std::size_t laplace_nd(const double* const*,const double* const*,const double*,double*,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t);
std::size_t laplace_nd(const std::uint8_t* const*,const std::uint8_t* const*,const std::uint8_t*,double*,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t);
std::size_t edge_mask(float*,const unsigned char*,std::size_t,std::ptrdiff_t);
std::size_t edge_mask(double*,const unsigned char*,std::size_t,std::ptrdiff_t);
std::size_t edge_half(const std::uint16_t*,float*,std::size_t);
std::size_t edge_swap(const unsigned char*,unsigned char*,std::size_t,std::size_t);
std::size_t small_edge_column(const float*,float*,std::size_t,std::size_t,std::size_t,int);
std::size_t small_edge_column(const double*,double*,std::size_t,std::size_t,std::size_t,int);
std::size_t small_edge_column(const std::uint8_t*,double*,std::size_t,std::size_t,std::size_t,int);
std::size_t small_edge(const float*,const float*,const float*,float*,std::size_t,std::size_t,std::size_t,int);
std::size_t small_edge(const double*,const double*,const double*,double*,std::size_t,std::size_t,std::size_t,int);
std::size_t small_edge(const std::uint8_t*,const std::uint8_t*,const std::uint8_t*,double*,std::size_t,std::size_t,std::size_t,int);
#ifdef SKIMAGEXX_AVX2
std::size_t gray_expand_mapped(const unsigned char*,const unsigned char*,unsigned char*,std::size_t,std::size_t,int,std::ptrdiff_t,std::ptrdiff_t,const std::size_t*);
std::size_t gray_expand(const unsigned char*,const unsigned char*,unsigned char*,std::size_t,std::size_t,int,std::ptrdiff_t,std::ptrdiff_t,std::size_t);
std::size_t rgba_composite(const void*,void*,std::size_t,PixelType,const double*,std::ptrdiff_t,std::ptrdiff_t);
std::size_t local_prefix_columns(double*,double*,std::size_t,std::size_t,std::size_t);
std::size_t local_threshold_rows(const double* const*,const double* const*,const int*,std::size_t,void*,std::size_t,std::size_t,bool,bool,LocalThreshold,const LocalThresholdParameters&);
std::size_t threshold_logs(const float*,float*,std::size_t);
// First two plateau-aware peaks; count saturates at three.
std::size_t minimum_histogram_peaks(const float*,std::size_t,std::size_t*);
std::size_t histogram_edges(const float*,std::size_t,const float*,std::size_t,std::uint64_t*);
std::size_t histogram_edges(const double*,std::size_t,const double*,std::size_t,std::uint64_t*);
std::size_t metric_distance(const void*,const void*,void*,std::size_t,PixelType,PixelType,DistanceMetric,
    const DistanceParameters&,bool,bool,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t);
std::size_t delta_e76(const void*,const void*,float*,std::size_t,PixelType,PixelType,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t);
std::size_t delta_e76(const void*,const void*,double*,std::size_t,PixelType,PixelType,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t,std::ptrdiff_t);
std::size_t stain_transform(const void*,double*,std::size_t,PixelType,const detail::StainSpec&,std::ptrdiff_t,std::ptrdiff_t);
bool boolean_pack(const std::uint8_t*,std::uint64_t*,std::size_t,std::size_t);
std::size_t boolean_unpack(const std::uint64_t*,std::uint8_t*,std::size_t);
std::size_t boolean_combine(const std::uint64_t*,const std::uint64_t*,std::uint64_t*,std::size_t,bool);
std::size_t boolean_shift_reduce(std::uint64_t*,std::size_t,std::size_t,bool);
bool morphology_has_nan(const void*, std::size_t, PixelType);
std::size_t morphology_difference(const void*,const void*,void*,std::size_t,PixelType,bool&,bool&);
std::size_t morphology_line(const void* const*, std::size_t, void*, std::size_t, PixelType, bool, double, bool);
std::size_t perceptual_transform(const void*, void*, std::size_t, PixelType, const detail::PerceptualSpec&,
                                std::ptrdiff_t, std::ptrdiff_t, PerceptualInfo&);
std::size_t color_transform(const void*, void*, std::size_t, PixelType, const detail::ColorSpec&,
                           std::ptrdiff_t pixel_stride, std::ptrdiff_t channel_stride);
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
