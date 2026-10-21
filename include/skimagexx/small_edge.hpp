#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "convert.hpp"
namespace skimagexx {
enum class SmallEdge { laplace, roberts, roberts_positive, roberts_negative };
// Nonempty contiguous 2-D arrays; reflected borders; no overlap. Float32
// convolution accumulates in double then rounds, matching scipy.ndimage.
template<class T> void small_edge_into(const T*,T*,std::size_t,std::size_t,SmallEdge);
void small_edge_into(const std::uint8_t*,double*,std::size_t,std::size_t,SmallEdge);
extern template void small_edge_into(const float*,float*,std::size_t,std::size_t,SmallEdge);
extern template void small_edge_into(const double*,double*,std::size_t,std::size_t,SmallEdge);
// Sparse N-D Laplacian in scipy's coefficient order, including singleton axes.
template<class T> void laplace_nd_into(const T*,T*,const std::vector<std::size_t>&);
void laplace_nd_into(const std::uint8_t*,double*,const std::vector<std::size_t>&);
extern template void laplace_nd_into(const float*,float*,const std::vector<std::size_t>&);
extern template void laplace_nd_into(const double*,double*,const std::vector<std::size_t>&);
// Exact binary16 decoding, without floating arithmetic or NaN quieting.
void edge_half_to_float(const std::uint16_t*,float*,std::size_t);
void edge_swap_into(const void*,void*,std::size_t,std::size_t);
// Complete numeric C++ entry points: normalized integers/bools yield float64;
// float32 and binary16 yield float32. Buffers must be aligned and not overlap.
void small_edge_numeric_into(const void*,void*,const std::vector<std::size_t>&,PixelType,SmallEdge);
void small_edge_half_into(const std::uint16_t*,float*,const std::vector<std::size_t>&,SmallEdge);
// Contiguous floating output; mask row offsets and pixel stride are bytes.
void edge_mask_rows_into(void*,bool,const unsigned char*,std::size_t,std::size_t,const std::ptrdiff_t*,std::ptrdiff_t);
}
