#pragma once
#include <skimagexx/image.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace skimagexx {
// Caller retains the source storage for the call. Strides are signed bytes;
// pointer denotes logical element zero, including reversed/broadcast views.
enum class GrayIntegerType { i8,u8,i16,u16,i32,u32,i64,u64 };
struct GrayImageView { const void* data; std::size_t rows,columns; std::ptrdiff_t row_stride,column_stride; GrayIntegerType type; bool swapped=false; bool writeable=true; };
struct GrayMatrix { std::size_t levels,distances,angles; bool normed; std::vector<std::uint32_t> counts; std::vector<double> probabilities; };
// Complete owning operation: count, optional uint32 symmetry, optional float64
// normalization. Finite offsets must round into ptrdiff_t. No image provider.
GrayMatrix graycomatrix(GrayImageView,const std::vector<double>& distances,const std::vector<double>& angles,std::size_t levels,bool symmetric=false,bool normed=false,int workers=0);
enum class GrayProperty { contrast,dissimilarity,homogeneity,ASM,energy,mean,variance,stddev,entropy,correlation,invalid };
enum class GrayNumericType { boolean,i8,u8,i16,u16,i32,u32,i64,u64,f16,f32,f64,c64,c128 };
struct GrayProbabilityView { const void* data; std::size_t levels,distances,angles; std::ptrdiff_t strides[4]; GrayNumericType type=GrayNumericType::f64; bool swapped=false; };
struct GrayProperties { std::size_t distances,angles;std::vector<double> values;bool fortran=false; };
// Complete owning operation includes a copy, normalization and the named
// property. Typed POD sources may be strided/swapped/nonfinite; source-ordered reductions
// remain serial within each independent distance/angle plane.
using GrayPolicyCallback = void(*)(const char* operation,unsigned flags,void* context);
GrayProperties graycoprops(GrayProbabilityView,GrayProperty,int workers=0,GrayPolicyCallback policy=nullptr,void* context=nullptr);
}
