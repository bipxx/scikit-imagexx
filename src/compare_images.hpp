#pragma once
#include "skimagexx/convert.hpp"
#include <cstddef>
namespace skimagexx::detail {
enum class CompareMethod { diff, blend, checkerboard };
struct CompareInput { const void* data; PixelType type; };
struct CompareGrid {
    std::size_t rows=0,columns=0,row_step=0,column_step=0,row_tiles=0,column_tiles=0;
};
// Prepared native aligned C f32/f64 inputs and disjoint fresh logical C output.
// Diff/blend output is f64 if either input is f64, otherwise f32. Checker output
// has first-input dtype. Python validates tile protocols before normalizing any
// nonpositive/zero-step grid to zero tile counts. Positive tile coverage fits
// rows/columns. Scalar public boxing and output layout belong to the caller.
// False discards output and requests literal NumPy math. Floating environments
// and flags are restored; exceptional values/arithmetic never synthesize warnings.
bool compare_images_into(CompareInput first,CompareInput second,void* output,
    std::size_t count,CompareMethod,const CompareGrid&,int workers);
}
namespace skimagexx::simd {
bool compare_finite(const void*,std::size_t,PixelType);
std::size_t compare_span(const void*,const void*,void*,std::size_t,
    PixelType,PixelType,PixelType,detail::CompareMethod,bool& all_finite);
}
