#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace skimagexx {
struct FootprintCross {std::ptrdiff_t vertical,horizontal;std::size_t repetitions;};
std::array<std::uint8_t,3> footprint_sphere_counts(int dimensions,std::ptrdiff_t radius);
std::vector<FootprintCross> footprint_cross_sums(const std::ptrdiff_t* sums,std::size_t count,std::size_t height);
std::vector<FootprintCross> footprint_cross_costs(const double* costs,std::size_t count,double limit);
// C-contiguous reference-free atoms. kind 0=T, 1=diamond, 2=cross.
// T/diamond dimensions are 2 or 3, all extents 3. Cross extents are
// 2*vertical+1 and 2*horizontal+1, and a zero-length arm is not filled.
void footprint_element_into(void* output,std::size_t itemsize,const void* zero,const void* one,
                            int kind,int dimensions,int axis,int direction,
                            std::size_t vertical=0,std::size_t horizontal=0);
}
