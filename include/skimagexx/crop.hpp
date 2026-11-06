#pragma once
#include <cstddef>
#include <vector>
namespace skimagexx {
// Non-owning basic-slice crop. Caller keeps valid input/output storage alive.
// Stops use ordinary Python step-one slice bounds; the binder saturates Python
// index objects to ptrdiff_t before entry. No element/type conversion occurs.
struct CropSlice { std::ptrdiff_t start=0, stop=0; };
struct CropView {
    const void* data=nullptr;
    std::vector<std::ptrdiff_t> shape, strides;
    std::size_t itemsize=1;
};
struct CropPlan {
    CropView view;
    std::vector<CropSlice> axes;
};
CropPlan crop_view(const CropView&,const std::vector<CropSlice>&);
// Complete cropped payload copy into caller-owned, disjoint, non-overlapping
// writable output with matching cropped shape/itemsize. POD words only: object
// reference ownership belongs to the Python/NumPy boundary. Serial traversal.
void crop_copy_into(const CropView&,const std::vector<CropSlice>&,
                    void* output,const std::vector<std::ptrdiff_t>& output_strides);
}
