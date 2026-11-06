#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
namespace skimagexx::detail {
// Borrowed disjoint mutable f64/f32 image, signed byte strides.
// Coordinate/color/alpha typed pointers are aligned contiguous arrays. Operand vectors may
// alias the image: all reads/validation and computed values precede any scatter.
struct SetColorView {
    void* image=nullptr;
    std::array<std::size_t,3> shape{};
    std::array<std::ptrdiff_t,3> strides{};
    const std::intptr_t* rows=nullptr;
    const std::intptr_t* columns=nullptr;
    std::size_t count=0;
    const double* color=nullptr;
    const double* alpha=nullptr;
    // Original color*alpha ufunc used int64. Coefficients are integral within
    // the existing [-16,16] frontier; products fit int64 and convert exactly.
    bool integer_product=false;
    // f32storage, f64originalRHS, once-roundedf32assignment. Selectedinput
    // andfinaloutput mustbezeroornormal; exceptionalcastsuffix staysprovider.
    bool image_float32=false;
};
// External typed boundary validates spans, nonoverlap, indices and bounded
// ordinary-FP operands once. False leaves image and FP context unchanged.
// Numerical stages reproduce NumPy f64 operations without contraction.
// Default1/global snapshot supplied by caller; independent compute may partition,
// final scatter is ordered serial (duplicate-coordinate last write wins).
bool draw_set_color(const SetColorView&,int workers=1);
}
