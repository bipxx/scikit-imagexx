#pragma once
#include <cstddef>

namespace skimagexx::detail {
struct RollingAxis {
    const double* coordinates;
    std::size_t count;
    double semi_axis;
};
struct RollingGrid {
    const RollingAxis* axes;
    std::size_t ndim;
    std::size_t count;
};
// Prepared boundary invariants: ndim in [1,32]; stable, naturally aligned,
// native-f64 contiguous axis spans; checked product of extents equals count.
// Output is disjoint contiguous f64 storage of count elements, private until
// acceptance. Ellipsoid semi axes are positive finite. workers is in [1,1024].
// Scalar coefficients are the actual source-stage values after promotion:
// radius_squared comes from the literal radius**2 operation, not native r*r.
// No Python callbacks, dtype conversion or profile/rank inference occurs here.
// false means discard the ENTIRE private output and use the literal provider.
// Non-nearest/FTZ/DAZ or exceptional arithmetic refuses, restoring caller FP
// state. Independent workers/SIMD lanes never change per-element association.
bool ball_kernel_into(const RollingGrid& grid, double radius,
                      double radius_squared, double* output, int workers);
bool ellipsoid_kernel_into(const RollingGrid& grid, double intensity,
                           double* output, int workers);
}
