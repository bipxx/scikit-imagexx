#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
constexpr std::size_t ellipsoid_max_elements = 16*1024*1024;
struct DrawEllipsoidPlan {
    std::array<std::size_t,3> shape{};
    std::size_t count = 0;
    std::array<double,3> axes{},spacing{},low{},high{};
    // Owned source-ordered ((index*spacing + low)/axis)^2, independently
    // reusable across voxels. No borrowed operands/dense coordinate grids.
    std::array<std::vector<double>,3> squared;
};
// Native scalar boundary: three finite positive double axes/spacings in
// [2^-128,2^128], ordinary nearest rounding; count<=ellipsoid_max_elements.
// Refusal leaves plan unchanged; Python scalar/protocol/error/truth and FP
// policy handling are separate frontend gates. Prepared shape may be EVEN.
// Stages: ceil(-axis-spacing), floor((axis+spacing)+1), parity low shift;
// exact arange probe with second=(low+spacing), effective_step=second-low;
// zero-missing correction from maximum negative probe value; mgrid uses the
// NOMINAL spacing, not the probe's effective step, and fresh ceil shape.
bool draw_ellipsoid_prepare(const std::array<double,3>& axes,
    const std::array<double,3>& spacing,DrawEllipsoidPlan& output);
// Caller owns aligned/disjoint contiguous C output of plan.count elements.
// Bool is canonical uint8 0/1. Per voxel: (x_square+y_square)+z_square, then
// <=1 for bool OR subtract1 for levelset, no reassociation/FMA. The caller
// snapshots the existing global budget once (global default1); independent
// row spans optionally parallelize only past existing execution threshold.
void draw_ellipsoid_bool_into(const DrawEllipsoidPlan&,std::uint8_t*,int workers);
void draw_ellipsoid_levelset_into(const DrawEllipsoidPlan&,double*,int workers);
}
