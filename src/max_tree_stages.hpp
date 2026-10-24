#pragma once
// Private staged protocol surface; not an installed SDK header. The frozen
// aggregate max-tree-prepared-v1 interface remains unchanged.
#include "skimagexx/max_tree.hpp"

namespace skimagexx::max_tree_stages {
inline constexpr const char* interface = "max-tree-stages-v1";
inline constexpr unsigned capabilities = 31; // build/area/extension/filter/maxima
struct Image { const void* data; MaxTreeType type; std::size_t elements; };
struct Tree {
    MaxTreeSpan<const std::int64_t> parent, traverser;
};
struct Coordinates {
    double* data;
    std::size_t elements, rank;
    std::size_t row_stride, column_stride; // ELEMENTS; dense C or F (N,rank)
};
// Every span is native, naturally aligned and backed by an allocation of its
// declared count. All active source/output/workspace extents are disjoint.
// Empty inputs and nonfinite image pixels are unsupported. All guards and
// any internal validation allocation finish before caller-buffer writes.
// Source consumer stages use the supplied bounded index order and current
// input/attribute/output values, rather than rebuilding a different tree.
// Initial area/extension/maxima/output and coordinate arrays are literal
// provider results: no implicit reset or allocation substitutes their values.
// Stages remain serial, matching source floating and label association.
void build(Image, MaxTreeSpan<const std::uint8_t> mask,
    MaxTreeSpan<const std::int32_t> structure,
    MaxTreeSpan<const std::int32_t> shape, MaxTreeSpan<const std::int32_t> points,
    MaxTreeSpan<std::int64_t> parent, MaxTreeSpan<const std::int64_t> traverser,
    MaxTreeSpan<std::int64_t> roots);
void area(Image, Tree, MaxTreeSpan<double> attribute);
void extension(Image, MaxTreeSpan<const std::int32_t> shape, Tree,
    MaxTreeSpan<double> attribute, Coordinates minimum, Coordinates maximum);
void direct_filter(Image, MaxTreeBuffer output, Tree,
    MaxTreeSpan<const double> attribute, double threshold);
void local_maxima(Image, MaxTreeSpan<std::uint64_t> output, Tree);
} // namespace skimagexx::max_tree_stages
