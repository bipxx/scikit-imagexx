#pragma once

#include "skimagexx/image.hpp"

namespace skimagexx {

enum class SegmentLabel { boolean, i8, u8, i16, u16, i32, u32, i64, u64 };

// Native-endian contiguous labels. The returned indices are in C order.
std::vector<std::size_t> segment_positive_indices(
    const void* labels, std::size_t count, SegmentLabel kind);

// Copy disjoint, caller-owned feature and label outputs. Each index must
// reference a complete row in both input buffers.
void segment_gather_rows(const std::vector<std::size_t>& indices,
    const void* features, std::size_t feature_bytes, const void* labels,
    std::size_t label_bytes, void* output_features, void* output_labels);

// Merge a polygon into an existing label image. The draw polygon plan and
// destination must have matching bounds. When selections overlap, the caller
// applies this function in selection order.
namespace detail { struct DrawPolygonPlan; }
void segment_stamp_polygon(const detail::DrawPolygonPlan& plan,
    std::size_t rows, std::size_t columns, std::intptr_t label,
    std::intptr_t* output);

} // namespace skimagexx
