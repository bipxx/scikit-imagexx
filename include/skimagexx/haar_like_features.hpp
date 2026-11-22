#pragma once

#include <array>
#include <cstddef>
#include "convert.hpp"

namespace skimagexx {

enum class HaarKind { two_x, two_y, three_x, three_y, four };
struct HaarLikeRectangle { int top, left, bottom, right; };
struct HaarLikeFeature {
    HaarKind kind;
    std::array<HaarLikeRectangle, 4> rectangles;
};

std::size_t haar_rectangle_count(HaarKind kind);
std::size_t haar_feature_count(int width, int height, HaarKind kind);

// Each output buffer must have exactly count entries and must not alias its input.
// Image rows are contiguous and input/output are aligned for their pixel type.
// The output for integral and boolean input is int64; floating output keeps its width.
void haar_coordinates(int width, int height, HaarKind kind, HaarLikeFeature* output, std::size_t count);
void haar_generated_values(const void* image, PixelType type, int width, int height,
                           HaarKind kind, void* output, std::size_t count);
void haar_selected_values(const void* image, PixelType type, int width, int height,
                          const HaarLikeFeature* features, void* output, std::size_t count);

} // namespace skimagexx
