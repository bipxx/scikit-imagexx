#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// Bounded, serial unit-vote Hough transform. Each image axis is in [1, 4096]
// with at most 1048576 pixels. There are 1 to 1024 angles and at most
// 134217728 pixel/angle pairs.
// Throws std::invalid_argument for a shape outside these bounds.
int hough_line_offset(std::size_t height, std::size_t width, std::size_t angles);

// Supply a C-order byte mask and one cosine and sine per angle, with absolute
// projections bounded by the offset on the image. Nonzero bytes vote once.
// The aligned, caller-owned output has (2 * offset + 1) * angles elements,
// distance first and angle second. All spans must be complete. Output must
// be disjoint from the inputs. Call under round-to-nearest and keep input
// storage stable until return. Validation precedes any output write.
void hough_line_into(const std::uint8_t* image, std::size_t image_count,
                     std::size_t height, std::size_t width,
                     const double* cosines, const double* sines, std::size_t angles,
                     std::uint64_t* accumulator, std::size_t accumulator_count);

}  // namespace skimagexx
