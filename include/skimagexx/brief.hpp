#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// Compare the pixels at each pair of offsets from each keypoint (row, column).
// The image and all pair buffers are contiguous. The caller supplies room for
// keypoint_count * descriptor_size Boolean bytes, in keypoint-major order.
// Offsets that would access memory outside the image throw std::out_of_range.
void brief_sample(const float* image, std::size_t rows, std::size_t cols,
                  const std::int64_t* keypoints, std::size_t keypoint_count,
                  const std::int32_t* first, const std::int32_t* second,
                  std::size_t descriptor_size, std::uint8_t* descriptors);
void brief_sample(const double* image, std::size_t rows, std::size_t cols,
                  const std::int64_t* keypoints, std::size_t keypoint_count,
                  const std::int32_t* first, const std::int32_t* second,
                  std::size_t descriptor_size, std::uint8_t* descriptors);

} // namespace skimagexx
