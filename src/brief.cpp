#include "skimagexx/brief.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <stdexcept>

namespace skimagexx {
namespace {

struct OffsetPair {
    std::int64_t first;
    std::int64_t second;
};

std::int64_t sample_index(std::int64_t row, std::int64_t col,
                          std::int64_t rows, std::int64_t cols, std::int64_t size) {
    // The original array accessor wraps negative indices and uses row-major
    // addressing. Guard the resulting access, including the last image row.
    if (row < 0) row += rows;
    if (col < 0) col += cols;
    if (row < 0 || row >= rows || col < 0)
        throw std::out_of_range("BRIEF sample is outside image storage");
    const auto start = row * cols;
    if (col >= size - start)
        throw std::out_of_range("BRIEF sample is outside image storage");
    return start + col;
}

template <typename Pixel>
void sample(const Pixel* image, std::size_t rows, std::size_t cols,
            const std::int64_t* keypoints, std::size_t keypoint_count,
            const std::int32_t* first, const std::int32_t* second,
            std::size_t descriptor_size, std::uint8_t* descriptors) {
    if (keypoint_count == 0 || descriptor_size == 0) return;
    constexpr auto limit = std::numeric_limits<std::int64_t>::max() / 2;
    if (!image || !keypoints || !first || !second || !descriptors ||
        rows == 0 || cols == 0 || cols > static_cast<std::uint64_t>(limit) ||
        rows > static_cast<std::uint64_t>(limit) / cols)
        throw std::invalid_argument("invalid BRIEF image or buffers");

    const auto height = static_cast<std::int64_t>(rows);
    const auto width = static_cast<std::int64_t>(cols);
    const auto size = height * width;
    std::int64_t min_row = 0, max_row = 0, min_col = 0, max_col = 0;
    for (std::size_t i = 0; i < descriptor_size; ++i) {
        const auto fr = std::int64_t(first[2 * i]);
        const auto fc = std::int64_t(first[2 * i + 1]);
        const auto sr = std::int64_t(second[2 * i]);
        const auto sc = std::int64_t(second[2 * i + 1]);
        min_row = std::min({min_row, fr, sr});
        max_row = std::max({max_row, fr, sr});
        min_col = std::min({min_col, fc, sc});
        max_col = std::max({max_col, fc, sc});
    }

    // Store the common 256-bit pattern without a heap allocation. Only
    // linearize offsets if the image can contain an entire pattern.
    std::array<OffsetPair, 256> local_offsets;
    std::unique_ptr<OffsetPair[]> large_offsets;
    auto* offsets = local_offsets.data();
    if (-min_row < height - max_row && -min_col < width - max_col) {
        if (descriptor_size > local_offsets.size()) {
            large_offsets.reset(new OffsetPair[descriptor_size]);
            offsets = large_offsets.get();
        }
        for (std::size_t i = 0; i < descriptor_size; ++i)
            offsets[i] = {std::int64_t(first[2 * i]) * width + first[2 * i + 1],
                          std::int64_t(second[2 * i]) * width + second[2 * i + 1]};
    }

    for (std::size_t point = 0; point < keypoint_count; ++point) {
        const auto row = keypoints[2 * point];
        const auto col = keypoints[2 * point + 1];
        if (row < -height || row >= height || col < -width || col >= width)
            throw std::out_of_range("BRIEF keypoint is outside image storage");
        if (row >= -min_row && row < height - max_row &&
            col >= -min_col && col < width - max_col) {
            const auto* center = image + row * width + col;
            for (std::size_t i = 0; i < descriptor_size; ++i)
                *descriptors++ = center[offsets[i].first] < center[offsets[i].second];
        } else {
            const auto* a_offset = first;
            const auto* b_offset = second;
            for (std::size_t i = 0; i < descriptor_size; ++i, a_offset += 2, b_offset += 2) {
                const auto a = sample_index(row + a_offset[0], col + a_offset[1], height, width, size);
                const auto b = sample_index(row + b_offset[0], col + b_offset[1], height, width, size);
                *descriptors++ = image[a] < image[b];
            }
        }
    }
}

} // namespace

void brief_sample(const float* image, std::size_t rows, std::size_t cols,
                  const std::int64_t* keypoints, std::size_t keypoint_count,
                  const std::int32_t* first, const std::int32_t* second,
                  std::size_t descriptor_size, std::uint8_t* descriptors) {
    sample(image, rows, cols, keypoints, keypoint_count, first, second, descriptor_size, descriptors);
}

void brief_sample(const double* image, std::size_t rows, std::size_t cols,
                  const std::int64_t* keypoints, std::size_t keypoint_count,
                  const std::int32_t* first, const std::int32_t* second,
                  std::size_t descriptor_size, std::uint8_t* descriptors) {
    sample(image, rows, cols, keypoints, keypoint_count, first, second, descriptor_size, descriptors);
}

} // namespace skimagexx
