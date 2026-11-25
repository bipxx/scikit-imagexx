#include "skimagexx/brief.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace skimagexx {
namespace {

struct OffsetPair {
    std::int64_t first;
    std::int64_t second;
};

std::int64_t sample_index(std::int64_t row, std::int64_t col,
                          std::int64_t rows, std::int64_t cols) {
    // The original array accessor wraps negative indices and uses row-major
    // addressing. Guard the resulting access, including the last image row.
    if (row < 0) row += rows;
    if (col < 0) col += cols;
    if (row < 0 || row >= rows || col < 0 || col >= (rows - row) * cols)
        throw std::out_of_range("BRIEF sample is outside image storage");
    return row * cols + col;
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
    std::vector<OffsetPair> offsets(descriptor_size);
    std::int64_t min_row = 0, max_row = 0, min_col = 0, max_col = 0;
    for (std::size_t i = 0; i < descriptor_size; ++i) {
        const auto fr = std::int64_t(first[2 * i]);
        const auto fc = std::int64_t(first[2 * i + 1]);
        const auto sr = std::int64_t(second[2 * i]);
        const auto sc = std::int64_t(second[2 * i + 1]);
        // Only offsets with a possible in-image center need a linear offset.
        if (fr > -height && fr < height && sr > -height && sr < height &&
            fc > -width && fc < width && sc > -width && sc < width)
            offsets[i] = {fr * width + fc, sr * width + sc};
        min_row = std::min({min_row, fr, sr});
        max_row = std::max({max_row, fr, sr});
        min_col = std::min({min_col, fc, sc});
        max_col = std::max({max_col, fc, sc});
    }

    for (std::size_t point = 0; point < keypoint_count; ++point) {
        const auto row = keypoints[2 * point];
        const auto col = keypoints[2 * point + 1];
        if (row < -height || row >= height || col < -width || col >= width)
            throw std::out_of_range("BRIEF keypoint is outside image storage");
        if (row >= -min_row && row < height - max_row &&
            col >= -min_col && col < width - max_col) {
            const auto* center = image + row * width + col;
            for (const auto& offset : offsets)
                *descriptors++ = center[offset.first] < center[offset.second];
        } else {
            for (std::size_t i = 0; i < descriptor_size; ++i) {
                const auto a = sample_index(row + first[2 * i], col + first[2 * i + 1], height, width);
                const auto b = sample_index(row + second[2 * i], col + second[2 * i + 1], height, width);
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
