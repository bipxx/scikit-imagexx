#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace skimagexx::io {

// Copy one image to a disjoint, contiguous output in row-major order.
// Shape, byte strides, and item size describe a bounded source buffer.
void copy_image(const void* source, void* output, const std::ptrdiff_t* shape,
                const std::ptrdiff_t* strides, std::size_t rank, std::size_t item_size);

struct NetpbmImage {
    std::size_t width;
    std::size_t height;
    std::size_t channels;
    std::string_view pixels;
};

// Canonical binary P5/P6 with maxval 255 and one complete raster.
std::optional<NetpbmImage> netpbm_view(std::string_view bytes);
std::string netpbm_header(std::size_t width, std::size_t height, std::size_t channels);

enum class FeatureFormat { sift, surf };

struct FeatureLayout {
    std::size_t count;
    std::size_t descriptor_size;
    std::size_t values_per_record;
    std::string_view values;
};

// The supplied text owns the storage referenced by the returned view.
std::optional<FeatureLayout> feature_layout(std::string_view text, FeatureFormat format);
// Parse exactly size whitespace-separated values into caller-owned storage.
bool read_feature_values(std::string_view text, double* output, std::size_t size);

}  // namespace skimagexx::io
