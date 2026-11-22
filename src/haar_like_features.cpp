// Rectangle layouts and inclusion-exclusion follow scikit-image 0.26.0.
// See third_party/scikit-image-LICENSE.txt.
#include "skimagexx/haar_like_features.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {

struct Layout { int rows, columns, blocks; };
Layout layout(HaarKind kind) {
    switch (kind) {
    case HaarKind::two_x: return {1, 2, 2};
    case HaarKind::two_y: return {2, 1, 2};
    case HaarKind::three_x: return {1, 3, 3};
    case HaarKind::three_y: return {3, 1, 3};
    case HaarKind::four: return {2, 2, 4};
    }
    throw std::invalid_argument("Invalid Haar feature type");
}

std::uint64_t placements(int length, int blocks) {
    const std::uint64_t steps = length / blocks;
    return steps * (length + 1ULL) - blocks * (steps * (steps + 1) / 2);
}

template <typename Visit>
void visit_features(int width, int height, HaarKind kind, Visit visit) {
    const auto tiles = layout(kind);
    for (int row = 0; row <= height - tiles.rows; ++row) {
        for (int column = 0; column <= width - tiles.columns; ++column) {
            for (int dy = 1; dy <= (height - row) / tiles.rows; ++dy) {
                for (int dx = 1; dx <= (width - column) / tiles.columns; ++dx) {
                    HaarLikeFeature feature{kind, {}};
                    for (int block = 0; block < tiles.blocks; ++block) {
                        const int y = row + (block / tiles.columns) * dy;
                        const int x = column + (kind == HaarKind::four && block >= 2
                                                ? 3 - block : block % tiles.columns) * dx;
                        feature.rectangles[block] = {y, x, y + dy - 1, x + dx - 1};
                    }
                    visit(feature, tiles.blocks);
                }
            }
        }
    }
}

void validate_storage(const void* image, const void* output, int width, int height, std::size_t count) {
    constexpr auto limit = std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t);
    if (width < 0 || height < 0 || (height > 0 && static_cast<std::size_t>(width) > limit / height)) {
        throw std::invalid_argument("Invalid Haar image shape");
    }
    if (count && (!image || !output || image == output || width == 0 || height == 0)) {
        throw std::invalid_argument("Invalid Haar input or output buffer");
    }
}

template <typename T>
std::uint64_t signed_word(T word) {
    std::make_signed_t<T> signed_value;
    std::memcpy(&signed_value, &word, sizeof(word));
    return static_cast<std::uint64_t>(signed_value);
}

template <typename T>
T rectangle_sum(const T* image, int width, const HaarLikeRectangle& box) {
    const auto bottom = static_cast<std::size_t>(box.bottom) * width;
    T sum = image[bottom + box.right];
    if constexpr (std::is_floating_point_v<T>) {
        // The first addition to zero is observable with negative zero.
        sum = T{0} + sum;
    }
    if (box.top) {
        const auto above = static_cast<std::size_t>(box.top - 1) * width;
        if (box.left) {
            sum = static_cast<T>(sum + image[above + box.left - 1]);
        }
        sum = static_cast<T>(sum - image[above + box.right]);
    }
    if (box.left) {
        sum = static_cast<T>(sum - image[bottom + box.left - 1]);
    }
    return sum;
}

template <typename T>
void evaluate(const T* image, int width, const HaarLikeFeature& feature, int blocks, void* output,
              std::size_t index) {
    using Result = std::conditional_t<std::is_floating_point_v<T>, T, std::uint64_t>;
    Result negative = 0, positive = 0;
    for (int i = 0; i < blocks; ++i) {
        const T value = rectangle_sum(image, width, feature.rectangles[i]);
        auto& sum = i % 2 ? positive : negative;
        if constexpr (std::is_floating_point_v<T>) {
            sum += value;
        } else {
            sum += signed_word(value);
        }
    }
    const Result result = positive - negative;
    std::memcpy(static_cast<unsigned char*>(output) + index * sizeof(Result), &result, sizeof(Result));
}

template <typename Visit>
void visit_pixel_type(PixelType type, const void* image, Visit visit) {
    switch (type) {
    case PixelType::boolean:
    case PixelType::uint8:
    case PixelType::int8: return visit(static_cast<const std::uint8_t*>(image));
    case PixelType::uint16:
    case PixelType::int16: return visit(static_cast<const std::uint16_t*>(image));
    case PixelType::uint32:
    case PixelType::int32: return visit(static_cast<const std::uint32_t*>(image));
    case PixelType::uint64:
    case PixelType::int64: return visit(static_cast<const std::uint64_t*>(image));
    case PixelType::float32: return visit(static_cast<const float*>(image));
    case PixelType::float64: return visit(static_cast<const double*>(image));
    }
    throw std::invalid_argument("Invalid Haar pixel type");
}

} // namespace

std::size_t haar_rectangle_count(HaarKind kind) {
    return layout(kind).blocks;
}

std::size_t haar_feature_count(int width, int height, HaarKind kind) {
    const auto tiles = layout(kind);
    if (width < tiles.columns || height < tiles.rows) {
        return 0;
    }
    const auto x = placements(width, tiles.columns);
    const auto y = placements(height, tiles.rows);
    if (x > std::numeric_limits<std::size_t>::max() / y) {
        throw std::overflow_error("Haar feature count exceeds addressable storage");
    }
    return static_cast<std::size_t>(x * y);
}

void haar_coordinates(int width, int height, HaarKind kind, HaarLikeFeature* output, std::size_t count) {
    if (count != haar_feature_count(width, height, kind) || (count && !output)) {
        throw std::invalid_argument("Invalid Haar coordinate output buffer");
    }
    if (count) {
        visit_features(width, height, kind, [&](const HaarLikeFeature& feature, int) { *output++ = feature; });
    }
}

void haar_generated_values(const void* image, PixelType type, int width, int height, HaarKind kind,
                           void* output, std::size_t count) {
    validate_storage(image, output, width, height, count);
    if (count != haar_feature_count(width, height, kind)) {
        throw std::invalid_argument("Invalid Haar value output buffer");
    }
    if (count) {
        visit_pixel_type(type, image, [&](const auto* data) {
            std::size_t index = 0;
            visit_features(width, height, kind, [&](const HaarLikeFeature& feature, int blocks) {
                evaluate(data, width, feature, blocks, output, index++);
            });
        });
    }
}

void haar_selected_values(const void* image, PixelType type, int width, int height,
                          const HaarLikeFeature* features, void* output, std::size_t count) {
    validate_storage(image, output, width, height, count);
    if (count && !features) {
        throw std::invalid_argument("Missing Haar rectangles");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const auto blocks = haar_rectangle_count(features[i].kind);
        for (std::size_t block = 0; block < blocks; ++block) {
            const auto& box = features[i].rectangles[block];
            if (box.top < 0 || box.top >= height || box.bottom < 0 || box.bottom >= height ||
                box.left < 0 || box.left >= width || box.right < 0 || box.right >= width) {
                throw std::invalid_argument("Haar rectangle lies outside the integral image ROI");
            }
        }
    }
    visit_pixel_type(type, image, [&](const auto* data) {
        for (std::size_t i = 0; i < count; ++i) {
            evaluate(data, width, features[i], static_cast<int>(haar_rectangle_count(features[i].kind)), output, i);
        }
    });
}

} // namespace skimagexx
