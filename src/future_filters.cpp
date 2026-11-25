#include "skimagexx/future_filters.hpp"
#include "skimagexx/draw_polygon.hpp"
#include "execution.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {
template<class T> std::vector<std::size_t> positive_indices(const void* labels, std::size_t count) {
    auto input = static_cast<const unsigned char*>(labels);
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < count; ++i) {
        T label;
        std::memcpy(&label, input + i * sizeof(T), sizeof(T));
        if (label > 0) selected.push_back(i);
    }
    return selected;
}
}

std::vector<std::size_t> segment_positive_indices(
    const void* labels, std::size_t count, SegmentLabel kind) {
    if (count && !labels) throw std::invalid_argument("missing segment labels");
    if (count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::uint64_t))
        throw std::invalid_argument("segment label extent overflow");
    switch (kind) {
        case SegmentLabel::boolean:
        case SegmentLabel::u8: return positive_indices<std::uint8_t>(labels, count);
        case SegmentLabel::i8: return positive_indices<std::int8_t>(labels, count);
        case SegmentLabel::i16: return positive_indices<std::int16_t>(labels, count);
        case SegmentLabel::u16: return positive_indices<std::uint16_t>(labels, count);
        case SegmentLabel::i32: return positive_indices<std::int32_t>(labels, count);
        case SegmentLabel::u32: return positive_indices<std::uint32_t>(labels, count);
        case SegmentLabel::i64: return positive_indices<std::int64_t>(labels, count);
        case SegmentLabel::u64: return positive_indices<std::uint64_t>(labels, count);
    }
    throw std::invalid_argument("invalid segment label dtype");
}

void segment_gather_rows(const std::vector<std::size_t>& indices,
    const void* features, std::size_t feature_bytes, const void* labels,
    std::size_t label_bytes, void* output_features, void* output_labels) {
    const auto* feature_input = static_cast<const unsigned char*>(features);
    const auto* label_input = static_cast<const unsigned char*>(labels);
    auto* feature_output = static_cast<unsigned char*>(output_features);
    auto* label_output = static_cast<unsigned char*>(output_labels);
    const auto bytes = feature_bytes + label_bytes;
    const int workers = bytes && indices.size() >= detail::parallel_min / bytes
        ? get_num_threads() : 1;
    detail::partition(indices.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t) {
        for (auto i = begin; i < end; ++i) {
            const auto source = indices[i];
            if (feature_bytes)
                std::memcpy(feature_output + i * feature_bytes,
                            feature_input + source * feature_bytes, feature_bytes);
            std::memcpy(label_output + i * label_bytes,
                        label_input + source * label_bytes, label_bytes);
        }
    });
}

void segment_stamp_polygon(const detail::DrawPolygonPlan& plan,
    std::size_t rows, std::size_t columns, std::intptr_t label,
    std::intptr_t* output) {
    if (!plan.count) return;
    const auto first_row = static_cast<std::size_t>(plan.first[0]);
    const auto first_column = static_cast<std::size_t>(plan.first[1]);
    const auto height = plan.shape[0];
    const auto width = plan.shape[1];
    const auto limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(*output);
    if (!output || (columns && rows > limit / columns) || first_row >= rows ||
        first_column >= columns || height > rows - first_row || width > columns - first_column ||
        !width || height > plan.labels.size() / width)
        throw std::invalid_argument("segment polygon extent exceeds destination");
    detail::partition(height, detail::active_workers(plan.labels.size()),
        [&](std::size_t begin, std::size_t end, std::size_t) {
            for (auto row = begin; row < end; ++row) {
                const auto* mask = plan.labels.data() + row * width;
                auto* destination = output + (row + first_row) * columns + first_column;
                for (std::size_t column = 0; column < width; ++column)
                    if (mask[column]) destination[column] = label;
            }
        });
}
} // namespace skimagexx
