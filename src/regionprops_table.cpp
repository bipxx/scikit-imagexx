#include <skimagexx/regionprops_table.hpp>

#include <algorithm>
#include <cstring>
#include <utility>

namespace skimagexx {
namespace {

struct Bounds {
    std::uint16_t top;
    std::uint16_t left;
    std::uint16_t bottom;
    std::uint16_t right;
};

std::int32_t label_at(const unsigned char* row, std::size_t column) noexcept {
    std::int32_t label;
    std::memcpy(&label, row + column * sizeof(label), sizeof(label));
    return label;
}

}  // namespace

bool regionprops_table_bounds(const void* labels, std::size_t height, std::size_t width,
                              std::vector<RegionPropsTableRow>& rows) {
    if (!labels || !height || !width || height > regionprops_table_max_extent
        || width > regionprops_table_max_extent) {
        return false;
    }

    std::array<Bounds, regionprops_table_max_label + 1> bounds{};
    const auto* image = static_cast<const unsigned char*>(labels);
    std::size_t count = 0;
    std::int32_t highest = 0;
    for (std::size_t y = 0; y < height; ++y) {
        const auto* row = image + y * width * sizeof(std::int32_t);
        for (std::size_t x = 0; x < width;) {
            const auto label = label_at(row, x);
            if (label < 0 || label > regionprops_table_max_label) {
                return false;
            }
            const auto first = x++;
            while (x < width && label_at(row, x) == label) {
                ++x;
            }
            if (label == 0) {
                continue;
            }

            auto& b = bounds[static_cast<std::size_t>(label)];
            const auto left = static_cast<std::uint16_t>(first);
            const auto right = static_cast<std::uint16_t>(x);
            const auto bottom = static_cast<std::uint16_t>(y + 1);
            if (b.right == 0) {
                b = {static_cast<std::uint16_t>(y), left, bottom, right};
                ++count;
                highest = std::max(highest, label);
            } else {
                b.left = std::min(b.left, left);
                b.right = std::max(b.right, right);
                b.bottom = bottom;
            }
        }
    }

    std::vector<RegionPropsTableRow> result;
    result.reserve(count);
    for (std::int32_t label = 1; label <= highest; ++label) {
        const auto& b = bounds[static_cast<std::size_t>(label)];
        if (b.right != 0) {
            result.push_back({label, {b.top, b.left, b.bottom, b.right}});
        }
    }
    rows = std::move(result);
    return true;
}

}  // namespace skimagexx
