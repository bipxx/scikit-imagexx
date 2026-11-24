#include <skimagexx/regionprops_table.hpp>

#include <array>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("regionprops_table SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::RegionPropsTableRow;
    using skimagexx::regionprops_table_bounds;
    using Box = std::array<std::intptr_t, 4>;

    const std::array<std::int32_t, 24> image{
        7, 0, 0, 3, 0, 7,
        0, 0, 0, 0, 0, 0,
        11, 0, 3, 0, 7, 0,
        0, 11, 0, 0, 0, 0,
    };
    std::vector<RegionPropsTableRow> rows;
    require(regionprops_table_bounds(image.data(), 4, 6, rows));
    require(rows.size() == 3);
    require(rows[0].label == 3 && rows[0].bbox == Box{0, 2, 3, 4});
    require(rows[1].label == 7 && rows[1].bbox == Box{0, 0, 3, 6});
    require(rows[2].label == 11 && rows[2].bbox == Box{2, 0, 4, 2});

    const std::array<std::int32_t, 2> too_high{1, skimagexx::regionprops_table_max_label + 1};
    const std::array<std::int32_t, 2> negative{1, -1};
    require(!regionprops_table_bounds(too_high.data(), 1, 2, rows));
    require(!regionprops_table_bounds(negative.data(), 1, 2, rows));
    require(!regionprops_table_bounds(image.data(), 0, 6, rows));
    require(!regionprops_table_bounds(nullptr, 4, 6, rows));
    require(!regionprops_table_bounds(image.data(), skimagexx::regionprops_table_max_extent + 1, 1, rows));
    require(rows.size() == 3 && rows[2].label == 11 && rows[2].bbox == Box{2, 0, 4, 2});

    const std::array<std::int32_t, 6> zeros{};
    require(regionprops_table_bounds(zeros.data(), 2, 3, rows));
    require(rows.empty());
}
