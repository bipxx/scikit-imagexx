#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace skimagexx::detail {

using PerimeterShape = std::optional<std::array<std::intptr_t, 2>>;

struct PerimeterVertex {
    double row, column;
};

struct PerimeterCoordinates {
    std::vector<std::intptr_t> rows, columns;
};

enum class PerimeterStatus { complete, outside_domain, no_polygon };

// Coordinates and shape entries with magnitude at most 2^26. The output is
// assigned only on success. No polygon means clipping produced no closed path.
// Small paths use ordered scalar clipping and the existing line recurrence.
PerimeterStatus polygon_perimeter(const std::vector<PerimeterVertex>& vertices,
                                  PerimeterShape shape, bool clip,
                                  PerimeterCoordinates& output);

} // namespace skimagexx::detail
