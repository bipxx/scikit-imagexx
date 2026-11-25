#pragma once

#include <skimagexx/polygon_perimeter.hpp>

namespace skimagexx::detail {

// For use_extent the second pair is a signed extent. Otherwise it is the
// inclusive end of the inner rectangle. The perimeter is outside that slice.
PerimeterStatus rectangle_perimeter(const std::array<std::intptr_t, 2>& start,
                                    const std::array<std::intptr_t, 2>& end_or_extent,
                                    bool use_extent, PerimeterShape shape, bool clip,
                                    PerimeterCoordinates& output);

} // namespace skimagexx::detail
