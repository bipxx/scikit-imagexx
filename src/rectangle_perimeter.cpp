#include <skimagexx/rectangle_perimeter.hpp>

#include <algorithm>

namespace skimagexx::detail {

PerimeterStatus rectangle_perimeter(const std::array<std::intptr_t, 2>& start,
                                    const std::array<std::intptr_t, 2>& end_or_extent,
                                    bool use_extent, PerimeterShape shape, bool clip,
                                    PerimeterCoordinates& output) {
    constexpr std::intptr_t limit = std::intptr_t{1} << 26;
    std::array<double, 2> first, last;
    for (std::size_t axis = 0; axis < first.size(); ++axis) {
        const auto origin = start[axis], other = end_or_extent[axis];
        if (origin < -limit || origin > limit || other < -limit || other > limit)
            return PerimeterStatus::outside_domain;
        const auto end = use_extent ? origin + other : other;
        first[axis] = static_cast<double>(std::min(origin, end) - 1);
        last[axis] = static_cast<double>(std::max(origin, end) + !use_extent);
    }
    const auto [r0, c0] = first;
    const auto [r1, c1] = last;
    return polygon_perimeter({{r0, c0}, {r0, c1}, {r1, c1}, {r1, c0}, {r0, c0}}, shape, clip, output);
}

} // namespace skimagexx::detail
