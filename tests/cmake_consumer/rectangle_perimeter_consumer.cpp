#include <skimagexx/rectangle_perimeter.hpp>

using namespace skimagexx::detail;

int main() {
    PerimeterCoordinates output;
    if (rectangle_perimeter({1, 1}, {2, 2}, false, {}, false, output) != PerimeterStatus::complete)
        return 1;
    if (output.rows != std::vector<std::intptr_t>{0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0}
        || output.columns != std::vector<std::intptr_t>{0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0})
        return 2;

    const std::array<std::intptr_t, 2> shape{3, 4};
    if (rectangle_perimeter({2, 3}, {-2, -2}, true, shape, true, output) != PerimeterStatus::complete)
        return 3;
    if (output.rows != std::vector<std::intptr_t>{0, 1, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0}
        || output.columns != std::vector<std::intptr_t>{3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0, 1, 2, 3}) return 4;
    if (rectangle_perimeter({10, 11}, {14, 20}, false, shape, true, output) != PerimeterStatus::no_polygon)
        return 5;
    if (output.rows.size() != 14 || output.columns.size() != 14) return 6;
    return 0;
}
