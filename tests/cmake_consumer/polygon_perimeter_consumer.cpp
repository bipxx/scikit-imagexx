#include <skimagexx/polygon_perimeter.hpp>

#include <cfenv>

using namespace skimagexx::detail;

int main() {
    PerimeterCoordinates output;
    if (polygon_perimeter({{0, 0}, {2, 1}, {0, 2}}, {}, false, output) != PerimeterStatus::complete)
        return 1;
    if (output.rows != std::vector<std::intptr_t>{0, 1, 2, 2, 1, 0, 0, 0, 0}
        || output.columns != std::vector<std::intptr_t>{0, 1, 1, 1, 2, 2, 2, 1, 0}) return 2;

    const std::vector<PerimeterVertex> outside{{-2, 1}, {1, 4}, {4, 1}, {1, -2}};
    const std::array<std::intptr_t, 2> shape{3, 3};
    if (std::feraiseexcept(FE_DIVBYZERO) != 0) return 3;
    if (polygon_perimeter(outside, shape, true, output) != PerimeterStatus::complete) return 4;
    if (!(std::fetestexcept(FE_DIVBYZERO) & FE_DIVBYZERO)) return 5;
    if (output.rows != std::vector<std::intptr_t>{0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 0}
        || output.columns != std::vector<std::intptr_t>{0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0}) return 6;
    if (polygon_perimeter(outside, shape, false, output) != PerimeterStatus::complete) return 7;
    if (!output.rows.empty() || !output.columns.empty()) return 8;
    if (polygon_perimeter({{12, 12}, {12, 13}, {13, 12}}, shape, true, output)
        != PerimeterStatus::no_polygon) return 9;
    return 0;
}
