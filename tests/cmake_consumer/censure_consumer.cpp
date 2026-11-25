#include "skimagexx/censure.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

int main() {
    constexpr std::size_t rows = 45;
    constexpr std::size_t columns = 45;
    std::array<double, rows * columns> image;
    image.fill(1.0);
    for (int row = -4; row <= 4; ++row) {
        for (int column = -4; column <= 4; ++column) {
            if (row * row + column * column <= 16) {
                image[static_cast<std::size_t>(22 + row) * columns
                      + static_cast<std::size_t>(22 + column)] = 0.0;
            }
        }
    }
    for (const auto mode : {skimagexx::CensureMode::dob, skimagexx::CensureMode::octagon,
                            skimagexx::CensureMode::star}) {
        skimagexx::CensureOptions options;
        options.max_scale = 4;
        options.mode = mode;
        const auto results = skimagexx::censure_detect(image.data(), rows, columns, options);
        if (!results || results->size() != 1 || (*results)[0].row != 22 || (*results)[0].column != 22
            || (*results)[0].scale != 3) {
            return 1;
        }
    }
    image[0] = std::numeric_limits<double>::infinity();
    if (skimagexx::censure_detect(image.data(), rows, columns)) {
        return 2;
    }
    try {
        skimagexx::censure_detect(image.data(), rows, 3);
        return 3;
    } catch (const std::invalid_argument&) {
        return 0;
    }
}
