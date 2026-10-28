#pragma once
#include "skimagexx/level_set.hpp"

namespace skimagexx::simd {
void level_set_checker_row(const std::int8_t* pattern, std::int8_t* output,
                            std::size_t count, bool invert);
std::size_t level_set_disk_row(const double* last_axis, std::int8_t* output,
                               std::size_t count, double prefix, double radius);
}
