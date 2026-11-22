#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace skimagexx {

inline constexpr std::size_t match_descriptors_max_rows = 1024;
inline constexpr std::size_t match_descriptors_max_width = 256;

struct DescriptorMatchOptions {
    double max_distance = std::numeric_limits<double>::infinity();
    bool cross_check = true;
    double max_ratio = 1.0;
};

using DescriptorMatch = std::array<std::intptr_t, 2>;

// Each input points to rows * width accessible contiguous bytes. Values
// must be canonical Boolean bytes. Unsupported calls leave matches intact.
bool match_descriptors_bool(const std::uint8_t* left, std::size_t left_rows,
                            const std::uint8_t* right, std::size_t right_rows,
                            std::size_t width, const DescriptorMatchOptions& options,
                            std::vector<DescriptorMatch>& matches);

} // namespace skimagexx
