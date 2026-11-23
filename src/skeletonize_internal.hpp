#pragma once

#include "skimagexx/skeletonize.hpp"

namespace skimagexx::detail {

// Produce the validated skeleton with its zero border, keeping the destination
// allocation for the caller. False leaves the padded state unchanged.
bool skeletonize_zhang_padded(
    const std::uint8_t* image, std::size_t rows, std::size_t columns,
    SkeletonizeInput kind, std::vector<std::uint8_t>& padded);

// Copy the interior of a successfully computed padded state to a destination
// with storage for rows * columns bytes.
void copy_skeletonize_zhang_rows(
    const std::vector<std::uint8_t>& padded, std::size_t rows, std::size_t columns,
    std::uint8_t* destination);

}  // namespace skimagexx::detail
