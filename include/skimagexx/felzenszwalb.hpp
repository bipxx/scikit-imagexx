#pragma once

#include <cstddef>

namespace skimagexx {

// Positive, addressable image dimensions. Costs are grouped by right, down,
// down-right, and up-right edges in row-major order within each group.
std::ptrdiff_t felzenszwalb_edge_count(std::ptrdiff_t height, std::ptrdiff_t width);
void felzenszwalb_costs_into(const double* image, double* costs, std::ptrdiff_t height,
                             std::ptrdiff_t width, std::ptrdiff_t channels);

// order is a permutation of the edge indices sorted by their corresponding
// cost, with ties ordered according to the caller's sorting policy.
void felzenszwalb_labels_into(const double* costs, const std::ptrdiff_t* order,
                              std::ptrdiff_t* labels, std::ptrdiff_t height,
                              std::ptrdiff_t width, double scale, std::ptrdiff_t min_size);

} // namespace skimagexx
