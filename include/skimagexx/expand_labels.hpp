#pragma once

#include <cstddef>

namespace skimagexx {

// Expand a one-, two-, or three-dimensional label image in C order. T is a
// fixed-width signed or unsigned integer (8, 16, 32, or 64 bits). The caller
// supplies aligned, complete and disjoint input/output spans. The shape and
// sampling arrays have rank entries. Each axis is in [1, 65536] and the image
// has at most 67108864 elements. Sampling is finite in [1/65536, 65536].
// Zero is background. Equal-distance features prefer the lower coordinate
// on the last differing axis, using the pinned Euclidean feature recurrence.
// Pass negative distance or quiet NaN for no labels, positive infinity for
// unrestricted expansion. Call under round-to-nearest.
template<class T>
void expand_labels_into(const T* labels, std::size_t label_count,
                        std::size_t rank, const std::size_t* shape,
                        const double* sampling, double distance,
                        T* result, std::size_t result_count);

}  // namespace skimagexx
