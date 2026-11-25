#pragma once

#include <cstddef>

namespace skimagexx {

enum class ObjectDistanceNorm { manhattan, euclidean, chebyshev };

inline constexpr std::size_t object_distance_max_extent = 32768;
inline constexpr std::size_t object_distance_max_elements = 4194304;
inline constexpr std::size_t object_distance_max_label = 1048576;

// Filter a rank-1 to rank-3 C-order label image with nonempty axes. Zero is
// background. Labels with higher priority survive first, breaking ties by
// higher label ID. A null priority pointer with count zero uses object areas.
// Spacing entries are multiples of 1/16 in [1/16, 16]. The radius is finite
// in [0, 1048576]. Call under round-to-nearest with complete, aligned and
// disjoint input, output, shape, spacing and optional double priority spans.
// T is a fixed-width signed or unsigned integer (8, 16, 32, or 64 bits).
// Returns false without writing the output for unsupported labels or priority
// values. Invalid descriptors throw std::invalid_argument.
template<class T>
bool remove_objects_by_distance_into(
    const T* labels, std::size_t label_count, std::size_t rank,
    const std::size_t* shape, const double* spacing, double min_distance,
    ObjectDistanceNorm norm, const double* priority, std::size_t priority_count,
    T* output, std::size_t output_count);

}  // namespace skimagexx
