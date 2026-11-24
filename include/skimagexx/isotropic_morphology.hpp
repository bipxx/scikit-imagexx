#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

enum class IsotropicOperation { erosion, dilation, opening, closing };

// Binary Euclidean morphology on a nonempty C-order image. Nonzero input bytes
// are true. Aligned input/output spans must be complete and disjoint, including
// the shape and spacing arrays. Rank is 1 to 3, axes are at most 65536, the
// image has at most 67108864 elements and finite sampling is in [1/65536, 65536].
// Call under round-to-nearest. The radius may be finite, infinite or quiet NaN.
// The output bytes are 0 or 1. No internal worker threads are used.
void isotropic_morphology_into(const std::uint8_t* image, std::size_t image_count,
                               std::size_t rank, const std::size_t* shape,
                               const double* spacing, double radius, IsotropicOperation operation,
                               std::uint8_t* output, std::size_t output_count);

}  // namespace skimagexx
