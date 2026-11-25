#pragma once

#include <cstddef>

namespace skimagexx {

// Contiguous finite float or double images with the same shape. The caller's
// disjoint output stores row displacement first, then column displacement.
// Lucas-Kanade returns false if an active local system is too poorly
// conditioned or produces a nonfinite result. Discard the output on false.
template<class T>
bool optical_flow_ilk_into(const T* reference, const T* moving, T* flow,
                           std::size_t height, std::size_t width,
                           int radius = 7, int num_warp = 10,
                           bool gaussian = false, bool prefilter = false);

template<class T>
void optical_flow_tvl1_into(const T* reference, const T* moving, T* flow,
                            std::size_t height, std::size_t width,
                            double attachment = 15, double tightness = 0.3,
                            int num_warp = 5, int num_iter = 10,
                            double tol = 1e-4, bool prefilter = false);

} // namespace skimagexx
