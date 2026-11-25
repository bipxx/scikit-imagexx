#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace skimagexx {

struct SlicOptions {
    std::size_t n_segments = 100;
    double compactness = 10.;
    std::size_t max_num_iter = 10;
    std::array<double, 3> spacing{1., 1., 1.};
    bool enforce_connectivity = true;
    double min_size_factor = 0.5;
    double max_size_factor = 3.;
    bool slic_zero = false;
    std::ptrdiff_t start_label = 1;
};

// Interleaved, contiguous (Z,Y,X,C) input and contiguous (Z,Y,X) labels.
// For 2D, use a depth of one. The caller owns disjoint input and output
// buffers, sized by their element counts. The iteration and connectivity are
// ordered and serial. False indicates an unsupported numerical domain.
template<class T>
bool slic_into(const T* image, std::size_t image_size,
               const std::array<std::size_t, 4>& shape, const SlicOptions& options,
               std::ptrdiff_t* output, std::size_t output_size);

// Same algorithm for caller-prepared color features, already normalized,
// optionally color-converted/smoothed, and scaled by inverse compactness.
template<class T>
bool slic_prepared_into(const T* features, std::size_t image_size,
                        const std::array<std::size_t, 4>& shape, const SlicOptions& options,
                        std::ptrdiff_t* output, std::size_t output_size);

extern template bool slic_into(const std::uint8_t*, std::size_t,
    const std::array<std::size_t, 4>&, const SlicOptions&, std::ptrdiff_t*, std::size_t);
extern template bool slic_into(const float*, std::size_t,
    const std::array<std::size_t, 4>&, const SlicOptions&, std::ptrdiff_t*, std::size_t);
extern template bool slic_into(const double*, std::size_t,
    const std::array<std::size_t, 4>&, const SlicOptions&, std::ptrdiff_t*, std::size_t);
extern template bool slic_prepared_into(const float*, std::size_t,
    const std::array<std::size_t, 4>&, const SlicOptions&, std::ptrdiff_t*, std::size_t);
extern template bool slic_prepared_into(const double*, std::size_t,
    const std::array<std::size_t, 4>&, const SlicOptions&, std::ptrdiff_t*, std::size_t);

}
