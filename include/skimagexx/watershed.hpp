#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct WatershedNeighborhood {
    std::vector<std::size_t> padding;
    std::vector<std::ptrdiff_t> offsets;
};

// Positive-rank contiguous C-order image, markers, optional mask, and output.
// Buffers have count elements, shape has rank elements, and output must not
// overlap the inputs. Nonzero mask bytes include a pixel. T is a fixed-width
// signed or unsigned integer. For boolean markers use uint8_t and set
// boolean_markers to normalize nonzero marker bytes to 1.
// A null neighborhood uses axis connectivity. An explicit neighborhood
// supplies symmetric padding and offsets in padded C-order search order.
// Callers must choose padding that accommodates each offset's coordinate
// displacement. The heap flood is ordered and runs serially.
template<class T>
void watershed_into(const double* image, const T* markers, const std::uint8_t* mask,
                    std::size_t count, const std::size_t* shape, std::size_t rank,
                    double compactness, bool watershed_line, T* output,
                    const WatershedNeighborhood* neighborhood = nullptr,
                    bool boolean_markers = false);

}  // namespace skimagexx
