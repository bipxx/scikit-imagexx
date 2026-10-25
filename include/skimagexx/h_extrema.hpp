#pragma once
#include "reconstruction.hpp"
#include <cstddef>
#include <cstdint>

namespace skimagexx {
// Private composition ABI. This integer increment is not full H-extrema API
// completion; floating preparation and height protocols remain separate work.
inline constexpr const char* h_extrema_interface = "h-extrema-integer-v1";
struct HExtremaRange {
    // Raw input-width result of max-min. Signed callers interpret its sign bit
    // before the source's h > ptp test; widening that subtraction is incorrect.
    std::uint64_t bits=0;
    std::size_t samples=0;
};
struct HExtremaInfo {
    int workers=1;                  // selected worker limit for this scan
    bool simd=false;                // no new ISA-specific leaf in this increment
};
// Metadata-only allocation/shape/extent validation, with zero pixel reads.
// Used before output allocation and by safe intake rejection controls.
std::size_t h_extrema_validate_view(const ReconstructionView&);
// All eight signed/unsigned integer widths, arbitrary validated byte strides
// and byte order. Nonempty input only; rank0 denotes one sample. Shape/stride
// spans themselves must be caller-owned readable metadata.
HExtremaRange h_extrema_integer_range(const ReconstructionView&);
// Writes native-byte-order, contiguous C-order seed words of the input dtype.
// Positive/zero height must fit the full unsigned width. Saturation uses exact
// integer distances, never signed overflow. No source image is modified.
HExtremaInfo h_extrema_integer_seed_into(const ReconstructionView&,std::uint64_t height,
    ReconstructionMethod,ReconstructionSpan<unsigned char> output);
// Compare source-ordered float64 residue to the source-converted height and
// write an owning-caller uint8 mask. Reconstruction must be finite float64 and
// have the same shape. Source integer->float64 rounding is observable for i/u64.
// Nearest rounding and gradual underflow are required; caller FP state restored.
// Revalidate the *current* image after footprint/warning callbacks before call.
HExtremaInfo h_extrema_integer_mask_into(const ReconstructionView& image,
    const ReconstructionView& reconstructed,double height,ReconstructionMethod,
    ReconstructionSpan<unsigned char> output);
// Every function validates dimensions/allocation extent before input reads.
// Mutable output must be disjoint from inputs and their metadata; every failure
// is detected before writes. No hidden array allocation or Python callback.
// Seed scans use the sole SDK worker limit (default1); the residue scan is serial.
}
