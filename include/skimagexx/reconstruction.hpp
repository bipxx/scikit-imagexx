#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
// Private v1 ABI; no public registration or proved capability is implied.
inline constexpr const char* reconstruction_interface = "reconstruction-v1";
// Output precisions; mask float16/float32 -> float32, other real masks -> float64.
inline constexpr unsigned reconstruction_capabilities = 3;
enum class ReconstructionType { boolean, uint8, int8, uint16, int16, uint32,
    int32, uint64, int64, float16, float32, float64 };
enum class ReconstructionMethod { dilation, erosion };
enum class ReconstructionQueue { automatic, bucket8, bucket16, radix32, radix64, indexed_heap };
template<class T> struct ReconstructionSpan { T* data=nullptr; std::size_t size=0; };
struct ReconstructionView {
    const void* data=nullptr;                 // logical index0, possibly unaligned
    const void* allocation=nullptr;           // complete pinned readable allocation
    std::size_t allocation_bytes=0;           // BYTES, not logical sample count
    ReconstructionSpan<const std::size_t> shape;
    ReconstructionSpan<const std::ptrdiff_t> strides; // signed BYTE strides
    ReconstructionType type=ReconstructionType::float64;
    bool swapped=false;
};
struct ReconstructionFootprint {
    ReconstructionView values;               // boolean; rank must match images
    ReconstructionSpan<const std::size_t> offset; // COORDINATES, rank entries
};
struct ReconstructionZero {
    bool supplied=false;
    bool negative=false;
    // Source rank_order chooses one global representative of equal +/-0.
    // A caller-supplied sign is needed only when any converted value is -0.
    // Its provenance must be independently source-qualified; this API does
    // not silently normalize source-visible floating words.
};
struct ReconstructionRequirements {
    std::size_t samples=0, padded_samples=0, neighbor_capacity=0;
    std::size_t output_bytes=0, mask_bytes=0, bucket_heads=0; // 33/65 for radix
    ReconstructionType output_type=ReconstructionType::float64;
    ReconstructionQueue queue=ReconstructionQueue::indexed_heap;
};
struct ReconstructionBuffers {
    ReconstructionSpan<unsigned char> output; // padded C-order, native F32/F64
    ReconstructionSpan<unsigned char> mask;   // samples contiguous native F32/F64
    ReconstructionSpan<std::size_t> to_padded; // samples
    ReconstructionSpan<std::size_t> from_padded; // padded_samples, SIZE_MAX border
    ReconstructionSpan<std::size_t> queue;    // samples: heap nodes OR intrusive next
    ReconstructionSpan<std::size_t> position; // samples: heap positions OR intrusive prev
    ReconstructionSpan<std::size_t> heads;    // bucket_heads; unused heap span mayempty
    ReconstructionSpan<std::ptrdiff_t> neighbors; // neighbor_capacity ELEMENT offsets
};
struct ReconstructionInfo {
    ReconstructionQueue queue=ReconstructionQueue::indexed_heap;
    int preprocessing_workers=1;
    bool preprocessing_simd=false;            // actual AVX2 lane work, not eligibility
    bool parallel_flood=false;                // dependent flood remains serial
};
// Read-only source-order predicate for intake before footprint/provider work.
// Nonempty real typed views only; the Python shell preserves other preentry cases.
bool reconstruction_ordered(const ReconstructionView& seed,const ReconstructionView& mask,
    ReconstructionMethod);
// Read-only negative-zero intake scan; avoids a Python image scan on ordinary calls.
bool reconstruction_negative_zero(const ReconstructionView& seed,const ReconstructionView& mask);
// Metadata-only sizing. No allocation and no writes. Rank0 means one scalar;
// non-scalar dimensions must be positive. Overflow and footprint geometry
// are rejected; the Python caller preserves its earlier source preentry errors.
ReconstructionRequirements reconstruction_requirements(
    const ReconstructionView& seed,const ReconstructionView& mask,
    const ReconstructionFootprint&,ReconstructionQueue=ReconstructionQueue::automatic);
// All active spans, byte extents, dtype/shape/offsets, input ordering, finite
// conversion domain, and zero-profile requirements are validated BEFORE ANY
// output/workspace write. Nearest rounding and gradual underflow are required;
// caller FP state is restored on success/rejection. Inputs may alias each other; all active mutable
// regions must be disjoint from inputs and each other. No hidden heap storage.
// Output padding is initialized too: a returned interior view exposes this
// owner through .base. The caller owns all storage and return-view creation.
// Structural failures throw invalid_argument/overflow_error; unsupported FP
// domains throw domain_error before writes. No callback occurs inside this call.
// Every call snapshots the sole image.cpp global worker limit (default1).
ReconstructionInfo reconstruction_into(
    const ReconstructionView& seed,const ReconstructionView& mask,
    const ReconstructionFootprint&,ReconstructionMethod,
    const ReconstructionBuffers&,ReconstructionZero={},
    ReconstructionQueue=ReconstructionQueue::automatic);
}
