#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace skimagexx {
inline constexpr const char* max_tree_interface = "max-tree-prepared-v1";
inline constexpr unsigned max_tree_type_capabilities = 0x3ff; // ten prepared real types
inline constexpr unsigned max_tree_operation_capabilities = 0x3f; // private six-operation surface
inline constexpr bool max_tree_parallel_traversal = false;

enum class MaxTreeType { u8, u16, u32, u64, i8, i16, i32, i64, f32, f64 };
enum class MaxTreeOperation { tree, local_maxima, area_opening, area_closing,
                              diameter_opening, diameter_closing };
template<class T> struct MaxTreeSpan { T* data; std::size_t elements; };
struct MaxTreeBuffer { void* data; std::size_t elements; };
struct MaxTreeImage {
    const void* data;
    MaxTreeType type;
    std::size_t elements;
    std::size_t rank; // 2 or 3, dense C; unused shape entries ignored
    std::array<std::size_t,3> shape;
};
struct MaxTreeProvided {
    MaxTreeSpan<const std::int64_t> parent, traverser;
};
struct MaxTreeWorkspace {
    // When no provided tree is present, all four are exact N words. Parent
    // and traverser are also the tree operation's caller-owned results.
    MaxTreeSpan<std::int64_t> parent, traverser, roots, sort_scratch;
    MaxTreeBuffer inverted; // exact N input-type elements, closing only
    MaxTreeSpan<double> attribute; // exact N, area/diameter only
    // Exact N*rank words each, diameter only; C pixel-major coordinates.
    MaxTreeSpan<std::int64_t> minimum_coordinates, maximum_coordinates;
};
struct MaxTreeParameters {
    MaxTreeOperation operation;
    int connectivity; // radius-one 1..rank for building; ignored with supplied tree
    double threshold; // direct-filter comparison, including +/-Inf or NaN
};
struct MaxTreeResult {
    bool tree_reused;
    int preparation_workers; // actual independent preparation team
};

// Counts are ELEMENTS, not bytes. Every active span must describe a real
// allocation of its declared type, native and naturally aligned. Shape entries
// are >=3 and <=INT32_MAX; products/byte extents fit ptrdiff_t and signed64.
// Builder's raveled radius-one offsets must also fit source int32 storage.
// All active input/provided/workspace/output extents must be disjoint. No
// initial workspace contents are required; inactive buffers are not accessed.
// Source-proven reference parity presently covers dimensions>=4, not unsafe
// original tiny/empty shapes. Finite typed input and nearest/gradual FP only.
// No provided tree: parent/traverser/roots/scratch required. Provided tree:
// all four are inactive; input tree is readonly, exact N words, stable intensity
// permutation, one self-root, parent-before-child. Closing's supplied tree is
// for T(1)-image (float) or bitwise complement (integer), NOT the input tree.
// Tree operation forbids provided tree and has empty output. Local maxima
// output is exact N uint64 words; four filters output exact N input-type words.
// Area/diameter require attribute; diameter requires both coordinate spans.
// Closing requires inverted workspace. No hidden allocation or shared state.
// Every guard finishes before any source/workspace/output write. It restores
// caller FP status/control. Unsupported metadata returns a static reason.
const char* max_tree_ineligibility(const MaxTreeImage&, const MaxTreeParameters&,
    const MaxTreeWorkspace&, MaxTreeBuffer output,
    MaxTreeProvided provided = {{nullptr,0},{nullptr,0}}) noexcept;

// Rechecks admission before writes. Stable typed mergesort (ties in original
// flat-index order), descending union/find, ascending canonicalization; reverse
// F64 attributes and source-ordered maxima labels, then forward direct filter.
// Root below threshold emits typed zero; comparisons are attribute<threshold.
// Inversion/copies/coordinate preparation may use runtime SIMD/shared optional
// workers (default1). Dependency traversals and scalar association stay serial.
// Never mutates source/provided trees. No Python provider, error/warning text,
// arbitrary dtype/layout/protocol, public registration or performance claim.
MaxTreeResult max_tree_into(const MaxTreeImage&, const MaxTreeParameters&,
    const MaxTreeWorkspace&, MaxTreeBuffer output,
    MaxTreeProvided provided = {{nullptr,0},{nullptr,0}});
} // namespace skimagexx
