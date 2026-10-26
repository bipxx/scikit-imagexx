#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace skimagexx {
// Private staged ABI. No installed/public or full-domain capability is implied.
inline constexpr const char* colocalization_interface = "colocalization-stages-v1";
inline constexpr std::size_t colocalization_max_rank = 64;
enum class ColocType { boolean, uint8, int8, uint16, int16, uint32, int32,
    uint64, int64, float16, float32, float64 };
enum class ColocProfile { numpy_1244, numpy_254 };
template<class T> struct ColocSpan { T* data=nullptr; std::size_t size=0; };
struct ColocView {
    const void* data=nullptr;                // logical index zero; may be unaligned
    const void* allocation=nullptr;          // pinned complete readable allocation
    std::size_t allocation_bytes=0;
    ColocSpan<const std::size_t> shape;
    ColocSpan<const std::ptrdiff_t> strides;  // signed BYTE strides
    ColocType type=ColocType::float64;
    bool swapped=false;
};
struct ColocBuffer { void* data=nullptr; std::size_t bytes=0; };
struct ColocOrder {
    std::array<std::size_t,colocalization_max_rank> axes{}; // fastest axis first
    std::size_t rank=0;
};
struct ColocReductionPlan {
    ColocProfile profile=ColocProfile::numpy_254;
    std::size_t buffer_elements=8192; // read at THIS source SUM stage
};
enum ColocFp : unsigned { coloc_divide=1, coloc_overflow=2,
    coloc_underflow=4, coloc_invalid=8 }; // NumPy status-bit order
struct ColocExecution { int worker_limit=1; bool simd=false; };
struct ColocScalar {
    ColocType type=ColocType::float64;
    std::uint64_t bits=0; // native-width low bits; not a conversion to double
    unsigned fp_status=0;
};
struct ColocPredicate { bool value=false; unsigned fp_status=0; };
// Explicit pre-write refusal. The binding translates ONLY this type into the
// named primitive-provider route; structural errors and all other exceptions
// propagate. It avoids duplicating full data scans before the actual kernel.
class ColocProviderRequired : public std::domain_error {
public:
    using std::domain_error::domain_error;
};

// Structural validation only: overflow-safe logical shape, signed extents and
// actual allocation containment. Empty views need no pixel pointer. Descriptor
// arrays must themselves be valid, live C++ objects owned by the caller.
std::size_t coloc_validate(const ColocView&);
std::size_t coloc_itemsize(ColocType);
bool coloc_fp_supported() noexcept; // nearest and gradual underflow, no changes
ColocOrder coloc_keep_order(const ColocView&); // stable KEEPORDER, no stride flip
ColocOrder coloc_keep_order(const ColocView&,const ColocView&);
bool coloc_sum_supported(const ColocView&,ColocReductionPlan); // metadata only
bool coloc_all_finite(const ColocView&); // bit inspection; no floating operation
ColocType coloc_square_type(ColocType); // bool square resolves to signed int8
ColocType coloc_sum_type(ColocType);    // bool/signed -> int64; unsigned -> uint64

// Operations below validate every output range, descriptor object and shape/
// stride array before any store. Inputs may alias one another; mutable output
// must not overlap any input allocation, descriptor or descriptor metadata.
// Exact output size is required. Binary/square/multiply outputs have native
// byte order and the provided dense axis order. Gather uses logical C order
// and preserves the input byte order/raw element bits, including bool bytes.
// Caller allocates storage; there is no Python, NumPy, callback or hidden pool.
// Any floating operation restores the caller's complete FEnv and returns only
// operation-raised flags. All floating arithmetic is serial and strict; no FMA
// or reassociation. Independent byte/integer work may use the sole global worker
// state (default1). worker_limit is the requested upper bound, not a claim
// about the runtime OpenMP team size. SIMD is runtime-guarded. Floating product
// and sum entry points admit finite values only; nonfinite inputs are a visible
// primitive-provider frontier in the Python shell, never silently normalized.
// The caller must prevent concurrent writes, resizes, or descriptor changes
// throughout a call, including from another thread while Python releases GIL.
ColocPredicate coloc_binary_valid(const ColocView&);
ColocExecution coloc_binary_into(const ColocView&,const ColocOrder&,ColocBuffer);
std::uint64_t coloc_count_nonzero(const ColocView&); // bool bytes use !=0
std::uint64_t coloc_joint_count(const ColocView&,const ColocView&);
ColocPredicate coloc_min_negative(const ColocView&); // NaN min <0 is false
ColocExecution coloc_gather_into(const ColocView&,const ColocView& boolean_mask,
    ColocBuffer,std::size_t selected);
ColocScalar coloc_sum(const ColocView&,ColocReductionPlan);
ColocExecution coloc_square_into(const ColocView&,const ColocOrder&,ColocBuffer,
    unsigned& fp_status);
// Same dtype, or a boolean second operand (MCC). Other dtype promotion is an
// explicit caller/provider frontier; do not reinterpret or force to double.
ColocExecution coloc_multiply_into(const ColocView&,const ColocView&,
    const ColocOrder&,ColocBuffer,unsigned& fp_status);
} // namespace skimagexx
