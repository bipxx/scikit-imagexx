#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
// PRIVATE matrix-transform-math-v1. Profiles describe executed arithmetic,
// not NumPy version ranges or a public/function capability registration.
inline constexpr const char* matrix_transform_math_v2_interface = "matrix-transform-math-v2";
inline constexpr std::uint32_t matrix_transform_math_v2_capabilities = 63;
inline constexpr const char* matrix_transform_math_v3_interface = "matrix-transform-math-v3";
inline constexpr std::uint32_t matrix_transform_math_v3_capabilities = 1023;
enum class MatrixArithmeticProfile : std::uint32_t {
    unsupported = 0,
    // Audited Haswell OpenBLAS 0.3.21 / 0.3.34.106 GEMM, K=3..5,
    // alpha=1,beta=0. Caller authenticates library/dispatch and GEMM selection.
    haswell_gemm_fma = 1,
    // NumPy1.24.4 DOUBLE_matmul_inner_noblas: ordered separate mul/add.
    numpy124_noblas = 2,
    // N=1 Haswell GEMV: literal transposed/nontransposed small-K trees.
    // Caller selects from original matmul's actual internal layout/iterator
    // buffering, not from the layout of a later converted view.
    haswell_gemv_transposed = 3,
    haswell_gemv_nontransposed = 4,
    // NumPy2.5.4 N=1 special-case, non-BLASable matrix scalar loop.
    numpy254_special_noblas = 5,
    // v2 ONLY: exact pinned OpenBLAS0.3.21 Cooperlake GEMV-N association.
    // N=1, K=3..5; this does not authenticate a CPU/runtime or other BLAS paths.
    cooperlake0321_gemv_nontransposed = 6,
    // Exact authenticated OpenBLAS0.3.21 Prescott separate-mul/add trees.
    // K3..5. GEMVT selects ORIGINAL BLAS A address modulo16 before copies;
    // the8byte-peel and K5 final-component remainder affect association.
    prescott0321_gemm = 7,
    prescott0321_gemv_nontransposed = 8,
    prescott0321_gemv_transposed_aligned = 9,
    prescott0321_gemv_transposed_peeled = 10,
};
struct MatrixReadView {
    const void* data;
    std::size_t rows, columns;
    std::ptrdiff_t row_stride, column_stride; // signed bytes; unaligned allowed
    const void* allocation;
    std::size_t allocation_bytes; // authentic owner extent, never fabricated
};
struct MatrixWriteView {
    void* data;
    std::size_t rows, columns;
    std::size_t allocation_bytes; // contiguous C double output, no self-overlap
};
enum class MatrixMapStatus : std::uint32_t {
    ok = 0, unsupported_profile, unsupported_shape, invalid_view,
    overlap, unsafe_fp_environment, nonfinite_or_range, unsafe_divisor,
};
// Allocation-free finite F64 mapping for D=2..4, N>0, arbitrary authentic
// input byte strides. Input abs<=2^20; nonzeros abs>=2^-200. All input bounds,
// output disjointness, nearest/gradual FP and all denominators are checked
// BEFORE any output store. Zero denominators become binary64 epsilon exactly.
// Nonzero denominators must abs>=2^-400. Source FP flags are restored.
// GEMV arithmetic requires the explicit separate profiles; GEMM rejects N=1.
// SIMD-disabled and portable paths use identical explicitly rounded arithmetic
// (std::fma for fused profiles), not an arithmetic-profile substitution.
MatrixMapStatus matrix_transform_into(const MatrixReadView& points,
    const MatrixReadView& matrix, const MatrixWriteView& output,
    MatrixArithmeticProfile profile);
// Strict separate v2 entry. The legacy entry still rejects profile6, keeping
// old v1/cap31 consumers interpretable. Views/status/old arithmetic are unchanged.
MatrixMapStatus matrix_transform_into_v2(const MatrixReadView& points,
    const MatrixReadView& matrix, const MatrixWriteView& output,
    MatrixArithmeticProfile profile);
// v3 adds only the four Prescott policies; v1/v2 continue rejecting them.
MatrixMapStatus matrix_transform_into_v3(const MatrixReadView& points,
    const MatrixReadView& matrix, const MatrixWriteView& output,
    MatrixArithmeticProfile profile);
namespace detail {
// Prepared scalar arithmetic shared by the mapping and standalone SDK controls.
// Caller supplies two read-only K-element arrays, K3..5, the admitted finite
// normal domain, nearest/gradual FP, one of profiles7..10 and component<K.
// No allocation, runtime/CPU query, worker read, provider or FP-policy mutation.
// x includes its actual final operand (mapping callers explicitly append+1).
double matrix_prescott_dot_f64(const double* x, const double* coefficients,
    std::size_t k, MatrixArithmeticProfile profile, std::size_t component);
}
namespace simd {
// Runtime-gated AVX2+FMA leaf, no validation; returns processed row count.
std::size_t matrix_transform_fma(const MatrixReadView&, const double* coefficients,
    double* output, std::size_t begin, std::size_t end, std::size_t dimensions);
// v2 one-point leaf: four output components, never a four-row/gather overread.
// Caller has validated all views/denominators and CPU+OS AVX2/FMA policy.
void matrix_transform_cooperlake_gemv_n(const MatrixReadView&, const double* coefficients,
    double* output, std::size_t dimensions);
}
} // namespace skimagexx
