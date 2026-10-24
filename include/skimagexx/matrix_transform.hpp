#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx {
// PRIVATE matrix-transform-math-v1. Profiles describe executed arithmetic,
// not NumPy version ranges or a public/function capability registration.
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
namespace simd {
// Runtime-gated AVX2+FMA leaf, no validation; returns processed row count.
std::size_t matrix_transform_fma(const MatrixReadView&, const double* coefficients,
    double* output, std::size_t begin, std::size_t end, std::size_t dimensions);
}
} // namespace skimagexx
