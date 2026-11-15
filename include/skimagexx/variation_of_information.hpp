#pragma once
#include <skimagexx/label_pair_metrics.hpp>
#include <array>
#include <cstddef>
#include <cstdint>

namespace skimagexx {
// Stable logical CSR spans, never a SciPy ABI. Caller owns data[nnz],
// indices[nnz], indptr[rows+1] throughout the call; row pointers are monotone.
// Unsorted indices and explicit zeros are retained in source reduction order.
// This proposal does not admit arbitrary sparse classes/protocols or mutation.
struct VariationTableView {
    const double* data=nullptr;
    const std::int64_t* indices=nullptr;
    const std::int64_t* indptr=nullptr;
    std::size_t rows=0,columns=0,nnz=0;
};
struct VariationInformation {
    // False splits H(image1|image0), false merges H(image0|image1).
    std::array<double,2> values{};
    unsigned normalization_fp_status=0;
    int captured_workers=1;
};
// Source-shaped entropy/log2/reduction proposal. Exact saved-word acceptance
// is a required gate, not implied by this interface. Nearest/gradual f64 only;
// finite nonnegative coefficients plus quiet NaNs from all-ignored source
// normalization. Arithmetic order stays serial; no invented SIMD/thread core.
// workers0 captures the existing process-global budget once, default1.
bool variation_of_information(VariationTableView,VariationInformation&,int workers=0);
// Complete image-label operation reuses admitted owned contingency counts,
// including per-pixel normalization before duplicate summation. NumPy policy
// callback delivery remains a separately staged Python boundary, not SDK work.
bool variation_of_information(const detail::LabelPairInput&,VariationInformation&,int workers=0);
}
