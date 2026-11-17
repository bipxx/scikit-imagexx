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
// Explicit elementary-log provider for hosts whose public log2 differs from
// platform libm. The owning operation calls exactly one whole nonzero batch
// per entropy direction, including empty batches, in source order. Pointers
// are valid only for the callback; count0 permits null pointers. The callback
// writes count logarithms, must not mutate input/CSR storage, and may return
// false to refuse. Exceptions propagate; output is published only on success.
// The host owns context for the whole call. No Python/NumPy ABI is required by
// this SDK interface. Provider FP/error effects are retained, never normalized.
struct VariationLog2Batch {
    using Function=bool(*)(const double*,double*,std::size_t,void*);
    Function function=nullptr;
    void* context=nullptr;
};
// Additive compatible entry; existing overloads retain scalar platform-libm
// semantics and ABI. The callback runs serially on the calling thread.
bool variation_of_information_compatible(VariationTableView,VariationInformation&,
    VariationLog2Batch,int workers=0);
bool variation_of_information_compatible(const detail::LabelPairInput&,VariationInformation&,
    VariationLog2Batch,int workers=0);
// Explicit dense inner-product provider. The callback receives two read-only
// contiguous vectors of count doubles and writes one scalar. Input spans live
// throughout the synchronous callback only; count0 permits null inputs.
// Call order is log2/inner-product/log2/inner-product. No later callback runs
// after false or an exception, and the complete output stays unpublished.
// Host arithmetic and FP/error effects are disclosed and retained unchanged.
struct VariationInnerProduct {
    using Function=bool(*)(const double*,const double*,std::size_t,double*,void*);
    Function function=nullptr;
    void* context=nullptr;
};
// Additive host-compatible operation. The scalar-libm and log2-only overloads
// above retain their existing ABI, native-dot graphs and semantics exactly.
// Marginals, inverses, sparse scaling/coalescing/products and reductions stay
// native; both providers run serially on the calling thread, with one captured
// worker budget. Hosts own both contexts for the entire operation.
bool variation_of_information_compatible(VariationTableView,VariationInformation&,
    VariationLog2Batch,VariationInnerProduct,int workers=0);
bool variation_of_information_compatible(const detail::LabelPairInput&,VariationInformation&,
    VariationLog2Batch,VariationInnerProduct,int workers=0);
}
