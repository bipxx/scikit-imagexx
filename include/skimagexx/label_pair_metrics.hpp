#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx::detail {
inline constexpr std::size_t label_pair_max_samples=std::size_t{1}<<20;
inline constexpr std::int64_t label_pair_max_label=(std::int64_t{1}<<20)-1;

// One boundary supplies stable, native aligned int64 buffers in logical C
// order. Original reshape/conversion/protocol/shape-error order is a frontend
// responsibility. Different image shapes with equal flattened size are valid
// for contingency_table, while the two metric APIs check shape first.
// Caller guarantees truth/test each address count int64 elements and ignored
// labels address ignored_count elements. Lifetimes and contents stay stable
// for this call; raw pointer capacities cannot be discovered by the SDK.
struct LabelPairInput {
    const std::int64_t* truth=nullptr;
    const std::int64_t* test=nullptr;
    std::size_t count=0;
    const std::int64_t* ignored_truth_labels=nullptr;
    std::size_t ignored_count=0;
    bool normalize=false;
};

// Logical CSR values, not a SciPy class/storage ABI. Unique row/column pairs
// are sorted lexicographically. Ignored truth labels retain explicit zero
// entries before normalization; all-ignored normalization instead follows the
// source 0/0 weights and reports invalid status. Both shape maxima include
// ignored positions. Normalization divides
// each source pixel weight BEFORE duplicate coalescing, never integer_count/N.
// Fresh vectors own their storage and do not alias input. Initial public CSR
// class/index-width/storage packaging remains an explicit SciPy provider until
// original observations establish its exact profile-dependent metadata.
struct LabelPairCsr {
    std::size_t rows=0,columns=0,total_samples=0,kept_samples=0;
    std::vector<std::int64_t> indptr,indices;
    std::vector<double> data;
    // SDK-local IEEE status: invalid1/divide2/overflow4/underflow8.
    // Delivery through NumPy policy/callback/warning hooks is frontend work.
    unsigned normalization_fp_status=0;
};

// Declaration-only prospective interface. Refuse count0, excessive sizes,
// negative/out-of-bound coordinates or malformed typed spans without changing
// output. First finite partition: count<=2^20, labels in[0,2^20-1], at most64
// ignored labels. Normalization requires nearest/gradual-underflow environment
// and restores caller FP state; unnormalized integer counts need no FP change.
// workers0 snapshots the existing global budget once (default1); an already
// captured positive budget is used directly. Ordered pair aggregation remains
// serial; only applicable independent preparation/fill spans may parallelize.
bool label_pair_counts(const LabelPairInput&,LabelPairCsr&,int workers=0);

// Finite table=None adapted-Rand stages. Counts are exact integers: N<=2^20
// bounds every dot/sum/pair total by2^40. These stages do not accept external
// CSR objects. Python conversion/shape/alpha-error order remains a frontend
// boundary. Deliver each scalar status before invoking the next stage because
// callbacks can change subsequent NumPy policy. No hidden batched arithmetic.
struct LabelPairRandTerms {double joint=0.,row=0.,column=0.;};
struct LabelPairScalar64 {std::uint64_t bits=0;unsigned fp_status=0;};
bool label_pair_rand_terms(const LabelPairInput&,LabelPairRandTerms&,int workers=0);
bool label_pair_rand_divide(double numerator,double denominator,LabelPairScalar64&);
bool label_pair_rand_fscore(const LabelPairRandTerms&,double alpha,LabelPairScalar64&);
bool label_pair_rand_error(std::uint64_t fscore_bits,LabelPairScalar64&);

// No generic sparse engine or VI reduction/log claim. Source-dependent CSR
// object packing, normalization callbacks and supplied-table mutations remain
// explicit ownership frontiers; no public API is advertised by these helpers.
}
