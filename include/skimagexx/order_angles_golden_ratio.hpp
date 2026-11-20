#pragma once
#include "components.hpp"
#include <cstdint>
#include <vector>

namespace skimagexx {
enum class GoldenScalarPolicy { wide, weak_input };
// Borrowed finite1D POD input; owner/storage remain alive. Negative/zero strides
// and swapped words are supported. Integer values currently require exact
// binary64 conversion (|value|<=2^53). Larger duplicate-sort ordering and
// nonfinite/protocol/FP domains are not upstream-parity admissions.
// Initialization sorts once. Each next() reads LIVE input with the frozen index
// list. Mutate values only BETWEEN calls; descriptor/data/shape may not change.
// Tiny ordered recurrence is serial. One process-global budget snapshot is
// captured at begin(), default1; no threads/SIMD are invented for metadata.
class GoldenAngleOrder {
public:
    GoldenAngleOrder(const ComponentsView&,GoldenScalarPolicy=GoldenScalarPolicy::wide,int workers=0);
    std::int64_t begin();
    // Called only AFTER the first yield: initialize Python's increment even
    // when no indices remain. A sequence wrapper may then raise its fancy-index
    // error before next(), preserving source lazy order.
    bool resume();
    std::int64_t next(const ComponentsView& live);
    int worker_budget()const{return workers_;}
    const std::vector<std::size_t>& remaining()const{return remaining_;}
private:
    ComponentsView view_;
    GoldenScalarPolicy policy_;
    std::vector<std::size_t> remaining_;
    int requested_,workers_=0;
    unsigned phase_=0,precision_=64;
    double angle_=0,increment_=0;
};
// Complete standalone owning INDEX operation (input payload remains borrowed).
// All yielded indices are returned; stateful interface above supports caller
// mutation/lazy consumption without a geometry-only preparation placeholder.
std::vector<std::int64_t> order_angles_golden_ratio(const ComponentsView&,
    GoldenScalarPolicy=GoldenScalarPolicy::wide,int workers=0);
}
