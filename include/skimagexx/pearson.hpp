#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

struct PearsonResult {
    double correlation;
    double p_value;
};

struct PearsonBetaTail {
    // Returns the upper tail of the symmetric beta distribution at x.
    // A false return declines the calculation.
    bool (*evaluate)(double shape, double x, double& upper_tail, void*) = nullptr;
    void* context = nullptr;
};

// Buffers contain count contiguous values each, where 2 <= count <= 2^20.
// A null mask selects every pair. Other masks select nonzero bytes. Inputs
// may alias and must stay valid and unchanged until the call returns.
//
// Floating-point inputs must be finite and within the bounded float64
// Pearson domain. The ordered reduction is serial. A false return leaves
// result unchanged. A distinct pair of observations has an exact p-value
// of one; larger samples use one explicitly supplied beta-tail calculation.
bool pearson_corr_coeff(const double* left, const double* right,
                        const std::uint8_t* mask, std::size_t count,
                        PearsonResult& result, PearsonBetaTail beta_tail = {});

} // namespace skimagexx
