#pragma once
#include "metrics.hpp"
#include <array>
#include <vector>

namespace skimagexx {
struct NmiPreparation {
    bool admitted = false;
    std::size_t samples = 0;
    // Exact representable common-dtype bounds, before constant-range expansion.
    std::array<double, 4> bounds{};
};
struct NmiStatistics {
    bool admitted = false;
    std::array<std::size_t, 2> bins{};
    std::vector<std::uint64_t> counts;
    // H(column marginal), H(row marginal), H(joint density).
    std::array<double, 3> entropy{};
};
// Equal ranks; the smaller extent on each axis is virtually padded with zeros.
// Arbitrary byte strides, byte order and binary16 use the existing metric view.
// Returns admitted=false outside the finite, exactly representable initial
// frontier. It does not execute an upstream/provider algorithm.
NmiPreparation nmi_prepare(const MetricImageView&, const MetricImageView&);
// Edges and widths are caller-prepared using the declared provider semantics.
// Each edge vector has bins+1 entries; each width vector has bins entries.
// Inputs must have passed nmi_prepare. Counts use right-open interior bins and
// include the final right edge. Workers share the global default-one setting;
// private integer histograms merge deterministically. Nonconstant densities
// with max probability>0.99 are not admitted for the initial entropy path.
NmiStatistics nmi_statistics(const MetricImageView&, const MetricImageView&,
    const std::vector<double>& edges0, const std::vector<double>& edges1,
    const std::vector<double>& widths0, const std::vector<double>& widths1);
} // namespace skimagexx
