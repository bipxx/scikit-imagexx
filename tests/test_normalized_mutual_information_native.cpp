#include "skimagexx/normalized_mutual_information.hpp"
#include "skimagexx/image.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    using namespace skimagexx;
    double a[] = {0, 0, 1, 1}, b[] = {0, 1, 0, 1};
    MetricImageView first{a, PixelType::float64, false, false, {4}, {8}};
    MetricImageView second{b, PixelType::float64, false, false, {4}, {8}};
    assert(get_num_threads() == 1);
    const auto prepared = nmi_prepare(first, second);
    assert(prepared.admitted && prepared.samples == 4);
    const std::vector<double> edges{0, 0.5, 1}, widths{0.5, 0.5};
    const auto independent = nmi_statistics(first, second, edges, edges, widths, widths);
    assert(independent.admitted && independent.counts == std::vector<std::uint64_t>({1, 1, 1, 1}));
    assert(std::abs((independent.entropy[0] + independent.entropy[1]) / independent.entropy[2] - 1) < 1e-14);
    first.data = a + 3; first.strides = {-8};
    const auto reversed = nmi_statistics(first, second, edges, edges, widths, widths);
    assert(reversed.counts == independent.counts);
    first.data = a; first.shape = {2}; first.strides = {8};
    const auto padding = nmi_statistics(first, second, edges, edges, widths, widths);
    assert(padding.counts == std::vector<std::uint64_t>({2, 2, 0, 0}));
    first.shape = {0}; first.data = nullptr;
    assert(nmi_prepare(first, second).admitted);
    second.shape = {0}; second.data = nullptr;
    assert(!nmi_prepare(first, second).admitted);
    first.data = a; first.shape = {2}; first.strides = {std::numeric_limits<std::ptrdiff_t>::min()};
    bool rejected = false;
    try { nmi_prepare(first, first); } catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
    first.strides = {8}; first.shape = {2147483648ULL};
    assert(!nmi_prepare(first, first).admitted);
    // A valid nonuniform direct-C++ density can normalize a tiny positive
    // density to zero. Its entropy contribution is0, never0*log(0).
    double x[] = {0, 0, 1}, y[] = {0, 1e-100, 1};
    first = {x, PixelType::float64, false, false, {3}, {8}};
    second = {y, PixelType::float64, false, false, {3}, {8}};
    const std::vector<double> wide_edges{0, 1e-100, 2e-100, 1e100}, wide_widths{1e-100, 1e-100, 1e100};
    const auto tiny = nmi_statistics(first, second, wide_edges, wide_edges, wide_widths, wide_widths);
    assert(tiny.admitted);
    for (auto value : tiny.entropy) assert(std::isfinite(value));
    std::cout << "NMI native counts, padding, strides, worker state and bounds passed\n";
}
