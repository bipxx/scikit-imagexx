#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <limits>
namespace skimagexx {
enum class HistogramThreshold { yen, triangle, isodata };
// Counts use upstream float32 rounding; center precision controls ISODATA arithmetic.
std::vector<std::size_t> threshold_indices(const double* counts, const double* centers,
    std::size_t bins, HistogramThreshold method, bool float_centers=false, unsigned* warnings=nullptr, bool numpy_log=true,
    double bin_width=std::numeric_limits<double>::quiet_NaN());
// Finite nonnegative probabilities/counts, 2..16 classes, at most 4096 bins.
// Returns first lexicographic maximum using the upstream float32 objective.
std::vector<std::size_t> threshold_multiotsu(const float* probabilities, std::size_t bins, std::size_t classes);
float threshold_log(float value, bool numpy_log=true);
void threshold_logs(const float* input, float* output, std::size_t count, bool numpy_log=true);
// Reflect-padded, float32 histogram smoothing with ordered double accumulation.
// Returns the first minimum between the two peaks; throws if not converged.
std::size_t threshold_minimum(const float* counts, std::size_t bins, std::size_t max_iterations=10000);
template<class T> std::vector<std::uint64_t> histogram_edges(const T* image,
    std::size_t count, const T* edges, std::size_t bins);
}
