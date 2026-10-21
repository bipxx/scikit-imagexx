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
float threshold_log(float value, bool numpy_log=true);
void threshold_logs(const float* input, float* output, std::size_t count, bool numpy_log=true);
template<class T> std::vector<std::uint64_t> histogram_edges(const T* image,
    std::size_t count, const T* edges, std::size_t bins);
}
