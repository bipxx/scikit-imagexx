#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

namespace skimagexx {

using PeakCoordinate = std::array<std::ptrdiff_t, 2>;
using PeakLabels = std::variant<std::monostate, const std::int32_t*, const std::int64_t*>;

enum class PeakNorm { chebyshev, manhattan, euclidean, minkowski };

struct LocalPeakOptions {
    int min_distance = 1;
    std::optional<double> threshold_abs;
    std::optional<double> threshold_rel;
    std::array<int, 2> border{1, 1};
    std::size_t max_peaks = (std::numeric_limits<std::size_t>::max)();
    std::size_t max_peaks_per_label = (std::numeric_limits<std::size_t>::max)();
    PeakNorm norm = PeakNorm::chebyshev;
    bool suppress_connected = false;
    bool wide_relative_product = false;
    double minkowski_p = 2;
};

// Contiguous 2-D buffers with at least height * width elements. The default
// detection footprint is a square. The native domain supports radii 1 through
// 8, finite inputs, and optional label IDs from 0 through 65535. Throws
// overflow_error when data fall outside the bounded computational domain.
template<class T>
std::vector<PeakCoordinate> peak_local_max(const T* image, int height, int width,
    const LocalPeakOptions& options = {}, PeakLabels labels = {});

// Refine integer corners against zero-padded image windows. Coordinates must
// be interior, and critical values are the F-distribution quantiles for the
// supplied window size. The output uses float for float input and double for
// double or uint8 input. The uint8 image is cast without intensity scaling.
template<class Input, class Output>
void corner_subpix_into(const Input* image, int height, int width,
    const PeakCoordinate* corners, std::size_t count, int window_size,
    double critical_dot, double critical_edge, Output* output);

}  // namespace skimagexx
