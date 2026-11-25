#pragma once

#include <array>
#include <complex>
#include <cstddef>

namespace skimagexx {

inline constexpr std::size_t phase_correlation_max_samples = 4 * 1024 * 1024;
inline constexpr std::size_t phase_correlation_max_axis = 8192;
inline constexpr int phase_correlation_max_upsample = 32;

struct PhaseCorrelationOptions {
    int upsample_factor = 1;
    bool normalize_phase = true;
};

template<class T> struct PhaseCorrelationResult {
    std::array<T, 2> shift;
    double error;
    T phasediff;
};

// Two aligned, contiguous row-major buffers of height * width elements.
// The floating-point overload transforms real images. The complex overload
// takes precomputed, full Fourier spectra. Both register periodic translations.
// Returns false for unsupported shapes, options, or nonfinite/degenerate
// arithmetic, including real peaks at the phase branch cut. The result
// remains unchanged on false. Supports float and double.
// FFTs honor the process-wide worker limit. The local DFT uses ordered sums.
template<class T>
bool phase_cross_correlation(
    const T* reference, const T* moving, std::size_t height, std::size_t width,
    PhaseCorrelationOptions options, PhaseCorrelationResult<T>& result);

template<class T>
bool phase_cross_correlation_fourier(
    const std::complex<T>* reference, const std::complex<T>* moving,
    std::size_t height, std::size_t width, PhaseCorrelationOptions options,
    PhaseCorrelationResult<T>& result);

} // namespace skimagexx
