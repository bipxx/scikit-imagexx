#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct ChanVeseOptions {
    double mu = 0.25;
    double lambda1 = 1.0;
    double lambda2 = 1.0;
    double tol = 1e-3;
    std::size_t max_num_iter = 500;
    double dt = 0.5;
    bool extended_output = false;
    // Zero uses an unbuffered pairwise sum. Set to the provider's reduction
    // buffer length when matching a buffered NumPy profile.
    std::size_t reduction_buffer = 0;
};

inline constexpr std::size_t chan_vese_max_pixels = 1024 * 1024;
inline constexpr std::size_t chan_vese_max_iterations = 10000;

template<class T>
struct ChanVeseResult {
    std::vector<std::uint8_t> segmentation;
    // Both are empty when extended_output is false. The history starts with
    // initial energy and has one value per completed iteration.
    std::vector<T> level_set;
    std::vector<T> energies;
};

// Evolve a supplied continuous level set on a real 2D image. Float and double
// overloads use their input precision. The two aligned, contiguous spans
// must each contain count = height * width elements and stay stable during
// the call. The function owns its output and does not mutate either span.
// Nonempty images of up to chan_vese_max_pixels are supported. Weights and
// dt must be in [0, 16], tol in [0, 2], and max_num_iter <= 10000.
// Nonfinite values, magnitudes above max(T)/4 for the image or 1e6 for the
// initial level set, and unsupported floating environments throw domain_error.
// Invalid shape, spans or options throw invalid_argument.
template<class T>
ChanVeseResult<T> chan_vese(const T* image, const T* initial_level_set,
                           std::size_t count, std::size_t height, std::size_t width,
                           const ChanVeseOptions& options = {});

extern template ChanVeseResult<float> chan_vese(
    const float*, const float*, std::size_t, std::size_t, std::size_t, const ChanVeseOptions&);
extern template ChanVeseResult<double> chan_vese(
    const double*, const double*, std::size_t, std::size_t, std::size_t, const ChanVeseOptions&);

}  // namespace skimagexx
