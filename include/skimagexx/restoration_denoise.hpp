#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// All buffers are contiguous and row major. The caller owns the output and
// provides nonoverlapping storage. Image dimensions must be positive.
template<class T>
bool denoise_nl_means(const T* image, T* out, std::size_t depth,
                      std::size_t rows, std::size_t cols, std::size_t channels,
                      bool volume, int patch_size, int patch_distance,
                      double h, double sigma);

template<class T>
double estimate_sigma(const T* image, std::size_t rows, std::size_t cols,
                      std::size_t channels, std::size_t channel, bool plane);

template<class T>
bool denoise_wavelet(const T* image, T* out, std::size_t rows, std::size_t cols,
                     std::size_t channels, bool plane, int filter_length,
                     int levels, double sigma, bool automatic_sigma,
                     bool soft, bool bayes);

template<class T>
bool inpaint_biharmonic(const T* image, const std::uint8_t* mask, T* out,
                        std::size_t rows, std::size_t cols, std::size_t channels);

template<class T>
bool unwrap_phase(const T* image, double* out, std::size_t depth, std::size_t rows,
                  std::size_t cols, int rank, bool deterministic_boundary);

template<class T>
bool richardson_lucy(const T* image, const T* psf, T* out, std::size_t rows,
                     std::size_t cols, std::size_t psf_rows, std::size_t psf_cols,
                     int iterations, bool clip, double filter_epsilon);

// A false result directs the caller to the compatibility implementation.
// Float and double instantiations are provided.
}
