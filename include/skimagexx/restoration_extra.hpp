#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace skimagexx {

// Dense, aligned, row-major buffers. Shape extents must be positive.
// Output buffers are caller-owned and must not overlap inputs.
using RestorationShape = std::vector<std::size_t>;

// The real spectrum has the final extent reduced to shape.back()/2 + 1.
template<class T>
void restoration_transfer(const T* kernel, const RestorationShape& kernel_shape,
                          const RestorationShape& shape, std::complex<T>* output,
                          int workers);

template<class T>
void restoration_laplacian(const RestorationShape& shape, T* output);

// Regularizer values are the real transfer function, before squaring.
// A null regularizer selects the centered Laplacian.
// A false return requests the caller's exceptional floating-point path.
template<class T>
bool restoration_wiener(const T* image, const RestorationShape& shape,
                        const std::complex<T>* transfer, const T* reg,
                        T balance, bool clip, T* output, int workers);

template<class T>
void restoration_gibbs_sample(const std::complex<T>* data,
                              const std::complex<T>* noise_weighted_transfer,
                              const T* precision, const double* normal_real,
                              const double* normal_imag, std::size_t count,
                              std::complex<T>* output);

// normals and residuals each store two consecutive spectra of count elements.
template<class T>
void restoration_gibbs_step(const std::complex<T>* data, const std::complex<T>* transfer,
                            const T* reg, const T* power, const T* reg_power,
                            T noise, T prior, const double* normals, std::size_t count,
                            std::complex<T>* sample, std::complex<T>* residuals);

// Orthonormal real transforms for Gibbs initialization and reconstruction.
template<class T>
void restoration_gibbs_forward(const T* image, const RestorationShape& shape,
                               std::complex<T>* data, int workers);

template<class T>
void restoration_gibbs_inverse(const std::complex<T>* data, const RestorationShape& shape,
                               T* output, int workers);

template<class T>
void restoration_interpolate(const T* image, const RestorationShape& shape,
                             std::size_t spatial_rank, T* output);

// Each shift is already reduced modulo its axis extent.
template<class T>
void restoration_roll(const T* image, const RestorationShape& shape,
                      const RestorationShape& shifts, T* output, bool add);

} // namespace skimagexx
