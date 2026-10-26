#pragma once
#include <complex>
#include <cstddef>
#include <vector>

namespace skimagexx::detail {
// Private C-contiguous, native aligned, caller-owned nonoverlapping buffers.
// shape describes the FULL real shape for r2c/c2r. Spectrum reduces axes.back()
// to shape[axis]/2+1; other dimensions stay unchanged, including batch axes.
// All dimensions positive; axes nonempty/unique. Caller supplies sufficient
// storage. Forward unnormalized; inverse scales by 1/product(transformed axes).
// workers is the consumer's single snapshot of the process-global budget.
void fft_r2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const float* input, std::complex<float>* output, int workers);
void fft_c2r(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const std::complex<float>* input, float* output, int workers);
void fft_c2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const std::complex<float>* input, std::complex<float>* output,
             bool forward, int workers);
void fft_r2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const double* input, std::complex<double>* output, int workers);
void fft_c2r(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const std::complex<double>* input, double* output, int workers);
void fft_c2c(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& axes,
             const std::complex<double>* input, std::complex<double>* output,
             bool forward, int workers);
}
