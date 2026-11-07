#pragma once
#include <skimagexx/convert.hpp>
#include <skimagexx/crop.hpp>
#include <cstddef>
#include <vector>
namespace skimagexx {
struct InverseGaussianPass { std::size_t axis; std::vector<double> weights; };
using InverseGaussianPasses=std::vector<std::vector<InverseGaussianPass>>;
// Prepared public boundary: native contiguous input/output preserve input
// scalar dtype. Complex storage consists of interleaved real/imaginary words.
// Coefficients are an explicitly named small NumPy metadata provider; image
// correlation, square, ordered sum and sqrt are exclusively owned here.
void inverse_gaussian_gradient_magnitude(const void*,void*,PixelType,bool complex,
    const std::vector<std::size_t>&,const InverseGaussianPasses&,int workers=0);
// Stable broadcast inputs in a common floating/complex dtype. A stage is
// multiply(alpha,gradient), add1, sqrt, reciprocal respectively; stages permit
// source warning/callback delivery between public expression operations.
unsigned inverse_gaussian_finish(const void*,const void*,void*,PixelType,bool complex,
    const std::vector<std::size_t>& gradient_shape,const std::vector<std::size_t>& alpha_shape,
    const std::vector<std::size_t>& output_shape,unsigned stage,int workers=0);
// Complete owning SDK operation: generates scalar Gaussian derivative kernels,
// computes typed separable magnitude, and applies the scalar alpha expression.
// The SDK covers float32/float64 only; no Python integer/promotion policy claim.
template<class T> struct InverseGaussianResult {std::vector<std::size_t> shape;std::vector<T> values;int captured_workers=1;};
template<class T> InverseGaussianResult<T> inverse_gaussian_gradient(const T*,
    const std::vector<std::size_t>& shape,const std::vector<double>& sigma,
    const T* alpha,const std::vector<std::size_t>& alpha_shape,int workers=0);
// Borrowed signed-strided numeric input. Existing admitted POD crop-copy
// supplies the dense working image; callers retain valid input/alpha storage.
template<class T> InverseGaussianResult<T> inverse_gaussian_gradient(const CropView&,
    const std::vector<double>& sigma,const T* alpha,
    const std::vector<std::size_t>& alpha_shape,int workers=0);
template<class T> std::vector<T> inverse_gaussian_gradient(const T*,
    const std::vector<std::size_t>& shape,const std::vector<double>& sigma,
    T alpha=T(100),int workers=0);
extern template std::vector<float> inverse_gaussian_gradient(const float*,const std::vector<std::size_t>&,const std::vector<double>&,float,int);
extern template std::vector<double> inverse_gaussian_gradient(const double*,const std::vector<std::size_t>&,const std::vector<double>&,double,int);
extern template InverseGaussianResult<float> inverse_gaussian_gradient(const float*,const std::vector<std::size_t>&,const std::vector<double>&,const float*,const std::vector<std::size_t>&,int);
extern template InverseGaussianResult<double> inverse_gaussian_gradient(const double*,const std::vector<std::size_t>&,const std::vector<double>&,const double*,const std::vector<std::size_t>&,int);
extern template InverseGaussianResult<float> inverse_gaussian_gradient(const CropView&,const std::vector<double>&,const float*,const std::vector<std::size_t>&,int);
extern template InverseGaussianResult<double> inverse_gaussian_gradient(const CropView&,const std::vector<double>&,const double*,const std::vector<std::size_t>&,int);
}
