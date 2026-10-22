#pragma once
#include "gaussian_nd.hpp"
#include <cstddef>

namespace skimagexx {
// Real 2-D image, interleaved complex kernel, separate real/imaginary outputs.
// All buffers are native-endian, contiguous and aligned. Output storage uses
// the image PixelType (including raw uint8 storage for boolean arrays). Kernel
// dimensions are odd. The outputs must not overlap input, kernel or each other.
// Accumulation is float64 in reversed-kernel row-major order, omitting weights
// at or below double epsilon independently for each component, as in ndimage.
// zero_stride_collision reproduces ndimage's coordinate-1/sentinel collision
// for aligned native arrays with all byte strides zero before preparation.
void gabor_filter_into(const void* image,PixelType type,void* real,void* imaginary,
    std::size_t height,std::size_t width,const void* kernel,bool single_kernel,
    std::size_t kernel_height,std::size_t kernel_width,GaussianBorder border,double cval=0,
    bool zero_stride_collision=false);
}
