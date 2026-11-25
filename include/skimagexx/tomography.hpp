#pragma once

namespace skimagexx {

enum class TomographyFilter { ramp, shepp_logan, cosine, hamming, hann, none };
enum class TomographyInterpolation { linear, nearest, cubic };

struct RadonRotation {
    double cosine, sine, column_offset, row_offset;
};

// All buffers are contiguous and row-major. Extents are positive. The caller
// supplies storage for each output and keeps it disjoint from every input.
// Only float and double are instantiated. The kernels expect finite values.
int radon_detector_size(int rows, int columns, bool circle);
bool radon_outside_circle(const float* image, int rows, int columns);
bool radon_outside_circle(const double* image, int rows, int columns);

// Each rotation has sine/cosine in [-1, 1] and offsets at most twice the
// detector size in magnitude. Output shape is (detector size, angles).
template<class T>
void radon_into(const T* image, int rows, int columns, const RadonRotation* rotations,
                int angles, bool circle, T* sinogram);

// Trigonometric vectors have one entry per angle. Output shape is (size, size).
// Cubic interpolation requires at least four detectors after circle padding.
template<class T>
void iradon_into(const T* sinogram, int detectors, int angles, const double* cosines,
                 const double* sines, int size, TomographyFilter filter,
                 TomographyInterpolation interpolation, bool circle, T* image);

// Angles in degrees are distinct and within [-360, 360]. Theta and shifts each
// have 'angles' entries. There are at least four detectors, shifts are at most
// 'detectors' in magnitude, and image contains a detector-by-detector initial
// estimate updated in place. With clip, clipped_image receives the clipped
// estimate and remaining updates, while image receives just the first update.
// Clip points to two bounds. Without clip, clipped_image is unused.
template<class T>
void iradon_sart_into(const T* sinogram, int detectors, int angles, const T* theta,
                      const T* shifts, T relaxation, bool weak_scalars, T* image,
                      const T* clip = nullptr, T* clipped_image = nullptr);

}
