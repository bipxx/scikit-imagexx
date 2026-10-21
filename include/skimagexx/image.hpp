#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <array>

namespace skimagexx {
enum class Border { nearest, reflect };
enum class EdgeFilter { sobel, prewitt, scharr };
enum class ExposureTransform { gamma, log, sigmoid };
// Process-wide worker limit, shared by all algorithms and calling threads.
// Default: 1 (serial). Set >1 to opt into parallel execution, or 1 to disable it.
// Each call snapshots the limit. Small images still run serially. SIMD is independent.
void set_num_threads(int count);
int get_num_threads();
bool has_avx2();

// Caller-owned contiguous output: height * width elements, no overlap with input.
// No output allocation/initialization. Uses the same global worker setting.
template<class T>
void rgb2gray_into(const T*, T*, std::size_t height, std::size_t width);
void rgb2gray_into(const std::uint8_t*, double*, std::size_t height, std::size_t width);
template<class T>
void gaussian_into(const T*, T*, std::size_t height, std::size_t width,
                   double sigma = 1.0, Border border = Border::nearest, double truncate = 4.0);
void gaussian_into(const std::uint8_t*, double*, std::size_t height, std::size_t width,
                   double sigma = 1.0, Border border = Border::nearest, double truncate = 4.0,
                   bool preserve_range = false);
template<class T>
void sobel_into(const T*, T*, std::size_t height, std::size_t width);
void sobel_into(const std::uint8_t*, double*, std::size_t height, std::size_t width);
// axis=-1: normalized magnitude; 0: horizontal edges; 1: vertical edges.
template<class T> void gradient_into(const T*, T*, std::size_t, std::size_t, EdgeFilter, int axis = -1);
void gradient_into(const std::uint8_t*, double*, std::size_t, std::size_t, EdgeFilter, int axis = -1);
template<class T> void gray2rgb_into(const T*, T*, std::size_t, std::size_t);
template<class T> void invert_into(const T*, T*, std::size_t, std::size_t, bool signed_float = false);
template<class T> double threshold_mean(const T*, std::size_t, std::size_t);
// Pointwise exposure correction. Float inputs must be nonnegative. Nonfinite
// transform parameters are rejected. Output dtype equals input dtype.
// gamma: parameter=exponent, gain=multiplier; log: inverse controls exp2;
// sigmoid: parameter=cutoff, gain=slope, inverse complements the curve.
template<class T> void exposure_into(const T*, T*, std::size_t height, std::size_t width,
                                     ExposureTransform, double parameter, double gain, bool inverse = false);
template<class T> std::pair<double,double> intensity_bounds(const T*, std::size_t height, std::size_t width);
template<class Input, class Output> void rescale_into(const Input*, Output*, std::size_t height, std::size_t width,
                                                      double input_min, double input_max, double output_min, double output_max);
std::array<std::uint64_t,256> histogram_u8(const std::uint8_t*, std::size_t height, std::size_t width);
void equalize_hist_into(const std::uint8_t*, double*, std::size_t height, std::size_t width);

// Contiguous row-major buffers. Every function allocates and owns its result.
// Pointers must reference at least height * width (* 3 for RGB) elements.
// Float and double preserve their input precision and range.
template<class T>
std::vector<T> rgb2gray(const T* rgb, std::size_t height, std::size_t width);
// uint8 input is normalized to [0,1] directly during conversion.
std::vector<double> rgb2gray(const std::uint8_t* rgb, std::size_t height, std::size_t width);

template<class T>
std::vector<T> gaussian(const T* image, std::size_t height, std::size_t width,
                        double sigma = 1.0, Border border = Border::nearest,
                        double truncate = 4.0);

template<class T>
std::vector<T> sobel(const T* image, std::size_t height, std::size_t width);

// Integer histogram with all 256 bins. First maximum wins ties.
std::uint8_t threshold_otsu(const std::uint8_t* image, std::size_t height,
                          std::size_t width, bool* constant = nullptr);

// Only float and double instantiations of the three templates are provided.
extern template std::vector<float> rgb2gray(const float*, std::size_t, std::size_t);
extern template std::vector<double> rgb2gray(const double*, std::size_t, std::size_t);
extern template std::vector<float> gaussian(const float*, std::size_t, std::size_t, double, Border, double);
extern template std::vector<double> gaussian(const double*, std::size_t, std::size_t, double, Border, double);
extern template std::vector<float> sobel(const float*, std::size_t, std::size_t);
extern template std::vector<double> sobel(const double*, std::size_t, std::size_t);
} // namespace skimagexx
