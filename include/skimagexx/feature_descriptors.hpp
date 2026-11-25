#pragma once

#include <cstddef>
#include <cstdint>

namespace skimagexx {

// Input and output buffers are contiguous, aligned, and disjoint. Callers
// provide valid dimensions and storage for the documented output shapes.
enum class LbpMethod { standard, rotation, uniform, oriented_uniform, variance };
// 1..64 points (1..31 for standard and rotation), finite radius in [0, 1e6].
void local_binary_pattern_into(const std::uint8_t* image, int rows, int cols,
                               int points, double radius, LbpMethod method, double* output);
// width and height describe one cell. The complete 3x3 grid fits the image.
int multiblock_lbp(const float* integral, int cols, int row, int col, int width, int height);

enum class HogNorm { l1, l1_sqrt, l2, l2_hys };
struct HogOptions {
    int rows, cols, channels;
    int cell_rows, cell_cols, block_rows, block_cols, orientations;
    HogNorm norm;
};
// Outputs are [block grid rows, block grid cols, block rows, block cols, bins]
// and optionally [rows, cols]. Float inputs contain finite values.
template<class Input, class Output>
void hog_into(const Input* image, const HogOptions& options, Output* features, Output* visualization);

enum class DaisyNorm { off, l1, l2, per_histogram };
struct DaisyOptions {
    int rows, cols, step, radius, rings, histograms, orientations;
    DaisyNorm norm;
};
// Sigmas has rings+1 entries, radii has rings entries in [0, radius].
// Output is [ceil((rows-2*radius)/step), ceil((cols-2*radius)/step), descriptor size].
template<class Input, class Output>
void daisy_into(const Input* image, const DaisyOptions& options, const double* sigmas,
                const double* radii, Output* output);

struct TemplateMatchOptions {
    int rows, cols, template_rows, template_cols;
    bool pad;
    double constant;
};
// Output is image-sized with padding, otherwise [rows-template_rows+1, cols-template_cols+1].
// Returns false for nonfinite or poorly conditioned inputs so callers can
// reproduce the upstream FFT and floating-point behavior.
template<class T>
bool match_template_into(const T* image, const T* pattern, const TemplateMatchOptions& options, T* output);

struct BasicFeatureOptions {
    int rows, cols, channels;
    bool intensity, edges, texture;
};
// Output is [rows, cols, channels * scales * features per scale], in that order.
template<class T>
void multiscale_basic_features_into(const T* image, const BasicFeatureOptions& options,
                                    const double* sigmas, int scales, float* output);

} // namespace skimagexx
