#pragma once

#include "skimagexx/image.hpp"

namespace skimagexx {

struct CircleParameters {
    std::array<double, 2> center{};
    double radius = 0;
};

struct EllipseParameters {
    std::array<double, 2> center{};
    std::array<double, 2> axis_lengths{};
    double theta = 0;
};

// Point buffers contain count interleaved (x, y) pairs. An empty result
// indicates a singular, ill-conditioned or unsupported fit.
bool fit_circle(const double* points, std::size_t count, CircleParameters& output);
bool fit_ellipse(const double* points, std::size_t count, EllipseParameters& output);
void circle_residuals(const double* points, std::size_t count, CircleParameters, double* output);
bool ellipse_residuals(const double* points, std::size_t count, EllipseParameters, double* output);

// Predict from count cosine and sine values into 2*count interleaved coordinates.
void predict_circle(const double* cosine, const double* sine, std::size_t count,
                    CircleParameters, double* output);
void predict_ellipse(const double* cosine, const double* sine, std::size_t count,
                     EllipseParameters, double* output);

double ransac_trial_limit(std::size_t inliers, std::size_t samples, std::size_t minimum,
                          double probability);

}  // namespace skimagexx
