#include "skimagexx/geometric_models.hpp"

#include <cmath>

namespace {

bool close(double first, double second) {
    return std::abs(first - second) < 1e-10;
}

}  // namespace

int main() {
    const double points[]{6, 1, 2, 5, -2, 1, 2, -3};
    skimagexx::CircleParameters circle;
    if (!skimagexx::fit_circle(points, 4, circle) || !close(circle.center[0], 2)
        || !close(circle.center[1], 1) || !close(circle.radius, 4)) return 1;
    double residuals[64]{};
    skimagexx::circle_residuals(points, 4, circle, residuals);
    for (int i = 0; i < 4; ++i) {
        if (!close(residuals[i], 0)) return 2;
    }

    const skimagexx::EllipseParameters expected{{2, 3}, {6, 2}, 0.4};
    double cosine[64], sine[64], ellipse_points[128];
    for (int i = 0; i < 64; ++i) {
        cosine[i] = std::cos(i * 0.1);
        sine[i] = std::sin(i * 0.1);
    }
    skimagexx::predict_ellipse(cosine, sine, 64, expected, ellipse_points);
    skimagexx::EllipseParameters ellipse;
    if (!skimagexx::fit_ellipse(ellipse_points, 64, ellipse) || !close(ellipse.center[0], 2)
        || !close(ellipse.center[1], 3) || !close(ellipse.axis_lengths[0], 6)
        || !close(ellipse.axis_lengths[1], 2) || !close(ellipse.theta, 0.4)) return 3;
    if (!skimagexx::ellipse_residuals(ellipse_points, 64, ellipse, residuals)) return 4;
    for (double residual : residuals) {
        if (residual > 1e-6) return 5;
    }
    return skimagexx::ransac_trial_limit(50, 100, 2, 0.99) == 17 ? 0 : 6;
}
