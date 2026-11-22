#include <skimagexx/ellipsoid_stats.hpp>

#include <algorithm>
#include <cmath>

namespace skimagexx {
namespace {

constexpr double min_radius = 0x1p-96;
constexpr double max_radius = 0x1p96;

bool in_range(double value, double low, double high) {
    return value >= low && value <= high;
}

// DLMF 19.21.10 relates R_G to R_F and R_D. Choose the median as the
// distinguished argument so the R_D term is added without cancellation.
// The two integrals share the duplication steps of DLMF 19.26.18.
double symmetric_rg(double low, double mid, double high) {
    if (low > mid) std::swap(low, mid);
    if (mid > high) std::swap(mid, high);
    if (low > mid) std::swap(low, mid);
    if (low == high) return std::sqrt(high);

    double x = low / high;
    double y = 1;
    double z = mid / high;
    const double rd_weight = (z - x) * (1 - z);
    const double root_ratio = std::sqrt(x / z);
    const double normalized_mid = z;
    double rd_sum = 0;
    double scale = 1;

    while (y - x > (x + y + z) / 3000) {
        const double sx = std::sqrt(x);
        const double sy = std::sqrt(y);
        const double sz = std::sqrt(z);
        const double lambda = sx * sy + sz * (sx + sy);
        if (rd_weight != 0) rd_sum += scale / (sz * (z + lambda));
        scale *= 0.25;
        x = 0.25 * (x + lambda);
        y = 0.25 * (y + lambda);
        z = 0.25 * (z + lambda);
    }

    // DLMF 19.36.1/2. At this cutoff the omitted degree-six terms are
    // smaller than binary64 roundoff.
    const double mean_f = (x + y + z) / 3;
    const double fx = (mean_f - x) / mean_f;
    const double fy = (mean_f - y) / mean_f;
    const double fz = (mean_f - z) / mean_f;
    const double f2 = fx * fy + fz * (fx + fy);
    const double f3 = fx * fy * fz;
    const double rf = (1 + (-0.1 + f2 / 24 - 3 * f3 / 44) * f2 + f3 / 14)
        / std::sqrt(mean_f);

    double rd_term = 0;
    if (rd_weight != 0) {
        const double mean_d = (x + y + 3 * z) / 5;
        const double dx = (mean_d - x) / mean_d;
        const double dy = (mean_d - y) / mean_d;
        const double dz = (mean_d - z) / mean_d;
        const double xy = dx * dy;
        const double zz = dz * dz;
        const double e2 = xy - 6 * zz;
        const double e3 = (3 * xy - 8 * zz) * dz;
        const double e4 = 3 * (xy - zz) * zz;
        const double e5 = xy * zz * dz;
        const double taylor = 1 - 3 * e2 / 14 + e3 / 6 + 9 * e2 * e2 / 88
            - 3 * e4 / 22 - 9 * e2 * e3 / 52 + 3 * e5 / 26;
        rd_term = rd_weight * (rd_sum + scale * taylor / (3 * mean_d * std::sqrt(mean_d)));
    }

    return 0.5 * std::sqrt(high) * (normalized_mid * rf + root_ratio + rd_term);
}

}

std::optional<EllipsoidStats> ellipsoid_stats(double a, double b, double c) noexcept {
    if (!in_range(a, min_radius, max_radius) || !in_range(b, min_radius, max_radius)
        || !in_range(c, min_radius, max_radius))
        return std::nullopt;
    constexpr double four_thirds_pi = 0x1.0c152382d7365p+2;
    // Keep the supplied order for multiplication. Only the integral is symmetric.
    const double volume = ((four_thirds_pi * a) * b) * c;
    return EllipsoidStats{volume, (3 * volume) * symmetric_rg(1 / (a * a), 1 / (b * b), 1 / (c * c))};
}

namespace detail {
std::optional<double> ellipsoid_carlson_rg(double x, double y, double z) noexcept {
    constexpr double low = 0x1p-192;
    constexpr double high = 0x1p192;
    if (!in_range(x, low, high) || !in_range(y, low, high) || !in_range(z, low, high))
        return std::nullopt;
    return symmetric_rg(x, y, z);
}
}
}
