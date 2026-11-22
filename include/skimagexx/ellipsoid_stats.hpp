#pragma once

#include <optional>

namespace skimagexx {

struct EllipsoidStats {
    double volume;
    double surface_area;
};

// Scalar binary64 kernel. Radii must be finite and in [2^-96, 2^96].
// A declined input returns nullopt. This small kernel has no worker dispatch.
std::optional<EllipsoidStats> ellipsoid_stats(double a, double b, double c) noexcept;

namespace detail {
// Carlson R_G for positive finite inputs in [2^-192, 2^192].
std::optional<double> ellipsoid_carlson_rg(double x, double y, double z) noexcept;
}

}
