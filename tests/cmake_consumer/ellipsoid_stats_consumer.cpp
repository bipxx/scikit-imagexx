#include <skimagexx/ellipsoid_stats.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void expect(double a, double b, double c, double volume, double surface) {
    const auto value = skimagexx::ellipsoid_stats(a, b, c);
    if (!value || value->volume != volume
        || std::abs(value->surface_area - surface) > std::abs(surface) * 2e-14)
        throw std::runtime_error("ellipsoid volume or surface differs from the reference");
}
}

int main() {
    expect(1, 1, 1, 0x1.0c152382d7365p+2, 0x1.921fb54442d18p+3);
    expect(1, 2, 3, 0x1.921fb54442d18p+4, 0x1.870ea2b87f04ap+5);
    expect(3, 2, 1, 0x1.921fb54442d18p+4, 0x1.870ea2b87f04ap+5);
    expect(1, 1.000000000000001, 1.000000000000002,
           0x1.0c152382d7373p+2, 0x1.921fb54442d26p+3);
    expect(0x1p-96, 0x1p95, 0x1p96, 0x1.0c152382d7365p+97, 0x1.921fb54442d18p+193);
    expect(0x1p-96, 0x1p-95, 0x1p96, 0x1.0c152382d7365p-93, 0x1.e6fe9921228b7p+3);
    for (double invalid : std::array<double, 5>{0, -1, 0x1p-97,
             std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
        if (skimagexx::ellipsoid_stats(1, invalid, 2))
            throw std::runtime_error("ellipsoid accepted unsupported radii");
    std::cout << "ellipsoid SDK: six saved pairs and five boundary refusals\n";
}
