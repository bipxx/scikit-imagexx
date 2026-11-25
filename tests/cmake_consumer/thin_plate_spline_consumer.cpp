#include <skimagexx/thin_plate_spline.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("thin plate spline SDK assertion failed");
}

} // namespace

int main() {
    const double src[]{0, 0, 4, 0, 0, 4, 4, 4, 2, 2};
    const double dst[]{2, 1, 6, 1, 2, 5, 6, 5, 4.5, 3.2};
    std::vector<double> weights;
    require(skimagexx::thin_plate_spline_fit(src, dst, 5, weights));
    require(weights.size() == 16);
    double result[10]{};
    require(skimagexx::thin_plate_spline_apply(src, weights.data(), 5, src, 5, result));
    for (std::size_t k = 0; k < 10; ++k) require(std::abs(result[k] - dst[k]) < 5e-7);
    require(skimagexx::thin_plate_spline_apply(src, weights.data(), 5, nullptr, 0, nullptr));

    const auto before = weights;
    const double singular[10]{};
    require(!skimagexx::thin_plate_spline_fit(singular, dst, 5, weights));
    require(weights == before);
    const double nonfinite[]{std::numeric_limits<double>::quiet_NaN(), 0};
    require(!skimagexx::thin_plate_spline_apply(src, weights.data(), 5, nonfinite, 1, result));
}
