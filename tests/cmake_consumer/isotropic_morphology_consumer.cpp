#include "skimagexx/isotropic_morphology.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

int main() {
    using skimagexx::IsotropicOperation;
    using skimagexx::isotropic_morphology_into;
    using Line = std::array<std::uint8_t, 5>;
    const Line line{0, 2, 3, 0, 0};
    Line output{};
    const std::size_t width = line.size();
    const double unit = 1.;
    const auto check = [&](IsotropicOperation operation, double radius, const Line& expected) {
        isotropic_morphology_into(line.data(), line.size(), 1, &width, &unit, radius,
                                  operation, output.data(), output.size());
        return output == expected;
    };
    if (!check(IsotropicOperation::erosion, 1., {0, 0, 0, 0, 0}) ||
        !check(IsotropicOperation::dilation, 1., {1, 1, 1, 1, 0}) ||
        !check(IsotropicOperation::opening, 1., {1, 0, 0, 0, 0}) ||
        !check(IsotropicOperation::closing, 1., {1, 1, 1, 0, 0}) ||
        !check(IsotropicOperation::erosion, -0., {0, 1, 1, 0, 0}) ||
        !check(IsotropicOperation::closing, std::numeric_limits<double>::quiet_NaN(),
               {0, 0, 0, 0, 0}))
        return 1;

    const std::array<std::size_t, 2> shape{2, 3};
    const std::array<double, 2> spacing{0.75, 1.5};
    const std::array<std::uint8_t, 6> no_sites{};
    std::array<std::uint8_t, 6> mask{};
    isotropic_morphology_into(no_sites.data(), no_sites.size(), shape.size(), shape.data(),
                              spacing.data(), 1.7, IsotropicOperation::dilation,
                              mask.data(), mask.size());
    if (mask != std::array<std::uint8_t, 6>{1, 1, 0, 1, 0, 0})
        return 2;

    const auto saved = output;
    try {
        isotropic_morphology_into(line.data(), line.size(), 1, &width, &unit, 1.,
                                  IsotropicOperation::erosion, output.data(), output.size() - 1);
        return 3;
    } catch (const std::invalid_argument&) {
        return output == saved ? 0 : 4;
    }
}
