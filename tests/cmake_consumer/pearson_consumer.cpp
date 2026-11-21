#include <skimagexx/pearson.hpp>

#include <array>
#include <cfenv>
#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("Pearson SDK assertion failed");
}

bool uniform_tail(double shape, double x, double& upper_tail, void* context) {
    require(shape == 1.0);
    ++*static_cast<unsigned*>(context);
    upper_tail = 1 - x;
    return true;
}

} // namespace

int main() {
    using skimagexx::pearson_corr_coeff;
    using skimagexx::PearsonResult;
    const std::array left{0., 1., 2., 3.};
    const std::array right{0., 2., 3., 1.};
    const std::array<std::uint8_t, 4> pair{0, 1, 0, 2};
    PearsonResult result{3., 4.};
    unsigned providers = 0;
    require(pearson_corr_coeff(left.data(), right.data(), pair.data(), left.size(),
                               result, {uniform_tail, &providers}));
    require(result.correlation == -1.0 && result.p_value == 1.0 && providers == 0);

    std::fenv_t saved;
    require(std::fegetenv(&saved) == 0);
    std::feclearexcept(FE_ALL_EXCEPT);
    std::feraiseexcept(FE_DIVBYZERO);
    const int incoming_flags = std::fetestexcept(FE_ALL_EXCEPT);
    require(pearson_corr_coeff(left.data(), right.data(), nullptr, left.size(),
                               result, {uniform_tail, &providers}));
    const int outgoing_flags = std::fetestexcept(FE_ALL_EXCEPT);
    require(std::abs(result.correlation - 0.4) < 1e-15 &&
            std::abs(result.p_value - 0.6) < 1e-15 && providers == 1);
    require(outgoing_flags == incoming_flags);
    require(std::fesetround(FE_UPWARD) == 0);
    result = {3., 4.};
    require(!pearson_corr_coeff(left.data(), right.data(), nullptr, left.size(),
                                result, {uniform_tail, &providers}));
    require(result.correlation == 3. && result.p_value == 4. && providers == 1);
    require(std::fegetround() == FE_UPWARD);
    require(std::fesetenv(&saved) == 0);

    require(!pearson_corr_coeff(left.data(), left.data(), nullptr, left.size(),
                                result, {uniform_tail, &providers}));
    require(result.correlation == 3. && result.p_value == 4. && providers == 1);
    require(!pearson_corr_coeff(left.data(), right.data(), nullptr, left.size(), result));
    const std::array near{1., 1. + 0x1p-45, 1., 1.};
    require(!pearson_corr_coeff(near.data(), right.data(), pair.data(), left.size(),
                                result, {uniform_tail, &providers}));
    require(providers == 1);
}
