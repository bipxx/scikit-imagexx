#include <skimagexx/phase_cross_correlation.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("phase_cross_correlation SDK assertion failed");
}

template<class T>
void check() {
    using skimagexx::PhaseCorrelationResult;
    using skimagexx::phase_cross_correlation;
    using skimagexx::phase_cross_correlation_fourier;
    const std::array<T, 6> reference{0, 1, 0, 0, 0, 0};
    const std::array<T, 6> moving{0, 0, 0, 0, 0, 1};
    PhaseCorrelationResult<T> result{};
    require(phase_cross_correlation(reference.data(), moving.data(), 2, 3, {}, result));
    require(result.shift == std::array<T, 2>{1, -1});
    require(std::abs(result.error) < 0.001 && std::abs(result.phasediff) < 0.001);
    require(phase_cross_correlation(reference.data(), moving.data(), 2, 3, {8, false}, result));
    require(result.shift == std::array<T, 2>{1, -1});
    require(std::abs(result.error) < 0.001 && std::abs(result.phasediff) < 0.001);
    const auto previous = result;
    require(!phase_cross_correlation(reference.data(), moving.data(), 0, 3, {}, result));
    require(!phase_cross_correlation(reference.data(), moving.data(), 2, 3, {33, true}, result));
    require(result.shift == previous.shift && result.error == previous.error &&
            result.phasediff == previous.phasediff);

    const std::array<std::complex<T>, 4> reference_spectrum{{1, 1, 1, 1}};
    const std::array<std::complex<T>, 4> moving_spectrum{{1, 1, -1, -1}};
    require(phase_cross_correlation_fourier(
        reference_spectrum.data(), moving_spectrum.data(), 2, 2, {}, result));
    require(result.shift == std::array<T, 2>{1, 0});
    require(result.error == 0 && result.phasediff == 0);
}

} // namespace

int main() {
    check<float>();
    check<double>();
}
