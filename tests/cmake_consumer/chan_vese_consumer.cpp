#include <skimagexx/chan_vese.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("chan_vese SDK result or validation differs");
}

template<class T>
void exercise() {
    const std::array<T, 6> image{T(0), T(1), T(1), T(0), T(0.5), T(0.25)};
    std::array<T, 6> initial{T(-1), T(1), T(0.5), T(-0.5), T(-0.1), T(0)};
    const std::array<std::uint8_t, 6> expected{0, 1, 1, 0, 0, 0};
    skimagexx::ChanVeseOptions options;
    options.extended_output = true;
    options.max_num_iter = 0;
    const auto no_iterations = skimagexx::chan_vese(image.data(), initial.data(), image.size(), 2, 3, options);
    require(no_iterations.segmentation.size() == image.size() && no_iterations.level_set.size() == image.size());
    require(std::equal(no_iterations.segmentation.begin(), no_iterations.segmentation.end(), expected.begin()));
    require(std::equal(no_iterations.level_set.begin(), no_iterations.level_set.end(), initial.begin()));
    require(no_iterations.energies.empty());

    options.dt = 0;
    options.max_num_iter = 10;
    const auto stationary = skimagexx::chan_vese(image.data(), initial.data(), image.size(), 2, 3, options);
    require(stationary.segmentation == no_iterations.segmentation && stationary.level_set == no_iterations.level_set);
    require(stationary.energies.size() == 1 && std::isfinite(stationary.energies.front()));

    options.dt = 0.5;
    options.extended_output = false;
    const auto evolved = skimagexx::chan_vese(image.data(), initial.data(), image.size(), 2, 3, options);
    require(evolved.segmentation.size() == image.size() && evolved.level_set.empty() && evolved.energies.empty());
    bool rejected = false;
    try {
        skimagexx::chan_vese(image.data(), initial.data(), image.size() - 1, 2, 3, options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);
    initial.back() = std::numeric_limits<T>::infinity();
    rejected = false;
    try {
        skimagexx::chan_vese(image.data(), initial.data(), image.size(), 2, 3, options);
    } catch (const std::domain_error&) {
        rejected = true;
    }
    require(rejected && std::isinf(initial.back()));
}

}  // namespace

int main() {
    exercise<float>();
    exercise<double>();
}
