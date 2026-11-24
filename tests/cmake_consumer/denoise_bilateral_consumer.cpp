#include "skimagexx/denoise_bilateral.hpp"

#include <array>
#include <stdexcept>

template<class T>
bool test() {
    const std::array<T, 4> input{T(0), T(0.25), T(0.5), T(1)};
    const std::array<T, 4> color{T(1), T(1), T(1), T(1)};
    std::array<T, 36> spatial{};
    spatial[0] = T(1);
    spatial[12] = T(1);
    std::array<T, 4> output{};
    skimagexx::denoise_bilateral_constant_into(
        input.data(), input.size(), 2, 2, 1, 1., 5, 0.5, color.data(), color.size(),
        spatial.data(), spatial.size(), output.data(), output.size());
    if (output != std::array<T, 4>{T(0.25), T(0.375), T(0.5), T(0.75)})
        return false;
    const auto saved = output;
    try {
        skimagexx::denoise_bilateral_constant_into(
            input.data(), input.size(), 2, 2, 1, 1., 5, 0.5, color.data(), color.size(),
            spatial.data(), spatial.size() - 1, output.data(), output.size());
    } catch (const std::invalid_argument&) {
        return output == saved;
    }
    return false;
}

int main() {
    return test<float>() && test<double>() ? 0 : 1;
}
