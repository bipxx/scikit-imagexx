#include <skimagexx/morphological_snakes.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

int main() {
    const std::array<double, 6> image{0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    std::array<double, 6> gradient{};
    skimagexx::inverse_gaussian_gradient_into(image.data(), gradient.data(), {2, 3}, 60.0, 1.0);
    if (!std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 1.0; }))
        return 1;

    const std::array<float, 4> two_regions{0, 0, 1, 1};
    const std::array<std::int8_t, 4> segments{0, 0, 1, 1};
    std::array<std::int8_t, 4> segmented{};
    skimagexx::morphological_chan_vese_into(two_regions.data(), segments.data(), segmented.data(),
                                          {2, 2}, 2, 0, 1.0, 1.0, false);
    if (segmented != segments)
        return 2;

    const std::array<float, 8> uniform{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<std::int8_t, 8> seed{1, 0, 0, 0, 0, 0, 0, 0};
    std::array<std::int8_t, 8> grown{};
    skimagexx::morphological_geodesic_active_contour_into(uniform.data(), seed.data(), grown.data(),
                                                         {2, 2, 2}, 1, 0, 0.5, 1, false);
    if (!std::all_of(grown.begin(), grown.end(), [](std::int8_t value) { return value == 1; }))
        return 3;
}
