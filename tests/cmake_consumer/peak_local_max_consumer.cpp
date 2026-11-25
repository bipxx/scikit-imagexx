#include <skimagexx/peak_local_max.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

int main() {
    using skimagexx::PeakCoordinate;
    std::array<double, 121> image{};
    std::array<std::int32_t, 121> labels{};
    image[2 * 11 + 2] = image[2 * 11 + 3] = 3;
    image[7 * 11 + 7] = 4;
    labels[2 * 11 + 2] = labels[2 * 11 + 3] = 1;
    labels[7 * 11 + 7] = 2;
    auto points = skimagexx::peak_local_max(image.data(), 11, 11);
    if (points != std::vector<PeakCoordinate>{{7, 7}, {2, 2}, {2, 3}})
        throw std::runtime_error("peak_local_max SDK intensity or tie order differs");
    skimagexx::LocalPeakOptions options;
    options.suppress_connected = true;
    options.norm = skimagexx::PeakNorm::euclidean;
    points = skimagexx::peak_local_max(image.data(), 11, 11, options, labels.data());
    if (points != std::vector<PeakCoordinate>{{2, 2}, {7, 7}})
        throw std::runtime_error("corner_peaks SDK suppression or label order differs");

    for (int row = 0; row < 11; ++row)
        for (int col = 0; col < 11; ++col)
            image[static_cast<std::size_t>(row * 11 + col)] = (row < 5) == (col < 5);
    const PeakCoordinate corner{5, 5};
    std::array<double, 2> subpixel{};
    skimagexx::corner_subpix_into(image.data(), 11, 11, &corner, 1, 7, 2.01, 0.49, subpixel.data());
    if (!(std::abs(subpixel[0] - 4.5) < 1e-12 && std::abs(subpixel[1] - 4.5) < 1e-12))
        throw std::runtime_error("corner_subpix SDK refinement differs");
}
