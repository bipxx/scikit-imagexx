#include <skimagexx/slic.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("SLIC installed SDK contract differs");
}
}

int main() {
    using Label = std::ptrdiff_t;
    const std::array<std::size_t, 4> plane{1, 2, 4, 1};
    const std::array<std::uint8_t, 8> pixels{0, 0, 255, 255, 0, 0, 255, 255};
    std::array<Label, 8> output{};
    skimagexx::SlicOptions options;
    options.n_segments = 2;
    options.max_num_iter = 3;
    require(skimagexx::slic_into(pixels.data(), pixels.size(), plane, options, output.data(), output.size()));
    require(output == std::array<Label, 8>{1, 1, 2, 2, 1, 1, 2, 2});

    const std::array<float, 8> features{0, 0, .1f, .1f, 0, 0, .1f, .1f};
    require(skimagexx::slic_prepared_into(features.data(), features.size(), plane, options,
                                         output.data(), output.size()));
    require(output == std::array<Label, 8>{1, 1, 2, 2, 1, 1, 2, 2});
    options.max_size_factor = .25;
    require(skimagexx::slic_into(pixels.data(), pixels.size(), plane, options, output.data(), output.size()));
    require(output == std::array<Label, 8>{1, 1, 1, 1, 1, 1, 1, 1});

    const std::array<std::size_t, 4> volume{3, 3, 4, 1};
    std::array<float, 36> ramp{};
    std::array<Label, 36> labels{};
    for (std::size_t i = 0; i < ramp.size(); ++i)
        ramp[i] = static_cast<float>(i);
    options.n_segments = 4;
    options.max_size_factor = 3;
    options.compactness = .25;
    options.spacing = {1.5, .75, 1.};
    options.slic_zero = true;
    options.start_label = 0;
    require(skimagexx::slic_into(ramp.data(), ramp.size(), volume, options, labels.data(), labels.size()));
    require(labels == std::array<Label, 36>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                            0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                                            0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1});
    try {
        skimagexx::slic_into(ramp.data(), ramp.size(), volume, options, labels.data(), labels.size() - 1);
        require(false);
    } catch (const std::invalid_argument&) {}
}
