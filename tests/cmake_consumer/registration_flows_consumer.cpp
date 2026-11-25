#include <skimagexx/registration_flows.hpp>

#include <algorithm>
#include <array>
#include <cmath>

int main() {
    constexpr std::size_t height = 35;
    constexpr std::size_t width = 41;
    std::array<float, height * width> image{};
    std::array<float, 2 * height * width> flow{};
    for (std::size_t i = 0; i < image.size(); ++i)
        image[i] = static_cast<float>((i * i + 3 * i) % 31) / 32;
    const auto zero_flow = [&] {
        return std::all_of(flow.begin(), flow.end(), [](float x) {
            return std::abs(x) <= 1e-6f;
        });
    };
    flow.fill(2);
    if (!skimagexx::optical_flow_ilk_into(image.data(), image.data(), flow.data(), height, width) ||
        !zero_flow())
        return 1;
    flow.fill(2);
    skimagexx::optical_flow_tvl1_into(image.data(), image.data(), flow.data(), height, width);
    return zero_flow() ? 0 : 2;
}
