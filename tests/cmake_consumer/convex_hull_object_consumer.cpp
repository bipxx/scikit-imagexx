#include "skimagexx/convex_hull_object.hpp"

#include <algorithm>
#include <array>
#include <cstdint>

int main() {
    const std::array<std::uint8_t, 20> input = {
        0, 1, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 0, 1, 0,
    };
    std::array<std::uint8_t, 20> output{};
    auto separate = input;
    separate[7] = 1;
    if (!skimagexx::convex_hull_object_into(input.data(), 4, 5, 1, output.data()) ||
        output != separate)
        return 1;
    auto joined = separate;
    joined[17] = 1;
    if (!skimagexx::convex_hull_object_into(input.data(), 4, 5, 2, output.data()) ||
        output != joined)
        return 2;
    const std::array<std::uint8_t, 4> background{};
    std::fill(output.begin(), output.end(), std::uint8_t{1});
    if (!skimagexx::convex_hull_object_into(background.data(), 2, 2, 2, output.data()) ||
        !std::all_of(output.begin(), output.begin() + 4, [](std::uint8_t b) { return b == 0; }))
        return 3;
    return skimagexx::convex_hull_object_into(input.data(), 4, 5, 3, output.data()) ? 4 : 0;
}
