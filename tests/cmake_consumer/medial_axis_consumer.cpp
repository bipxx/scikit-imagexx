#include "skimagexx/medial_axis.hpp"

#include <array>
#include <vector>

int main() {
    using skimagexx::medial_axis_ordered;
    const std::array<std::uint8_t, 4> foreground{1, 1, 1, 1};
    const std::array<double, 4> distances{1.0, 1.0, 1.0, 1.0};
    const std::array<std::int64_t, 4> forward{0, 1, 2, 3};
    const std::array<std::int64_t, 4> reverse{3, 2, 1, 0};
    const std::array<std::int64_t, 4> duplicate{0, 1, 1, 3};
    std::vector<std::uint8_t> result;
    if (!medial_axis_ordered(foreground.data(), distances.data(), 4, forward.data(), 4, 2, 2, result)
        || result != std::vector<std::uint8_t>{0, 0, 1, 1}) {
        return 1;
    }
    if (!medial_axis_ordered(foreground.data(), distances.data(), 4, reverse.data(), 4, 2, 2, result)
        || result != std::vector<std::uint8_t>{1, 1, 0, 0}) {
        return 2;
    }
    if (medial_axis_ordered(foreground.data(), distances.data(), 4, duplicate.data(), 4, 2, 2, result)
        || result != std::vector<std::uint8_t>{1, 1, 0, 0}) {
        return 3;
    }
}
