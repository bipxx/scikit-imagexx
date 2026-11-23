#include "skimagexx/skeletonize.hpp"

#include <array>
#include <vector>

int main() {
    using skimagexx::SkeletonizeInput;
    using skimagexx::skeletonize_zhang;
    const std::array<std::uint8_t, 9> block{255, 2, 3, 4, 250, 6, 7, 8, 9};
    std::vector<std::uint8_t> result{42};
    if (!skeletonize_zhang(block.data(), 3, 3, SkeletonizeInput::uint8, result)
        || result != std::vector<std::uint8_t>{0, 1, 0, 1, 0, 0, 0, 0, 0}) {
        return 1;
    }
    const std::array<std::uint8_t, 5> line{0, 1, 1, 0, 1};
    if (!skeletonize_zhang(line.data(), 1, 5, SkeletonizeInput::boolean, result)
        || result != std::vector<std::uint8_t>(line.begin(), line.end())) {
        return 2;
    }
    const std::array<std::uint8_t, 1> ordinary_uint8{2};
    if (skeletonize_zhang(ordinary_uint8.data(), 1, 1, SkeletonizeInput::boolean, result)
        || skeletonize_zhang(line.data(), 0, 5, SkeletonizeInput::boolean, result)
        || result != std::vector<std::uint8_t>(line.begin(), line.end())) {
        return 3;
    }
}
