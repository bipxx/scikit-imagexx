#include <skimagexx/data_checkerboard.hpp>

#include <array>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("data_checkerboard SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::data_checkerboard_elements;
    using skimagexx::data_checkerboard_u8_into;

    std::vector<std::uint8_t> image(data_checkerboard_elements, 1);
    require(data_checkerboard_u8_into(image.data(), image.size()));
    require(image[0] == 255);
    require(image[data_checkerboard_elements - 1] == 255);

    std::array<std::uint8_t, 4> small{7, 8, 9, 10};
    const auto saved = small;
    require(!data_checkerboard_u8_into(small.data(), small.size()));
    require(!data_checkerboard_u8_into(nullptr, data_checkerboard_elements));
    require(small == saved);
}
