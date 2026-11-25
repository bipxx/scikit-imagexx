#include <skimagexx/data_loaders_b.hpp>

#include <array>
#include <limits>
#include <stdexcept>

int main() {
    const std::array<std::uint8_t, 6> input{1, 2, 3, 4, 5, 6};
    const std::array<std::uint8_t, 6> reversed{4, 5, 6, 1, 2, 3};
    std::array<std::uint8_t, 6> output{};
    skimagexx::copy_data_rows(input.data(), input.size(), output.data(), output.size(), 2, 3);
    if (output != input) {
        return 1;
    }
    skimagexx::copy_data_rows(input.data(), input.size(), output.data(), output.size(), 2, 3,
                             skimagexx::DataRowOrder::bottom_up);
    if (output != reversed) {
        return 2;
    }
    try {
        skimagexx::copy_data_rows(input.data(), input.size(), output.data(), output.size() - 1, 2, 3);
        return 3;
    } catch (const std::invalid_argument&) {
    }
    try {
        skimagexx::copy_data_rows(input.data(), input.size(), output.data(), output.size(),
                                 std::numeric_limits<std::size_t>::max(), 2);
        return 4;
    } catch (const std::invalid_argument&) {
    }
    return output == reversed ? 0 : 5;
}
