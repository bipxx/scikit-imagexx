#include <skimagexx/expand_labels.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("expand_labels installed SDK output or validation differs");
}

template<class T>
void check_line() {
    const std::array<std::size_t, 1> shape{5};
    const std::array<double, 1> spacing{0.5};
    const T label = std::numeric_limits<T>::max();
    const std::array<T, 5> input{label, 0, 0, 0, T(2)};
    std::array<T, 5> output{};
    auto expand = [&](double distance) {
        skimagexx::expand_labels_into(input.data(), input.size(), 1, shape.data(), spacing.data(),
                                      distance, output.data(), output.size());
    };
    expand(1);
    require(output == std::array<T, 5>{label, label, label, T(2), T(2)});
    expand(-0.0);
    require(output == input);
    expand(std::numeric_limits<double>::quiet_NaN());
    require(output == std::array<T, 5>{});
}

void check_volume() {
    const std::array<std::size_t, 3> shape{3, 3, 3};
    const std::array<double, 3> spacing{0.75, 1.5, 0.75};
    std::array<std::int16_t, 27> input{};
    input[1] = 4;
    input[3] = 8;
    input[9] = 16;
    input[26] = -7;
    std::array<std::int16_t, 27> output{};
    skimagexx::expand_labels_into(input.data(), input.size(), shape.size(), shape.data(),
                                  spacing.data(), 1.5, output.data(), output.size());
    const std::array<std::int16_t, 27> expected{
        16, 4, 4, 8, 8, 8, 8, 0, -7,
        16, 16, 4, 8, 8, 0, 0, -7, -7,
        16, 16, 0, 8, 0, -7, -7, -7, -7};
    require(output == expected);

    input.fill(0);
    skimagexx::expand_labels_into(input.data(), input.size(), shape.size(), shape.data(),
                                  spacing.data(), std::numeric_limits<double>::infinity(),
                                  output.data(), output.size());
    require(output == input);
    try {
        skimagexx::expand_labels_into(input.data(), input.size(), shape.size(), shape.data(),
                                      spacing.data(), 1.5, output.data(), output.size() - 1);
        require(false);
    } catch (const std::invalid_argument&) {}
    require(output == input);
}

}  // namespace

int main() {
    check_line<std::int8_t>();
    check_line<std::uint8_t>();
    check_line<std::int16_t>();
    check_line<std::uint16_t>();
    check_line<std::int32_t>();
    check_line<std::uint32_t>();
    check_line<std::int64_t>();
    check_line<std::uint64_t>();
    check_volume();
}
