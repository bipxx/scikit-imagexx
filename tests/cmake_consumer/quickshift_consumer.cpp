#include <skimagexx/quickshift.hpp>

#include <array>

int main() {
    constexpr std::array<double, 3> input{0, 0, 12};
    constexpr std::array<double, 3> noise{0, 0.001, 0};
    std::array<std::ptrdiff_t, 3> labels{};
    skimagexx::quickshift_into(input.data(), noise.data(), labels.data(), 1, 3, 1, 1.0, 2.0);
    return labels == std::array<std::ptrdiff_t, 3>{0, 0, 1} ? 0 : 1;
}
