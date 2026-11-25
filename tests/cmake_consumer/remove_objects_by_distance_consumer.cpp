#include <skimagexx/remove_objects_by_distance.hpp>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("remove_objects_by_distance installed SDK result differs");
}

template<class T>
void check_line() {
    const std::array<std::size_t, 1> shape{5};
    const std::array<double, 1> spacing{0.5};
    const std::array<T, 5> labels{2, 0, 1, 0, 3};
    std::array<T, 5> output{};
    auto filter = [&](double radius, const double* priority, std::size_t count) {
        return skimagexx::remove_objects_by_distance_into(
            labels.data(), labels.size(), shape.size(), shape.data(), spacing.data(),
            radius, skimagexx::ObjectDistanceNorm::euclidean, priority, count,
            output.data(), output.size());
    };
    require(filter(1, nullptr, 0));
    require(output == std::array<T, 5>{2, 0, 0, 0, 3});
    const std::array<double, 4> priority{0, 4, 1, 2};
    require(filter(1, priority.data(), priority.size()));
    require(output == std::array<T, 5>{0, 0, 1, 0, 0});
    require(!filter(1, priority.data(), priority.size() - 1));
    require(output == std::array<T, 5>{0, 0, 1, 0, 0});
    require(filter(0, nullptr, 0));
    require(output == labels);
}

void check_volume() {
    const std::array<std::size_t, 3> shape{2, 3, 3};
    const std::array<double, 3> spacing{2, 0.75, 1.25};
    std::array<std::int32_t, 18> labels{};
    labels[1] = 7;
    labels[9] = 8;
    labels[16] = 9;
    std::array<std::int32_t, 18> output{};
    require(skimagexx::remove_objects_by_distance_into(
        labels.data(), labels.size(), shape.size(), shape.data(), spacing.data(),
        2., skimagexx::ObjectDistanceNorm::chebyshev, nullptr, 0,
        output.data(), output.size()));
    std::array<std::int32_t, 18> expected{};
    expected[16] = 9;
    require(output == expected);
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
