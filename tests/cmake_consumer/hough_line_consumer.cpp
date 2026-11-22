#include <skimagexx/hough_line.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("hough_line installed SDK output or validation differs");
}

template<class Operation>
void rejects(Operation operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    require(false);
}

}  // namespace

int main() {
    constexpr std::size_t height = 2;
    constexpr std::size_t width = 3;
    std::array<std::uint8_t, height * width> image{0, 0, 0, 7, 0, 0};
    const std::array<double, 3> cosines{1.0, 1.0, 1.0};
    const std::array<double, 3> sines{0.49999999999999994, -0.49999999999999994, -0.0};
    const auto angles = sines.size();
    const int offset = skimagexx::hough_line_offset(height, width, angles);
    require(offset == 4);
    std::array<std::uint64_t, 27> accumulator{};
    auto vote = [&](const std::uint8_t* mask, const double* sine, std::size_t output_count) {
        skimagexx::hough_line_into(mask, image.size(), height, width, cosines.data(), sine, angles,
                                  accumulator.data(), output_count);
    };
    vote(image.data(), sines.data(), accumulator.size());
    std::array<std::uint64_t, 27> expected{};
    expected[(offset + 1) * angles] = 1;
    expected[(offset - 1) * angles + 1] = 1;
    expected[offset * angles + 2] = 1;
    require(accumulator == expected);

    rejects([&] { vote(image.data(), sines.data(), accumulator.size() - 1); });
    rejects([&] { vote(reinterpret_cast<const std::uint8_t*>(accumulator.data()),
                       sines.data(), accumulator.size()); });
    auto invalid_sines = sines;
    invalid_sines[0] = std::numeric_limits<double>::infinity();
    rejects([&] { vote(image.data(), invalid_sines.data(), accumulator.size()); });
    invalid_sines[0] = 99.0;
    rejects([&] { vote(image.data(), invalid_sines.data(), accumulator.size()); });
    rejects([&] { skimagexx::hough_line_offset(4097, 1, 1); });
    require(accumulator == expected);
}
