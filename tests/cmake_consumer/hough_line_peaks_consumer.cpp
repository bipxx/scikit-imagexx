#include <skimagexx/hough_line_peaks.hpp>

#include <array>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("hough_line_peaks SDK result or validation differs");
}

}  // namespace

int main() {
    constexpr int height = 11, width = 12;
    std::array<std::uint64_t, height * width> input{};
    auto scan = [&](std::optional<double> threshold = std::nullopt) {
        return skimagexx::hough_line_peak_indices(input.data(), input.size(), height, width, 1, 1, threshold);
    };
    require(scan().empty());
    input[3 * width] = 12;
    input[8 * width + 11] = 10;
    input[2 * width + 11] = 9;
    const auto periodic = scan();
    require(periodic.size() == 2);
    require(periodic[0].votes == 12 && periodic[0].distance == 3 && periodic[0].angle == 0);
    require(periodic[1].votes == 9 && periodic[1].distance == 2 && periodic[1].angle == 11);
    require(scan(9.0).size() == 1);
    input.fill(0);
    for (int row : {2, 3})
        for (int column : {2, 3})
            input[row * width + column] = 7;
    const auto plateau = scan();
    require(plateau.size() == 1 && plateau[0].votes == 7 && plateau[0].distance == 2 && plateau[0].angle == 2);
    bool failed = false;
    try {
        skimagexx::hough_line_peak_indices(input.data(), input.size() - 1, height, width, 1, 1);
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    require(failed);
    input[0] = 1048577;
    failed = false;
    try {
        scan();
    } catch (const std::overflow_error&) {
        failed = true;
    }
    require(failed && input[0] == 1048577);
}
