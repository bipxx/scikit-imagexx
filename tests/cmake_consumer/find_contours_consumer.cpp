#include <skimagexx/find_contours.hpp>

#include <array>
#include <limits>

namespace {

bool point(skimagexx::ContourPoint actual, double row, double column) {
    return actual.row == row && actual.column == column;
}

}  // namespace

int main() {
    const std::array<double, 4> saddle{0, 1, 1, 0};
    skimagexx::Contours output;
    if (!skimagexx::find_contours(saddle.data(), nullptr, 2, 2, {0.5, false, false}, output)
        || output.size() != 2 || output[0].size() != 2 || output[1].size() != 2
        || !point(output[0][0], 0.5, 1) || !point(output[0][1], 0, 0.5)
        || !point(output[1][0], 0.5, 0) || !point(output[1][1], 1, 0.5)) {
        return 1;
    }
    if (!skimagexx::find_contours(saddle.data(), nullptr, 2, 2, {0, true, true}, output)
        || output.size() != 2 || !point(output[0][0], 0.5, 0)
        || !point(output[1][0], 0.5, 1)) {
        return 2;
    }
    const std::array<std::uint8_t, 4> mask{1, 1, 1, 0};
    const std::array<float, 4> floats{0, 1, 1, 0};
    if (!skimagexx::find_contours(floats.data(), mask.data(), 2, 2, {0.5, false, true}, output)
        || !output.empty()) {
        return 3;
    }
    const std::array<std::uint8_t, 4> bytes{0, 11, 11, 0};
    if (!skimagexx::find_contours(bytes.data(), nullptr, 2, 2, {5.5, false, true}, output)
        || output.size() != 2 || !point(output[0][0], 0.5, 0)) {
        return 4;
    }
    auto nonfinite = saddle;
    nonfinite[3] = std::numeric_limits<double>::quiet_NaN();
    if (skimagexx::find_contours(nonfinite.data(), nullptr, 2, 2, {0.5, false, false}, output)
        || skimagexx::find_contours(bytes.data(), nullptr, 2, 2, {0, true, false}, output)
        || skimagexx::find_contours(saddle.data(), nullptr, 1, 4, {0.5, false, false}, output)
        || output.size() != 2 || !point(output[0][0], 0.5, 0)) {
        return 5;
    }
}
