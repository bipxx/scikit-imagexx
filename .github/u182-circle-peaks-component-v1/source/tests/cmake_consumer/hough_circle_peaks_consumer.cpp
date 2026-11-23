#include "skimagexx/hough_circle_peaks.hpp"

#include <array>
#include <limits>
#include <stdexcept>

int main() {
    std::array<double, 10 * 12> field{};
    field[2 * 12] = 20;
    field[8 * 12 + 11] = 18;
    field[11] = 17;
    const auto peaks = skimagexx::hough_circle_peak_indices(field.data(), field.size(), 10, 12, 1, 1, 0.0);
    if (!peaks || peaks->size() != 2 || (*peaks)[0].vote != 20 || (*peaks)[0].column != 0 ||
        (*peaks)[0].row != 2 || (*peaks)[1].vote != 17 || (*peaks)[1].column != 11 || (*peaks)[1].row != 0)
        return 1;

    field.fill(0);
    field[2 * 12 + 2] = 3.125;
    field[2 * 12 + 3] = 2.75;
    field[4 * 12 + 5] = 2.5;
    field[4 * 12 + 6] = 2.5;
    const auto plateaus = skimagexx::hough_circle_peak_indices(field.data(), field.size(), 10, 12, 0, 0);
    if (!plateaus || plateaus->size() != 2 || (*plateaus)[0].column != 2 || (*plateaus)[0].vote != 3.125 ||
        (*plateaus)[1].column != 6 || (*plateaus)[1].vote != 2.5)
        return 2;

    const auto empty = skimagexx::hough_circle_peak_indices(field.data(), field.size(), 10, 12, 1, 1, 20.0);
    if (!empty || !empty->empty())
        return 3;
    field[0] = std::numeric_limits<double>::infinity();
    if (skimagexx::hough_circle_peak_indices(field.data(), field.size(), 10, 12, 1, 1))
        return 4;
    try {
        skimagexx::hough_circle_peak_indices(field.data(), field.size() - 1, 10, 12, 1, 1);
        return 5;
    } catch (const std::invalid_argument&) {
    }
    return 0;
}
