#include <skimagexx/orb.hpp>
#include <skimagexx/image.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("ORB SDK assertion failed");
}

} // namespace

int main(int argc, char**) {
    require(skimagexx::get_num_threads() == 1);
    if (argc > 1) require(!skimagexx::has_avx2());
    constexpr std::size_t rows = 73, cols = 81;
    const std::array<double, 9> harris_weights{
        0.00013383062461474175, 0.0044318616200312655, 0.05399112742070441,
        0.24197144565660073, 0.39894346935609776, 0.24197144565660073,
        0.05399112742070441, 0.0044318616200312655, 0.00013383062461474175,
    };
    const std::vector<double> smooth_weights{
        3.4256063613481042e-06, 0.040387616547241253, 0.9192179156927948,
        0.040387616547241253, 3.4256063613481042e-06,
    };
    std::vector<std::uint8_t> image(rows * cols);
    for (std::size_t row = 24; row <= 54; ++row)
        std::fill_n(image.data() + row * cols + 27, 28, std::uint8_t(255));
    std::vector<double> prepared(image.size());
    require(skimagexx::orb_prepare(image.data(), prepared.data(), skimagexx::PixelType::uint8, rows, cols));
    require(prepared[24 * cols + 27] == 1 && prepared[23 * cols + 27] == 0);

    const auto result = skimagexx::orb_detect_octave(
        prepared.data(), false, rows, cols, 9, .08, .04, harris_weights);
    require(result.had_fast_peaks && result.points.size() == 4);
    const auto& first = result.points.front();
    require(first.row == 24 && first.col == 27);
    require(std::abs(first.orientation - std::atan2(1., 1.)) < 1e-13);
    require(first.response > 0);

    constexpr double points[]{24, 27, 5, 10, 19.25, 19.25};
    constexpr double orientations[]{0, 0, -1};
    const auto descriptors = skimagexx::orb_extract_octave(
        prepared.data(), false, rows, cols, points, orientations, 3);
    require(descriptors.mask == std::vector<std::uint8_t>{1, 0, 1});
    require(descriptors.data.size() == 2 * skimagexx::orb_descriptor_size);
    require(descriptors.data.front() == 1);

    constexpr std::size_t reduced_rows = 61, reduced_cols = 68;
    std::vector<double> reduced(reduced_rows * reduced_cols);
    skimagexx::orb_reduce(prepared.data(), reduced.data(), false, rows, cols,
                          reduced_rows, reduced_cols, smooth_weights);
    require(reduced.front() == 0 && std::abs(reduced[29 * reduced_cols + 29] - 1) < 1e-14);

    constexpr std::size_t large_rows = 517, large_cols = 523;
    std::vector<float> noise(large_rows * large_cols);
    std::uint32_t state = 66133;
    for (auto& value : noise) {
        state = state * 1664525u + 1013904223u;
        value = float(state >> 8) / float(1u << 24);
    }
    const auto detect = [&] {
        return skimagexx::orb_detect_octave(noise.data(), true, large_rows, large_cols,
                                           9, .08, .04, harris_weights);
    };
    const auto serial = detect();
    skimagexx::set_num_threads(2);
    require(skimagexx::get_num_threads() == 2);
    const auto parallel = detect();
    skimagexx::set_num_threads(1);
    require(serial.had_fast_peaks && parallel.had_fast_peaks);
    require(!serial.points.empty() && serial.points.size() == parallel.points.size());
    for (std::size_t i = 0; i < serial.points.size(); ++i) {
        const auto& a = serial.points[i];
        const auto& b = parallel.points[i];
        require(a.row == b.row && a.col == b.col
                && a.orientation == b.orientation && a.response == b.response);
    }
}
