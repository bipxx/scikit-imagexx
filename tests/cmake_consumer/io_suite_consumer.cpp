#include "skimagexx/io_suite.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

namespace io = skimagexx::io;

int main() {
    const std::array<int, 6> pixels{0, 1, 2, 3, 4, 5};
    std::array<int, 6> output{};
    const std::ptrdiff_t shape[]{2, 3};
    const std::ptrdiff_t strides[]{3 * sizeof(int), -static_cast<std::ptrdiff_t>(sizeof(int))};
    io::copy_image(pixels.data() + 2, output.data(), shape, strides, 2, sizeof(int));
    if (output != std::array<int, 6>{2, 1, 0, 5, 4, 3}) return 1;
    const std::ptrdiff_t dense_strides[]{3 * sizeof(int), sizeof(int)};
    io::copy_image(pixels.data(), output.data(), shape, dense_strides, 2, sizeof(int));
    if (output != pixels) return 2;

    const auto netpbm = io::netpbm_header(2, 1, 1) + std::string("\0\xff", 2);
    const auto decoded = io::netpbm_view(netpbm);
    if (!decoded || decoded->width != 2 || decoded->channels != 1 || decoded->height != 1
        || decoded->pixels != std::string_view("\0\xff", 2)) return 3;
    if (io::netpbm_view("P6\n2 1\n255\nabcde")) return 4;
    if (io::netpbm_view("P5\n18446744073709551615 2\n255\nx")) return 5;

    const auto sift = io::feature_layout("1 2\r\n-0 .5 -1e1 +2\n 127 255\n", io::FeatureFormat::sift);
    std::array<double, 6> values{};
    if (!sift || sift->count != 1 || sift->descriptor_size != 2 || sift->values_per_record != 6
        || !io::read_feature_values(sift->values, values.data(), values.size())
        || values != std::array<double, 6>{0, .5, -10, 2, 127, 255} || !std::signbit(values[0])) return 6;
    const auto surf = io::feature_layout("2\n1\n2 3 0 1 0 -1 .125\n", io::FeatureFormat::surf);
    if (!surf || surf->descriptor_size != 1 || surf->values_per_record != 7) return 7;
    if (io::read_feature_values("1 2junk", values.data(), 2)
        || io::read_feature_values("1 2 3", values.data(), 2)) return 8;
}
