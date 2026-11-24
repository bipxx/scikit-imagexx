#include <skimagexx/data_shepp_logan_phantom.hpp>

#include <array>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("data_shepp_logan_phantom SDK assertion failed");
    }
}

}  // namespace

int main() {
    using skimagexx::data_shepp_logan_phantom_columns;
    using skimagexx::data_shepp_logan_phantom_elements;
    using skimagexx::data_shepp_logan_phantom_f64_into;

    std::vector<double> image(data_shepp_logan_phantom_elements, -1.0);
    require(data_shepp_logan_phantom_f64_into(image.data(), image.size()));
    require(image[0] == 0.0);
    require(image[200 * data_shepp_logan_phantom_columns + 200] == 0.2);
    require(image[data_shepp_logan_phantom_elements - 1] == 0.0);

    std::array<double, 4> small{1.0, 2.0, 3.0, 4.0};
    const auto saved = small;
    require(!data_shepp_logan_phantom_f64_into(small.data(), small.size()));
    require(!data_shepp_logan_phantom_f64_into(nullptr, data_shepp_logan_phantom_elements));
    require(small == saved);
}
