#include <skimagexx/draw_haar_like_feature.hpp>

#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace skimagexx;
    const std::array<std::uint8_t,6> image{};
    std::array<double,18> output{};
    const std::array<double,3> red{1,0,0}, green{0,1,0};
    HaarFeatures features{{{-2,-3,1,2},{0,1,0,2}}};
    const auto draw=[&](const std::vector<std::size_t>* order=nullptr) {
        return draw_haar_like_feature_into(image.data(),image.size(),PixelType::uint8,2,3,1,
            output.data(),output.size(),features,order,red,green,.5);
    };
    // The first wrapped rectangle overlaps itself but blends only once.
    if (!draw() || output[0]!=0 || output[1]!=.5 || output[2]!=0
        || output[3]!=.5 || output[4]!=.25 || output[5]!=0 || output[16]!=.5)
        throw std::runtime_error("Haar wrapped/ordered paint differs");
    const std::vector<std::size_t> empty{};
    if (!draw(&empty) || output!=std::array<double,18>{})
        throw std::runtime_error("Haar empty selection differs");
    const std::vector<std::size_t> invalid{1};
    if (draw(&invalid)) throw std::runtime_error("Haar admitted an invalid selection");
    features[0][0].row_start=-3;
    if (draw()) throw std::runtime_error("Haar admitted an out-of-bounds rectangle");
    std::cout<<"Haar SDK owning render: wrapped overlap, selection and bounds passed\n";
}
