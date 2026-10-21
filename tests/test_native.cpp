#include "skimagexx/image.hpp"
#include "skimagexx/convert.hpp"
#include "skimagexx/color.hpp"
#include "skimagexx/perceptual.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <algorithm>

void require(bool condition) {
    if (!condition) throw std::runtime_error("native check failed");
}
int main() {
    using namespace skimagexx;
    require(get_num_threads() == 1);
#ifdef _OPENMP
    const int thread_counts[] = {1, 4};
#else
    const int thread_counts[] = {1};
#endif
    for (int threads : thread_counts) {
        set_num_threads(threads);
        require(get_num_threads() == threads);
        const double rgb[] = {1, 0, 0, 0, 1, 0};
        const auto gray = rgb2gray(rgb, 1, 2);
        require(gray[0] == 0.2125 && gray[1] == 0.7154);
        const std::uint8_t rgb8[] = {255,0,0,0,255,0};
        require(rgb2gray(rgb8,1,2) == gray);
        std::vector<double> direct(2);
        rgb2gray_into(rgb,direct.data(),1,2);
        require(direct == gray);
        std::vector<double> a(513*515, 0.5);
        const auto blur = gaussian(a.data(),513,515);
        const auto edges = sobel(a.data(),513,515);
        direct.assign(a.size()+2,-999.0);
        gaussian_into(a.data(),direct.data()+1,513,515);
        require(direct.front() == -999.0 && direct.back() == -999.0);
        require(std::equal(blur.begin(),blur.end(),direct.begin()+1));
        sobel_into(a.data(),direct.data()+1,513,515);
        require(direct.front() == -999.0 && direct.back() == -999.0);
        require(std::equal(edges.begin(),edges.end(),direct.begin()+1));
        for (std::size_t i = 0; i < a.size(); ++i) {
            require(std::abs(blur[i]-0.5) < 1e-14);
            require(edges[i] == 0);
        }
        const std::uint8_t binary[] = {10, 10, 200, 200};
        require(threshold_otsu(binary,2,2) == 10);
        require(threshold_mean(binary,2,2) == 105);
        std::vector<std::uint8_t> inverted(4), tripled(12);
        invert_into(binary,inverted.data(),2,2);
        gray2rgb_into(binary,tripled.data(),2,2);
        for (std::size_t i = 0; i < 4; ++i) {
            require(inverted[i] == 255-binary[i]);
            for (std::size_t c = 0; c < 3; ++c) require(tripled[3*i+c] == binary[i]);
        }
        for (auto filter : {EdgeFilter::sobel,EdgeFilter::prewitt,EdgeFilter::scharr}) {
            gradient_into(a.data(),direct.data()+1,513,515,filter,0);
            for (std::size_t i = 1; i <= a.size(); ++i) require(direct[i] == 0);
        }
        const auto counts = histogram_u8(binary,2,2);
        require(counts[10] == 2 && counts[200] == 2);
        std::vector<double> equalized(4);
        equalize_hist_into(binary,equalized.data(),2,2);
        require(equalized[0] == 0.5 && equalized[3] == 1.0);
        const auto range = intensity_bounds(binary,2,2);
        require(range.first == 10 && range.second == 200);
        rescale_into(binary,inverted.data(),2,2,10,200,0,255);
        require(inverted[0] == 0 && inverted[3] == 255);
        exposure_into(binary,inverted.data(),2,2,ExposureTransform::gamma,1,1,false);
        require(std::equal(inverted.begin(),inverted.end(),binary));
        const double values[] = {0.0,0.25,1.0};
        std::vector<double> corrected(3);
        exposure_into(values,corrected.data(),1,3,ExposureTransform::gamma,0.5,1,false);
        require(corrected[0] == 0 && corrected[1] == 0.5 && corrected[2] == 1);
        const std::uint8_t bytes[] = {0,1,127,128,255};
        float normalized[5];
        convert_into(bytes,normalized,5,PixelType::uint8,PixelType::float32);
        require(normalized[0] == 0 && normalized[4] == 1);
        std::vector<double> color_rgb(3*19), yuv(color_rgb.size()), roundtrip(color_rgb.size());
        for (std::size_t j = 0; j < color_rgb.size(); ++j) color_rgb[j] = double(j)/double(color_rgb.size());
        color_transform_into(color_rgb.data(),yuv.data(),19,PixelType::float64,ColorTransform::rgb2yuv);
        color_transform_into(yuv.data(),roundtrip.data(),19,PixelType::float64,ColorTransform::yuv2rgb);
        for (std::size_t j = 0; j < color_rgb.size(); ++j) require(std::abs(roundtrip[j]-color_rgb[j]) < 2e-15);
        std::vector<double> planar(color_rgb.size()), strided(yuv.size());
        for (std::size_t j = 0; j < 19; ++j)
            for (std::size_t channel = 0; channel < 3; ++channel) planar[19*channel+j] = color_rgb[3*j+channel];
        color_transform_strided_into(planar.data(),strided.data(),19,1,19,ColorTransform::rgb2yuv);
        for (std::size_t j = 0; j < yuv.size(); ++j) require(strided[j] == yuv[j]);
        color_transform_strided_into(planar.data()+18,strided.data(),19,-1,19,ColorTransform::rgb2yuv);
        for (std::size_t j = 0; j < 19; ++j)
            for (std::size_t channel = 0; channel < 3; ++channel) require(strided[3*j+channel] == yuv[3*(18-j)+channel]);
        const std::uint8_t white[] = {255,255,255};
        double xyz[3];
        color_transform_into(white,xyz,1,PixelType::uint8,ColorTransform::rgb2xyz);
        require(std::abs(xyz[0]-0.950456) < 1e-14 && std::abs(xyz[1]-1) < 1e-14 && std::abs(xyz[2]-1.088754) < 1e-14);
        color_transform_into(nullptr,nullptr,0,PixelType::float32,ColorTransform::rgb2yuv);
        const double hsv_red[] = {1,0,0}, hsv_gray[] = {0.4,0.4,0.4};
        double hsv[3];
        auto perception = perceptual_into(hsv_red,hsv,1,PixelType::float64,PerceptualTransform::rgb2hsv);
        require(!perception.unsupported && hsv[0] == 0 && hsv[1] == 1 && hsv[2] == 1);
        perceptual_into(hsv_gray,hsv,1,PixelType::float64,PerceptualTransform::rgb2hsv);
        require(hsv[0] == 0 && hsv[1] == 0 && hsv[2] == 0.4);
        const double xyz_source[] = {0.1,0.4,0.7};
        double lab[3], xyz_back[3];
        perceptual_into(xyz_source,lab,1,PixelType::float64,PerceptualTransform::xyz2lab);
        perceptual_into(lab,xyz_back,1,PixelType::float64,PerceptualTransform::lab2xyz);
        for (int j = 0; j < 3; ++j) require(std::abs(xyz_source[j]-xyz_back[j]) < 2e-15);
        const float invalid_lab[] = {0,0,100};
        float xyz32[3];
        perception = perceptual_into(invalid_lab,xyz32,1,PixelType::float32,PerceptualTransform::lab2xyz);
        require(perception.negative_z == 1);
        lab_scalar_negative_z_into(invalid_lab,xyz_back,PerceptualTransform::lab2xyz,{0.95047,1,1.08883});
        require(xyz_back[0] < 0 && xyz_back[2] < 0);
        std::uint16_t expanded[5];
        convert_into(bytes,expanded,5,PixelType::uint8,PixelType::uint16);
        require(expanded[1] == 257 && expanded[4] == 65535);
        std::uint8_t restored[5];
        convert_into(expanded,restored,5,PixelType::uint16,PixelType::uint8);
        require(std::equal(restored,restored+5,bytes));
        const std::int64_t small[] = {-65535,0,127};
        const auto info = convert_into(small,restored,3,PixelType::int64,PixelType::uint8);
        require(info.downcast_without_scaling && info.maximum_signed == 127);
        require(restored[0] == 1 && restored[2] == 127);
    }
    set_num_threads(1);
    require(get_num_threads() == 1);
    bool rejected = false;
    try { sobel<double>(nullptr,2,2); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    rejected = false;
    try { convert_into(nullptr,nullptr,1,PixelType::uint8,PixelType::float32); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    convert_into(nullptr,nullptr,0,PixelType::uint8,PixelType::float32);
    rejected = false;
    const float pixel = 1;
    try { sobel_into(&pixel,static_cast<float*>(nullptr),1,1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    rejected = false;
    const double value = 0;
    try { sobel(&value,std::numeric_limits<std::size_t>::max(),2); }
    catch (const std::overflow_error&) { rejected = true; }
    require(rejected);
    std::cout << "Native C++ checks passed; AVX2=" << has_avx2() << '\n';
}
