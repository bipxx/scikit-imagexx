#include "skimagexx/image.hpp"
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
    }
    set_num_threads(1);
    require(get_num_threads() == 1);
    bool rejected = false;
    try { sobel<double>(nullptr,2,2); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
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
