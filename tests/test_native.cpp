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
