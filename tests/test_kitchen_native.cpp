#include "skimagexx/kitchen.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void require(bool condition){if(!condition)throw std::runtime_error("native Kitchen-Rosenfeld check failed");}
template<class T> void quadratic(skimagexx::PixelType type,std::size_t height,std::size_t width){
    using namespace skimagexx;using O=std::conditional_t<std::is_same_v<T,float>,float,double>;
    std::vector<T> image(height*width);std::vector<O> output(image.size()),serial;
    for(std::size_t r=0;r<height;++r)for(std::size_t c=0;c<width;++c){
        const auto y=std::ptrdiff_t(r)-std::ptrdiff_t(height/2),x=std::ptrdiff_t(c)-std::ptrdiff_t(width/2);
        image[r*width+c]=T(y*y+x*x);
    }
    const int worker_counts[]={1
#ifdef _OPENMP
        ,4
#endif
    };
    for(auto workers:worker_counts){
        set_num_threads(workers);
        require(kitchen_rosenfeld_into(image.data(),output.data(),type,height,width,{GaussianBorder::reflect,GaussianBorder::reflect}));
        require(output[(height/2)*width+width/2]==0);
        // Sobel of x²+y² gives (16y,16x), second derivatives128.
        require(output[(height/2)*width+width/2+1]==128);
        if(workers==1)serial=output;else require(serial==output);
    }
    set_num_threads(1);
}
}
int main(){
    using namespace skimagexx;require(get_num_threads()==1);
    quadratic<unsigned char>(PixelType::uint8,9,11);
    quadratic<float>(PixelType::float32,9,11);quadratic<double>(PixelType::float64,9,11);
    quadratic<float>(PixelType::float32,513,519);quadratic<double>(PixelType::float64,513,519);
    std::array<double,9> input{},output{},scratch{};
    const auto borders=std::array<GaussianBorder,2>{GaussianBorder::reflect,GaussianBorder::reflect};
    sobel_derivative_2d_into(input.data(),output.data(),scratch.data(),3,3,false,0,borders);
    for(auto value:output)require(value==0);
    input[4]=std::numeric_limits<double>::quiet_NaN();
    require(!kitchen_rosenfeld_into(input.data(),output.data(),PixelType::float64,3,3,borders));
    require(kitchen_rosenfeld_into(nullptr,nullptr,PixelType::float64,0,3,borders));
}
