#include "skimagexx/gabor_filter.hpp"
#include "skimagexx/gabor.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition){if(!condition)throw std::runtime_error("native Gabor check failed");}
}
int main() {
    using namespace skimagexx;
    require(get_num_threads()==1);
    // Reversed row-major accumulation, epsilon cutoff and integer cast
    // constants captured independently from scipy.ndimage.convolve.
    const std::vector<double> kernel{0,0, .25,1e-16, 0,-.5,
                                    .3,.2, 1,0, 0,-.7,
                                    .25,0, 0,0, .125,-.25};
    std::vector<std::int64_t> image(35),real(35),imaginary(35);
    for(std::size_t i=0;i<35;++i)image[i]=std::int64_t(i)*12345678901234567LL;
    const std::size_t positions[]={0,8,17,34};
    const std::int64_t reflect_real[]={28395061472839504LL,184567899573456768LL,398456786537345728LL,774074067107407360LL};
    const std::int64_t reflect_imag[]={-40740740374074072LL,-124691356902469120LL,-263580244541358016LL,-485185180818518464LL};
    const std::int64_t constant_real[]={25308641747530860LL,184567899573456768LL,398456786537345728LL,459876539070987648LL};
    const std::int64_t constant_imag[]={2469135780246918LL,-124691356902469120LL,-263580244541358016LL,-365432095476543168LL};
    for(auto border:{GaussianBorder::reflect,GaussianBorder::constant}) {
        gabor_filter_into(image.data(),PixelType::int64,real.data(),imaginary.data(),5,7,kernel.data(),false,3,3,border,-3.25);
        for(std::size_t i=0;i<4;++i){
            require(real[positions[i]]==(border==GaussianBorder::reflect ? reflect_real[i] : constant_real[i]));
            require(imaginary[positions[i]]==(border==GaussianBorder::reflect ? reflect_imag[i] : constant_imag[i]));
        }
    }
    {
        std::vector<double> broadcast(35,7.),r(35),im(35);
        const double reflect_r[]={7.635,2.16,2.16,13.475};
        const double reflect_i[]={-6.559999999999999,-5.1,-5.1,-8.75};
        const double constant_r[]={6.7225,2.16,2.16,7.635};
        const double constant_i[]={.375,-5.1,-5.1,-6.559999999999999};
        for(auto border:{GaussianBorder::reflect,GaussianBorder::constant}) {
            gabor_filter_into(broadcast.data(),PixelType::float64,r.data(),im.data(),5,7,kernel.data(),false,3,3,border,-.3,true);
            for(std::size_t i=0;i<4;++i){
                require(std::abs(r[positions[i]]-(border==GaussianBorder::reflect ? reflect_r[i] : constant_r[i]))<2e-14);
                require(std::abs(im[positions[i]]-(border==GaussianBorder::reflect ? reflect_i[i] : constant_i[i]))<2e-14);
            }
        }
    }
    constexpr double pi=3.14159265358979323846;
    const GaborParameters p{std::cos(.53),std::sin(.53),4,9,2*pi*.2,.2,1/(2*pi*2*3)};
    std::vector<double> generated(2*17*13);
    gabor_kernel_accurate_into(generated.data(),false,17,13,p);
    require(std::abs(generated[220]-.02599707340376524)<1e-17);
    require(std::abs(generated[221]-.005269867672798817)<1e-17);
    std::vector<double> large(257*263),serial_real(large.size()),serial_imag(large.size()),parallel_real(large.size()),parallel_imag(large.size());
    for(std::size_t i=0;i<large.size();++i)large[i]=double((i*117+i/37)%1000)/1000;
    for(auto border:{GaussianBorder::reflect,GaussianBorder::constant,GaussianBorder::mirror,GaussianBorder::nearest,GaussianBorder::wrap}) {
        set_num_threads(1);
        gabor_filter_into(large.data(),PixelType::float64,serial_real.data(),serial_imag.data(),257,263,kernel.data(),false,3,3,border,-.3);
#ifdef _OPENMP
        set_num_threads(4);
#endif
        gabor_filter_into(large.data(),PixelType::float64,parallel_real.data(),parallel_imag.data(),257,263,kernel.data(),false,3,3,border,-.3);
        require(serial_real==parallel_real && serial_imag==parallel_imag);
    }
    set_num_threads(1);
    bool rejected=false;
    try{gabor_filter_into(image.data(),PixelType::int64,real.data(),imaginary.data(),5,7,kernel.data(),false,2,3,GaussianBorder::reflect);}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected);
    std::cout<<"Standalone Gabor checks passed\n";
}
