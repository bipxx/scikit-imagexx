// Targeted candidate controls. Baseline scalar helper is unchanged; saved rows are original-only.
#include <skimagexx/stain.hpp>
#include <skimagexx/image.hpp>
#include "stain_detail.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t checks=0,records=0;
void check(bool condition){++checks;if(!condition)throw std::runtime_error("stain F32 check="+std::to_string(checks));}
void near(float actual,float expected,double atol){
    check(std::isnan(expected)?std::isnan(actual):std::isinf(expected)?actual==expected:
          std::fabs(double(actual)-expected)<=atol+2e-6*std::fabs(double(expected)));
    if(std::isfinite(expected))check(actual>=0 && actual<=1);
}
std::uint64_t u(std::ifstream& input){std::uint64_t value;input.read(reinterpret_cast<char*>(&value),8);check(bool(input));return value;}
std::string text(std::ifstream& input){std::string value(u(input),'\0');input.read(value.data(),value.size());check(bool(input));return value;}
template<class T>std::vector<T> values(std::ifstream& input,std::size_t count){std::vector<T> out(count);input.read(reinterpret_cast<char*>(out.data()),count*sizeof(T));check(bool(input));return out;}
void saved(const char* path){
    std::ifstream input(path,std::ios::binary);check(bool(input));const auto count=u(input);
    for(std::size_t row=0;row<count;++row){
        const auto id=text(input);(void)id;const auto n=u(input),k=u(input);
        const auto source=values<float>(input,3*n);const auto weights=values<double>(input,3*k);
        const auto expected=values<float>(input,n*k);double atol;input.read(reinterpret_cast<char*>(&atol),8);check(bool(input));
        std::vector<float> actual(n*k+2,987654.f);
        stain_transform_into(source.data(),actual.data()+1,n,PixelType::float32,StainTransform::combine,weights.data(),k,PixelType::float32,PixelType::float32);
        for(std::size_t i=0;i<n*k;++i)near(actual[i+1],expected[i],atol);
        check(actual.front()==987654.f && actual.back()==987654.f);++records;
    }
    check(input.peek()==std::char_traits<char>::eof());
}
void controls(int workers){
    check(get_num_threads()==workers);
    for(auto n:{std::size_t(0),std::size_t(1),std::size_t(7),std::size_t(8),std::size_t(9),std::size_t(31),std::size_t(257)})
    for(auto k:{std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(3),std::size_t(7),std::size_t(17)})
    for(int layout=0;layout<4;++layout){
        const std::ptrdiff_t pixels=layout==1?-5:layout==2?1:layout==3?0:3;
        const std::ptrdiff_t channels=layout==1?-1:layout==2?std::ptrdiff_t(n+3):layout==3?0:1;
        std::vector<float> storage(8*n+32,.125f);
        const auto origin=std::size_t(layout==1?5*n+5:1);const float* source=storage.data()+origin;
        for(std::size_t i=0;i<n;++i)for(int c=0;c<3;++c)
            storage[std::ptrdiff_t(origin)+std::ptrdiff_t(i)*pixels+c*channels]=float((i*13+c*3)%97)/128;
        const auto input_before=storage;
        std::vector<double> weights(3*k);
        for(std::size_t j=0;j<weights.size();++j)weights[j]=double(int(j%9)-4)/8;
        double matrix_norm=1;
        for(std::size_t j=0;j<k;++j)matrix_norm=std::max(matrix_norm,std::fabs(weights[3*j])+std::fabs(weights[3*j+1])+std::fabs(weights[3*j+2]));
        const auto atol=2e-6*matrix_norm;
        detail::StainSpec spec{StainTransform::combine,weights.data(),k,true,13.815510557964274};
        std::vector<float> expected(n*k),actual(n*k+2,987654.f);
        for(std::size_t i=0;i<n;++i)detail::stain_pixel(source+std::ptrdiff_t(i)*pixels,expected.data()+i*k,spec,channels);
        stain_transform_into(source,actual.data()+1,n,PixelType::float32,StainTransform::combine,weights.data(),k,PixelType::float32,PixelType::float32,pixels,channels);
        for(std::size_t i=0;i<n*k;++i)near(actual[i+1],expected[i],atol);
        check(storage==input_before && actual.front()==987654.f && actual.back()==987654.f);
#ifdef SKIMAGEXX_AVX2
        if(has_avx2() && n>=8 && k){
            std::vector<float> direct(n*k,987654.f);
            const auto done=simd::stain_combine_float32(source,direct.data(),n,spec,pixels,channels);
            check(done==n/8*8);for(std::size_t i=0;i<done*k;++i)near(direct[i],expected[i],atol);
            for(std::size_t i=done*k;i<n*k;++i)check(direct[i]==987654.f);
        }
#endif
    }
    // Transform and double-output neighbors must retain their original routes.
    std::vector<float> neighbor(51,.25f),separate(17*3),separate_expected(17*3);
    const double neighbor_weights[9]={.5,.25,.125,.125,.5,.25,.25,.125,.5};
    const detail::StainSpec old_separate{StainTransform::separate,neighbor_weights,3,true,13.815510557964274};
    for(std::size_t i=0;i<17;++i)detail::stain_pixel(neighbor.data()+3*i,separate_expected.data()+3*i,old_separate,1);
    stain_transform_into(neighbor.data(),separate.data(),17,PixelType::float32,StainTransform::separate,neighbor_weights,3,PixelType::float32,PixelType::float32);
    for(std::size_t i=0;i<separate.size();++i)check(std::fabs(separate[i]-separate_expected[i])<=2e-6+2e-6*std::fabs(separate_expected[i]));
    std::vector<double> wide(51,.25),wide_result(51),wide_expected(51);
    const detail::StainSpec old_double{StainTransform::combine,neighbor_weights,3,false,13.815510557964274};
    for(std::size_t i=0;i<17;++i)detail::stain_pixel(wide.data()+3*i,wide_expected.data()+3*i,old_double,1);
    stain_transform_into(wide.data(),wide_result.data(),17,PixelType::float64,StainTransform::combine,neighbor_weights,3,PixelType::float64,PixelType::float64);
    for(std::size_t i=0;i<wide_result.size();++i)check(std::fabs(wide_result[i]-wide_expected[i])<=2e-13+2e-13*std::fabs(wide_expected[i]));
    // Exceptional exp8 range/NaN fallback must retain native scalar values and four NumPy FP flags.
    for(float value:{0.f,-0.f,1.f,-1.f,100.f,-100.f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        std::vector<float> source(27,value),actual(9),expected(9);const double weights[3]={1,0,0};
        const detail::StainSpec spec{StainTransform::combine,weights,1,true,13.815510557964274};
        constexpr int observed=FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW;
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        for(std::size_t i=0;i<9;++i)detail::stain_pixel(source.data()+3*i,expected.data()+i,spec,1);
        const auto flags=std::fetestexcept(observed);
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        stain_transform_into(source.data(),actual.data(),9,PixelType::float32,StainTransform::combine,weights,1,PixelType::float32,PixelType::float32);
        check(std::fetestexcept(observed)==flags);
        for(std::size_t i=0;i<9;++i)near(actual[i],expected[i],2e-6);
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    // Eligible parallel path with a non-multiple-of-eight partition tail.
    constexpr std::size_t n=262147;std::vector<float> input(3*n,.25f),actual(n+2,987654.f);const double weights[3]={.5,.25,.125};
    stain_transform_into(input.data(),actual.data()+1,n,PixelType::float32,StainTransform::combine,weights,1,PixelType::float32,PixelType::float32);
    const detail::StainSpec spec{StainTransform::combine,weights,1,true,13.815510557964274};float expected;
    detail::stain_pixel(input.data(),&expected,spec,1);for(std::size_t i=0;i<n;++i)near(actual[i+1],expected,2e-6);
    check(actual.front()==987654.f && actual.back()==987654.f);
}
}
int main(int argc,char**argv){
    check(argc==2 && get_num_threads()==1);
    for(int workers:{1,4}){set_num_threads(workers);saved(argv[1]);controls(workers);}
    set_num_threads(1);check(get_num_threads()==1);
    std::cout<<"PASS stain_f32 records="<<records<<" checks="<<checks<<" workers=1,4 default=1 avx2="<<has_avx2()<<'\n';
}
