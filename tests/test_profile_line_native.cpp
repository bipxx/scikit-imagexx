#include "resample_detail.hpp"
#include "execution.hpp"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t controls=0;std::ofstream journal;
void check(bool value){++controls;if(!value)std::fprintf(stderr,"profile check %zu failed\n",controls);assert(value);}
// BEGIN SOURCE-DERIVED CONTROLS
// Literal SciPy source-schedule controls, NOT new genuine reference calls.
constexpr std::size_t query_count=15;
const double queries[][2]={{-0x1.1000000000000p+2,0x1.f000000000000p+2},{-0x1.8000000000000p+0,-0x1.0000000000000p-1},{-0x1.0000000000000p+0,0x0.0p+0},{-0x1.0000000000000p-1,0x1.c000000000000p+1},{-0x1.0000000000000p-54,0x1.0000000000000p-1},{0x0.0p+0,0x0.0p+0},{0x1.0000000000000p-54,0x1.8000000000000p+0},{0x1.0000000000000p-1,0x1.0000000000000p-1},{0x1.0000000000000p+0,0x1.0000000000000p+1},{0x1.8000000000000p+0,0x1.8000000000000p+1},{0x1.0000000000000p+1,-0x1.0000000000000p-1},{0x1.4000000000000p+1,0x1.1000000000000p+2},{0x1.8000000000000p+1,0x1.0000000000000p+0},{0x1.c000000000000p+1,-0x1.8000000000000p+0},{0x1.f000000000000p+2,-0x1.1000000000000p+2}};
const double answers[2][5][query_count]={{{0x1.0000000000000p+3,0x1.0000000000000p+2,0x0.0p+0,0x1.8000000000000p+1,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.6000000000000p+3,0x1.0000000000000p+3,0x1.6000000000000p+3,0x1.2000000000000p+3,0x1.2000000000000p+3,0x1.6000000000000p+3},{0x1.0000000000000p+1,0x1.2000000000000p+3,0x1.0000000000000p+2,0x1.8000000000000p+2,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.6000000000000p+3,0x1.2000000000000p+3,0x1.8000000000000p+2,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.0000000000000p+1},{0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.6000000000000p+3,0x1.0000000000000p+3,0x1.6000000000000p+3,0x1.2000000000000p+3,0x1.0000000000000p+3,0x1.0000000000000p+3},{-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.6000000000000p+3,0x1.0000000000000p+3,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1},{0x1.0000000000000p+3,0x1.0000000000000p+3,0x1.0000000000000p+3,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+1,0x1.4000000000000p+2,0x1.8000000000000p+2,0x1.6000000000000p+3,0x1.0000000000000p+3,0x0.0p+0,0x1.0000000000000p+0,0x1.c000000000000p+2,0x1.0000000000000p+3}},{{0x1.c000000000000p+2,0x1.0000000000000p+1,0x0.0p+0,0x1.8000000000000p+1,0x1.0000000000000p-1,0x0.0p+0,0x1.8000000000000p+0,0x1.4000000000000p+1,0x1.8000000000000p+2,0x1.2000000000000p+3,0x1.0000000000000p+3,0x1.5800000000000p+3,0x1.2000000000000p+3,0x1.a000000000000p+2,0x1.4000000000000p+3},{0x1.6000000000000p+1,0x1.a000000000000p+2,0x1.0000000000000p+2,0x1.2000000000000p+2,0x1.0000000000000p-1,0x0.0p+0,0x1.8000000000000p+0,0x1.4000000000000p+1,0x1.8000000000000p+2,0x1.2000000000000p+3,0x1.1000000000000p+3,0x1.f000000000000p+2,0x1.4000000000000p+2,0x1.c000000000000p+1,0x1.6000000000000p+1},{0x1.8000000000000p+1,0x0.0p+0,0x0.0p+0,0x1.8000000000000p+1,0x1.0000000000000p-1,0x0.0p+0,0x1.8000000000000p+0,0x1.4000000000000p+1,0x1.8000000000000p+2,0x1.2000000000000p+3,0x1.0000000000000p+3,0x1.6000000000000p+3,0x1.2000000000000p+3,0x1.0000000000000p+3,0x1.0000000000000p+3},{-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.b000000000000p+0,0x1.0000000000000p-1,0x0.0p+0,0x1.8000000000000p+0,0x1.4000000000000p+1,0x1.8000000000000p+2,0x1.2000000000000p+3,0x1.3000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1,-0x1.a000000000000p+1},{0x1.f000000000000p+2,0x1.e000000000000p+2,0x1.0000000000000p+3,0x1.6000000000000p+2,0x1.0000000000000p-1,0x0.0p+0,0x1.8000000000000p+0,0x1.4000000000000p+1,0x1.8000000000000p+2,0x1.2000000000000p+3,0x1.3000000000000p+3,0x1.1000000000000p+2,0x1.0000000000000p+0,0x1.2000000000000p+2,0x1.f000000000000p+2}}};
const unsigned char bool_answers[5][query_count]={{0,0,0,1,1,0,0,1,0,1,0,1,1,1,1},{0,1,0,0,1,0,0,1,0,1,1,0,1,0,0},{1,0,0,1,1,0,0,1,0,1,0,1,1,0,0},{0,0,0,0,1,0,0,1,0,1,0,0,0,0,0},{0,0,0,0,1,0,0,1,0,1,0,0,1,1,0}};
// END SOURCE-DERIVED CONTROLS
const GaussianBorder modes[]={GaussianBorder::reflect,GaussianBorder::mirror,GaussianBorder::nearest,GaussianBorder::constant,GaussianBorder::wrap};
template<class T>constexpr PixelType type(){return sizeof(T)==1?PixelType::uint8:sizeof(T)==4?PixelType::float32:PixelType::float64;}
template<class T>std::vector<T> run(const std::vector<T>& input,std::size_t height,std::size_t width,
        const std::vector<double>& coordinates,int order,GaussianBorder mode,double cval,int workers,bool boolean=false){
    const auto count=coordinates.size()/2;std::vector<T> output(count+2,T(117));
    const auto before=input;const auto coords=coordinates;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    check(profile_sample({input.data(),boolean?PixelType::boolean:type<T>(),height,width},coordinates.data(),count,output.data()+1,order,mode,cval,workers));
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    check(std::memcmp(input.data(),before.data(),input.size()*sizeof(T))==0&&coordinates==coords);
    check(output.front()==T(117)&&output.back()==T(117));
    journal.write(reinterpret_cast<const char*>(output.data()+1),std::streamsize(count*sizeof(T)));
    return std::vector<T>(output.begin()+1,output.end()-1);
}
template<class T>void ordinary(int workers){
    std::vector<T> image(12);for(std::size_t i=0;i<image.size();++i)image[i]=T(i);
    for(int order:{0,1})for(std::size_t mode=0;mode<5;++mode)
    for(std::size_t count:{0u,1u,3u,4u,5u,7u,8u,9u,31u,32u,33u,65u}){
        std::vector<double> coords(count*2);
        for(std::size_t i=0;i<count;++i){coords[i]=queries[i%query_count][0];coords[count+i]=queries[i%query_count][1];}
        const auto output=run(image,3,4,coords,order,modes[mode],-3.25,workers);
        for(std::size_t i=0;i<count;++i){
            const double expected=answers[order][mode][i%query_count];
            const auto rounded=sizeof(T)==1?std::floor(std::max(0.,expected)+.5):expected;
            check(output[i]==T(rounded));
        }
    }
}
void cohort(int workers){
    ordinary<std::uint8_t>(workers);ordinary<float>(workers);ordinary<double>(workers);
    // Saved genuine horizontal source pixels; coordinates are literal and
    // source-provable, not claimed as captured coordinate-array bytes.
    std::vector<std::uint8_t> image(99);for(std::size_t i=0;i<image.size();++i)image[i]=std::uint8_t(i%23+1);
    std::vector<double> coords(16,2.);for(std::size_t i=0;i<8;++i)coords[8+i]=double(i+1);
    check(run(image,9,11,coords,1,GaussianBorder::reflect,0.,workers)==std::vector<std::uint8_t>({1,2,3,4,5,6,7,8}));
    // Raw npy_bool output is not nearest-rounding or truth normalization.
    image.resize(12);for(std::size_t i=0;i<12;++i)image[i]=std::uint8_t(i%2);
    coords.resize(query_count*2);
    for(std::size_t i=0;i<query_count;++i){coords[i]=queries[i][0];coords[query_count+i]=queries[i][1];}
    for(std::size_t mode=0;mode<5;++mode){
        const auto output=run(image,3,4,coords,0,modes[mode],0.,workers,true);
        for(std::size_t i=0;i<query_count;++i)check(output[i]==bool_answers[mode][i]);
    }
    // Difference between 1-(1-delta) and direct delta is observable here.
    std::vector<double> sensitive={0.,64.,128.,256.};
    check(run(sensitive,2,2,{0.,std::ldexp(1.,-54)},1,GaussianBorder::reflect,0.,workers)[0]==0.);
    // Independent large queries cross the actual optional-worker threshold.
    const std::size_t count=8192*33;coords.resize(count*2);
    std::vector<double> large_input(12);for(std::size_t i=0;i<12;++i)large_input[i]=double(i);
    for(std::size_t i=0;i<count;++i){coords[i]=queries[i%query_count][0];coords[count+i]=queries[i%query_count][1];}
    const auto output=run(large_input,3,4,coords,1,GaussianBorder::reflect,-3.25,workers);
    for(std::size_t i=0;i<count;++i)check(output[i]==answers[1][0][i%query_count]);
    for(auto mode:{GaussianBorder::reflect,GaussianBorder::mirror,GaussianBorder::nearest,GaussianBorder::wrap})
        check(run(std::vector<double>{4.},1,1,{-1.5,0.,.5,3.,-2.,0.,.5,4.},1,mode,0.,workers)==std::vector<double>(4,4.));
}
void refusal(int workers){
    std::vector<double> image(12,1.),coords{0.,0.},output(3,-777.);
    auto rejected=[&](){
        const auto before=image;const auto positions=coords;
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        const bool accepted=profile_sample({image.data(),PixelType::float64,3,4},coords.data(),1,output.data()+1,1,GaussianBorder::constant,0.,workers);
        check(!accepted);check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
        check(std::memcmp(before.data(),image.data(),image.size()*sizeof(double))==0&&std::memcmp(positions.data(),coords.data(),coords.size()*sizeof(double))==0);
        check(output.front()==-777.&&output.back()==-777.);
    };
    for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){image[0]=bad;rejected();image[0]=1.;coords[0]=bad;rejected();coords[0]=0.;}
    coords[0]=0x1p63;rejected();coords[0]=0.;
    const auto rounding=std::fegetround();check(std::fesetround(FE_DOWNWARD)==0);rejected();check(std::fegetround()==FE_DOWNWARD);check(std::fesetround(rounding)==0);
    check(!profile_sample({image.data(),PixelType::float64,3,4},coords.data(),1,output.data()+1,1,GaussianBorder::constant,std::numeric_limits<double>::infinity(),workers));
    float target=0;std::vector<float> small(12,1.);coords={-1.,-1.};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    check(!profile_sample({small.data(),PixelType::float32,3,4},coords.data(),1,&target,1,GaussianBorder::constant,1e300,workers));
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
}
}
int main(int argc,char** argv){
    check(argc==2);journal.open(argv[1],std::ios::binary);check(bool(journal));check(get_num_threads()==1);
    cohort(1);refusal(1);
#ifdef _OPENMP
    set_num_threads(4);cohort(4);refusal(4);set_num_threads(1);
#else
    bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument& error){refused=std::string(error.what())=="this build has no OpenMP support";}
    check(refused&&get_num_threads()==1);
    refused=false;try{profile_sample({nullptr,PixelType::float64,1,1},nullptr,0,nullptr,0,GaussianBorder::reflect,0.,4);}catch(const std::invalid_argument&){refused=true;}check(refused);
#endif
    journal.close();check(bool(journal));
    std::printf("%zu Profile controls PASS avx2=%d workers=%d\n",controls,int(has_avx2()),get_num_threads());
}
