#include "rolling_ball_kernels.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
using namespace skimagexx;using namespace skimagexx::detail;
namespace {
std::size_t checks=0,accepted=0,refused=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("rolling-kernel control "+std::to_string(checks));}
std::uint64_t word(std::ifstream& f){std::uint64_t x;f.read(reinterpret_cast<char*>(&x),8);require(bool(f));return x;}
double number(std::ifstream& f){const auto x=word(f);double value;std::memcpy(&value,&x,8);return value;}
// Independent full point reduction: no cached row prefixes or tree spines.
double numpy_sum(const std::vector<double>& terms){
    double result=-0.;
    if(terms.size()<8){for(double t:terms)result+=t;}
    else{
        std::array<double,8> lane;std::copy_n(terms.begin(),8,lane.begin());std::size_t i=8;
        for(;i+8<=terms.size();i+=8)for(std::size_t j=0;j<8;++j)lane[j]+=terms[i+j];
        const double a=lane[0]+lane[1],b=lane[2]+lane[3],c=lane[4]+lane[5],d=lane[6]+lane[7];
        result=(a+b)+(c+d);for(;i<terms.size();++i)result+=terms[i];
    }
    volatile double identity=0.;return identity+result;
}
std::vector<double> oracle(const RollingGrid& grid,bool ball,double first,double second){
    std::vector<double> expected(grid.count),terms(grid.ndim);
    for(std::size_t i=0;i<grid.count;++i){
        auto remaining=i;
        for(std::size_t a=grid.ndim;a-->0;){const auto& axis=grid.axes[a];const auto x=axis.coordinates[remaining%axis.count];remaining/=axis.count;const double q=ball?x:x/axis.semi_axis;terms[a]=q*q;}
        const double sum=numpy_sum(terms),scaling=(ball?second:1.)-sum;
        const double height=std::sqrt(scaling<0.?0.:scaling);
        if(ball){const double distance=std::sqrt(sum);expected[i]=distance>first?std::numeric_limits<double>::infinity():height;}
        else{expected[i]=first*height;if(scaling<0.)expected[i]=std::numeric_limits<double>::infinity();}
    }
    return expected;
}
void exercise(std::vector<std::vector<double>> coordinates,const std::vector<double>& semi,
              bool ball,double first,double second,int workers,std::ofstream& stream,
              const std::vector<double>* frozen=nullptr,bool expect_accept=true){
    std::vector<RollingAxis> axes;std::size_t count=1;for(std::size_t a=0;a<coordinates.size();++a){axes.push_back({coordinates[a].data(),coordinates[a].size(),semi[a]});count*=coordinates[a].size();}
    const auto before=coordinates;const RollingGrid grid{axes.data(),axes.size(),count};
    const auto expected=frozen?*frozen:oracle(grid,ball,first,second);
    constexpr std::uint64_t guard=0x7fe123456789abcdULL;double sentinel;std::memcpy(&sentinel,&guard,8);
    std::vector<double> output(count+2,sentinel);const int rounding=std::fegetround(),global=get_num_threads();
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    const bool ok=ball?ball_kernel_into(grid,first,second,output.data()+1,workers):ellipsoid_kernel_into(grid,first,output.data()+1,workers);
    require(ok==expect_accept);require(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    require(coordinates==before);require(std::memcmp(output.data(),&sentinel,8)==0&&std::memcmp(output.data()+count+1,&sentinel,8)==0);
    if(ok){++accepted;require(!count||std::memcmp(output.data()+1,expected.data(),count*8)==0);if(count)stream.write(reinterpret_cast<const char*>(output.data()+1),std::streamsize(count*8));}
    else ++refused;
}
void fixtures(const char* path,int workers,std::ofstream& stream){
    std::ifstream input(path,std::ios::binary);require(bool(input));const auto n=word(input);require(n==36||n==37);
    for(std::size_t record=0;record<n;++record){
        const bool ball=word(input)!=0;const auto rank=word(input),count=word(input);const auto first=number(input),second=number(input);const bool accept=word(input)!=0;
        std::vector<std::vector<double>> coordinates(rank);std::vector<double> semi(rank);
        for(std::size_t a=0;a<rank;++a){const auto length=word(input);semi[a]=number(input);coordinates[a].resize(length);for(auto& value:coordinates[a])value=number(input);}
        std::vector<double> expected(count);for(auto& value:expected)value=number(input);
        exercise(coordinates,semi,ball,first,second,workers,stream,&expected,accept);
    }
}
void environment_refusals(){
    const double x[]={-1,0,1};const RollingAxis axis{x,3,1};const RollingGrid grid{&axis,1,3};double output[]={17,18,19};
    const int saved=std::fegetround();
    for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        require(std::fesetround(mode)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(!ball_kernel_into(grid,1.,1.,output,1));require(!ellipsoid_kernel_into(grid,1.,output,4));require(std::fegetround()==mode&&std::fetestexcept(FE_ALL_EXCEPT)==flags);require(output[0]==17&&output[1]==18&&output[2]==19);
    }
    require(std::fesetround(saved)==0);
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|(1u<<15)|(1u<<6));const auto unusual=_mm_getcsr();
    require(!ellipsoid_kernel_into(grid,1.,output,1));require(_mm_getcsr()==unusual);_mm_setcsr(csr);
#endif
}
}
int main(int argc,char** argv){
    require(argc==2||argc==4);std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
    for(int workers:{1,4}){
        if(argc==4){fixtures(argv[2],workers,stream);fixtures(argv[3],workers,stream);}
        for(std::size_t rank=1;rank<=32;++rank)
        for(std::size_t columns:{1,3,4,5,7,8,9,15,16,17}){
            std::vector<std::vector<double>> axes(rank);std::vector<double> semi(rank);
            for(std::size_t a=0;a<rank;++a){const auto n=a+1==rank?columns:(a<2?std::size_t(3):std::size_t(1));semi[a]=double(a+3);for(std::size_t i=0;i<n;++i)axes[a].push_back((double((a*37+i*11)%101)-50.)/97.);}
            for(double intensity:{0.,-0.,-2.,1.5})exercise(axes,semi,false,intensity,0.,workers,stream);
            exercise(axes,semi,true,3.25,10.5625,workers,stream);
        }
        for(std::size_t rank:{1,2,3,5,8,16,24,31,32}){
            std::vector<std::vector<double>> axes(rank,std::vector<double>{0.});std::vector<double> semi(rank,1.);
            exercise(axes,semi,true,0.,0.,workers,stream);exercise(axes,semi,true,-.25,.0625,workers,stream);
            for(std::size_t a=0;a<rank;++a)axes[a][0]=134217728.+double(a%3);
            exercise(axes,semi,true,1000000000.,1000000000000000000.,workers,stream);
        }
        exercise({{-1.,0.,1.}},{1.},false,std::numeric_limits<double>::infinity(),0.,workers,stream,nullptr,false);
        exercise({{-1.,0.,1.}},{1.},false,-std::numeric_limits<double>::infinity(),0.,workers,stream,nullptr,false);
        exercise({{1.}},{std::numeric_limits<double>::denorm_min()},false,2.,0.,workers,stream,nullptr,false);
        exercise({{0x1p-520}},{1.},false,2.,0.,workers,stream);
        exercise({{std::nextafter(0x1p-520,1.)}},{1.},false,2.,0.,workers,stream,nullptr,false);
        // Force partitioning with disjoint row tiles and a long last axis.
        std::vector<std::vector<double>> large(2);for(int i=0;i<513;++i)large[0].push_back(double(i)-256.);for(int i=0;i<513;++i)large[1].push_back(double(i)-256.);
        exercise(large,{256.,256.},true,256.,65536.,workers,stream);exercise(large,{256.,256.},false,2.,0.,workers,stream);
    }
    environment_refusals();std::feclearexcept(FE_ALL_EXCEPT);
    std::cout<<"rolling-kernels native "<<checks<<" checks PASS accepted="<<accepted<<" refused="<<refused<<" avx2="<<has_avx2()<<"\n";
}
