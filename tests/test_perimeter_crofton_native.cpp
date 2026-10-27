#include "perimeter_crofton.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t checks=0;std::ofstream journal;
void require(bool v){++checks;if(!v)throw std::runtime_error("Crofton control "+std::to_string(checks));}
using Histogram=std::array<std::ptrdiff_t,16>;
const std::array<double,16> coefficients{0.,1.,.5,1.5,0.,2.,0.,.25,1.,2.,.75,.5,1.,2.,0.,0.};
void save(const Histogram& histogram,double value){
    journal.write(reinterpret_cast<const char*>(histogram.data()),sizeof(histogram));
    journal.write(reinterpret_cast<const char*>(&value),sizeof(value));
}
Histogram reference(const std::vector<unsigned char>& binary,std::size_t height,std::size_t width){
    // Independent literal zero-pad/convolution oracle; saved originals below
    // also pin bit orientation, padding and source coefficient arithmetic.
    std::vector<unsigned char> padded((height+2)*(width+2),0);
    for(std::size_t y=0;y<height;++y)for(std::size_t x=0;x<width;++x)padded[(y+1)*(width+2)+x+1]=binary[y*width+x];
    Histogram result{};
    for(std::size_t y=0;y<height+2;++y)for(std::size_t x=0;x<width+2;++x){
        unsigned code=padded[y*(width+2)+x];
        if(x)code+=4*padded[y*(width+2)+x-1];
        if(y)code+=2*padded[(y-1)*(width+2)+x];
        if(x&&y)code+=8*padded[(y-1)*(width+2)+x-1];
        ++result[code];
    }
    return result;
}
void check(const detail::CroftonInput& input,const Histogram& expected,const double* coefs,double original,int workers){
    std::array<std::ptrdiff_t,18> output{};output.front()=output.back()=73;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto before=std::fetestexcept(FE_ALL_EXCEPT);
    detail::crofton_counts(input,output.data()+1,workers);
    require(std::fetestexcept(FE_ALL_EXCEPT)==before);
    require(output.front()==73&&output.back()==73);
    Histogram actual{};std::copy_n(output.data()+1,16,actual.data());require(actual==expected);
    const auto value=detail::crofton_perimeter(actual.data(),coefs,workers);
    require(std::fetestexcept(FE_ALL_EXCEPT)==before);std::feclearexcept(FE_ALL_EXCEPT);
    long double magnitude=0;for(std::size_t i=0;i<16;++i)magnitude+=std::abs(static_cast<long double>(coefs[i]))*expected[i];
    const long double u=std::numeric_limits<double>::epsilon()/2.L,g=32*u/(1-32*u);
    require(std::isfinite(value)&&std::abs(static_cast<long double>(value)-original)<=2*g*magnitude);
    if(!magnitude)require(value==0&&!std::signbit(value));save(actual,value);
}
void captured(PixelType type,std::size_t h,std::size_t w,const char* hex,const Histogram& histogram,
              const std::array<double,16>& coefs,double original,int workers){
    const auto count=std::strlen(hex)/2;std::vector<std::uint64_t> storage((count+7)/8);
    auto* bytes=reinterpret_cast<unsigned char*>(storage.data());
    for(std::size_t i=0;i<count;++i)bytes[i]=static_cast<unsigned char>(std::stoul(std::string(hex+2*i,2),nullptr,16));
    const auto before=storage;check({storage.data(),type,h,w},histogram,coefs.data(),original,workers);require(storage==before);
}
void originals(int workers){
    // geometry-empty00
    captured(PixelType::uint8,0,0,
        "",
        {4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x0.0p+0,workers);
    // geometry-empty05
    captured(PixelType::uint8,0,5,
        "",
        {14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x0.0p+0,workers);
    // geometry-empty40
    captured(PixelType::uint8,4,0,
        "",
        {12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x0.0p+0,workers);
    // geometry-single1
    captured(PixelType::uint8,1,1,
        "01",
        {5,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.573bf3790c7fep+1,workers);
    // dtype-bool
    captured(PixelType::boolean,5,7,
        "0001010100010101000101010001010100010101000101010001010100010101000101",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.e4836c7093466p+4,workers);
    // dtype-int8
    captured(PixelType::int8,5,7,
        "fe030100fe030100fe030100fe030100fe030100fe030100fe030100fe030100fe0301",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // dtype-uint8
    captured(PixelType::uint8,5,7,
        "0003020100030201000302010003020100030201000302010003020100030201000302",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.e4836c7093466p+4,workers);
    // dtype-int16
    captured(PixelType::int16,5,7,
        "feff030001000000feff030001000000feff030001000000feff030001000000feff030001000000feff030001000000feff030001000000feff030001000000feff03000100",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // dtype-uint16
    captured(PixelType::uint16,5,7,
        "00000300020001000000030002000100000003000200010000000300020001000000030002000100000003000200010000000300020001000000030002000100000003000200",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.e4836c7093466p+4,workers);
    // dtype-int32
    captured(PixelType::int32,5,7,
        "feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff030000000100000000000000feffffff0300000001000000",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // dtype-uint32
    captured(PixelType::uint32,5,7,
        "0000000003000000020000000100000000000000030000000200000001000000000000000300000002000000010000000000000003000000020000000100000000000000030000000200000001000000000000000300000002000000010000000000000003000000020000000100000000000000030000000200000001000000000000000300000002000000",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.e4836c7093466p+4,workers);
    // dtype-int64
    captured(PixelType::int64,5,7,
        "0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000ffffffffffffffff0000000000000080ffffffffffffff7f0000000000000000",
        {33,3,9,0,9,0,0,0,3,6,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.4cd2e89795e73p+4,workers);
    // dtype-uint64
    captured(PixelType::uint64,5,7,
        "0000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff010000000000008000000000000000800000000000000000ffffffffffffffff0100000000000080",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.e4836c7093466p+4,workers);
    // dtype-float16
    captured(PixelType::float32,5,7,
        "000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // dtype-float32
    captured(PixelType::float32,5,7,
        "000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f00000000000000c0000040400000803f",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // dtype-float64
    captured(PixelType::float64,5,7,
        "00000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f000000000000000000000000000000c00000000000000840000000000000f03f",
        {21,3,9,1,9,2,0,0,3,0,2,6,1,6,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.dd26f4372c7c3p+4,workers);
    // directions-two
    captured(PixelType::uint8,5,7,
        "0003020100030201000302010003020100030201000302010003020100030201000302",
        {17,3,3,2,3,3,6,0,3,0,3,6,2,6,0,6},
        {0x0.0p+0,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0,0x1.921fb54442d18p+0,0x1.921fb54442d18p+1,0x0.0p+0,0x0.0p+0,0x1.921fb54442d18p+0,0x1.921fb54442d18p+1,0x0.0p+0,0x0.0p+0},0x1.2106ca4910069p+5,workers);
    // nonfinite
    captured(PixelType::float64,2,4,
        "00000000000000000000000000000080000000000000f87f000000000000f07f000000000000f0ff00000000000000c00000000000000840000000000000f87f",
        {17,2,1,0,1,0,1,0,2,0,0,0,0,0,0,0},
        {0x0.0p+0,0x1.573bf3790c7fep+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p+0,0x0.0p+0,0x1.573bf3790c7fep+0,0x0.0p+0,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x1.1c5831add62e4p-1,0x1.1c5831add62e4p-1,0x1.921fb54442d18p-1,0x1.921fb54442d18p+0,0x0.0p+0,0x0.0p+0},0x1.33b0ed4351ba2p+2,workers);
}
template<class T>void typed(PixelType type,int workers){
    for(std::size_t h:{0,1,3,513})for(std::size_t w:{0,1,7,31,32,33,513}){
        std::vector<T> values(h*w);std::vector<unsigned char> binary(values.size());
        for(std::size_t i=0;i<values.size();++i){
            const auto k=i%8;
            if constexpr(std::is_floating_point_v<T>){
                const std::array<T,8> v{-std::numeric_limits<T>::infinity(),T(-1),-T(0),T(0),
                    std::numeric_limits<T>::denorm_min(),std::numeric_limits<T>::max(),std::numeric_limits<T>::infinity(),std::numeric_limits<T>::quiet_NaN()};
                values[i]=v[k];binary[i]=k>=4&&k<=6;
            }else if(type==PixelType::boolean){values[i]=T(k%2);binary[i]=k%2;}
            else if constexpr(std::is_signed_v<T>){
                const std::array<T,4> v{std::numeric_limits<T>::min(),0,std::numeric_limits<T>::max(),1};values[i]=v[k%4];binary[i]=k%4>=2;
            }else{
                const std::array<T,4> v{0,1,std::numeric_limits<T>::max(),T(std::numeric_limits<T>::max()/2+1)};values[i]=v[k%4];binary[i]=k%4!=0;
            }
        }
        const auto before=values;const auto histogram=reference(binary,h,w);
        double original=0;for(std::size_t i=0;i<16;++i)original+=coefficients[i]*histogram[i];
        check({values.data(),type,h,w},histogram,coefficients.data(),original,workers);
        require(values.empty()||std::memcmp(values.data(),before.data(),values.size()*sizeof(T))==0);
    }
}
void invalid(int workers){
    const unsigned char pixel=1;
    for(int kind:{0,1,2,3,4}){
        Histogram output;output.fill(73);bool rejected=false;
        try{
            detail::CroftonInput input{&pixel,PixelType::uint8,1,1};
            if(kind==0)input.data=nullptr;
            if(kind==1)input.height=std::numeric_limits<std::size_t>::max();
            if(kind==2)input.type=static_cast<PixelType>(99);
            detail::crofton_counts(input,output.data(),kind==3?0:kind==4?1025:workers);
        }catch(const std::invalid_argument&){rejected=true;}catch(const std::overflow_error&){rejected=true;}
        require(rejected&&std::all_of(output.begin(),output.end(),[](auto x){return x==73;}));
    }
}
}
int main(int argc,char**argv){
    if(argc!=2)return 2;journal.open(argv[1],std::ios::binary);require(bool(journal));require(get_num_threads()==1);
    std::vector<int> workers{1};
#ifdef _OPENMP
    workers.push_back(4);
#else
    bool refused=false;try{set_num_threads(4);}catch(const std::invalid_argument& e){refused=std::string(e.what())=="this build has no OpenMP support";}
    require(refused&&get_num_threads()==1);
#endif
    for(int worker:workers){set_num_threads(worker);originals(worker);
        typed<std::uint8_t>(PixelType::boolean,worker);typed<std::uint8_t>(PixelType::uint8,worker);typed<std::int8_t>(PixelType::int8,worker);
        typed<std::uint16_t>(PixelType::uint16,worker);typed<std::int16_t>(PixelType::int16,worker);
        typed<std::uint32_t>(PixelType::uint32,worker);typed<std::int32_t>(PixelType::int32,worker);
        typed<std::uint64_t>(PixelType::uint64,worker);typed<std::int64_t>(PixelType::int64,worker);
        typed<float>(PixelType::float32,worker);typed<double>(PixelType::float64,worker);invalid(worker);
    }
    set_num_threads(1);journal.close();std::cout<<checks<<" Crofton controls PASS avx2="<<has_avx2()<<" workers="<<get_num_threads()<<'\n';
}
