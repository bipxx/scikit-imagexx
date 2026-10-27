#include "perimeter.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Histogram=std::array<std::ptrdiff_t,256>;
using skimagexx::detail::perimeter_counts;
using skimagexx::detail::perimeter_weighted;
namespace {
std::size_t controls=0;
std::ofstream journal;
void check(bool condition){++controls;assert(condition);}
struct Original {
    const char* id;std::vector<std::uint8_t> image;std::size_t height,width;
    bool cross;std::vector<std::ptrdiff_t> histogram;bool success;double result,bound;
};
const double source_weights[50]={0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.3504f333f9de6p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.6a09e667f3bcdp+0,0x0.0p+0,0x1.3504f333f9de6p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x1.0000000000000p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x1.6a09e667f3bcdp+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0,0x0.0p+0};
const std::vector<Original> originals={
    {"geometry-empty00",{},0,0,true,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"geometry-empty05",{},0,5,true,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"geometry-empty40",{},4,0,true,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"geometry-single0",{0},1,1,true,{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"geometry-single1",{1},1,1,true,{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"geometry-row",{1,1,1,1,1,1,1},1,7,true,{0,0,0,2,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.4000000000000p+2,0x1.f400000000063p-44},
    {"geometry-column",{1,1,1,1,1,1},6,1,true,{0,0,0,2,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.0000000000000p+2,0x1.900000000004fp-44},
    {"geometry-rectangle",{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},5,7,true,{3,0,0,0,0,12,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.4000000000000p+4,0x1.f400000000063p-42},
    {"geometry-ring",{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},7,9,true,{3,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0,8,0,4,8,0,16,0,0,0,0,0,0,0,8,0,0,4,0,0,0,0,0,0,0,0,0,0,0},true,0x1.2000000000000p+5,0x1.c200000000059p-41},
    {"geometry-diagonal",{1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1},6,6,true,{12,0,0,0,10,0,0,0,0,0,8,2,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.6a09e667f3bcdp+2,0x1.1ad7bc01366f0p-43},
    {"dtype-bool",{0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1},6,8,true,{0,1,1,0,0,1,0,0,0,0,0,0,6,6,5,4,0,0,0,0,0,0,0,0,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.83e1db337db36p+4,0x1.2f0873403a37fp-41},
    {"dtype-uint8",{0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1},6,8,true,{0,1,1,0,0,1,0,0,0,0,0,0,6,6,5,4,0,0,0,0,0,0,0,0,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.83e1db337db36p+4,0x1.2f0873403a37fp-41},
    {"dtype-int64",{0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0},6,8,true,{0,0,0,0,2,0,10,0,12,0,0,2,0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0},true,0x1.c48c6001f0ac0p+3,0x1.618dab01840acp-42},
    {"dtype-uint64",{0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1},6,8,true,{0,0,1,0,0,0,0,0,0,0,1,0,4,6,6,6,0,0,0,0,0,0,0,0,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.93e1db337db36p+4,0x1.3b8873403a381p-41},
    {"neighborhood-eight",{0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,0,0,1},6,8,false,{0,1,1,0,0,1,0,0,0,0,0,0,6,6,5,4,0,0,0,0,0,0,0,0,12,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.83e1db337db36p+4,0x1.2f0873403a37fp-41},
    {"nonbinary-single2",{2},1,1,true,{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x0.0p+0,0x0.0p+0},
    {"nonbinary-single255",{255},1,1,true,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},false,0x0.0p+0,0x0.0p+0},
    {"nonbinary-full2",{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},6,8,true,{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,0,8,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,0,0,0,0,0,0,0,0,0,4},false,0x0.0p+0,0x0.0p+0},
    {"nonbinary-castwrap",{0,1,0,1,0,1},2,3,true,{0,0,0,0,2,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},true,0x1.6a09e667f3bcdp+0,0x1.1ad7bc01366f0p-45}
};


Histogram literal(const std::vector<std::uint8_t>& image,std::size_t h,std::size_t w,bool cross){
    std::vector<std::uint8_t> border(image.size());
    for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){
        bool eroded=true;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            if(cross&&dx&&dy)continue;
            const auto yy=std::ptrdiff_t(y)+dy,xx=std::ptrdiff_t(x)+dx;
            if(yy<0||xx<0||yy>=std::ptrdiff_t(h)||xx>=std::ptrdiff_t(w)||!image[std::size_t(yy)*w+std::size_t(xx)])eroded=false;
        }
        border[y*w+x]=std::uint8_t(image[y*w+x]-std::uint8_t(eroded));
    }
    const int weights[3][3]={{10,2,10},{2,1,2},{10,2,10}};Histogram result{};
    for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){
        unsigned value=0;
        for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            const auto yy=std::ptrdiff_t(y)+dy,xx=std::ptrdiff_t(x)+dx;
            if(yy>=0&&xx>=0&&yy<std::ptrdiff_t(h)&&xx<std::ptrdiff_t(w))
                value+=unsigned(border[std::size_t(yy)*w+std::size_t(xx)])*unsigned(weights[dy+1][dx+1]);
        }
        ++result[value&255u];
    }
    return result;
}
Histogram run(const std::vector<std::uint8_t>& input,std::size_t h,std::size_t w,bool cross,int workers,std::size_t& length){
    std::vector<std::uint8_t> storage(input.size()+2,0xa5);
    std::copy(input.begin(),input.end(),storage.begin()+1);const auto before=storage;
    std::array<std::ptrdiff_t,258> result;result.fill(-777);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    length=perimeter_counts(input.empty()?nullptr:storage.data()+1,h,w,cross,result.data()+1,workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);check(storage==before);
    check(result.front()==-777&&result.back()==-777);
    Histogram counts;std::copy_n(result.begin()+1,256,counts.begin());
    check(std::accumulate(counts.begin(),counts.end(),std::ptrdiff_t(0))==std::ptrdiff_t(h*w));
    std::size_t expected=256;while(expected>50&&!counts[expected-1])--expected;
    check(length==expected);
    journal.write(reinterpret_cast<const char*>(counts.data()),sizeof(counts));
    return counts;
}
double weighted(const Histogram& h,const double* weights,int workers){
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const auto result=perimeter_weighted(h.data(),weights,workers);
    check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    journal.write(reinterpret_cast<const char*>(&result),sizeof(result));return result;
}
void cohort(int workers){
    for(const auto& row:originals){
        std::size_t length;const auto got=run(row.image,row.height,row.width,row.cross,workers,length);
        check(length==row.histogram.size());
        for(std::size_t i=0;i<256;++i)check(got[i]==(i<row.histogram.size()?row.histogram[i]:0));
        if(row.success){
            const auto result=weighted(got,source_weights,workers);
            check(std::abs(result-row.result)<=row.bound);
            if(row.bound==0.)check(result==row.result&&std::signbit(result)==std::signbit(row.result));
        }else check(length>50); // Source shape error: no weighted stage.
    }
    for(auto h:{std::size_t(0),std::size_t(1),std::size_t(3),std::size_t(513)})
    for(auto w:{std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(31),std::size_t(32),std::size_t(33),std::size_t(65),std::size_t(513)})
    for(bool cross:{false,true})for(int pattern=0;pattern<3;++pattern){
        std::vector<std::uint8_t> image(h*w);
        for(std::size_t i=0;i<image.size();++i)image[i]=pattern==0?std::uint8_t((i*7+i/w)%3!=0):pattern==1?std::uint8_t(i*73+19):std::uint8_t(255);
        const auto expected=literal(image,h,w,cross);std::size_t length;
        const auto got=run(image,h,w,cross,workers,length);check(got==expected);
        if(length==50){
            double dyadic[50];for(std::size_t i=0;i<50;++i)dyadic[i]=double(i%7)/8.;
            double total=0.;for(std::size_t i=0;i<50;++i)total+=dyadic[i]*double(expected[i]);
            check(weighted(got,dyadic,workers)==total);
        }
    }
}
void rejection(){
    Histogram out;out.fill(51);const auto before=out;std::uint8_t byte=1;
    for(int workers:{0,1025}){
        bool refused=false;try{perimeter_counts(&byte,1,1,true,out.data(),workers);}catch(const std::invalid_argument&){refused=true;}
        check(refused&&out==before);
    }
    bool refused=false;try{perimeter_counts(nullptr,1,1,true,out.data(),1);}catch(const std::invalid_argument&){refused=true;}
    check(refused&&out==before);
    refused=false;try{perimeter_counts(&byte,1,1,true,nullptr,1);}catch(const std::invalid_argument&){refused=true;}
    check(refused);
    refused=false;try{perimeter_counts(&byte,std::size_t(std::numeric_limits<std::ptrdiff_t>::max()),2,true,out.data(),1);}catch(const std::overflow_error&){refused=true;}
    check(refused&&out==before);
}
}
int main(int argc,char** argv){
    check(argc==2);journal.open(argv[1],std::ios::binary);check(bool(journal));
    check(skimagexx::get_num_threads()==1);rejection();cohort(1);
#ifdef _OPENMP
    skimagexx::set_num_threads(4);cohort(4);skimagexx::set_num_threads(1);
#else
    bool refused=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument& error){refused=std::string(error.what())=="this build has no OpenMP support";}
    check(refused&&skimagexx::get_num_threads()==1);
#endif
    journal.close();check(bool(journal));
    std::printf("%zu Perimeter controls PASS avx2=%d workers=%d\n",controls,int(skimagexx::has_avx2()),skimagexx::get_num_threads());
}
