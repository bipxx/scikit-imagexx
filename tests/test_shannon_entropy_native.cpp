#include "shannon_entropy.hpp"
#include "skimagexx/histogram_matching.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t checks=0;std::ofstream journal;
void require(bool b){++checks;if(!b)throw std::runtime_error("Shannon native control "+std::to_string(checks));}
void counts(const void* data,std::size_t n,PixelType type,const std::vector<std::size_t>& expected,int workers){
    auto actual=histogram_sorted_counts(data,n,type,workers);require(actual==expected);
    journal.write(reinterpret_cast<const char*>(actual.data()),std::streamsize(actual.size()*sizeof(std::size_t)));
}
void entropy(const std::vector<std::ptrdiff_t>& c,double expected,int workers){
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const int before=std::fetestexcept(FE_ALL_EXCEPT);
    const double actual=detail::shannon_entropy_counts(c.data(),c.size(),workers);
    require(std::fetestexcept(FE_ALL_EXCEPT)==before);std::feclearexcept(FE_ALL_EXCEPT);
    require(std::abs(actual-expected)<=2e-13+2e-12*std::abs(expected));
    if(expected==0)require(actual==0&&!std::signbit(actual));
    journal.write(reinterpret_cast<const char*>(&actual),sizeof(actual));
}
}
int main(int argc,char**argv){
    if(argc!=2)return 2;journal.open(argv[1],std::ios::binary);require(bool(journal));
    std::vector<int> budgets{1};
#ifdef _OPENMP
    budgets.push_back(4);
#endif
    for(int workers:budgets){set_num_threads(workers);
        counts(nullptr,0,PixelType::uint8,{},workers);
        const std::uint64_t wide[]={UINT64_C(1)<<63,(UINT64_C(1)<<63)+1,(UINT64_C(1)<<63)+1,UINT64_MAX};
        counts(wide,4,PixelType::uint64,{1,2,1},workers);
        const std::int64_t signed_wide[]={INT64_MIN,INT64_MIN+1,INT64_MAX,INT64_MIN+1};
        counts(signed_wide,4,PixelType::int64,{1,2,1},workers);
        const unsigned char booleans[]={0,1,1,0,1};counts(booleans,5,PixelType::boolean,{2,3},workers);
        const std::uint64_t words[]={0,UINT64_C(1)<<63,UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0xfff8000000000007)};
        double symbols[6];std::memcpy(symbols,words,sizeof(words));counts(symbols,6,PixelType::float64,{1,2,1,2},workers);
        const std::uint32_t fwords[]={0,UINT32_C(0x80000000),UINT32_C(0x7f800000),UINT32_C(0xff800000),UINT32_C(0x7fc00001),UINT32_C(0xffc00007)};
        float fsymbols[6];std::memcpy(fsymbols,fwords,sizeof(fwords));counts(fsymbols,6,PixelType::float32,{1,2,1,2},workers);
        const std::int8_t bytes[]={-128,-1,0,127,-128};counts(bytes,5,PixelType::int8,{2,1,1,1},workers);
        for(auto n:{1u,7u,127u,128u,129u,8192u,8193u,262147u}){
            std::vector<std::uint8_t> dense(n);std::vector<std::size_t> want(std::min(n,17u));
            for(std::size_t i=0;i<n;++i){dense[i]=std::uint8_t(i%17);++want[i%17];}
            counts(dense.data(),n,PixelType::uint8,want,workers);
            std::vector<double> hashed(n);for(std::size_t i=0;i<n;++i)hashed[i]=double(i%17)*1000003.;
            counts(hashed.data(),n,PixelType::float64,want,workers);
            std::vector<double> sorted(n);for(std::size_t i=0;i<n;++i)sorted[i]=double(n-i)*1.25;
            counts(sorted.data(),n,PixelType::float64,std::vector<std::size_t>(n,1),workers);
            entropy(std::vector<std::ptrdiff_t>(n,1),std::log(double(n)),workers);
        }
        entropy({},0.,workers);entropy({19},0.,workers);entropy({4,4},std::log(2.),workers);
        entropy({1,1,1,1},std::log(4.),workers);
        entropy({7,1},0.37677016125643675,workers);
        bool refused=false;try{detail::shannon_entropy_counts(std::vector<std::ptrdiff_t>{1,0}.data(),2,workers);}catch(const std::invalid_argument&){refused=true;}require(refused);
    }
    {std::vector<double> input(4097),out(4097);
     for(std::size_t i=0;i<input.size();++i)input[i]=double(i%17)*1000003.;
     match_histograms_into(input.data(),input.data(),out.data(),input.size(),input.size(),1,PixelType::float64,PixelType::float64,PixelType::float64);
     require(out==input);}
    // Existing matching controls from test_native.cpp: values mode remains default.
    set_num_threads(1);
    require(get_num_threads() == 1);
    {
        const std::uint8_t first[]={0,0,1,2},reference[]={10,20};
        double output[4];
        require(match_histograms_into(first,reference,output,4,2,1,PixelType::uint8,PixelType::uint8,PixelType::float64,true)==0);
        require(output[0]==10 && output[1]==10 && output[2]==15 && output[3]==20);
        const std::uint8_t color[]={0,0,1,1,2,2,3,3},palette[]={0,10,3,20};
        std::uint8_t colored[8];
        match_histograms_into(color,palette,colored,4,2,2,PixelType::uint8,PixelType::uint8,PixelType::uint8,true);
        const std::uint8_t expected[]={0,10,0,10,1,15,3,20};
        require(std::equal(colored,colored+8,expected));
        const float planes[]={0,1,2,3,10,11,12,13},template_planes[]={0,3,10,20};
        float strided[8];
        match_histograms_strided_into(planes,template_planes,strided,4,2,2,
            PixelType::float32,PixelType::float32,PixelType::float32,{1,4,1,2});
        const float plane_expected[]={0,10,0,10,1.5f,15,3,20};
        require(std::equal(strided,strided+8,plane_expected));
        match_histograms_strided_into(planes+3,template_planes+1,strided,4,2,2,
            PixelType::float32,PixelType::float32,PixelType::float32,{-1,4,-1,2});
        for(int i=0;i<4;++i)for(int c=0;c<2;++c)require(strided[2*i+c]==plane_expected[2*(3-i)+c]);
        const float broadcast_pixel[]={1,2};
        match_histograms_strided_into(broadcast_pixel,template_planes,strided,4,2,2,
            PixelType::float32,PixelType::float32,PixelType::float32,{0,1,1,2});
        for(int i=0;i<4;++i)require(strided[2*i]==3 && strided[2*i+1]==20);
        const auto extreme=std::numeric_limits<std::ptrdiff_t>::min();
        match_histograms_strided_into(planes,template_planes,strided,1,1,1,
            PixelType::float32,PixelType::float32,PixelType::float32,{extreme,extreme,extreme,extreme});
        require(strided[0]==0);
        bool stride_overflow=false;
        try {match_histograms_strided_into(planes,template_planes,strided,2,2,1,
            PixelType::float32,PixelType::float32,PixelType::float32,{extreme,0,1,0});}
        catch(const std::overflow_error&){stride_overflow=true;}
        require(stride_overflow);
        const float half_input[]={0,1};const double precise[]={1.00048828125+0x1p-40,1.5};
        float rounded[2];
        match_histograms_into(half_input,precise,rounded,2,2,1,PixelType::float32,PixelType::float64,PixelType::float32,false,true);
        require(rounded[0]==1.0009765625f && rounded[1]==1.5f);
        require(match_histograms_into(nullptr,nullptr,nullptr,0,0,1,PixelType::float32,PixelType::float32,PixelType::float32)==0);
        for(auto type:{PixelType::float32,PixelType::float64}) {
            const std::size_t count=262149;
            std::vector<float> fsource(count),ftemplate(count),fout(count);
            std::vector<double> dsource(count),dtemplate(count),dout(count);
            for(std::size_t i=0;i<count;++i){fsource[i]=float(count-1-i);ftemplate[i]=float(i);dsource[i]=double(count-1-i);dtemplate[i]=double(i);}
            const int workers[]={1
#ifdef _OPENMP
                ,4
#endif
            };
            for(auto worker:workers) {
                set_num_threads(worker);
                if(type==PixelType::float32) {
                    match_histograms_into(fsource.data(),ftemplate.data(),fout.data(),count,count,1,type,type,type);
                    require(fout==fsource);
                } else {
                    match_histograms_into(dsource.data(),dtemplate.data(),dout.data(),count,count,1,type,type,type);
                    require(dout==dsource);
                }
            }
        }
        set_num_threads(1);
    }
#ifndef _OPENMP
    bool refused=false;try{histogram_sorted_counts(nullptr,0,PixelType::uint8,4);}catch(const std::invalid_argument&){refused=true;}require(refused);
    refused=false;try{detail::shannon_entropy_counts(nullptr,0,4);}catch(const std::invalid_argument&){refused=true;}require(refused);
#endif
    set_num_threads(1);journal.close();std::cout<<checks<<" native controls PASS\n";
}
