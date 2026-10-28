#include "skimagexx/approximate_polygon.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,cases=0,calls=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("RDP native check "+std::to_string(checks));}
std::uint64_t bits(double value){std::uint64_t result;std::memcpy(&result,&value,8);return result;}
double real(std::uint64_t value){double result;std::memcpy(&result,&value,8);return result;}
std::uint64_t word(std::ifstream& stream){std::uint64_t result;stream.read(reinterpret_cast<char*>(&result),8);check(bool(stream));return result;}
std::ptrdiff_t stride(std::ifstream& stream){const auto raw=word(stream);std::int64_t value;std::memcpy(&value,&raw,8);return std::ptrdiff_t(value);}
std::vector<unsigned char> bytes(std::ifstream& stream,std::size_t count){std::vector<unsigned char> result(count);if(count)stream.read(reinterpret_cast<char*>(result.data()),std::streamsize(count));check(bool(stream));return result;}
struct TrigTable {
    std::map<std::pair<std::uint64_t,std::uint64_t>,ApproximatePolygonTrig> values;
    std::thread::id caller=std::this_thread::get_id();
    std::size_t calls=0;
};
ApproximatePolygonTrig table_trig(void* context,double dr,double dc) {
    auto& table=*static_cast<TrigTable*>(context);check(std::this_thread::get_id()==table.caller);++table.calls;
    const auto found=table.values.find({bits(dr),bits(dc)});check(found!=table.values.end());return found->second;
}
void exercise(const std::vector<double>& backing,std::size_t origin,std::size_t count,
              std::ptrdiff_t rs,std::ptrdiff_t cs,double tolerance,
              const std::vector<std::uint8_t>& expected,int workers,std::ofstream& stream,
              TrigTable* table=nullptr,const std::vector<unsigned char>* frozen_gather=nullptr) {
    ++cases;const auto before=backing;const auto global=get_num_threads();
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        check(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);const auto prior=table?table->calls:0;
        std::vector<std::uint8_t> mask(count+32,0xa5);std::vector<double> scratch(count+4,42.);
        const ApproximatePolygonInput input{backing.data()+origin,count,rs,cs};++calls;
        const bool accepted=approximate_polygon_select(input,tolerance,mask.data()+16,scratch.data()+2,workers,table?table_trig:nullptr,table);
        check(accepted==(rounding==FE_TONEAREST));
        check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags&&get_num_threads()==global);
        check(backing.size()==before.size()&&std::memcmp(backing.data(),before.data(),backing.size()*8)==0);
        check(std::all_of(mask.begin(),mask.begin()+16,[](auto value){return value==0xa5;})&&std::all_of(mask.end()-16,mask.end(),[](auto value){return value==0xa5;}));
        check(scratch[0]==42.&&scratch[1]==42.&&scratch[count+2]==42.&&scratch[count+3]==42.);
        if(accepted) {
            check(expected.size()==count&&std::equal(expected.begin(),expected.end(),mask.begin()+16));
            if(table)check(table->calls>prior);
            if(frozen_gather) {
                std::vector<unsigned char> gathered;
                for(std::size_t i=0;i<count;++i)if(mask[i+16])for(std::size_t axis=0;axis<2;++axis) {
                    const auto* value=input.data+std::ptrdiff_t(i)*rs+std::ptrdiff_t(axis)*cs;
                    const auto* raw=reinterpret_cast<const unsigned char*>(value);gathered.insert(gathered.end(),raw,raw+8);
                }
                check(gathered==*frozen_gather);
            }
        }else {
            check(std::all_of(mask.begin(),mask.end(),[](auto value){return value==0xa5;}));
            check(std::all_of(scratch.begin(),scratch.end(),[](auto value){return value==42.;}));
            if(table)check(table->calls==prior);
        }
        std::uint64_t hash=1469598103934665603ULL;
        for(auto value:mask){hash^=value;hash*=1099511628211ULL;}
        hash^=accepted;stream.write(reinterpret_cast<const char*>(&hash),8);
    }
    std::fesetround(FE_TONEAREST);std::feclearexcept(FE_ALL_EXCEPT);
}
void genuine(const char* path,int workers,std::ofstream& stream) {
    std::ifstream file(path,std::ios::binary);check(bool(file));const auto records=word(file);
    for(std::size_t record=0;record<records;++record) {
        const auto id=bytes(file,word(file));check(!id.empty());const auto count=word(file);
        const auto rs=stride(file),cs=stride(file);const auto origin=word(file),storage=word(file);const auto tolerance=real(word(file));
        auto raw=bytes(file,storage*8);std::vector<double> backing(storage);std::memcpy(backing.data(),raw.data(),raw.size());
        const auto mask=bytes(file,count),gather=bytes(file,word(file));const auto pairs=word(file);TrigTable table;
        for(std::size_t i=0;i<pairs;++i) {
            const auto dr=word(file),dc=word(file);const auto sine=real(word(file)),cosine=real(word(file));
            check(table.values.emplace(std::make_pair(dr,dc),ApproximatePolygonTrig{sine,cosine}).second);
        }
        exercise(backing,origin,count,rs,cs,tolerance,mask,workers,stream,&table,&gather);
    }
    check(file.peek()==std::char_traits<char>::eof());
}
void line(std::size_t count,int layout,int workers,std::ofstream& stream) {
    std::ptrdiff_t rs=2,cs=1;
    if(layout==1)rs=-2;else if(layout==2)cs=-1;else if(layout==3){rs=1;cs=std::ptrdiff_t(count);}else if(layout==4)rs=0;else if(layout==5){rs=3;cs=0;}
    const auto low=std::min(std::ptrdiff_t(0),std::ptrdiff_t(count-1)*rs)+std::min(std::ptrdiff_t(0),cs);
    const auto high=std::max(std::ptrdiff_t(0),std::ptrdiff_t(count-1)*rs)+std::max(std::ptrdiff_t(0),cs);
    const auto origin=std::size_t(-low);std::vector<double> backing(std::size_t(high-low)+1,17.);
    for(std::size_t i=0;i<count;++i) {
        backing[std::size_t(std::ptrdiff_t(origin)+std::ptrdiff_t(i)*rs)]=layout==5?double(i):0.;
        backing[std::size_t(std::ptrdiff_t(origin)+std::ptrdiff_t(i)*rs+cs)]=layout==4?0.:double(i);
    }
    std::vector<std::uint8_t> expected(count,0);expected[0]=expected[count-1]=1;
    if(count>256*1024) {
        TrigTable table;table.values.emplace(std::make_pair(bits(0.),bits(double(count-1))),ApproximatePolygonTrig{-0.,1.});
        exercise(backing,origin,count,rs,cs,.25,expected,workers,stream,&table);
    }else exercise(backing,origin,count,rs,cs,.25,expected,workers,stream);
}
ApproximatePolygonTrig throwing(void*,double,double){throw std::runtime_error("trig failure sentinel");}
ApproximatePolygonTrig nonfinite(void*,double,double){return {std::numeric_limits<double>::quiet_NaN(),1.};}
void refusal(std::vector<double> input,double tolerance,int workers,std::ofstream& stream,ApproximatePolygonTrigProvider provider=nullptr) {
    ++cases;std::vector<std::uint8_t> mask(input.size()/2+32,0xa5);std::vector<double> scratch(input.size()/2+4,42.);const auto before=input;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);const auto global=get_num_threads();++calls;
    check(!approximate_polygon_select({input.data(),input.size()/2,2,1},tolerance,mask.data()+16,scratch.data()+2,workers,provider));
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==FE_TONEAREST&&get_num_threads()==global);
    check(std::memcmp(input.data(),before.data(),input.size()*8)==0);
    check(std::all_of(mask.begin(),mask.begin()+16,[](auto value){return value==0xa5;})&&std::all_of(mask.end()-16,mask.end(),[](auto value){return value==0xa5;}));
    check(scratch[0]==42.&&scratch[1]==42.&&scratch[scratch.size()-2]==42.&&scratch.back()==42.);
    const std::uint64_t marker=0;stream.write(reinterpret_cast<const char*>(&marker),8);std::feclearexcept(FE_ALL_EXCEPT);
}
}
int main(int argc,char** argv) {
    if(argc!=2&&argc!=3)throw std::runtime_error("usage: approximate-polygon-native output [frozen-fixtures]");
    std::ofstream stream(argv[1],std::ios::binary);check(bool(stream));const auto old=get_num_threads();
#ifdef _OPENMP
    set_num_threads(3);
#endif
    const auto global=get_num_threads();
    for(int workers:{1,4}) {
        if(argc==3)genuine(argv[2],workers,stream);
        for(std::size_t count:{1,2,3,4,5,6,7,8,9,17,257})for(int layout=0;layout<6;++layout)line(count,layout,workers,stream);
        // Above the shared parallel threshold, with an analytic two-endpoint
        // oracle; this is not a quadratic worst-case zigzag workload.
        line(256*1024+3,0,workers,stream);
        const auto tiny=std::numeric_limits<double>::denorm_min();
        std::vector<double> perpendicular{0.,0.,tiny,.2,tiny,.4,tiny,.6,tiny,.8,0.,1.};
        std::vector<std::uint8_t> endpoints{1,0,0,0,0,1};exercise(perpendicular,0,6,2,1,.25,endpoints,workers,stream);
        std::vector<double> euclidean{1.,1.,tiny,tiny,tiny,tiny,tiny,tiny,tiny,tiny,2.,2.};exercise(euclidean,0,6,2,1,10.,endpoints,workers,stream);
        std::vector<double> large{1e200,0.,1e200,.2,1e200,.4,1e200,.6,1e200,.8,1e200,1.};exercise(large,0,6,2,1,.25,endpoints,workers,stream);
        for(auto invalid:{UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001)}) {
            for(std::size_t point:{0,3,6})for(std::size_t axis:{0,1}) {
                std::vector<double> values(14,0.);values[point*2+axis]=real(invalid);refusal(values,.5,workers,stream);
            }
            refusal({0.,0.,1.,1.,2.,2.},real(invalid),workers,stream);
        }
        for(double tolerance:{0.,-0.,-1.})refusal({0.,0.,1.,1.,2.,2.},tolerance,workers,stream);
        refusal({0.,0.,1.,1.,2.,2.},.5,workers,stream,nonfinite);
        const auto maximum=std::numeric_limits<double>::max();refusal({maximum,0.,-maximum,1.,maximum,2.},.5,workers,stream);
        refusal({0.,0.,1e-200,1e-200,2e-200,2e-200},.5,workers,stream);
        std::array<double,6> values{0.,0.,1.,1.,2.,2.},scratch{};std::array<std::uint8_t,3> mask{};
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);bool caught=false;++calls;++cases;
        try{approximate_polygon_select({values.data(),3,2,1},.5,mask.data(),scratch.data(),workers,throwing);}catch(const std::runtime_error& error){caught=std::string(error.what())=="trig failure sentinel";}
        check(caught&&std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==FE_TONEAREST&&get_num_threads()==global);std::feclearexcept(FE_ALL_EXCEPT);
    }
    set_num_threads(old);check(bool(stream));std::cout<<"RDP native case-worker combinations="<<cases<<" calls="<<calls<<" checks="<<checks<<" PASS\n";
}
