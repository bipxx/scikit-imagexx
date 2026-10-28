#include "skimagexx/subdivide_polygon.hpp"
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

using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,cases=0,calls=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("subdivision native check "+std::to_string(checks));}
double real(std::uint64_t bits){double value;std::memcpy(&value,&bits,8);return value;}
std::uint64_t word(std::ifstream& f){std::uint64_t value;f.read(reinterpret_cast<char*>(&value),8);check(bool(f));return value;}
std::ptrdiff_t stride(std::ifstream& f){const auto bits=word(f);std::int64_t value;std::memcpy(&value,&bits,8);return std::ptrdiff_t(value);}
std::vector<unsigned char> bytes(std::ifstream& f,std::size_t n){std::vector<unsigned char> v(n);if(n)f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n));check(bool(f));return v;}
unsigned length(unsigned degree){return(degree+3)/2;}
std::size_t rows(std::size_t count,unsigned degree,bool circular,bool preserve){const auto n=circular?count:count-length(degree)+1;return n*2+(circular?1:preserve?2:0);}
// Independent output-row-centric stencil oracle. Integer masks come from the
// pinned original table, not the native private header. Explicitly compare the
// observed minimum sequential and modern first-four grouping policies too.
std::vector<double> oracle(const SubdividePolygonInput& input,unsigned degree,bool circular,bool preserve) {
    static const int even[7][5]={{1,1},{3,1},{1,6,1},{5,10,1},{1,15,15,1},{7,35,21,1},{1,28,70,28,1}};
    static const int odd[7][5]={{1,1},{1,3},{0,4,4},{1,10,5},{0,6,20,6},{1,21,35,7},{0,8,56,56,8}};
    std::vector<double> result(rows(input.count,degree,circular,preserve)*2);
    const auto prefix=!circular&&preserve?1:0;const auto middle=(result.size()/2-(circular?1:preserve?2:0));
    for(std::size_t output_row=0;output_row<middle;++output_row)for(std::size_t channel=0;channel<2;++channel) {
        const auto n=length(degree);const auto* mask=output_row%2?even[degree-1]:odd[degree-1];double products[5]{};
        for(unsigned k=0;k<n;++k) {
            auto index=std::int64_t(output_row/2)+(circular?std::int64_t((n-1)/2):std::int64_t(n-1))-std::int64_t(k);
            if(circular){while(index<0)index+=std::int64_t(input.count);while(std::size_t(index)>=input.count)index-=std::int64_t(input.count);}
            const auto value=input.data[index*input.row_stride+std::ptrdiff_t(channel)*input.column_stride];
            products[k]=(double(mask[k])/double(1U<<degree))*value;
        }
        double sequential=0.;for(unsigned k=0;k<n;++k)sequential+=products[k];
        double modern=0.;unsigned k=0;
        if(n>=4){auto group=products[0]+products[1];group+=products[2];group+=products[3];modern+=group;k=4;}
        for(;k<n;++k)modern+=products[k];
        check(std::memcmp(&sequential,&modern,8)==0);
        result[(output_row+prefix)*2+channel]=sequential;
    }
    if(circular)std::memcpy(result.data()+result.size()-2,result.data(),16);
    else if(preserve)for(std::size_t channel=0;channel<2;++channel) {
        result[channel]=input.data[std::ptrdiff_t(channel)*input.column_stride];
        result[result.size()-2+channel]=input.data[std::ptrdiff_t(input.count-1)*input.row_stride+std::ptrdiff_t(channel)*input.column_stride];
    }
    return result;
}
void exercise(const std::vector<double>& backing,std::size_t origin,std::size_t count,
    std::ptrdiff_t rs,std::ptrdiff_t cs,unsigned degree,bool circular,bool preserve,
    const std::vector<double>& expected,int workers,std::ofstream& stream) {
    ++cases;const auto before=backing;const auto global=get_num_threads();
    check(expected.size()==rows(count,degree,circular,preserve)*2);
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        check(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        std::vector<double> actual(expected.size()+6,42.);++calls;
        const bool accepted=subdivide_polygon_into({backing.data()+origin,count,rs,cs},degree,circular,preserve,actual.data()+3,workers);
        check(accepted==(rounding==FE_TONEAREST));
        check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags&&get_num_threads()==global);
        check(std::memcmp(backing.data(),before.data(),backing.size()*8)==0);
        check(std::all_of(actual.begin(),actual.begin()+3,[](auto x){return x==42.;})&&std::all_of(actual.end()-3,actual.end(),[](auto x){return x==42.;}));
        if(accepted) {
            if(std::memcmp(actual.data()+3,expected.data(),expected.size()*8)!=0)for(std::size_t i=0;i<expected.size();++i)if(std::memcmp(actual.data()+3+i,expected.data()+i,8)!=0){std::cerr<<"native difference word="<<i<<" actual="<<std::hexfloat<<actual[i+3]<<" expected="<<expected[i]<<"\n";break;}
            check(std::memcmp(actual.data()+3,expected.data(),expected.size()*8)==0);
        }
        else check(std::all_of(actual.begin(),actual.end(),[](auto x){return x==42.;}));
        std::uint64_t hash=1469598103934665603ULL;const auto* raw=reinterpret_cast<const unsigned char*>(actual.data()+3);
        for(std::size_t i=0;i<expected.size()*8;++i){hash^=raw[i];hash*=1099511628211ULL;}
        hash^=accepted;stream.write(reinterpret_cast<const char*>(&hash),8);
    }
    std::fesetround(FE_TONEAREST);std::feclearexcept(FE_ALL_EXCEPT);
}
void genuine(const char* path,int workers,std::ofstream& stream) {
    std::ifstream f(path,std::ios::binary);check(bool(f));const auto records=word(f);
    for(std::size_t r=0;r<records;++r) {
        const auto ident=bytes(f,word(f));check(!ident.empty());const auto count=word(f);const auto rs=stride(f),cs=stride(f);
        const auto origin=word(f),storage=word(f);const auto degree=unsigned(word(f));const bool circular=word(f),preserve=word(f);const auto output_rows=word(f);
        const auto raw=bytes(f,storage*8),output=bytes(f,output_rows*16);std::vector<double> backing(storage),expected(output_rows*2);
        std::memcpy(backing.data(),raw.data(),raw.size());std::memcpy(expected.data(),output.data(),output.size());
        std::cerr<<"genuine "<<std::string(ident.begin(),ident.end())<<" checks="<<checks<<"\n";
        const auto independent=oracle({backing.data()+origin,count,rs,cs},degree,circular,preserve);
        if(independent.size()==expected.size()&&std::memcmp(independent.data(),expected.data(),expected.size()*8)!=0)for(std::size_t i=0;i<expected.size();++i)if(std::memcmp(independent.data()+i,expected.data()+i,8)!=0){std::cerr<<"oracle difference word="<<i<<" actual="<<std::hexfloat<<independent[i]<<" expected="<<expected[i]<<"\n";break;}
        check(independent.size()==expected.size()&&std::memcmp(independent.data(),expected.data(),expected.size()*8)==0);
        exercise(backing,origin,count,rs,cs,degree,circular,preserve,expected,workers,stream);
    }
    check(f.peek()==std::char_traits<char>::eof());
}
void control(std::size_t count,unsigned degree,bool circular,bool preserve,int layout,int kind,int workers,std::ofstream& stream) {
    std::ptrdiff_t rs=2,cs=1;if(layout==1)rs=-2;else if(layout==2)cs=-1;else if(layout==3){rs=1;cs=std::ptrdiff_t(count);}else if(layout==4){rs=5;cs=2;}else if(layout==5)rs=0;
    const auto low=std::min(std::ptrdiff_t(0),std::ptrdiff_t(count-1)*rs)+std::min(std::ptrdiff_t(0),cs);
    const auto high=std::max(std::ptrdiff_t(0),std::ptrdiff_t(count-1)*rs)+std::max(std::ptrdiff_t(0),cs);const auto origin=std::size_t(-low);std::vector<double> backing(std::size_t(high-low)+1,17.);
    for(std::size_t i=0;i<count;++i)for(std::size_t channel=0;channel<2;++channel) {
        double value=(double(i%11)-3.)*(channel?-.3:.1);
        if(kind==1)value=(i+channel)%2?-0.:0.;
        else if(kind==2){const double sequence[]={0x1p53,1.,-0x1p53,3.,-2.};value=sequence[(i+channel)%5];}
        else if(kind==3)value=std::ldexp(double(int(i%3)-1),-900);
        else if(kind==4)value=std::ldexp(double(int(i%3)-1),900);
        else if(kind==5)value=channel?-5.:3.;
        else if(kind==6)value=std::numeric_limits<double>::max();
        backing[std::size_t(std::ptrdiff_t(origin)+std::ptrdiff_t(i)*rs+std::ptrdiff_t(channel)*cs)]=value;
    }
    std::vector<double> expected;
    if(kind==5){expected.resize(rows(count,degree,circular,preserve)*2);for(std::size_t i=0;i<expected.size();i+=2){expected[i]=3.;expected[i+1]=-5.;}}
    else expected=oracle({backing.data()+origin,count,rs,cs},degree,circular,preserve);
    exercise(backing,origin,count,rs,cs,degree,circular,preserve,expected,workers,stream);
}
void refusal(std::vector<double> input,std::size_t count,unsigned degree,bool circular,int workers,std::ofstream& stream) {
    ++cases;const auto before=input;std::vector<double> output(128,42.);const auto global=get_num_threads();
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);++calls;
    check(!subdivide_polygon_into({input.data(),count,2,1},degree,circular,false,output.data()+3,workers));
    check(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==FE_TONEAREST&&get_num_threads()==global);
    check(input.empty()||std::memcmp(input.data(),before.data(),input.size()*8)==0);
    check(std::all_of(output.begin(),output.begin()+3,[](auto x){return x==42.;})&&std::all_of(output.end()-3,output.end(),[](auto x){return x==42.;}));
    const std::uint64_t marker=0;stream.write(reinterpret_cast<const char*>(&marker),8);std::feclearexcept(FE_ALL_EXCEPT);
}
}
int main(int argc,char** argv) {
    if(argc!=2&&argc!=3)throw std::runtime_error("usage: subdivide-native output [frozen-fixtures]");
    std::ofstream stream(argv[1],std::ios::binary);check(bool(stream));const auto previous=get_num_threads();
#ifdef _OPENMP
    set_num_threads(3);
#endif
    for(int workers:{1,4}) {
        if(argc==3)genuine(argv[2],workers,stream);
        for(unsigned degree=1;degree<=7;++degree) {
            for(bool circular:{false,true})for(bool preserve:{false,true})for(std::size_t count:{1,2,5,9})
                if(circular||count>=length(degree))control(count,degree,circular,preserve,0,0,workers,stream);
            for(int layout=1;layout<=5;++layout)control(9,degree,true,layout%2,layout,0,workers,stream);
            for(int kind=1;kind<=6;++kind)control(9,degree,kind%2,kind%3==0,0,kind,workers,stream);
            refusal(std::vector<double>(2*(length(degree)-1),0.),length(degree)-1,degree,false,workers,stream);
        }
        // Real independent-output parallel entry: M>256k, no callback/pool.
        control(256*1024+3,7,true,true,0,5,workers,stream);
        for(unsigned degree:{0U,8U,std::numeric_limits<unsigned>::max()})refusal({0.,0.,1.,1.},2,degree,false,workers,stream);
        refusal({},0,2,true,workers,stream);
        for(auto invalid:{UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001)})for(std::size_t point:{0,3,6})for(std::size_t axis:{0,1}) {
            std::vector<double> values(14,1.);values[point*2+axis]=real(invalid);refusal(values,7,2,true,workers,stream);
        }
        const auto tiny=std::numeric_limits<double>::denorm_min();refusal({tiny,tiny,tiny,tiny},2,7,true,workers,stream);
    }
    set_num_threads(previous);check(bool(stream));std::cout<<"subdivide native case-worker combinations="<<cases<<" calls="<<calls<<" checks="<<checks<<" PASS\n";
}
