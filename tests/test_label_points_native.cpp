#include "label_points.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("label-points control "+std::to_string(checks));}
std::uint64_t word(std::ifstream& in){std::uint64_t v=0;in.read(reinterpret_cast<char*>(&v),8);require(bool(in));return v;}
void exercise(const std::vector<std::size_t>& dimensions,
              const std::vector<std::vector<std::int64_t>>& logical,
              std::ptrdiff_t stride,int workers,std::ofstream& stream,
              const std::vector<std::uint64_t>* genuine=nullptr) {
    const auto ndim=dimensions.size(),points=logical[0].size();
    std::size_t count=1;for(auto d:dimensions)count*=d;
    const auto slots=std::size_t(stride<0?-stride:stride)/8;
    std::vector<std::vector<std::int64_t>> storage(ndim);
    std::vector<const std::int64_t*> axes(ndim);std::vector<std::ptrdiff_t> strides(ndim,stride);
    for(std::size_t a=0;a<ndim;++a) {
        storage[a].assign(points?1+(points-1)*slots:1,-77);
        auto* origin=storage[a].data()+(stride<0&&points?(points-1)*slots:0);
        axes[a]=origin;
        for(std::size_t p=0;p<points;++p)
            std::memcpy(reinterpret_cast<char*>(origin)+std::ptrdiff_t(p)*stride,&logical[a][p],8);
    }
    const auto before=storage;
    constexpr auto sentinel=std::uint64_t(0xa55aa55adeadbeefULL);
    std::vector<std::uint64_t> output(count+4,sentinel),expected(count,0);
    // Independent reverse search per destination: last matching point wins.
    std::vector<std::size_t> coordinate(ndim);
    for(std::size_t destination=0;destination<count;++destination) {
        auto rest=destination;
        for(std::size_t a=ndim;a-->0;){coordinate[a]=rest%dimensions[a];rest/=dimensions[a];}
        for(std::size_t p=points;p-->0;) {
            bool equal=true;
            for(std::size_t a=0;a<ndim;++a)equal=equal&&std::size_t(logical[a][p])==coordinate[a];
            if(equal){expected[destination]=std::uint64_t(p)+1;break;}
        }
    }
    if(genuine)require(expected==*genuine);
    const auto global=get_num_threads();
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INVALID);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
    const LabelPointsInput input{axes.data(),strides.data(),dimensions.data(),ndim,points,count};
    label_points_into(input,output.data()+2,workers);
    require(get_num_threads()==global);require(std::fegetround()==rounding);
    require(std::fetestexcept(FE_ALL_EXCEPT)==flags);require(storage==before);
    require(output[0]==sentinel&&output[1]==sentinel&&output[count+2]==sentinel&&output[count+3]==sentinel);
    require(std::equal(expected.begin(),expected.end(),output.begin()+2));
    std::uint64_t hash=1469598103934665603ULL;
    for(auto value:output)for(unsigned byte=0;byte<8;++byte){hash^=(value>>(8*byte))&255;hash*=1099511628211ULL;}
    stream.write(reinterpret_cast<const char*>(&hash),8);
}
void fixtures(const std::string& path,int workers,std::ofstream& stream) {
    std::ifstream in(path,std::ios::binary);require(bool(in));const auto records=word(in);require(records==62);
    for(std::size_t record=0;record<records;++record) {
        const auto ndim=word(in),points=word(in),count=word(in);
        std::vector<std::size_t> dimensions(ndim);for(auto& d:dimensions)d=word(in);
        std::vector<std::vector<std::int64_t>> logical(ndim,std::vector<std::int64_t>(points));
        for(auto& axis:logical)for(auto& index:axis)index=std::int64_t(word(in));
        std::vector<std::uint64_t> expected(count);for(auto& v:expected)v=word(in);
        for(auto stride:{std::ptrdiff_t(8),std::ptrdiff_t(24),std::ptrdiff_t(-16)})
            exercise(dimensions,logical,stride,workers,stream,&expected);
    }
    require(in.peek()==std::char_traits<char>::eof());
}
}
int main(int argc,char** argv) {
    require(argc==2||argc==3);std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
    const auto saved_round=std::fegetround();
#ifdef _OPENMP
    set_num_threads(3);
#else
    set_num_threads(1);
#endif
    for(int workers:{1,4}) {
#ifndef _OPENMP
        if(workers==4)continue;
#endif
        for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            require(std::fesetround(rounding)==0);
            for(auto count:std::vector<std::size_t>{0,1,3,4,15,16,17,31,32,33,127,256,262143,262144,262149}) {
                std::vector<std::vector<std::int64_t>> points(1);
                if(count)points[0]={0,std::int64_t(count-1),0,std::int64_t(count/2),0};
                for(auto stride:{std::ptrdiff_t(8),std::ptrdiff_t(24),std::ptrdiff_t(-16)})
                    exercise({count},points,stride,workers,stream);
            }
            exercise({3,4,5},{{0,2,1,2,2},{1,3,2,3,3},{2,4,0,4,4}},-24,workers,stream);
            exercise({2,3,2,4},{{0,1,0},{1,2,1},{0,1,0},{2,3,2}},16,workers,stream);
            exercise({0,5},{{},{}},8,workers,stream);
            exercise({1},{{}},8,workers,stream);
        }
        if(argc==3)fixtures(argv[2],workers,stream);
    }
    require(std::fesetround(saved_round)==0);std::feclearexcept(FE_ALL_EXCEPT);set_num_threads(1);
    std::cout<<"label-points native "<<checks<<" checks PASS\n";
}
