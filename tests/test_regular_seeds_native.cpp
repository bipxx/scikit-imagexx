#include "regular_seeds.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("regular-seeds check "+std::to_string(checks));}
std::uint64_t word(std::ifstream& f){std::uint64_t v=0;f.read(reinterpret_cast<char*>(&v),8);require(bool(f));return v;}
std::size_t width(RegularSeedsType t){switch(t){case RegularSeedsType::u16:case RegularSeedsType::i16:return 2;case RegularSeedsType::u32:case RegularSeedsType::i32:case RegularSeedsType::f32:return 4;case RegularSeedsType::u64:case RegularSeedsType::i64:case RegularSeedsType::f64:return 8;default:return 1;}}
void store(unsigned char* destination,RegularSeedsType type,std::uint64_t label){
    if(type==RegularSeedsType::boolean){*destination=1;return;}
    if(type==RegularSeedsType::f32){const auto value=float(label);std::memcpy(destination,&value,4);return;}
    if(type==RegularSeedsType::f64){const auto value=double(label);std::memcpy(destination,&value,8);return;}
    std::memcpy(destination,&label,width(type));
}
// Recursive cartesian enumeration is independent of the leaf's flat-index/run
// decomposition. It also marks selected zero labels after narrow integer wrap.
void oracle(const RegularSeedsGrid& grid,RegularSeedsType type,std::size_t axis,
            std::size_t offset,std::uint64_t& label,std::vector<unsigned char>& result,
            std::vector<unsigned char>& selected){
    if(axis==grid.ndim){store(result.data()+offset*width(type),type,++label);selected[offset]=1;return;}
    const auto& a=grid.axes[axis];
    for(std::size_t coordinate=0;coordinate<a.count;++coordinate)
        oracle(grid,type,axis+1,offset+(a.start+coordinate*a.step)*a.output_stride,label,result,selected);
}
void exercise(const std::vector<RegularSeedsAxis>& axes,std::size_t count,
              RegularSeedsType type,int workers,std::ofstream& stream,
              const std::vector<unsigned char>* genuine=nullptr){
    std::size_t n=1;for(const auto& a:axes)n*=a.count;
    const RegularSeedsGrid grid{axes.data(),axes.size(),n,count};
    const auto bytes=count*width(type);
    std::vector<unsigned char> expected(bytes,0),selected(count,0);
    std::uint64_t label=0;oracle(grid,type,0,0,label,expected,selected);require(label==n);
    if(genuine)require(expected==*genuine);
    const auto policy=get_num_threads();
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        require(std::fesetround(rounding)==0);
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INVALID);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        // uint64 storage guarantees destination alignment; guard all bytes.
        std::vector<std::uint64_t> storage((bytes+7)/8+4,0xa55aa55aa55aa55aULL);
        auto* output=reinterpret_cast<unsigned char*>(storage.data()+2);
        regular_seeds_into(grid,output,type,workers);
        require(std::equal(expected.begin(),expected.end(),output));
        require(storage[0]==0xa55aa55aa55aa55aULL&&storage[1]==0xa55aa55aa55aa55aULL);
        const auto* all=reinterpret_cast<const unsigned char*>(storage.data());
        for(std::size_t i=16+bytes;i<storage.size()*8;++i)require(all[i]==((i%2)?0xa5:0x5a));
        require(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
        require(get_num_threads()==policy);
        std::memset(output,0x6d,bytes);
        regular_seeds_assign(grid,output,type,workers);
        for(std::size_t i=0;i<count;++i)for(std::size_t b=0;b<width(type);++b)
            require(output[i*width(type)+b]==(selected[i]?expected[i*width(type)+b]:0x6d));
        require(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
        require(get_num_threads()==policy);
        std::uint64_t hash=1469598103934665603ULL;
        for(auto value:expected){hash^=value;hash*=1099511628211ULL;}
        stream.write(reinterpret_cast<const char*>(&hash),8);
    }
    std::fesetround(FE_TONEAREST);std::feclearexcept(FE_ALL_EXCEPT);
}
}
int main(int argc,char** argv){
    if(argc!=2 && argc!=4)throw std::runtime_error("usage: native output [fixture-min fixture-modern]");
    std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
    for(int workers:{1,4}){
        for(int file=2;file<argc;++file){
            std::ifstream f(argv[file],std::ios::binary);require(bool(f));const auto records=word(f);
            for(std::size_t record=0;record<records;++record){
                const auto type=RegularSeedsType(word(f));
                const auto ndim=word(f),selected=word(f),count=word(f),bytes=word(f);
                std::vector<RegularSeedsAxis> axes(ndim);
                for(auto& a:axes)a={word(f),word(f),word(f),word(f)};
                std::vector<unsigned char> genuine(bytes);f.read(reinterpret_cast<char*>(genuine.data()),bytes);require(bool(f));
                std::size_t product=1;for(const auto& a:axes)product*=a.count;require(product==selected);
                exercise(axes,count,type,workers,stream,&genuine);
            }
        }
        for(int kind=0;kind<=10;++kind){
            const auto type=RegularSeedsType(kind);
            exercise({},1,type,workers,stream);
            exercise({{0,1,0,1}},0,type,workers,stream);
            exercise({{2,3,73,1}},225,type,workers,stream);
            exercise({{1,2,3,99},{2,3,5,9},{0,2,4,1}},693,type,workers,stream);
            exercise({{0,1,300111,1}},300111,type,workers,stream);
            exercise({{1,2,3,200007},{0,3,66669,1}},1200042,type,workers,stream);
        }
    }
    require(bool(stream));std::cout<<"regular-seeds checks "<<checks<<" PASS\n";
}
