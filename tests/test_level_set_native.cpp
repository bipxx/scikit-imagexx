#include "level_set.hpp"
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
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("level-set control "+std::to_string(checks));}
LevelSetShape view(const std::vector<std::size_t>& dimensions){
    std::size_t count=1;for(auto n:dimensions)count*=n;
    return {dimensions.data(),dimensions.size(),count};
}
void write(std::ofstream& stream,const std::int8_t* values,std::size_t count){stream.write(reinterpret_cast<const char*>(values),std::streamsize(count));}
// Independent direct coordinate oracle: no row patterns or axis-cost tables.
std::vector<std::int8_t> checker_oracle(LevelSetShape shape,std::int64_t square){
    std::vector<std::int8_t> output(shape.count);
    for(std::size_t i=0;i<shape.count;++i){
        auto index=i;unsigned bit=0;
        for(std::size_t d=shape.ndim;d-->0;){
            const auto x=index%shape.dimensions[d];index/=shape.dimensions[d];
            if(square==std::numeric_limits<std::int64_t>::min())bit^=unsigned(x!=0);
            else {const auto divisor=square<0?-square:square;auto q=std::int64_t(x)/square;
                if(square<0&&std::int64_t(x)%divisor) --q;bit^=unsigned(std::uint64_t(q)&1);}
        }
        output[i]=std::int8_t(bit);
    }
    return output;
}
std::vector<std::int8_t> disk_oracle(LevelSetShape shape,const std::vector<std::int64_t>& center,double radius){
    std::vector<std::int8_t> output(shape.count);
    for(std::size_t i=0;i<shape.count;++i){
        auto index=i;std::int64_t sum=0;
        for(std::size_t d=shape.ndim;d-->0;){
            const auto x=std::int64_t(index%shape.dimensions[d]);index/=shape.dimensions[d];
            const auto delta=x-center[d];sum+=delta*delta;
        }
        const double distance=std::sqrt(double(sum));const double phi=radius-distance;
        output[i]=std::int8_t(phi>0.0);
    }
    return output;
}
void controls(std::ofstream& stream,int workers){
    std::vector<std::vector<std::size_t>> dimensions={{},{0},{0,7},{7,0},{3,0,5},{2,3,2,5},{1,1,1,1,1,1},{128,64,32},{1,262177},{262177,1}};
    for(std::size_t width:{1u,2u,3u,4u,5u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,257u,1025u})dimensions.push_back({3,width});
    for(const auto& dims:dimensions){
        const auto shape=view(dims);std::vector<std::int8_t> output(shape.count+2,-99);
        for(std::int64_t square:std::initializer_list<std::int64_t>{std::int64_t(1),2,3,5,32,4096,-1,-2,-3,-5,-32,std::numeric_limits<std::int64_t>::max(),std::numeric_limits<std::int64_t>::min()}){
            const auto expected=checker_oracle(shape,square);checkerboard_level_set_into(shape,square,output.data()+1,workers);
            require(output.front()==-99&&output.back()==-99);require(std::equal(expected.begin(),expected.end(),output.begin()+1));write(stream,output.data()+1,shape.count);
        }
        std::vector<std::int64_t> center(shape.ndim);for(std::size_t d=0;d<shape.ndim;++d)center[d]=std::int64_t(shape.dimensions[d]/2);
        for(int displacement:{0,-7,13}){
            for(auto& c:center)c+=displacement;
            for(double radius:{-1.0,0.0,0.5,2.5,5.0,33.0}){
                require(disk_level_set_geometry_admissible(shape,center.data(),radius));
                const auto expected=disk_oracle(shape,center,radius);
                std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto before=std::fetestexcept(FE_ALL_EXCEPT);
                require(disk_level_set_into(shape,center.data(),radius,output.data()+1,workers));
                require(std::fetestexcept(FE_ALL_EXCEPT)==before);require(output.front()==-99&&output.back()==-99);
                require(std::equal(expected.begin(),expected.end(),output.begin()+1));write(stream,output.data()+1,shape.count);
            }
        }
    }
}
void admission(){
    const std::vector<std::size_t> dims{1};const auto shape=view(dims);std::int64_t center=94906265;
    require(disk_level_set_geometry_admissible(shape,&center,1));center=94906266;
    require(!disk_level_set_geometry_admissible(shape,&center,1));center=std::numeric_limits<std::int64_t>::min();
    require(!disk_level_set_geometry_admissible(shape,&center,1));center=0;
    for(double radius:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()})require(!disk_level_set_geometry_admissible(shape,&center,radius));
    const std::vector<std::size_t> pair{1,1};std::int64_t centers[2]={67108864,67108864};
    require(disk_level_set_geometry_admissible(view(pair),centers,1));++centers[0];require(!disk_level_set_geometry_admissible(view(pair),centers,1));
    for(int rounding:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        require(std::fesetround(rounding)==0);std::int8_t output=-99;std::feraiseexcept(FE_DIVBYZERO);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);require(!disk_level_set_into(shape,&center,1,&output,1));
        require(output==-99&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
    require(std::fesetround(FE_TONEAREST)==0);std::feclearexcept(FE_ALL_EXCEPT);
}
void fixtures(const char* path,std::ofstream& stream,int workers){
    std::ifstream file(path,std::ios::binary);require(bool(file));std::uint64_t records=0;file.read(reinterpret_cast<char*>(&records),8);require(records>0);
    for(std::uint64_t r=0;r<records;++r){
        std::uint64_t row[3];file.read(reinterpret_cast<char*>(row),sizeof(row));
        std::vector<std::size_t> dimensions(static_cast<std::size_t>(row[1]));for(auto& d:dimensions){std::uint64_t x;file.read(reinterpret_cast<char*>(&x),8);d=std::size_t(x);}
        const auto shape=view(dimensions);require(shape.count==row[2]);std::int64_t square=1;double radius=0;std::vector<std::int64_t> center(shape.ndim);
        if(row[0]==0)file.read(reinterpret_cast<char*>(&square),8);
        else {file.read(reinterpret_cast<char*>(&radius),8);file.read(reinterpret_cast<char*>(center.data()),std::streamsize(8*shape.ndim));}
        std::vector<std::int8_t> expected(shape.count),output(shape.count+2,-99);file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(shape.count));require(bool(file));
        if(row[0]==0)checkerboard_level_set_into(shape,square,output.data()+1,workers);
        else {require(disk_level_set_geometry_admissible(shape,center.data(),radius));require(disk_level_set_into(shape,center.data(),radius,output.data()+1,workers));}
        require(output.front()==-99&&output.back()==-99);require(std::equal(expected.begin(),expected.end(),output.begin()+1));write(stream,output.data()+1,shape.count);
    }
    require(file.peek()==std::ifstream::traits_type::eof());
}
}
int main(int argc,char** argv){
    if(argc!=2&&argc!=3){std::cerr<<"usage: level-set-native OUTPUT [FROZEN_FIXTURES]\n";return 2;}
    try {
        std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));admission();
        for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers!=1)continue;
#endif
            set_num_threads(workers);require(get_num_threads()==workers);controls(stream,get_num_threads());
            if(argc==3)fixtures(argv[2],stream,get_num_threads());
        }
        set_num_threads(1);std::cout<<checks<<" level-set checks PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
