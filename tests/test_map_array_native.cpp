#include "map_array.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
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
void require(bool value){++checks;if(!value)throw std::runtime_error("map-array control "+std::to_string(checks));}
void save(std::ofstream& stream,const std::vector<unsigned char>& bytes){
    std::uint64_t hash=1469598103934665603ULL;for(auto b:bytes){hash^=b;hash*=1099511628211ULL;}
    stream.write(reinterpret_cast<const char*>(&hash),8);
}
void exercise(const std::vector<std::uint64_t>& indices,const std::vector<std::uint64_t>& keys,
    const std::vector<unsigned char>& values,std::size_t width,unsigned bits,
    std::ptrdiff_t values_stride,std::ptrdiff_t out_stride,bool alias,int workers,
    std::ofstream& stream,const std::vector<unsigned char>* genuine=nullptr){
    const auto count=indices.size(),size=keys.size();
    const auto vm=std::size_t(values_stride<0?-values_stride:values_stride),om=std::size_t(out_stride<0?-out_stride:out_stride);
    std::vector<unsigned char> output((count?count-1:0)*om+width+32,0xa5);
    auto* destination=output.data()+16+(out_stride<0&&count?(count-1)*om:0);
    std::vector<unsigned char> value_storage((size?size-1:0)*vm+width+32,0x5a);
    auto* source=value_storage.data()+16+(values_stride<0&&size?(size-1)*vm:0);
    if(alias){require(count==size&&values_stride==out_stride);source=destination;}
    for(std::size_t k=0;k<size;++k)std::memcpy(source+std::ptrdiff_t(k)*values_stride,values.data()+k*width,width);
    auto expected=output;auto* expected_target=expected.data()+(destination-output.data());
    const auto mask=bits==64?~std::uint64_t(0):(std::uint64_t(1)<<bits)-1;
    std::vector<unsigned char> logical(count*width,0);
    // Independent reverse linear lookup: duplicate keys use the last value.
    for(std::size_t i=0;i<count;++i){
        for(std::size_t k=size;k-->0;)if((keys[k]&mask)==(indices[i]&mask)){
            std::memcpy(logical.data()+i*width,values.data()+k*width,width);break;
        }
        std::memcpy(expected_target+std::ptrdiff_t(i)*out_stride,logical.data()+i*width,width);
    }
    if(genuine)require(logical==*genuine);
    MapArrayInput input{indices.data(),count,keys.data(),source,size,width,bits,values_stride};
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_INVALID);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    map_array_into(input,{destination,out_stride},workers);
    require(std::fetestexcept(FE_ALL_EXCEPT)==flags);require(output==expected);save(stream,output);
}
std::vector<unsigned char> words(std::size_t count,std::size_t width){
    std::vector<unsigned char> result(count*width);
    for(std::size_t k=0;k<count;++k)for(std::size_t j=0;j<width;++j)result[k*width+j]=static_cast<unsigned char>(255-k*17-j*29);
    return result;
}
void controls(std::ofstream& stream,int workers){
    const auto top=std::numeric_limits<std::uint64_t>::max();
    for(std::size_t width:{1u,2u,4u,8u,16u})for(unsigned bits:{8u,16u,32u,64u}){
        for(int kind:{0,1,2}){
            const std::vector<std::uint64_t> keys=kind==0?std::vector<std::uint64_t>{3,1,7,0,3,8,257}:
                kind==1?std::vector<std::uint64_t>{top-7,top-4,top,top-1,top-4,top-5,top-6}:
                        std::vector<std::uint64_t>{0,top,top/2,1,top,999999,1000001};
            const auto values=words(keys.size(),width);
            for(std::size_t count:{0u,1u,2u,3u,4u,7u,8u,9u,15u,16u,17u,31u,32u,33u,257u,513u}){
                std::vector<std::uint64_t> indices(count);
                for(std::size_t i=0;i<count;++i)indices[i]=i%4==0?keys[i%keys.size()]:i%4==1?top:i%4==2?std::uint64_t(i%23):0;
                for(int layout:{0,1,2})exercise(indices,keys,values,width,bits,
                    layout==2?-std::ptrdiff_t(width*2):std::ptrdiff_t(width*(layout+1)),
                    layout==2?-std::ptrdiff_t(width*2):std::ptrdiff_t(width*(layout+1)),false,workers,stream);
            }
            std::vector<std::uint64_t> reversed(keys.rbegin(),keys.rend());
            exercise(reversed,keys,values,width,bits,std::ptrdiff_t(width),std::ptrdiff_t(width),true,workers,stream);
            exercise(reversed,keys,values,width,bits,-std::ptrdiff_t(width),-std::ptrdiff_t(width),true,workers,stream);
        }
        // Empty mappings must zero every destination without a value read.
        exercise({0,1,top},{},{},width,bits,std::ptrdiff_t(width),std::ptrdiff_t(width),false,workers,stream);
    }
    for(std::size_t width:{1u,2u,4u,8u,16u})for(unsigned bits:{8u,16u,32u,64u})
        for(std::size_t size:{8u,9u}) {
            std::vector<std::uint64_t> keys(size);
            for(std::size_t k=0;k<size;++k)keys[k]=k%3==0?top-k:k*100003;
            keys.back()=keys.front();const auto values=words(size,width);
            for(std::size_t count:{3u,4u,5u,17u,263177u}) {
                std::vector<std::uint64_t> indices(count);
                for(std::size_t i=0;i<count;++i)indices[i]=i%3?keys[i%size]:top-19;
                for(int stride:{-2,1,3})exercise(indices,keys,values,width,bits,
                    -std::ptrdiff_t(width*2),std::ptrdiff_t(width)*stride,false,workers,stream);
            }
            exercise(keys,keys,values,width,bits,-std::ptrdiff_t(width*2),-std::ptrdiff_t(width*2),true,workers,stream);
        }
    for(std::size_t width:{1u,2u,4u,8u,16u})for(bool sparse:{false,true}){
        std::vector<std::uint64_t> keys=sparse?std::vector<std::uint64_t>{0,top,top/2,1}:std::vector<std::uint64_t>{0,1,3,7};
        std::vector<std::uint64_t> indices(263177);for(std::size_t i=0;i<indices.size();++i)indices[i]=i%5==0?top:std::uint64_t(i%9);
        const auto values=words(keys.size(),width);
        exercise(indices,keys,values,width,64,std::ptrdiff_t(width),std::ptrdiff_t(width),false,workers,stream);
        exercise(indices,keys,values,width,64,-std::ptrdiff_t(width*2),-std::ptrdiff_t(width*2),false,workers,stream);
    }
}
void rounding(std::ofstream& stream,int workers){
    const auto previous=std::fegetround();
    for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        require(std::fesetround(mode)==0);
        for(std::size_t width:{1u,2u,4u,8u,16u}){
            exercise({0,1,7,9,3,1,0,7},{0,1,3,7},words(4,width),width,64,
                     std::ptrdiff_t(width),std::ptrdiff_t(width),false,workers,stream);
            require(std::fegetround()==mode);
        }
    }
    require(std::fesetround(previous)==0);
}
void fixtures(const char* path,std::ofstream& stream,int workers){
    std::ifstream file(path,std::ios::binary);require(bool(file));std::uint64_t records;file.read(reinterpret_cast<char*>(&records),8);require(records==74);
    for(std::uint64_t r=0;r<records;++r){
        std::uint64_t row[7];file.read(reinterpret_cast<char*>(row),sizeof(row));
        const auto count=std::size_t(row[0]),size=std::size_t(row[1]),width=std::size_t(row[2]);
        std::vector<std::uint64_t> indices(count),keys(size);std::vector<unsigned char> values(size*width),expected(count*width);
        file.read(reinterpret_cast<char*>(indices.data()),std::streamsize(count*8));file.read(reinterpret_cast<char*>(keys.data()),std::streamsize(size*8));
        file.read(reinterpret_cast<char*>(values.data()),std::streamsize(values.size()));file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(expected.size()));require(bool(file));
        exercise(indices,keys,values,width,unsigned(row[3]),std::ptrdiff_t(row[4]),std::ptrdiff_t(row[5]),bool(row[6]),workers,stream,&expected);
    }
    require(file.peek()==std::ifstream::traits_type::eof());
}
}
int main(int argc,char** argv){
    if(argc!=2&&argc!=3){std::cerr<<"usage: map-array-native OUTPUT [FROZEN_FIXTURES]\n";return 2;}
    try{
        std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
        for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers!=1)continue;
#endif
            require(get_num_threads()==1);controls(stream,workers);rounding(stream,workers);if(argc==3)fixtures(argv[2],stream,workers);require(get_num_threads()==1);
        }
        std::cout<<checks<<" map-array controls PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
