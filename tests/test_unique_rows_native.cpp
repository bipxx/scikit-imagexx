#include "unique_rows.hpp"
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
void require(bool value){++checks;if(!value)throw std::runtime_error("unique-row control "+std::to_string(checks));}
void save(std::ofstream& output,const std::vector<std::int64_t>& indices) {
    std::uint64_t hash=1469598103934665603ULL;
    for(auto index:indices)for(unsigned byte=0;byte<8;++byte) {hash^=(std::uint64_t(index)>>(byte*8))&255;hash*=1099511628211ULL;}
    const auto count=std::uint64_t(indices.size());output.write(reinterpret_cast<const char*>(&count),8);output.write(reinterpret_cast<const char*>(&hash),8);
}
int compare(const unsigned char* a,const unsigned char* b,std::size_t width) {
    for(std::size_t i=0;i<width;++i)if(a[i]!=b[i])return a[i]<b[i]?-1:1;
    return 0;
}
std::vector<std::int64_t> oracle(const unsigned char* rows,std::size_t count,std::size_t width) {
    // Incremental insertion of unique rows, independent of index sort/merge.
    std::vector<std::int64_t> result;
    for(std::size_t row=0;row<count;++row) {
        auto position=result.begin();
        while(position!=result.end() && compare(rows+std::size_t(*position)*width,rows+row*width,width)<0)++position;
        if(position==result.end() || compare(rows+std::size_t(*position)*width,rows+row*width,width)!=0)
            result.insert(position,std::int64_t(row));
    }
    return result;
}
void exercise(const unsigned char* rows,std::size_t count,std::size_t width,int workers,
    std::ofstream& stream,const std::vector<std::int64_t>* frozen=nullptr,const std::vector<unsigned char>* words=nullptr) {
    const std::vector<unsigned char> before(rows,rows+count*width);
    auto expected=oracle(rows,count,width);if(frozen)require(expected==*frozen);
    std::vector<std::int64_t> output(count+2,-17);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID|FE_DIVBYZERO);
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();
    const auto unique=unique_rows_indices({rows,count,width},output.data()+1,workers);
    require(std::fetestexcept(FE_ALL_EXCEPT)==flags);require(std::fegetround()==rounding);
    require(unique==expected.size());require(output.front()==-17 && output.back()==-17);
    require(std::equal(expected.begin(),expected.end(),output.begin()+1));
    require(std::equal(before.begin(),before.end(),rows));
    if(words) {
        std::vector<unsigned char> gathered;
        for(auto index:expected)gathered.insert(gathered.end(),rows+std::size_t(index)*width,rows+(std::size_t(index)+1)*width);
        require(gathered==*words);
    }
    save(stream,expected);
}
void controls(std::ofstream& stream,int workers) {
    for(std::size_t width:{1u,2u,3u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,97u})
        for(std::size_t count:{0u,1u,2u,3u,4u,7u,17u,65u})for(int kind:{0,1,2,3}) {
            std::vector<unsigned char> storage(count*width+3,0xab);auto* rows=storage.data()+1;
            for(std::size_t row=0;row<count;++row) {
                std::fill(rows+row*width,rows+(row+1)*width,0xff);
                if(kind!=0) {
                    const auto key=kind==1?row%7:kind==2?(count-row)%11:row;
                    rows[row*width+(kind==2?width-1:0)]=static_cast<unsigned char>(key);
                }
            }
            exercise(rows,count,width,workers,stream);
        }
    // Equal prefixes with a differing byte at/after the cache boundary.
    // Every position participates, including the first and final byte.
    for(std::size_t width:{7u,8u,9u,15u,16u,17u,31u,32u,33u}) {
        for(std::size_t position=0;position<width;++position) {
            std::vector<unsigned char> rows(13*width+1,0x81);auto* data=rows.data()+1;
            for(std::size_t row=0;row<13;++row)data[row*width+position]=static_cast<unsigned char>((13-row)%5);
            exercise(data,13,width,workers,stream);
        }
    }
    // Prefix ties straddle worker chunks; stable first indices include the
    // late single differing row after an otherwise all-equal linear prefix.
    for(std::size_t width:{8u,9u,16u,17u,32u,33u}) {
        const auto count=std::size_t(65539);
        std::vector<unsigned char> rows(count*width+1,0xff);auto* data=rows.data()+1;
        data[(count-1)*width+width-1]=0x7f;
        exercise(data,count,width,workers,stream);
        for(std::size_t row=0;row<count;++row)data[row*width+width-1]=static_cast<unsigned char>((count-row)%17);
        exercise(data,count,width,workers,stream);
    }
    for(std::size_t width:{1u,33u,65u}) {
        const auto count=std::size_t(263177);
        std::vector<unsigned char> rows(count*width,0xff);
        for(std::size_t row=0;row<count;++row)rows[row*width+width-1]=static_cast<unsigned char>((count-row)%17);
        exercise(rows.data(),count,width,workers,stream);
    }
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())for(std::size_t width:{31u,32u,33u,63u,64u,65u,97u}) {
        std::vector<unsigned char> a(width+3,0xff),b(width+3,0xff);
        require(simd::unique_rows_compare(a.data()+1,b.data()+2,width)==0);
        for(std::size_t position=0;position<width;++position) {
            b[position+2]=0x7f;require(simd::unique_rows_compare(a.data()+1,b.data()+2,width)>0);
            require(simd::unique_rows_compare(b.data()+2,a.data()+1,width)<0);b[position+2]=0xff;
        }
    }
#endif
}
void fixtures(const char* path,std::ofstream& stream,int workers) {
    std::ifstream file(path,std::ios::binary);require(bool(file));std::uint64_t records;file.read(reinterpret_cast<char*>(&records),8);require(records==43);
    for(std::uint64_t record=0;record<records;++record) {
        std::uint64_t header[3];file.read(reinterpret_cast<char*>(header),24);
        const auto count=std::size_t(header[0]),width=std::size_t(header[1]),unique=std::size_t(header[2]);
        std::vector<unsigned char> rows(count*width+1),words(unique*width);std::vector<std::int64_t> expected(unique);
        file.read(reinterpret_cast<char*>(rows.data()),std::streamsize(count*width));file.read(reinterpret_cast<char*>(expected.data()),std::streamsize(unique*8));file.read(reinterpret_cast<char*>(words.data()),std::streamsize(words.size()));require(bool(file));
        exercise(rows.data(),count,width,workers,stream,&expected,&words);
    }
    require(file.peek()==std::ifstream::traits_type::eof());
}
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=4){std::cerr<<"usage: unique-rows-native OUTPUT [MINIMUM MODERN]\n";return 2;}
    try {
        std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
        const auto previous=std::fegetround();
        for(int workers:{1,4}) {
#ifndef _OPENMP
            if(workers!=1)continue;
#endif
            require(get_num_threads()==1);controls(stream,workers);
            for(int mode:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
                require(std::fesetround(mode)==0);unsigned char rows[]={0xff,0,0x7f,0xff,0,0xff};
                exercise(rows,3,2,workers,stream);
                for(std::size_t width:{8u,9u,16u,17u,32u,33u}) {
                    std::vector<unsigned char> raw(7*width+1,0xff);auto* data=raw.data()+1;
                    data[width-1]=0x7f;data[3*width+width-1]=0;
                    exercise(data,7,width,workers,stream);
                    require(std::fegetround()==mode);
                }
                require(std::fegetround()==mode);
            }
            if(argc==4){fixtures(argv[2],stream,workers);fixtures(argv[3],stream,workers);}
            require(get_num_threads()==1);
        }
        require(std::fesetround(previous)==0);std::cout<<checks<<" unique-row controls PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
