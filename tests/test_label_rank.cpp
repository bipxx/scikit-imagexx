#include "skimagexx/label_color.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool ok) {if(!ok)throw std::runtime_error("label rank contract failed");}
int main() {
    using namespace skimagexx;
    LabelSummary summary;
    {
        std::array<std::uint8_t,256> input;
        for(std::size_t i=0;i<input.size();++i)input[i]=std::uint8_t(i);
        std::array<std::uint64_t,256> ranks;
        check(label_ranks(input.data(),input.size(),PixelType::uint8,false,256,true,ranks.data(),summary));
        check(summary.regions==257 && !summary.has_background);
        check(ranks.front()==1 && ranks.back()==0);
        for(std::size_t i=0;i<255;++i)check(ranks[i]==i+1);
    }
    {
        const std::uint8_t input[]={0,1,2,3};std::uint64_t ranks[4];
        check(label_ranks(input,4,PixelType::uint8,false,2,true,ranks,summary));
        check(summary.regions==4 && summary.has_background && summary.background==2);
        check(ranks[0]==1 && ranks[1]==2 && ranks[2]==0 && ranks[3]==3);
    }
    {
        const std::uint8_t input[]={5,7,9};std::uint64_t ranks[3];
        check(label_ranks(input,3,PixelType::uint8,false,6,true,ranks,summary));
        check(summary.background==1 && !summary.has_background);
        check(ranks[0]==1 && ranks[1]==2 && ranks[2]==3);
    }
    {
        const std::int16_t input[]={-1,0,2,-1};std::uint64_t ranks[4]={99,99,99,99};
        check(!label_ranks(input,4,PixelType::int16,false,0,true,ranks,summary));
        check(std::all_of(ranks,ranks+4,[](auto x){return x==99;}));
        check(label_ranks(input,4,PixelType::int16,false,0,false,ranks,summary));
        check(ranks[0]==1 && ranks[1]==0 && ranks[2]==2 && ranks[3]==1);
    }
    {
        const float input[]={.1f,.9f,1.9f};std::uint64_t ranks[3];
        check(label_ranks(input,3,PixelType::float32,false,0,true,ranks,summary));
        check(ranks[0]==0 && ranks[1]==0 && ranks[2]==1);
        check(label_ranks(input,3,PixelType::float32,false,0,false,ranks,summary));
        check(ranks[0]==1 && ranks[1]==2 && ranks[2]==3);
    }
    {
        const std::uint16_t values[]={0,7,33,1024};
        for(bool swap:{false,true}) {
            std::array<unsigned char,10> storage;storage.fill(0xa5);
            for(std::size_t i=0;i<4;++i) {
                const auto* p=reinterpret_cast<const unsigned char*>(values+i);
                storage[1+2*i]=p[swap ? 1 : 0];storage[2+2*i]=p[swap ? 0 : 1];
            }
            const auto saved=storage;std::uint64_t ranks[4];
            check(label_ranks(storage.data()+1,4,PixelType::uint16,swap,0,true,ranks,summary));
            for(std::size_t i=0;i<4;++i)check(ranks[i]==i);
            check(storage==saved);
        }
    }
    {
        const std::uint64_t values[]={UINT64_C(0xffffffffffffffff),0,UINT64_C(0x8000000000000000),UINT64_C(0xffffffffffffffff)};
        std::uint64_t ranks[4];
        check(label_ranks(values,4,PixelType::uint64,false,0,true,ranks,summary));
        check(ranks[0]==2 && ranks[1]==0 && ranks[2]==1 && ranks[3]==2);
    }
    {
        std::uint32_t bits=0x7f800001u;float value;std::memcpy(&value,&bits,4);
        std::uint64_t rank=99;
        std::feclearexcept(FE_ALL_EXCEPT);
        check(!label_ranks(&value,1,PixelType::float32,false,0,true,&rank,summary));
        check(!(std::fetestexcept(FE_INVALID|FE_OVERFLOW)));
        check(rank==99);
    }
    check(label_ranks(nullptr,0,PixelType::uint8,false,0,false,nullptr,summary));
    {
        const auto count=std::size_t(513)*517;
        std::vector<std::uint8_t> input(count);std::vector<std::uint64_t> ranks(count);
        for(std::size_t i=0;i<count;++i)input[i]=std::uint8_t(i%7);
        for(int workers:{1,4}) {
            set_num_threads(workers);
            check(label_ranks(input.data(),count,PixelType::uint8,false,0,true,ranks.data(),summary));
            for(std::size_t i=0;i<count;++i)check(ranks[i]==input[i]);
        }
        set_num_threads(1);
    }
    std::cout<<"Label rank C++ contracts passed\n";
}
