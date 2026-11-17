#pragma once
#include <skimagexx/variation_of_information.hpp>
#include <skimagexx/image.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
// Exact admitted original return words, not candidate-derived expectations.
inline int vi_large_saved_controls() {
    const std::uint64_t expected[4][2]={
        {UINT64_C(0x4007cbe4873c419a),UINT64_C(0x40064266ad26a8cd)},
        {UINT64_C(0x401190faa8ca5c4c),UINT64_C(0x40117a1cdc29005c)},
        {UINT64_C(0x4007cbbfce8962e6),UINT64_C(0x4006423b60ce96dd)},
        {UINT64_C(0x401191ad2ded50af),UINT64_C(0x40117acb63349d16)},
    };
    if(skimagexx::get_num_threads()!=1)return 1;
    unsigned records=0;
    for(unsigned c=0;c<4;++c) {
        const std::size_t n=c<2?16384:65536,labels=c%2?64:8;
        std::vector<std::int64_t> truth(n),test(n);const std::int64_t ignore=0;
        for(std::size_t i=0;i<n;++i){truth[i]=i%labels;test[i]=(i/3+i%7)%labels;}
        const auto a=truth,b=test;
        for(int workers:{0,1,4}) {
            skimagexx::VariationInformation result;
            skimagexx::detail::LabelPairInput in{truth.data(),test.data(),n,&ignore,1,true};
            if(!skimagexx::variation_of_information(in,result,workers))return 2;
            if(result.captured_workers!=(workers?workers:1)||truth!=a||test!=b)return 3;
            for(unsigned j=0;j<2;++j){std::uint64_t word;std::memcpy(&word,&result.values[j],8);
                if(word!=expected[c][j]){std::cerr<<"case="<<c<<" word="<<j<<" actual="<<std::hex<<word<<" expected="<<expected[c][j]<<'\n';return 4;}}
            ++records;
        }
    }
    if(skimagexx::get_num_threads()!=1)return 5;
    std::cout<<"large_saved_records="<<records<<" strict_words="<<records*2<<" originals=0\n";
    return 0;
}
