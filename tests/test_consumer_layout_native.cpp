#include "skimagexx/component_consumers.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
void require(bool value){if(!value)throw std::runtime_error("consumer width/transport regression");}
void run(int workers) {
    set_num_threads(workers);
    for(auto bytes:{1u,2u,4u,8u})for(auto n:{1u,3u,4u,7u,8u,15u,16u,31u,32u,33u,63u,64u,65u,129u,8193u}) {
        std::vector<std::intptr_t> labels(n);for(unsigned i=0;i<n;++i)labels[i]=std::intptr_t(i%9)-1;
        ConsumerCounts counts;counts.keys={0,1,2,3,4,5,6};counts.sizes={0,1,2,4,9,100,std::intptr_t(n)};counts.maximum=6;counts.empty=false;
        std::vector<unsigned char> seed(n*bytes);for(std::size_t i=0;i<seed.size();++i)seed[i]=static_cast<unsigned char>((i*131+17)&255);
        for(double threshold:{-std::numeric_limits<double>::infinity(),-1.,0.,1.,4.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            auto expected=seed;for(unsigned i=0;i<n;++i){const auto key=std::uint64_t(labels[i]);if(double(key<counts.sizes.size() ? counts.sizes[std::size_t(key)]:0)<=threshold)std::memset(expected.data()+i*bytes,0,bytes);}
            auto actual=std::make_unique<unsigned char[]>(seed.size()+1);std::memcpy(actual.get()+1,seed.data(),seed.size());
            ConsumerRaster out{actual.get()+1,{n},{bytes},bytes};consumer_filter_into(labels.data(),counts,threshold,out);
            require(!std::memcmp(actual.get()+1,expected.data(),expected.size()));
        }
        const std::uint64_t fill=0xfff000017f800001ULL;const auto* raw=reinterpret_cast<const unsigned char*>(&fill);
        const std::vector<std::uint8_t> remove{1,0,1,0,1,0,1};auto expected=seed;const auto number=std::min(n,6u);
        for(unsigned i=0;i<n;++i)if(labels[i]>=0 && std::uint64_t(labels[i])<=number && remove[std::size_t(labels[i])])std::memcpy(expected.data()+i*bytes,raw,bytes);
        auto actual=std::make_unique<unsigned char[]>(seed.size()+1);std::memcpy(actual.get()+1,seed.data(),seed.size());
        ConsumerRaster out{actual.get()+1,{n},{bytes},bytes};bool caught=false;
        try{consumer_clear_into(labels.data(),remove.data(),number,raw,out);}catch(const std::invalid_argument&){caught=true;}
        require(caught && !std::memcmp(actual.get()+1,expected.data(),expected.size()));
        // Duplicate modulo keys retain the last value and missing-key zeros.
        const std::vector<std::uint64_t> keys{1,2,257},index{0,1,2,3,255,256,257,258};std::vector<unsigned char> values(3*bytes),mapped(index.size()*bytes,0xcd),oracle(mapped.size(),0);
        for(std::size_t i=0;i<values.size();++i)values[i]=static_cast<unsigned char>(i*17+0x81);
        for(std::size_t i=0;i<index.size();++i)for(std::size_t j=0;j<keys.size();++j)if((index[i]&255)==(keys[j]&255))std::memcpy(oracle.data()+i*bytes,values.data()+j*bytes,bytes);
        consumer_map_into(index.data(),index.size(),keys.data(),values.data(),keys.size(),bytes,8,mapped.data());require(mapped==oracle);
    }
}
void feedback() {
    set_num_threads(1);
    // Reader/output overlap must retain serial byte feedback, not full-block
    // preloading. This deliberately mutates future remove-table decisions.
    std::vector<std::intptr_t> labels(65);for(std::size_t i=0;i<labels.size();++i)labels[i]=std::intptr_t(i%7);
    std::vector<std::uint8_t> actual(65,1),expected=actual;const std::uint8_t fill=0;
    for(std::size_t i=0;i<labels.size();++i)if(expected[std::size_t(labels[i])])expected[i]=fill;
    consumer_clear_into(labels.data(),actual.data(),6,&fill,{actual.data(),{actual.size()},{1},1});require(actual==expected);
    ConsumerCounts counts;counts.keys={0,1,2,3,4,5,6};counts.sizes={6,5,4,3,2,1,0};counts.maximum=6;counts.empty=false;
    auto sizes=counts.sizes;const std::intptr_t reverse[]{6,5,4,3,2,1,0};
    for(unsigned i=0;i<7;++i)if(double(sizes[std::size_t(reverse[i])])<=3)sizes[i]=0;
    consumer_filter_into(reverse,counts,3,{counts.sizes.data(),{7},{8},8});require(counts.sizes==sizes);
    // Zero-stride output aliases itself; last logical store/error behavior
    // stays on raster's original serial path.
    std::uint64_t output=~std::uint64_t{0};std::vector<std::uint8_t> remove{0,1};const std::uint64_t word=0x8000000000000000ULL;
    std::vector<std::intptr_t> ids(65,1);consumer_clear_into(ids.data(),remove.data(),1,&word,{&output,{65},{0},8});require(output==word);
}
int main(){require(get_num_threads()==1);run(1);
#ifdef _OPENMP
    run(4);
#else
    bool rejected=false;try{set_num_threads(4);}catch(const std::invalid_argument&){rejected=true;}require(rejected && get_num_threads()==1);
#endif
    feedback();set_num_threads(1);std::cout<<"Consumer widths, exact blocks/tails, rawNaN/zero stores, modulo maps andalias/partial-error checks passed\n";}
