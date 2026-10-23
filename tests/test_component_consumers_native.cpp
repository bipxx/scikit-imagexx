#include "skimagexx/component_consumers.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
using namespace skimagexx;
namespace {
void require(bool value){if(!value)throw std::runtime_error("component consumer native assertion failed");}
void count_filter(int workers) {
    set_num_threads(workers);constexpr std::size_t n=513*513;
    std::vector<std::intptr_t> labels(n);std::array<std::intptr_t,8> expected{};
    for(std::size_t i=0;i<n;++i){labels[i]=std::intptr_t(i%8);++expected[i%8];}
    auto counts=consumer_count(labels.data(),n);require(!counts.empty && counts.maximum==7);
    require(std::equal(expected.begin(),expected.end(),counts.sizes.begin()));
    std::vector<std::int16_t> pixels(n+2,-99);for(std::size_t i=0;i<n;++i)pixels[n-i]=std::int16_t(labels[i]);
    ConsumerRaster output{pixels.data()+n,{n},{-2},2};
    const double threshold=double(n/8);consumer_filter_into(labels.data(),counts,threshold,output);
    for(std::size_t i=0;i<n;++i)require(pixels[n-i]==(double(expected[i%8])<=threshold ? 0:std::int16_t(labels[i])));
    require(pixels.front()==-99 && pixels.back()==-99);
    std::vector<std::uint8_t> border(n);consumer_border_into(border.data(),{513,513},{1,1},{512,512});
    for(std::size_t y=0;y<513;++y)for(std::size_t x=0;x<513;++x)require(bool(border[y*513+x])==(x==0 || y==0 || x==512 || y==512));
    auto remove=consumer_border_ids(labels.data(),border.data(),n,7);require(std::all_of(remove.begin(),remove.end(),[](auto x){return x==1;}));
    const std::int16_t replacement=11;consumer_clear_into(labels.data(),remove.data(),7,&replacement,output);
    for(std::size_t i=1;i<=n;++i)require(pixels[i]==11);
}
#ifdef _OPENMP
void concurrent_configuration() {
    constexpr std::size_t n=513*513;
    std::vector<std::intptr_t> labels(n,1),pixels(n);
    ComponentsView input;input.data=labels.data();input.type=PixelType::int64;input.shape={n};input.strides={8};
    ConsumerRaster output{pixels.data(),{n},{8},8};const std::uint8_t remove[]{0,1};const std::intptr_t fill=7;
    std::atomic<bool> stop{false};std::thread setter([&]{while(!stop.load(std::memory_order_relaxed)){set_num_threads(1);set_num_threads(4);}});
    try {
        for(int repeat=0;repeat<20;++repeat) {
            require(!consumer_min_negative(input));
            consumer_clear_into(labels.data(),remove,1,&fill,output);
            require(std::all_of(pixels.begin(),pixels.end(),[](auto value){return value==7;}));
        }
    } catch(...) {stop.store(true);setter.join();throw;}
    stop.store(true);setter.join();set_num_threads(1);
}
#endif
}
int main() {
    require(consumer_count(nullptr,0).empty);
    const std::intptr_t sparse[]{0,1000000000000LL,3,1000000000000LL};
    auto counts=consumer_count(sparse,4);require(counts.keys==std::vector<std::uint64_t>({0,3,1000000000000ULL}));
    require(counts.sizes==std::vector<std::intptr_t>({1,1,2}));
    const std::intptr_t negative[]{-1,0};bool rejected=false;try{consumer_count(negative,2);}catch(const std::invalid_argument&){rejected=true;}require(rejected);
    std::array<std::intptr_t,33> truth{};std::array<std::uint8_t,33> bytes{},inverted{};
    for(std::size_t i=0;i<truth.size();++i)truth[i]=std::intptr_t(i%3)-1;
    consumer_truth_into(truth.data(),bytes.data(),truth.size(),false);
    ConsumerRaster boolean{inverted.data(),{33},{1},1};consumer_bool_write(bytes.data(),boolean,true);
    for(std::size_t i=0;i<truth.size();++i){require(bytes[i]==(truth[i]!=0));require(inverted[i]==(truth[i]==0));}
    std::vector<std::uint16_t> half(65536);std::vector<std::uint8_t> nonzero(65536);
    for(std::size_t i=0;i<half.size();++i)half[i]=std::uint16_t(i);
    ComponentsView view;view.data=half.data();view.type=PixelType::uint16;view.half=true;view.shape={half.size()};view.strides={2};
    consumer_nonzero_into(view,nonzero.data());for(std::size_t i=0;i<half.size();++i)require(bool(nonzero[i])==bool(i&0x7fff));
    require(!consumer_min_negative(view));
    const std::uint64_t raw[]{std::numeric_limits<std::uint64_t>::max(),3,0,1000000000000ULL,3};
    auto unique=consumer_sorted_unique(raw,5);require(unique==std::vector<std::uint64_t>({0,3,1000000000000ULL,std::numeric_limits<std::uint64_t>::max()}));
    const std::intptr_t invalid_labels[]{-1,0};const std::uint8_t invalid_border[]{1,0};rejected=false;try{consumer_border_ids(invalid_labels,invalid_border,2,1);}catch(const std::invalid_argument&){rejected=true;}require(rejected);
    const std::uint64_t keys[]{0,3,1000000000000ULL},index[]{3,1000000000000ULL,99,0};const std::int16_t values[]{0,7,9};std::array<std::int16_t,4> mapped{};
    consumer_map_into(index,4,keys,values,3,2,64,mapped.data());require(mapped==std::array<std::int16_t,4>{7,9,0,0});
    std::array<std::int16_t,4> dense{};const std::uint64_t dense_keys[]{1,3};const std::int16_t dense_values[]{5,8};
    consumer_dense_into(dense_keys,dense_values,2,2,dense.data(),4);require(dense==std::array<std::int16_t,4>{0,5,0,8});
    const std::uint64_t scatter[]{0,2,2};const std::int16_t updates[]{1,2,3};consumer_scatter_into(scatter,updates,3,2,dense.data(),4);require(dense==std::array<std::int16_t,4>{1,5,3,8});
    for(int workers:{1,
#ifdef _OPENMP
        4
#else
        1
#endif
    }) {
        count_filter(workers);
        std::vector<std::uint64_t> large(300001);for(std::size_t i=0;i<large.size();++i)large[i]=1000000000000ULL+(i*17)%257;
        const auto result=consumer_sorted_unique(large.data(),large.size());require(result.size()==257);
        for(std::size_t i=0;i<result.size();++i)require(result[i]==1000000000000ULL+i);
    }
#ifdef _OPENMP
    concurrent_configuration();
#endif
    set_num_threads(1);std::cout<<"component consumer native contracts passed\n";
}
