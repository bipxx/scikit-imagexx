#include "skimagexx/components.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void require(bool value){if(!value)throw std::runtime_error("component native contract failed");}
std::vector<std::intptr_t> oracle(const std::vector<std::intptr_t>& input,const std::vector<std::size_t>& shape,int connectivity,std::intptr_t background) {
    std::vector<std::size_t> stride(shape.size());std::size_t count=1;
    for(auto axis=shape.size();axis-->0;){stride[axis]=count;count*=shape[axis];}
    std::vector<std::intptr_t> output(count);std::intptr_t number=0;
    std::size_t choices=1;for(auto dim:shape){(void)dim;choices*=3;}
    for(std::size_t start=0;start<count;++start)if(input[start]!=background && !output[start]) {
        output[start]=++number;std::queue<std::size_t> pending;pending.push(start);
        while(!pending.empty()) {
            const auto current=pending.front();pending.pop();
            for(std::size_t code=0;code<choices;++code) {
                auto value=code;std::ptrdiff_t neighbor=std::ptrdiff_t(current);int distance=0;bool valid=true;
                for(std::size_t axis=0;axis<shape.size();++axis) {
                    const int delta=int(value%3)-1;value/=3;distance+=std::abs(delta);
                    const auto coordinate=current/stride[axis]%shape[axis];
                    if((delta<0 && !coordinate)||(delta>0 && coordinate+1==shape[axis]))valid=false;
                    neighbor+=delta*std::ptrdiff_t(stride[axis]);
                }
                if(valid && distance && distance<=connectivity && !output[std::size_t(neighbor)] && input[std::size_t(neighbor)]==input[start]) {
                    output[std::size_t(neighbor)]=number;pending.push(std::size_t(neighbor));
                }
            }
        }
    }return output;
}
void check(const std::vector<std::size_t>& shape,int connectivity,int workers,bool boolean) {
    std::size_t count=1;for(auto n:shape)count*=n;
    std::mt19937 random(77);std::vector<std::intptr_t> input(count),output(count);
    for(auto& value:input)value=random()%(boolean ? 2:4);
    const auto expected=oracle(input,shape,connectivity,0);skimagexx::set_num_threads(workers);
    if(boolean) {
        std::vector<std::uint8_t> bits(input.begin(),input.end());
        std::vector<std::int32_t> narrow(count);
        const auto number=skimagexx::components_label_bool_into(bits.data(),narrow.data(),false,shape,0,connectivity);
        require(std::equal(expected.begin(),expected.end(),narrow.begin()));
        require(number==std::size_t(*std::max_element(expected.begin(),expected.end())));
    } else {
        const auto number=skimagexx::components_label_index_into(input.data(),output.data(),shape,0,connectivity);
        require(output==expected);require(number==std::size_t(*std::max_element(expected.begin(),expected.end())));
    }
}
}
int main() {
    for(const auto& shape:std::vector<std::vector<std::size_t>>{{67},{7,13},{3,5,7},{1,3,5},{3,5,1}})
        for(int c=1;c<=int(shape.size());++c)for(bool boolean:{false,true})check(shape,c,1,boolean);
    for(int c=1;c<=4;++c)check({2,3,2,3},c,1,true);
    std::vector<std::uint8_t> raw{0,1,2,128,255,0},keys(6);
    skimagexx::ComponentsView view;view.data=raw.data();view.type=skimagexx::PixelType::boolean;view.shape={6};view.strides={1};
    skimagexx::components_prepare_bool_into(view,keys.data());require(keys==std::vector<std::uint8_t>({0,1,1,1,1,0}));
    std::vector<double> floats{0.,-0.9,1.9,-1.9,0x1p63,-0x1p63};std::vector<std::intptr_t> indexes(6);
    view.data=floats.data();view.type=skimagexx::PixelType::float64;view.strides={8};
    require(skimagexx::components_prepare_index_into(view,indexes.data()).invalid_cast);
    require(indexes[0]==0 && indexes[1]==0 && indexes[2]==1 && indexes[3]==-1 && indexes[4]==indexes[5]);
    // All half encodings are decoded without touching signaling NaNs in FP.
    std::vector<std::uint16_t> half(65536);for(std::size_t i=0;i<half.size();++i)half[i]=std::uint16_t(i);
    indexes.resize(half.size());view.data=half.data();view.type=skimagexx::PixelType::uint16;view.half=true;view.shape={half.size()};view.strides={2};
    require(skimagexx::components_prepare_index_into(view,indexes.data()).invalid_cast);
    for(std::size_t i=0;i<half.size();++i)if((i&0x7c00)==0x7c00)require(indexes[i]==std::numeric_limits<std::intptr_t>::min());
    // Above the shared parallel threshold, compare independently at worker1/4.
    for(int worker:{1,
#ifdef _OPENMP
        4
#else
        1
#endif
    }) {
        check({513,513},1,worker,false);check({37,73,101},3,worker,true);
        check({300001,1,1},3,worker,false);check({1,1,300001},1,worker,true);
    }
    skimagexx::set_num_threads(1);std::cout<<"components native contracts passed\n";
}
