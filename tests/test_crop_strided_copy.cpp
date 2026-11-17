#include <skimagexx/crop.hpp>
#include <skimagexx/image.hpp>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
using I=std::ptrdiff_t;
namespace {
std::vector<I> strides(const std::vector<I>& shape,int layout,I bytes) {
    std::vector<I> s(shape.size());I value=bytes;
    if(layout==1){for(std::size_t a=0;a<shape.size();++a){s[a]=value;value*=shape[a];}}
    else {for(std::size_t a=shape.size();a-->0;){s[a]=value;value*=shape[a];}}
    if(layout==2)s.back()*=-1;
    if(layout==3)for(auto& v:s)v*=-1;
    if(layout==4)for(auto& v:s)v=0;
    if(layout==5){I span=bytes;for(std::size_t a=shape.size();a-->0;){s[a]=span+bytes;span=s[a]*(shape[a]-1)+span;}}
    return s;
}
struct Buffer {
    std::vector<unsigned char> bytes;I origin=0;
    Buffer(const std::vector<I>& shape,const std::vector<I>& s,I word) {
        I span=word;for(std::size_t a=0;a<shape.size();++a){I delta=(shape[a]-1)*s[a];if(delta<0)origin-=delta;span+=delta<0?-delta:delta;}
        bytes.resize(std::size_t(span)+64,0xa5);origin+=32;
    }
    unsigned char* data(){return bytes.data()+origin;}
};
}
int main() {
    using namespace skimagexx;std::size_t cases=0;
    for(I word:{1,2,3,4,5,6,8,16,24,32,64})for(int rank:{2,3}) {
        const std::vector<I> shape=rank==2?std::vector<I>{4,5}:std::vector<I>{3,4,5};
        auto output_shape=shape;output_shape[0]-=1;output_shape.back()-=2;
        std::vector<CropSlice> slices;for(std::size_t a=0;a<shape.size();++a)slices.push_back({a==0?1:0,a+1==shape.size()?shape[a]-2:shape[a]});
        for(int in_layout:{0,1,2,3,4,5})for(int out_layout:{0,1,3,5}) {
            auto in_stride=strides(shape,in_layout,word),out_stride=strides(output_shape,out_layout,word);
            Buffer source(shape,in_stride,word),target(output_shape,out_stride,word);
            for(std::size_t b=0;b<source.bytes.size();++b)source.bytes[b]=static_cast<unsigned char>(b*37+11);
            const auto original=source.bytes;crop_copy_into({source.data(),shape,in_stride,std::size_t(word)},slices,target.data(),out_stride);
            std::vector<I> index(shape.size(),0);
            for(;;) {
                I from=in_stride[0],to=0;for(std::size_t a=0;a<shape.size();++a){from+=index[a]*in_stride[a];to+=index[a]*out_stride[a];}
                assert(std::memcmp(source.data()+from,target.data()+to,std::size_t(word))==0);
                std::size_t a=shape.size();while(a){--a;if(++index[a]<output_shape[a])break;index[a]=0;}if(a==0&&index[0]==0)break;
            }
            assert(source.bytes==original);++cases;
        }
    }
    assert(cases==528);assert(get_num_threads()==1);set_num_threads(4);set_num_threads(1);
    std::cout<<"PASS528 strided POD copy descriptors including reverse/F/C/padded/broadcast/odd words, serial default1\n";
}
