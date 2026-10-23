#include "../src/resample_detail.hpp"
#include "../src/resize_cast.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace skimagexx;
namespace {
void require(bool value) {if(!value)throw std::runtime_error("typed zoom native check failed");}
std::size_t count(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());
}
void put(void* destination,std::size_t bytes,std::uint64_t word) {
    if(bytes==1){const auto v=std::uint8_t(word);std::memcpy(destination,&v,1);}
    else if(bytes==2){const auto v=std::uint16_t(word);std::memcpy(destination,&v,2);}
    else if(bytes==4){const auto v=std::uint32_t(word);std::memcpy(destination,&v,4);}
    else std::memcpy(destination,&word,8);
}
std::uint64_t get(const void* source,std::size_t bytes) {
    if(bytes==1){std::uint8_t v;std::memcpy(&v,source,1);return v;}
    if(bytes==2){std::uint16_t v;std::memcpy(&v,source,2);return v;}
    if(bytes==4){std::uint32_t v;std::memcpy(&v,source,4);return v;}
    std::uint64_t v;std::memcpy(&v,source,8);return v;
}
void check(PixelType type,const std::vector<std::uint64_t>& words,
           const std::vector<std::size_t>& shape,const std::vector<std::size_t>& target,
           int order,GaussianBorder border,const std::vector<std::uint64_t>& expected,
           bool legacy,bool identity) {
    const auto bytes=pixel_size(type),n=count(target);require(words.size()==count(shape) && expected.size()==n);
    std::vector<std::uint64_t> source((words.size()*bytes+7)/8),output((n*bytes+7)/8),scalar(output.size());
    for(std::size_t i=0;i<words.size();++i)put(reinterpret_cast<char*>(source.data())+i*bytes,bytes,words[i]);
    const auto snapshot=source;
    detail::ZoomOptions options;options.order=order;options.border=border;options.cval=.3;
    options.legacy_reflect=legacy;options.identity_copy=identity;
    detail::zoom_typed_into(source.data(),output.data(),type,shape,target,options);require(source==snapshot);
    for(std::size_t i=0;i<n;++i) {
        const auto actual=get(reinterpret_cast<const char*>(output.data())+i*bytes,bytes);
        bool equal=actual==expected[i];
        if(type==PixelType::float32) {
            const auto a=std::uint32_t(actual),e=std::uint32_t(expected[i]);float av,ev;
            std::memcpy(&av,&a,4);std::memcpy(&ev,&e,4);equal=std::abs(double(av)-ev)<=3e-6*std::max(1.,std::abs(double(ev)));
        } else if(type==PixelType::float64) {
            double av,ev;std::memcpy(&av,&actual,8);std::memcpy(&ev,&expected[i],8);
            equal=std::abs(av-ev)<=3e-13*std::max(1.,std::abs(ev));
        }
        if(!equal) {std::cerr<<"type="<<int(type)<<" order="<<order<<" border="<<int(border)<<" legacy="<<legacy<<" index="<<i<<" actualbits="<<actual<<" expectedbits="<<expected[i]<<"\n";require(false);}
    }
    options.allow_simd=false;detail::zoom_typed_into(source.data(),scalar.data(),type,shape,target,options);
    require(std::memcmp(scalar.data(),output.data(),n*bytes)==0);
}
void casts() {
    const auto infinity=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN();
    require(detail::resize_raw_cast<std::int8_t>(-300.9)==-44);
    require(detail::resize_raw_cast<std::uint8_t>(-300.9)==212);
    require(detail::resize_raw_cast<std::int32_t>(infinity)==std::numeric_limits<std::int32_t>::min());
    require(detail::resize_raw_cast<std::uint32_t>(nan)==0);
    require(detail::resize_raw_cast<std::int64_t>(0x1p63)==std::numeric_limits<std::int64_t>::min());
    require(detail::resize_raw_cast<std::uint64_t>(0x1p64)==0);
    require(detail::resize_raw_cast<std::uint64_t>(nan)==(std::uint64_t{1}<<63));
}
void workers() {
    const std::vector<std::size_t> shape{257,1025},target{513,513};
    std::vector<float> source(count(shape)),one(count(target)),four(one.size()),scalar(one.size());
    for(std::size_t i=0;i<source.size();++i)source[i]=float(int((i*17)%29)-14)/8.f;
    detail::ZoomOptions options;options.cval=.3;
    for(int order:{0,1,3})for(auto border:{GaussianBorder::constant,GaussianBorder::nearest,GaussianBorder::mirror,GaussianBorder::reflect,GaussianBorder::wrap}) {
        options.order=order;options.border=border;options.allow_simd=true;
        set_num_threads(1);detail::zoom_typed_into(source.data(),one.data(),PixelType::float32,shape,target,options);
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        detail::zoom_typed_into(source.data(),four.data(),PixelType::float32,shape,target,options);require(one==four);
        options.allow_simd=false;detail::zoom_typed_into(source.data(),scalar.data(),PixelType::float32,shape,target,options);
        require(std::memcmp(one.data(),scalar.data(),one.size()*sizeof(float))==0);
    }
    set_num_threads(1);
}
}
int main() {
    require(get_num_threads()==1);casts();
    #include "data/zoom_typed.inc"
    #include "data/zoom_typed_legacy.inc"
    workers();
    std::cout<<"232 typed SciPy zoom fixtures, defined casts and exact worker/SIMD checks passed\n";
}
