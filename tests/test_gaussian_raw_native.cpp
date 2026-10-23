#include "../src/resize_gaussian.hpp"
#include "skimagexx/image.hpp"
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

using namespace skimagexx;
namespace {
void require(bool ok){if(!ok)throw std::runtime_error("raw Gaussian check failed");}
std::size_t count(const std::vector<std::size_t>& shape){return std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());}
void put(void* p,std::size_t bytes,std::uint64_t word){
    if(bytes==1){auto v=std::uint8_t(word);std::memcpy(p,&v,1);}
    else if(bytes==2){auto v=std::uint16_t(word);std::memcpy(p,&v,2);}
    else if(bytes==4){auto v=std::uint32_t(word);std::memcpy(p,&v,4);}
    else std::memcpy(p,&word,8);
}
std::uint64_t get(const void* p,std::size_t bytes){
    if(bytes==1){std::uint8_t v;std::memcpy(&v,p,1);return v;}
    if(bytes==2){std::uint16_t v;std::memcpy(&v,p,2);return v;}
    if(bytes==4){std::uint32_t v;std::memcpy(&v,p,4);return v;}
    std::uint64_t v;std::memcpy(&v,p,8);return v;
}
void check(PixelType type,const std::vector<std::uint64_t>& words,const std::vector<std::size_t>& shape,
           const std::vector<std::vector<double>>& weights,GaussianBorder border,double cval,
           const std::vector<std::uint64_t>& expected){
    const auto bytes=pixel_size(type),n=count(shape);require(n==words.size() && expected.size()==n);
    std::vector<std::uint64_t> input((n*bytes+7)/8),output(input.size());
    for(std::size_t i=0;i<n;++i)put(reinterpret_cast<char*>(input.data())+i*bytes,bytes,words[i]);
    const auto snapshot=input;
    std::vector<GaussianPass> passes;
    for(std::size_t axis=0;axis<weights.size();++axis)passes.push_back({axis,weights[axis],border,cval});
    detail::gaussian_raw_into(input.data(),output.data(),type,shape,passes);require(input==snapshot);
    for(std::size_t i=0;i<n;++i){const auto actual=get(reinterpret_cast<const char*>(output.data())+i*bytes,bytes);
        if(actual!=expected[i]){std::cerr<<"type="<<int(type)<<" rank="<<shape.size()<<" border="<<int(border)<<" index="<<i<<" actualbits="<<actual<<" expectedbits="<<expected[i]<<"\n";require(false);}}
}
template<class T> void workers(PixelType type){
    const std::vector<std::size_t> shape{257,1025};std::vector<T> input(count(shape)),one(input.size()),four(input.size());
    for(std::size_t i=0;i<input.size();++i){const auto word=std::make_unsigned_t<T>(std::uint64_t{0x9e3779b97f4a7c15ULL}*(i+1));std::memcpy(&input[i],&word,sizeof(T));}
    for(auto border:{GaussianBorder::reflect,GaussianBorder::constant,GaussianBorder::nearest,GaussianBorder::mirror,GaussianBorder::wrap}){
        const std::vector<GaussianPass> passes{{0,{.25,.5,.25},border,.3},{1,{.25,.5,.25},border,.3}};
        set_num_threads(1);detail::gaussian_raw_into(input.data(),one.data(),type,shape,passes);
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        detail::gaussian_raw_into(input.data(),four.data(),type,shape,passes);require(one==four);
    }
    set_num_threads(1);
}
template<class E,class F> void rejects(F body){bool caught=false;try{body();}catch(const E&){caught=true;}require(caught);}
void guards(){
    std::uint64_t input=std::numeric_limits<std::uint64_t>::max(),output=1;
    detail::gaussian_raw_into(nullptr,nullptr,PixelType::uint64,{0,7},{});
    detail::gaussian_raw_into(&input,&output,PixelType::uint64,{1},{});require(output==input);
    detail::gaussian_raw_into(&input,&output,PixelType::uint64,{1},{{0,{1.},GaussianBorder::reflect,0}});require(output==0);
    rejects<std::invalid_argument>([&]{detail::gaussian_raw_into(&input,&input,PixelType::uint64,{1},{});});
    rejects<std::invalid_argument>([&]{detail::gaussian_raw_into(&input,&output,PixelType::boolean,{1},{});});
    rejects<std::invalid_argument>([&]{detail::gaussian_raw_into(&input,&output,PixelType::uint64,{1},{{1,{1.},GaussianBorder::reflect,0}});});
    rejects<std::invalid_argument>([&]{detail::gaussian_raw_into(&input,&output,PixelType::uint64,{1},{{0,{.1,.5,.2},GaussianBorder::reflect,0}});});
}
}
int main(){
    require(get_num_threads()==1);guards();
    #include "data/gaussian_raw.inc"
    #include "data/gaussian_raw_legacy.inc"
    workers<std::uint8_t>(PixelType::uint8);workers<std::int8_t>(PixelType::int8);
    workers<std::uint16_t>(PixelType::uint16);workers<std::int16_t>(PixelType::int16);
    workers<std::uint32_t>(PixelType::uint32);workers<std::int32_t>(PixelType::int32);
    workers<std::uint64_t>(PixelType::uint64);workers<std::int64_t>(PixelType::int64);
    std::cout<<"240 raw Gaussian SciPy fixtures, guards and exact worker checks passed\n";
}
