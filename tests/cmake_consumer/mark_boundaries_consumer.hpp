#pragma once
#include "../fixtures/mark_boundaries_native_saved_v1.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
namespace skimagexx_test {
struct MarkStorage {
    std::vector<std::uint64_t> words;
    skimagexx::ComponentsView view;
    MarkStorage(skimagexx::ComponentsView input,const std::vector<unsigned char>& logical):view(input){
        const auto bytes=view.half?2:skimagexx::pixel_size(view.type);std::ptrdiff_t low=0,high=0;std::size_t count=1;
        for(std::size_t d=0;d<view.shape.size();++d){const auto span=std::ptrdiff_t(view.shape[d]-1)*view.strides[d];if(span<0)low+=span;else high+=span;count*=view.shape[d];}
        words.resize((std::size_t(high-low)+bytes+7)/8);auto* origin=reinterpret_cast<unsigned char*>(words.data())-low;view.data=origin;
        if(logical.size()!=count*bytes)throw std::runtime_error("mark saved storage mismatch");
        for(std::size_t i=0;i<count;++i){auto rest=i;std::ptrdiff_t offset=0;for(auto d=view.shape.size();d-->0;){offset+=std::ptrdiff_t(rest%view.shape[d])*view.strides[d];rest/=view.shape[d];}std::memcpy(origin+offset,logical.data()+i*bytes,bytes);}
    }
};
inline void mark_compare(const skimagexx::MarkImage& out,const MarkSaved& c){
    if(out.shape!=c.result_shape||out.strides!=c.result_strides)throw std::runtime_error("mark saved result layout: "+c.id);
    const auto bytes=skimagexx::pixel_size(out.type);const auto* raw=static_cast<const unsigned char*>(out.data());const auto n=c.expected.size()/bytes;
    for(std::size_t i=0;i<n;++i){auto rest=i;std::ptrdiff_t offset=0;for(auto d=out.shape.size();d-->0;){offset+=std::ptrdiff_t(rest%out.shape[d])*out.strides[d];rest/=out.shape[d];}
        if(std::memcmp(raw+offset*std::ptrdiff_t(bytes),c.expected.data()+i*bytes,bytes))throw std::runtime_error("mark saved output words: "+c.id+" element"+std::to_string(i));}
}
inline std::size_t mark_boundaries_consumer(int workers){
    std::size_t checked=0;
    for(const auto& c:mark_saved_cases()){
        skimagexx::MarkImage result;
        {MarkStorage image(c.image,c.image_bytes),labels(c.labels,c.label_bytes);const auto a=image.words,b=labels.words;
         result=skimagexx::mark_boundaries(image.view,labels.view,c.parameters,workers);
         if(image.words!=a||labels.words!=b)throw std::runtime_error("mark mutated input: "+c.id);
         if(result.data()==image.view.data||result.data()==labels.view.data)throw std::runtime_error("mark result aliases input");}
        mark_compare(result,c);++checked; // input owners destroyed: result owns lifetime.
    }return checked;
}
}
