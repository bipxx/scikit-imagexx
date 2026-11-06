#include <skimagexx/mark_boundaries.hpp>
#include <skimagexx/image.hpp>
#include <cstring>
#include <iostream>
#include <stdexcept>
int main(){try{
    using namespace skimagexx;constexpr std::size_t h=512,w=513;
    for(bool column_major:{false,true}){
        std::vector<std::uint16_t> pixels(h*w),labels(h*w);
        for(std::size_t r=0;r<h;++r)for(std::size_t c=0;c<w;++c){const auto i=column_major?r+c*h:r*w+c;pixels[i]=std::uint16_t((r*17+c)%65535);labels[i]=std::uint16_t((r/16)*100+c/16);}
        ComponentsView image{pixels.data(),PixelType::uint16,false,false,{h,w},column_major?std::vector<std::ptrdiff_t>{2,2*h}:std::vector<std::ptrdiff_t>{2*w,2}};
        ComponentsView label{labels.data(),PixelType::uint16,false,false,{h,w},image.strides};MarkBoundariesParameters p;p.outline=true;
        set_num_threads(1);auto a=mark_boundaries(image,label,p);set_num_threads(4);auto b=mark_boundaries(image,label,p);
        if(a.shape!=b.shape||a.strides!=b.strides||a.f64!=b.f64||get_num_threads()!=4)throw std::runtime_error("mark actual parallel1/4 mismatch");
        for(std::size_t r=0;r<h;++r)for(std::size_t c=0;c<w;++c){const auto i=column_major?r+c*h:r*w+c;if(pixels[i]!=std::uint16_t((r*17+c)%65535)||labels[i]!=std::uint16_t((r/16)*100+c/16))throw std::runtime_error("mark parallel mutated inputs");}
    }
    set_num_threads(1);std::cout<<"PASS large C/F composition workers1/4 threshold_crossed=262656 default1=true\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
