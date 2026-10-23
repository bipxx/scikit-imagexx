#include "skimagexx/footprint_dense.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value) {if(!value)throw std::runtime_error("footprint native assertion failed");}
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}require(caught);}
}
int main() {
    using namespace skimagexx;
    const std::uint8_t zero=0,one=1;
    for(const int workers:{1,4}) {
#ifndef _OPENMP
        if(workers>1)continue;
#endif
        set_num_threads(workers);
        for(const auto kind:{DenseFootprint::diamond,DenseFootprint::disk,DenseFootprint::ball}) {
            for(const auto width:{std::size_t{0},std::size_t{1},std::size_t{3},std::size_t{8},std::size_t{67}}) {
                const bool ball=kind==DenseFootprint::ball;
                const auto count=width*width*(ball ? width:1);
                std::vector<double> axis(width);
                for(std::size_t i=0;i<width;++i) {
                    const auto d=double(i)-double(width)/2;
                    axis[i]=kind==DenseFootprint::diamond ? std::abs(d):d*d;
                }
                std::vector<std::uint8_t> output(count+2,0xac);
                footprint_dense_into(output.data()+1,count,1,&zero,&one,axis.data(),width,kind,7.3);
                require(output.front()==0xac && output.back()==0xac);
                for(std::size_t i=0;i<count;++i) {
                    auto cost=axis[i%width]+axis[(i/width)%width];
                    if(ball)cost+=axis[i/(width*width)];
                    require(output[i+1]==std::uint8_t(cost<=7.3));
                }
            }
        }
        for(const auto bytes:{1u,2u,3u,8u,16u,24u}) {
            std::vector<unsigned char> atom(bytes),off(bytes,0),output(300001*bytes+2,0xac);
            for(unsigned i=0;i<bytes;++i)atom[i]=static_cast<unsigned char>(i+3);
            footprint_dense_into(output.data()+1,300001,bytes,off.data(),atom.data(),nullptr,0,DenseFootprint::rectangle,0.);
            require(output.front()==0xac && output.back()==0xac);
            for(std::size_t i=0;i<300001;++i)require(std::memcmp(output.data()+1+i*bytes,atom.data(),bytes)==0);
        }
    }
    rejects([&]{footprint_dense_into(nullptr,1,1,&zero,&one,nullptr,0,DenseFootprint::rectangle,0.);});
    rejects([&]{footprint_dense_into(nullptr,3,1,&zero,&one,nullptr,2,DenseFootprint::disk,0.);});
    rejects([&]{footprint_dense_into(nullptr,1,1,&zero,&one,nullptr,1,DenseFootprint::disk,0.);});
    rejects([&]{footprint_dense_into(nullptr,0,1,&zero,&one,nullptr,std::size_t(-1),DenseFootprint::ball,0.);});
    rejects([&]{footprint_dense_into(nullptr,std::size_t(-1),16,&zero,&one,nullptr,0,DenseFootprint::rectangle,0.);});
    rejects([&]{footprint_dense_into(nullptr,0,1,&zero,&one,nullptr,0,static_cast<DenseFootprint>(4),0.);});
    footprint_dense_into(nullptr,0,0,nullptr,nullptr,nullptr,0,DenseFootprint::rectangle,0.);
    set_num_threads(1);
    std::cout<<"footprint dense native guards, tails, atom widths and worker parity passed\n";
}
