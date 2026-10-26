#include "window_internal.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef NDEBUG
#error Window controls require active assertions
#endif
using skimagexx::detail::window_sample;
static std::size_t checks=0;
void check(bool value) {++checks;if(!value)throw std::runtime_error("window control "+std::to_string(checks));}
void literal(int workers) {
    const double window[]={2,6,10};
    const std::vector<float> coords{-1.25f,-1.f,-.75f,-.5f,-.25f,0.f,.25f,.5f,.75f,1.f,1.5f,2.f,2.25f,2.5f,2.75f,3.f,3.25f};
    const std::vector<double> linear{0,0,.5,1,1.5,2,3,4,5,6,8,10,7.5,5,2.5,0,0};
    const std::vector<double> nearest{0,0,0,2,2,2,2,6,6,6,10,10,10,0,0,0,0};
    for(int order:{0,1}) {
        std::vector<double> output(coords.size()+2,987.);
        window_sample(window,3,coords.data(),coords.size(),output.data()+1,order,workers);
        check(output.front()==987 && output.back()==987);
        for(std::size_t i=0;i<coords.size();++i)check(output[i+1]==(order?linear[i]:nearest[i]));
        // All short tails, including empty output, use the same literal oracle.
        for(std::size_t n=0;n<=9;++n) {
            std::fill(output.begin(),output.end(),987.);
            window_sample(window,3,coords.data(),n,output.data()+1,order,workers);
            for(std::size_t i=0;i<n;++i)check(output[i+1]==(order?linear[i]:nearest[i]));
            check(output[n+1]==987 && output.front()==987);
        }
    }
    const double single[]={4};const float positions[]={-.75f,-.5f,0.f,.5f,.75f,1.f};
    double out[6];window_sample(single,1,positions,6,out,1,workers);
    const double expected[]={1,2,4,2,1,0};for(int i=0;i<6;++i)check(out[i]==expected[i]);
}
void nonfinite(int workers) {
    const auto inf=std::numeric_limits<double>::infinity();
    const double window[]={2,inf,6};const float coords[]={-1.f,-.5f,0.f,.5f,1.f,1.5f,2.f,2.5f,3.f};double out[9];
    window_sample(window,3,coords,9,out,1,workers);
    check(out[0]==0 && out[1]==1);check(std::isnan(out[2]));
    check(out[3]==inf && out[4]==inf && out[5]==inf);
    check(out[6]==6 && out[7]==3 && out[8]==0);
    const double lower[]={inf,2};const float edge[]={-1.f,-.5f,0.f,1.f};
    window_sample(lower,2,edge,4,out,1,workers);
    check(std::isnan(out[0]));check(out[1]==inf && out[2]==inf && out[3]==2);
    const double zeros[]={-0.,-0.};const float zcoords[]={0.f,.25f,.5f,1.f};
    for(int order:{0,1}) {window_sample(zeros,2,zcoords,4,out,order,workers);for(int i=0;i<4;++i)check(out[i]==0 && !std::signbit(out[i]));}
    const double negative[]={-2,-6};const float mid[]={.25f,.5f,.75f};
    window_sample(negative,2,mid,3,out,1,workers);check(out[0]==-3 && out[1]==-4 && out[2]==-5);
    const double nanwindow[]={std::numeric_limits<double>::quiet_NaN(),2};
    window_sample(nanwindow,2,edge,4,out,1,workers);check(std::isnan(out[0])&&std::isnan(out[1])&&std::isnan(out[2])&&out[3]==2);
}
void guards(int workers) {
    const double window[]={2,6};const float coords[]={0,.5f,1};double output[]={99,99,99};
    for(int control=0;control<5;++control) {
        bool rejected=false;
        try {
            window_sample(control==4?nullptr:window,control==0?0:control==3?std::numeric_limits<std::size_t>::max():2,coords,3,output,control==1?2:1,control==2?0:workers);
        } catch(const std::invalid_argument&) {rejected=true;}
        check(rejected);for(auto v:output)check(v==99);
    }
    window_sample(nullptr,1,nullptr,0,nullptr,1,workers);check(output[0]==99);
}
void large(int workers) {
    const auto n=std::size_t(256*1024+37);const double window[]={2,6,10};
    const float table[]={-.5f,0,.25f,.5f,.75f,1,1.5f,2,2.5f,3,4};
    const double expected[]={1,2,3,4,5,6,8,10,5,0,0};
    std::vector<float> coords(n);std::vector<double> output(n+2,77.);
    for(std::size_t i=0;i<n;++i)coords[i]=table[i%11];
    window_sample(window,3,coords.data(),n,output.data()+1,1,workers);
    check(output.front()==77 && output.back()==77);
    for(std::size_t i=0;i<n;++i)check(output[i+1]==expected[i%11]);
    check(skimagexx::get_num_threads()==workers);
}
int main() {
    check(skimagexx::get_num_threads()==1);
    for(int workers:{1,4}) {
#ifndef _OPENMP
        if(workers==4) {bool refused=false;try{skimagexx::set_num_threads(4);}catch(const std::invalid_argument&){refused=true;}check(refused);break;}
#endif
        skimagexx::set_num_threads(workers);literal(workers);nonfinite(workers);guards(workers);large(workers);
    }
    skimagexx::set_num_threads(1);std::cout<<"window controls "<<checks<<" PASS avx2="<<skimagexx::has_avx2()<<"\n";
}
