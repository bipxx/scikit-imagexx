#include "../src/resample_detail.hpp"
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
void require(bool ok) {if(!ok)throw std::runtime_error("native zoom check failed");}
std::size_t count(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());
}
std::vector<double> pattern(std::size_t n) {
    std::vector<double> result(n);
    for(std::size_t i=0;i<n;++i)result[i]=double(int((i*17)%29)-14)/8.;
    return result;
}
void check(const std::vector<std::size_t>& shape,const std::vector<std::size_t>& target,
           int order,GaussianBorder border,const std::vector<double>& expected,
           bool legacy,bool identity) {
    const auto input=pattern(count(shape)),snapshot=input;
    std::vector<double> output(count(target)),scalar(output.size());require(output.size()==expected.size());
    detail::ZoomOptions options;options.order=order;options.border=border;options.cval=.3;
    options.legacy_reflect=legacy;options.identity_copy=identity;
    detail::zoom_into(input.data(),output.data(),shape,target,options);require(input==snapshot);
    for(std::size_t i=0;i<output.size();++i) {
        if(!(std::abs(output[i]-expected[i])<=3e-13*std::max(1.,std::abs(expected[i])))) {
            std::cerr<<"order="<<order<<" border="<<int(border)<<" legacy="<<legacy
                     <<" rank="<<shape.size()<<" index="<<i<<" actual="<<output[i]<<" reference="<<expected[i]<<"\n";
            require(false);
        }
    }
    options.allow_simd=false;
    detail::zoom_into(input.data(),scalar.data(),shape,target,options);
    require(std::memcmp(scalar.data(),output.data(),output.size()*sizeof(double))==0);
}
template<class E,class F> void rejects(F body) {
    bool caught=false;try {body();}catch(const E&) {caught=true;}require(caught);
}
void guards() {
    double input=3.,output=0.;detail::ZoomOptions options;
    #ifndef _OPENMP
    rejects<std::invalid_argument>([]{set_num_threads(4);});require(get_num_threads()==1);
    #endif
    detail::zoom_into(nullptr,nullptr,{2,3},{0,4},options);
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{}, {},options);});
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{1},{1,1},options);});
    rejects<std::invalid_argument>([&]{detail::zoom_into(nullptr,&output,{1},{1},options);});
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{0},{1},options);});
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&input,{1},{1},options);});
    options.order=6;
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{1},{1},options);});
    options.order=1;options.border=static_cast<GaussianBorder>(99);
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{1},{1},options);});
    options.border=GaussianBorder::reflect;
    rejects<std::overflow_error>([&]{detail::zoom_into(&input,&output,{std::numeric_limits<std::size_t>::max()},{1},options);});
    options.identity_copy=true;
    rejects<std::invalid_argument>([&]{detail::zoom_into(&input,&output,{1},{2},options);});
    detail::zoom_into(&input,&input,{1},{1},options);require(input==3.);
    const std::vector<double> special{-0.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()};
    std::vector<double> copied(3);
    detail::zoom_into(special.data(),copied.data(),{3},{3},options);
    require(std::memcmp(special.data(),copied.data(),3*sizeof(double))==0);
}
void workers() {
    const std::vector<std::size_t> shape{257,1025},target{513,513};const auto input=pattern(count(shape));
    std::vector<double> one(count(target)),four(one.size()),scalar(one.size());
    detail::ZoomOptions options;options.cval=-.3;
    for(bool legacy:{false,true})for(int order:{0,1,3})
    for(auto border:{GaussianBorder::constant,GaussianBorder::nearest,GaussianBorder::mirror,GaussianBorder::reflect,GaussianBorder::wrap}) {
        options.order=order;options.border=border;options.legacy_reflect=legacy;options.allow_simd=true;
        set_num_threads(1);detail::zoom_into(input.data(),one.data(),shape,target,options);
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        detail::zoom_into(input.data(),four.data(),shape,target,options);require(one==four);
        options.allow_simd=false;detail::zoom_into(input.data(),scalar.data(),shape,target,options);
        require(std::memcmp(one.data(),scalar.data(),one.size()*sizeof(double))==0);
    }
    set_num_threads(1);
}
}
int main() {
    require(get_num_threads()==1);guards();
    #include "data/zoom_samples.inc"
    #include "data/zoom_samples_legacy.inc"
    workers();
    std::cout<<"480 SciPy zoom fixtures, guards and exact worker/SIMD checks passed; AVX2 available="
             <<std::boolalpha<<has_avx2()<<"\n";
}
