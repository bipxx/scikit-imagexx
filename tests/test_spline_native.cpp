#include "../src/spline_detail.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace skimagexx;
namespace {
void require(bool ok) {if(!ok)throw std::runtime_error("spline native check failed");}
std::size_t count(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<std::size_t>());
}
std::vector<double> pattern(std::size_t n) {
    std::vector<double> result(n);
    for(std::size_t i=0;i<n;++i)result[i]=double(int((i*17)%29)-14)/8.;
    return result;
}
void check(const std::vector<std::size_t>& shape,int order,GaussianBorder border,
           const std::vector<double>& expected,bool legacy=false) {
    const auto input=pattern(count(shape)),snapshot=input;
    std::vector<double> output(input.size());require(output.size()==expected.size());
    detail::spline_filter_into(input.data(),output.data(),shape,order,border,legacy);
    require(input==snapshot);
    for(std::size_t i=0;i<output.size();++i)
        require(std::abs(output[i]-expected[i])<=3e-14*std::max(1.,std::abs(expected[i])));
    auto inplace=input;
    detail::spline_filter_into(inplace.data(),inplace.data(),shape,order,border,legacy);
    require(inplace==output);
    detail::spline_filter_into(input.data(),inplace.data(),shape,order,border,legacy,false);
    require(inplace==output);
}
template<class E,class F> void rejects(F body) {
    bool caught=false;try {body();}catch(const E&) {caught=true;}require(caught);
}
void guards() {
    double input=3.,output=0.;
    detail::spline_filter_into(nullptr,nullptr,{2,0,3},3,GaussianBorder::reflect);
    detail::spline_filter_into(&input,&output,{},3,GaussianBorder::reflect);require(output==3.);
    detail::spline_filter_into(&input,&output,{1,1,1},5,GaussianBorder::wrap);require(output==3.);
    rejects<std::invalid_argument>([&]{detail::spline_filter_into(&input,&output,{1},6,GaussianBorder::reflect);});
    rejects<std::invalid_argument>([&]{detail::spline_filter_into(nullptr,&output,{1},3,GaussianBorder::reflect);});
    rejects<std::invalid_argument>([&]{detail::spline_filter_into(&input,&output,{1},3,static_cast<GaussianBorder>(99));});
    rejects<std::overflow_error>([&]{detail::spline_filter_into(&input,&output,{std::numeric_limits<std::size_t>::max()},3,GaussianBorder::reflect);});
    rejects<std::overflow_error>([&]{detail::spline_filter_into(&input,&output,{std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double),2},3,GaussianBorder::reflect);});
    rejects<std::invalid_argument>([]{detail::spline_weights(0.,-1);});
    rejects<std::invalid_argument>([]{detail::spline_weights(std::numeric_limits<double>::infinity(),3);});
    for(int order=0;order<=5;++order)for(double x:{-2.7,-.5,-.1,0.,.1,.5,.9,1.3}) {
        const auto weights=detail::spline_weights(x,order);
        require(std::abs(std::accumulate(weights.begin(),weights.end(),0.)-1.)<4e-16);
        for(int k=0;k<=order;++k)require(weights[k]>=0 && weights[k]<=1);
    }
}
void workers() {
    const std::vector<std::size_t> shape{129,2051};const auto input=pattern(count(shape));
    std::vector<double> one(input.size()),four(input.size()),scalar(input.size());
    for(bool legacy:{false,true})
    for(auto border:{GaussianBorder::constant,GaussianBorder::nearest,GaussianBorder::mirror,GaussianBorder::reflect,GaussianBorder::wrap}) {
        set_num_threads(1);detail::spline_filter_into(input.data(),one.data(),shape,5,border,legacy);
        set_num_threads(4);detail::spline_filter_into(input.data(),four.data(),shape,5,border,legacy);
        require(one==four);
        detail::spline_filter_into(input.data(),scalar.data(),shape,5,border,legacy,false);
        require(scalar==one);
    }
    set_num_threads(1);
}
}
int main() {
    require(get_num_threads()==1);
    guards();
    #include "data/spline_coefficients.inc"
    #include "data/spline_coefficients_legacy.inc"
    workers();
    std::cout<<"240 SciPy spline coefficient fixtures, guards and worker checks passed; AVX2 available="
             <<std::boolalpha<<has_avx2()<<"\n";
}
