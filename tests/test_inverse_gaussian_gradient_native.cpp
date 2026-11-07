#include <skimagexx/inverse_gaussian_gradient.hpp>
#include <skimagexx/image.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {void check(bool v){if(!v)throw std::runtime_error("IGG native invariant failed");}}
int main(){try{
    check(skimagexx::get_num_threads()==1);
    // SciPy generic correlate initializes the last coefficient, then scans
    // left through penultimate. This cancellation witnesses that order.
    const double image[]={1e16,1.,2e16};double gradient[3];
    skimagexx::InverseGaussianPasses p{{{0,{1.,1.,-.5}}}};
    skimagexx::inverse_gaussian_gradient_magnitude(image,gradient,skimagexx::PixelType::float64,false,{3},p,1);
    check(gradient[1]==1.);
    // Complete operation, actual independent-output parallel threshold.
    std::vector<float> zeros(512*512);const auto before=zeros;
    skimagexx::set_num_threads(4);const float alpha=.5f;
    auto result=skimagexx::inverse_gaussian_gradient(zeros.data(),{512,512},{1.},&alpha,{},0);
    check(result.captured_workers==4&&result.values.size()==zeros.size()&&zeros==before);
    for(auto value:result.values)check(value==1.f);
    auto single=skimagexx::inverse_gaussian_gradient(zeros.data(),{512,512},{1.},&alpha,{},1);
    check(single.values==result.values&&single.captured_workers==1&&skimagexx::get_num_threads()==4);
    skimagexx::set_num_threads(1);
    bool refused=false;try{skimagexx::inverse_gaussian_gradient(zeros.data(),{2,2},{1.},&alpha,{3},1);}catch(const std::invalid_argument&){refused=true;}
    check(refused&&skimagexx::get_num_threads()==1);
    std::cout<<"generic-order + largeparallel/explicit-budget/ownership/refusal controls PASS\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
