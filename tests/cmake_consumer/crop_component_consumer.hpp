#pragma once
#include <skimagexx/crop.hpp>
#include <skimagexx/image.hpp>
#include <stdexcept>
#include <vector>
namespace skimagexx_test {
inline std::size_t crop_component_consumer(int workers){
    using namespace skimagexx;std::size_t count=0;
    auto check=[&](bool v){++count;if(!v)throw std::runtime_error("installed crop component mismatch");};
    const auto old=get_num_threads();struct Restore{int n;~Restore(){set_num_threads(n);}}restore{old};
    set_num_threads(workers);check(get_num_threads()==workers);
    std::int16_t a[20];for(int i=0;i<20;++i)a[i]=std::int16_t(i);
    CropView input{a,{4,5},{10,2},2};auto plan=crop_view(input,{{1,3},{1,4}});
    check((plan.view.shape==std::vector<std::ptrdiff_t>{2,3}));check(plan.view.data==a+6);
    std::int16_t output[6]={};crop_copy_into(input,{{1,3},{1,4}},output,{6,2});
    for(int i=0;i<6;++i)check(output[i]==a[6+(i/3)*5+i%3]);
    crop_copy_into(input,{{1,3},{1,4}},output,{2,4});
    for(int r=0;r<2;++r)for(int c=0;c<3;++c)check(output[r+c*2]==a[(r+1)*5+c+1]);
    CropView reverse{a+19,{4,5},{-10,-2},2};crop_copy_into(reverse,{{1,3},{1,4}},output,{6,2});
    for(int i=0;i<6;++i)check(output[i]==a[13-(i/3)*5-i%3]);
    check(get_num_threads()==workers);return count;
}
}
