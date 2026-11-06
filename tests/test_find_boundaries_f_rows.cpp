#include "cmake_consumer/find_boundaries_consumer.hpp"
#include <iostream>
int main(){
    using namespace skimagexx;using namespace skimagexx::detail;
    set_num_threads(1);std::size_t checks=0;
    for(int workers:{1,4})checks+=skimagexx_test::find_boundaries_f_rows(workers);
    std::fenv_t saved;std::fegetenv(&saved);
    std::uint16_t input[72]{};for(std::size_t i=0;i<72;++i)input[i]=std::uint16_t(i%5);
    ComponentsView view{input,PixelType::uint16,false,false,{8,9},{2,16}};
    std::uint8_t output[72];FindBoundaryParameters p{FindBoundaryMode::inner,2,0,true};
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        if(std::fesetround(rounding)||std::feclearexcept(FE_ALL_EXCEPT)||std::feraiseexcept(FE_DIVBYZERO|FE_INEXACT))return 2;
        if(!find_boundaries_into(view,p,output,{1,8},0)||std::fegetround()!=rounding||std::fetestexcept(FE_ALL_EXCEPT)!=(FE_DIVBYZERO|FE_INEXACT))return 3;
        ++checks;
    }
    std::fesetenv(&saved);if(get_num_threads()!=1)return 4;
    std::cout<<"PASS F_row_controls="<<checks<<" workers=1,4,default restored=1\n";
}
