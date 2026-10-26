#include "color_detail.hpp"
#include "skimagexx/image.hpp"
#include <cfenv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
namespace {
std::size_t checks=0;std::ofstream journal;
void require(bool value){++checks;if(!value)throw std::runtime_error("color status control "+std::to_string(checks));}
template<class T>void run(std::size_t n,int kind){
    std::vector<T> input(n*3,T(.25)),old(n*3),actual(n*3+2,T(1234));
    if(n&&kind==1){input[3*(n-1)]=std::numeric_limits<T>::infinity();input[3*(n-1)+1]=-std::numeric_limits<T>::infinity();input[3*(n-1)+2]=0;}
    if(n&&kind==2){input[0]=std::numeric_limits<T>::max();input[1]=-input[0];input[2]=-input[0];}
    if(n&&kind==3){input[0]=std::numeric_limits<T>::denorm_min();input[1]=input[2]=0;}
    const auto type=std::is_same_v<T,float>?PixelType::float32:PixelType::float64;
    color_transform_into(input.data(),old.data(),n,type,ColorTransform::rgb2yuv);
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    const int prior=std::fetestexcept(FE_ALL_EXCEPT);
    const auto flags=detail::color_transform_status_into(input.data(),actual.data()+1,n,ColorTransform::rgb2yuv);
    require(std::fetestexcept(FE_ALL_EXCEPT)==prior);std::feclearexcept(FE_ALL_EXCEPT);
    require(actual.front()==T(1234)&&actual.back()==T(1234));
    require(!n||std::memcmp(actual.data()+1,old.data(),n*3*sizeof(T))==0);
    if(kind==0)require(flags==0);
    if(kind==1)require(flags&8u);
    if(kind==2)require(flags&2u);
    if(kind==3)require(flags&4u);
    journal.write(reinterpret_cast<const char*>(&flags),sizeof(flags));
    journal.write(reinterpret_cast<const char*>(actual.data()+1),std::streamsize(n*3*sizeof(T)));
}
}
int main(int argc,char**argv){
    if(argc!=2)return 2;journal.open(argv[1],std::ios::binary);require(bool(journal));
    std::vector<int> workers{1};
#ifdef _OPENMP
    workers.push_back(4);
#endif
    for(int worker:workers){set_num_threads(worker);
        for(auto n:{0u,1u,3u,4u,7u,8u,9u,33u,262147u}){run<float>(n,0);run<double>(n,0);}
        for(int kind:{1,2,3}){run<float>(9,kind);run<double>(9,kind);}
        run<float>(262147,1);run<double>(262147,1);
    }
    set_num_threads(1);journal.close();std::cout<<checks<<" color status controls PASS\n";
}
