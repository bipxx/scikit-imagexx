#include "skimagexx/footprint_shapes.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("footprint control failed");}
template<class F>void rejects(F body){bool failed=false;try{body();}catch(const std::exception&){failed=true;}require(failed);}
using namespace skimagexx;
void run(){
    for(std::size_t h:{0u,1u,2u,4u,8u})for(std::size_t w:{0u,1u,3u,5u,8u}){
        const auto rows=2*h+1,cols=2*w+1;std::vector<unsigned char> output(rows*cols+2,99);
        footprint_shapes_ellipse({output.data()+1,rows*cols},rows,cols,double(h),double(w));
        require(output.front()==99&&output.back()==99);
        for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){
            const auto dr=std::ptrdiff_t(r)-std::ptrdiff_t(h),dc=std::ptrdiff_t(c)-std::ptrdiff_t(w);
            // Exact integer inequality is an independent boundary control for
            // these small integral radii, not the implementation's FP tree.
            const auto a=(h+1)*(h+1),b=(w+1)*(w+1);
            const auto lhs=std::size_t(dr*dr)*b+std::size_t(dc*dc)*a,rhs=a*b;
            // Exact rational boundary belongs to the source binary64 policy,
            // not an idealized conic. Full saved-word parity covers it.
            if(lhs!=rhs)require((output[1+r*cols+c]!=0)==(lhs<rhs));
        }
    }
    std::array<unsigned char,25> vertices{},hull{};vertices[0]=vertices[4]=vertices[20]=vertices[24]=1;
    footprint_shapes_hull({vertices.data(),25},{hull.data(),25},5,5);
    for(auto x:hull)require(x==1);
    vertices.fill(0);vertices[12]=1;
    footprint_shapes_hull({vertices.data(),25},{hull.data(),25},5,5);
    for(std::size_t i=0;i<25;++i)require(hull[i]==(i==12));
    std::array<double,5> x{-2,-1,0,1,2},y{},z{};std::array<unsigned char,5> l1{};
    footprint_shapes_l1({x.data(),40},{y.data(),40},{z.data(),40},{l1.data(),5},5,1);
    require((l1==std::array<unsigned char,5>{0,1,1,1,0}));
    std::array<std::uint16_t,5> atoms{};const std::uint16_t zero=0x1234,one=0xabcd;
    footprint_shapes_atoms({l1.data(),5},{atoms.data(),10},{&zero,2},{&one,2},5,2);
    for(std::size_t i=0;i<5;++i)require(atoms[i]==(l1[i]?one:zero));
    auto unchanged=atoms;
    rejects([&]{footprint_shapes_atoms({l1.data(),5},{atoms.data(),9},{&zero,2},{&one,2},5,2);});require(atoms==unchanged);
    rejects([&]{footprint_shapes_atoms({atoms.data(),5},{atoms.data(),10},{&zero,2},{&one,2},5,2);});require(atoms==unchanged);
    auto previous=hull;
    rejects([&]{footprint_shapes_hull({hull.data(),25},{hull.data(),25},5,5);});require(hull==previous);
    rejects([&]{footprint_shapes_ellipse({hull.data(),25},0,5,1,1);});require(hull==previous);
    const auto rounding=std::fegetround();require(std::fesetround(FE_DOWNWARD)==0);
    rejects([&]{footprint_shapes_ellipse({hull.data(),25},5,5,2,2);});require(hull==previous);
    require(std::fegetround()==FE_DOWNWARD);require(std::fesetround(rounding)==0);
    std::feraiseexcept(FE_DIVBYZERO);auto flags=std::fetestexcept(FE_ALL_EXCEPT);
    footprint_shapes_ellipse({hull.data(),25},5,5,2,2);
    require(std::fetestexcept(FE_ALL_EXCEPT)==flags);std::feclearexcept(FE_ALL_EXCEPT);
}
}
int main(){
    require(skimagexx::get_num_threads()==1);
    for(int workers:{1,4}){
#ifdef SKIMAGEXX_FOOTPRINT_SHAPES_PORTABLE
        if(workers==4){rejects([&]{skimagexx::set_num_threads(4);});continue;}
#endif
        skimagexx::set_num_threads(workers);run();
    }
    skimagexx::set_num_threads(1);std::cout<<"footprint native controls="<<checks<<'\n';
}
