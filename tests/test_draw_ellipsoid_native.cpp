#include "skimagexx/draw_ellipsoid.hpp"
#include "skimagexx/image.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace skimagexx::detail;
static std::size_t controls=0;
static void require(bool ok) { ++controls; if(!ok) throw std::runtime_error("ellipsoid control "+std::to_string(controls)); }
template<class T> static T read(std::ifstream& f) { T x{}; f.read(reinterpret_cast<char*>(&x),sizeof x); if(!f) throw std::runtime_error("fixture truncated"); return x; }
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    std::ifstream f(argv[1],std::ios::binary);
    const auto records=read<std::uint32_t>(f);
    const int old_workers=skimagexx::get_num_threads();
    require(old_workers==1);
    for(std::uint32_t record=0;record<records;++record) {
        std::array<double,3> axes{},spacing{};
        for(auto& x:axes)x=read<double>(f);
        for(auto& x:spacing)x=read<double>(f);
        const auto level=read<std::uint8_t>(f);
        std::array<std::size_t,3> shape{};
        for(auto& x:shape)x=read<std::uint64_t>(f);
        const auto bytes=read<std::uint64_t>(f);
        std::vector<char> expected(bytes);f.read(expected.data(),bytes);require(bool(f));
        DrawEllipsoidPlan p; require(draw_ellipsoid_prepare(axes,spacing,p));require(p.shape==shape);
        for(int workers:{1,4}) {
            skimagexx::set_num_threads(workers);
            const int snapshot=skimagexx::get_num_threads();
            std::feclearexcept(FE_ALL_EXCEPT);
            if(level) {
                std::vector<double> out(p.count+2,0x1.234p5);
                draw_ellipsoid_levelset_into(p,out.data()+1,snapshot);
                require(bytes==p.count*8);require(std::memcmp(out.data()+1,expected.data(),bytes)==0);
                require(out.front()==0x1.234p5&&out.back()==0x1.234p5);
            } else {
                std::vector<std::uint8_t> out(p.count+2,231);
                draw_ellipsoid_bool_into(p,out.data()+1,snapshot);
                require(bytes==p.count);require(std::memcmp(out.data()+1,expected.data(),bytes)==0);
                require(out.front()==231&&out.back()==231);
            }
            require(!(std::fetestexcept(FE_INVALID|FE_OVERFLOW|FE_DIVBYZERO|FE_UNDERFLOW)));
        }
    }
    DrawEllipsoidPlan large;
    require(draw_ellipsoid_prepare({32,32,32},{1,1,1},large));
    require(large.shape==std::array<std::size_t,3>{67,67,67});require(large.count==300763);
    for(int workers:{1,4}) {
        skimagexx::set_num_threads(workers);const int snapshot=skimagexx::get_num_threads();
        std::vector<double> level(large.count);std::vector<std::uint8_t> mask(large.count);
        draw_ellipsoid_bool_into(large,mask.data(),snapshot);
        draw_ellipsoid_levelset_into(large,level.data(),snapshot);
        for(int i=0;i<67;++i)for(int j=0;j<67;++j)for(int k=0;k<67;++k) {
            const int square=(i-33)*(i-33)+(j-33)*(j-33)+(k-33)*(k-33);
            const auto index=(i*67+j)*67+k;
            require(mask[index]==(square<=1024));require(level[index]==(square-1024)/1024.0);
        }
    }
    DrawEllipsoidPlan unchanged=large;
    for(double bad:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN(),0x1p-129,0x1p129}) {
        require(!draw_ellipsoid_prepare({bad,1,1},{1,1,1},large));
        require(!draw_ellipsoid_prepare({1,1,1},{bad,1,1},large));
        require(large.shape==unchanged.shape&&large.squared==unchanged.squared);
    }
    require(!draw_ellipsoid_prepare({4096,4096,4096},{1,1,1},large));
    require(!draw_ellipsoid_prepare({1,1,1},{0x1p-128,1,1},large));
    skimagexx::set_num_threads(old_workers);require(skimagexx::get_num_threads()==old_workers);
    std::cout<<"{\"records\":"<<records<<",\"controls\":"<<controls<<",\"workers\":[1,4],\"restored\":true}"<<std::endl;
}
