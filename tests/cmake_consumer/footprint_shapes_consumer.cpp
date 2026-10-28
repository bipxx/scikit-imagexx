// Installed prepared footprint stages: fixed masks and typed atoms, no Python.
#include <skimagexx/footprint_shapes.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
using namespace skimagexx;
namespace { void check(bool ok){if(!ok)throw std::runtime_error("installed footprint client mismatch");} }
int main(){
    try{
        check(get_num_threads()==1);
        for(int workers:{1,4}){
            set_num_threads(workers);check(get_num_threads()==workers);
            check(footprint_shapes_octahedron_axis(1)==3);
            std::array<unsigned char,27> octahedron{};
            footprint_shapes_octahedron({octahedron.data(),octahedron.size()},1);
            const std::array<unsigned char,27> cross={0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,1,0,0,0,0};
            check(octahedron==cross);
            const unsigned char corners[]={1,0,1,0,0,0,1,0,1};std::array<unsigned char,9> hull{},ellipse{};
            footprint_shapes_hull({corners,sizeof(corners)},{hull.data(),hull.size()},3,3);
            footprint_shapes_ellipse({ellipse.data(),ellipse.size()},3,3,1,1);
            check(std::all_of(hull.begin(),hull.end(),[](auto n){return n==1;})&&ellipse==hull);
            const double x[]={-1,0,1,1},y[]={0,0,0,1},z[]={0,0,0,0};std::array<unsigned char,4> mask{};
            footprint_shapes_l1({x,sizeof(x)},{y,sizeof(y)},{z,sizeof(z)},{mask.data(),mask.size()},4,1);
            check((mask==std::array<unsigned char,4>{1,1,1,0}));
            footprint_shapes_l1({x,sizeof(x)},{y,sizeof(y)},{z,sizeof(z)},{mask.data(),mask.size()},4,1,false);
            check((mask==std::array<unsigned char,4>{0,1,0,0}));
            const std::uint16_t zero=0x1234,one=0xabcd;std::array<std::uint16_t,4> atoms{};
            footprint_shapes_atoms({mask.data(),mask.size()},{atoms.data(),sizeof(atoms)},{&zero,sizeof(zero)},{&one,sizeof(one)},4,sizeof(one));
            check((atoms==std::array<std::uint16_t,4>{zero,one,zero,zero}));
        }
        set_num_threads(1);check(get_num_threads()==1);
        std::cout<<"installed footprint stages PASS avx2="<<has_avx2()<<" workers=1\n";
    }catch(const std::exception& error){set_num_threads(1);std::cerr<<error.what()<<'\n';return 1;}
}
