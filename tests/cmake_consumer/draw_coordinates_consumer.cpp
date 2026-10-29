// Installed SDK consumer only: no private source headers, Python or originals.
#include <skimagexx/draw_coordinates.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void check(bool value){++checks;if(!value)throw std::runtime_error("installed drawing coordinate mismatch "+std::to_string(checks));}
void stages(int workers){
    const auto global=get_num_threads(),rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
    std::array<std::intptr_t,7> line_r,line_c;line_r.fill(999);line_c.fill(999);
    draw_line_into(0,0,2,4,{line_r.data()+1,line_c.data()+1,1,1},workers);
    check((line_r==std::array<std::intptr_t,7>{999,0,1,1,2,2,999}));
    check((line_c==std::array<std::intptr_t,7>{999,0,1,2,3,4,999}));
    draw_line_into(2,4,0,0,{line_r.data()+5,line_c.data()+1,-1,1},workers);
    check((line_r==std::array<std::intptr_t,7>{999,0,0,1,1,2,999}));
    check((line_c==std::array<std::intptr_t,7>{999,4,3,2,1,0,999}));
    std::array<std::intptr_t,3> point_r{999,999,999},point_c=point_r;
    draw_line_into(-3,7,-3,7,{point_r.data()+1,point_c.data()+1,1,1},workers);
    check(point_r[0]==999&&point_r[1]==-3&&point_r[2]==999&&point_c[0]==999&&point_c[1]==7&&point_c[2]==999);

    const std::intptr_t starts[]{-1,2,4};const std::size_t lengths[]{2,3,2};
    std::array<std::array<std::intptr_t,14>,3> grids;for(auto& grid:grids)grid.fill(999);
    std::intptr_t* outputs[]{grids[0].data()+1,grids[1].data()+1,grids[2].data()+1};
    draw_rectangle_into(starts,lengths,3,12,outputs,workers);
    for(std::intptr_t c=0;c<3;++c)for(std::intptr_t r=0;r<2;++r)for(std::intptr_t z=0;z<2;++z){
        const auto index=std::size_t((c*2+r)*2+z);check(outputs[0][index]==r-1&&outputs[1][index]==c+2&&outputs[2][index]==z+4);
    }
    for(const auto& grid:grids)check(grid.front()==999&&grid.back()==999);
    const std::size_t empty[]{0,3};
    draw_rectangle_into(starts,empty,2,0,outputs,workers);
    draw_rectangle_into(nullptr,nullptr,0,0,nullptr,workers);
    for(const auto& grid:grids)check(grid.front()==999&&grid.back()==999);
    // Above the existing partition threshold: actual independent SDK blocks.
    const std::size_t big[]{513,517};std::vector<std::intptr_t> r(big[0]*big[1]),c(r.size());std::intptr_t* big_outputs[]{r.data(),c.data()};
    draw_rectangle_into(starts,big,2,r.size(),big_outputs,workers);
    for(std::size_t col=0;col<big[1];++col)for(std::size_t row=0;row<big[0];++row){const auto i=col*big[0]+row;check(r[i]==std::intptr_t(row)-1&&c[i]==std::intptr_t(col)+2);}

    DrawEllipseGrid ellipse{7,11,3.,5.,2.,4.,0.,1.,-7,9};
    std::array<std::uint8_t,79> mask;mask.fill(0x5a);std::array<std::size_t,10> prefix;prefix.fill(999);
    check(draw_ellipse_mask_into(ellipse,mask.data()+1,prefix.data()+1,workers));
    check(mask.front()==0x5a&&mask.back()==0x5a&&prefix.front()==999&&prefix.back()==999&&prefix[1]==0);
    std::vector<std::intptr_t> expected;
    for(std::intptr_t row=0;row<7;++row)for(std::intptr_t col=0;col<11;++col)if(4*(row-3)*(row-3)+(col-5)*(col-5)<16){expected.push_back(row-7);expected.push_back(col+9);}
    check(prefix[8]*2==expected.size());std::vector<std::intptr_t> pairs(expected.size()+2,999);
    draw_ellipse_emit_into(ellipse,mask.data()+1,prefix.data()+1,{pairs.data()+1,pairs.data()+2,2,2},workers);
    check(pairs.front()==999&&pairs.back()==999&&std::equal(expected.begin(),expected.end(),pairs.begin()+1));
    check(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    for(bool zero_rows:{false,true}){
        auto zero=ellipse;zero.rows=zero_rows?0:5;zero.columns=zero_rows?5:0;std::array<std::size_t,6> empty_prefix;empty_prefix.fill(999);
        check(draw_ellipse_mask_into(zero,nullptr,empty_prefix.data(),workers));
        check(std::all_of(empty_prefix.begin(),empty_prefix.begin()+std::ptrdiff_t(zero.rows+1),[](auto value){return value==0;}));
        draw_ellipse_emit_into(zero,nullptr,empty_prefix.data(),{nullptr,nullptr,2,2},workers);
    }
    std::fesetround(FE_DOWNWARD);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
    mask.fill(0x5a);prefix.fill(999);
    check(!draw_ellipse_mask_into(ellipse,mask.data()+1,prefix.data()+1,workers));
    check(std::fegetround()==FE_DOWNWARD&&std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
    check(std::all_of(mask.begin(),mask.end(),[](auto value){return value==0x5a;}));
    check(std::all_of(prefix.begin(),prefix.end(),[](auto value){return value==999;}));
    std::fesetround(rounding);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(flags);
    check(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
}
}
int main(){
    check(get_num_threads()==1);
    for(int workers:{1,4}){set_num_threads(workers);stages(workers);}
    set_num_threads(1);check(get_num_threads()==1);
    std::cout<<"installed drawing SDK controls="<<checks<<"; workers=1,4; default=1\n";
}
