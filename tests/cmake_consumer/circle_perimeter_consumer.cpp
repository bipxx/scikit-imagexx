// Installed SDK client: frozen original coordinate words, no Python/source includes.
#include <skimagexx/circle_perimeter.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
using Int=std::intptr_t;
std::size_t checks=0;
void check(bool ok){++checks;if(!ok)throw std::runtime_error("installed circle mismatch "+std::to_string(checks));}
const std::vector<Int> bresenham_rows{7,1,7,1,4,4,4,4,7,1,7,1,5,3,5,3,6,2,6,2,6,2,6,2};
const std::vector<Int> bresenham_columns{5,5,5,5,8,8,2,2,6,6,4,4,8,8,2,2,7,7,3,3,7,7,3,3};
const std::vector<Int> andres_rows{7,1,7,1,4,4,4,4,7,1,7,1,5,3,5,3,7,1,7,1,6,2,6,2,6,2,6,2,6,2,6,2};
const std::vector<Int> andres_columns{5,5,5,5,8,8,2,2,6,6,4,4,8,8,2,2,7,7,3,3,8,8,2,2,7,7,3,3,7,7,3,3};
void expected(const CirclePerimeter& prepared,const std::vector<Int>& rr,const std::vector<Int>& cc,int workers) {
    const auto n=circle_perimeter_count(prepared,workers);check(n==rr.size()&&n==cc.size());
    constexpr Int guard=887799;
    std::vector<Int> rows(3*n+2,guard),columns(rows.size(),guard);
    circle_perimeter_fill(prepared,{rows.data()+3*n,columns.data()+1,-3,3},workers);
    for(std::size_t i=0;i<n;++i){check(rows[3*n-3*i]==rr[i]);check(columns[1+3*i]==cc[i]);}
    for(std::size_t i=0;i<rows.size();++i){
        if(i<3||i%3)check(rows[i]==guard);
        if(i<1||(i-1)%3||i>=3*n)check(columns[i]==guard);
    }
}
void stages(int workers) {
    const auto global=get_num_threads(),rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
    for(auto method:{CirclePerimeterMethod::bresenham,CirclePerimeterMethod::andres}) {
        const auto& rr=method==CirclePerimeterMethod::bresenham?bresenham_rows:andres_rows;
        const auto& cc=method==CirclePerimeterMethod::bresenham?bresenham_columns:andres_columns;
        CirclePerimeter prepared;CirclePerimeterRequest request{4,5,3,method,false,0,0};
        check(prepare_circle_perimeter(request,prepared));expected(prepared,rr,cc,workers);
        request.clipped=true;request.height=6;request.width=7;
        check(prepare_circle_perimeter(request,prepared));std::vector<Int> clipped_r,clipped_c;
        for(std::size_t i=0;i<rr.size();++i)if(rr[i]>=0&&rr[i]<6&&cc[i]>=0&&cc[i]<7){clipped_r.push_back(rr[i]);clipped_c.push_back(cc[i]);}
        expected(prepared,clipped_r,clipped_c,workers);
        request.radius=0;request.clipped=false;
        check(prepare_circle_perimeter(request,prepared));expected(prepared,std::vector<Int>(8,4),std::vector<Int>(8,5),workers);
        request.radius=-1;check(prepare_circle_perimeter(request,prepared));check(circle_perimeter_count(prepared,workers)==0);
        circle_perimeter_fill(prepared,{nullptr,nullptr,1,1},workers);
        request.radius=3;request.clipped=true;request.height=0;
        check(prepare_circle_perimeter(request,prepared));check(circle_perimeter_count(prepared,workers)==0);
        circle_perimeter_fill(prepared,{nullptr,nullptr,1,1},workers);
        request.radius=32768;request.clipped=false;
        check(prepare_circle_perimeter(request,prepared));check(circle_perimeter_count(prepared,workers)>50000);
    }
    check(get_num_threads()==global&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
}
}
int main(){
    check(get_num_threads()==1);
    for(int workers:{1,4}){set_num_threads(workers);stages(workers);}
    set_num_threads(1);check(get_num_threads()==1);
    std::cout<<"installed circle SDK controls="<<checks<<"; workers=1,4; default=1; frozen Bresenham/Andres order\n";
}
