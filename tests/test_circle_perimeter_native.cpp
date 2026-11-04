#include <skimagexx/circle_perimeter.hpp>
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace skimagexx::detail;
namespace {
using Int=std::intptr_t;
std::size_t checks=0,fixture_records=0;
const char* context="initial";
void require(bool ok){++checks;if(!ok)throw std::runtime_error(context);}
template<class T>T read(std::ifstream& stream){T value{};stream.read(reinterpret_cast<char*>(&value),sizeof(T));require(bool(stream));return value;}
std::vector<Int> output(const CirclePerimeter& prepared,int workers) {
    const auto n=circle_perimeter_count(prepared,workers);
    require(n<=prepared.max_points);
    constexpr Int guard=918273;
    std::vector<Int> rows(3*n+2,guard),columns(rows.size(),guard),pairs;
    circle_perimeter_fill(prepared,{rows.data()+3*n,columns.data()+1,-3,3},workers);
    pairs.reserve(2*n);
    for(std::size_t i=0;i<n;++i){pairs.push_back(rows[3*n-3*i]);pairs.push_back(columns[1+3*i]);}
    for(std::size_t i=0;i<rows.size();++i){
        if(i<3||i%3)require(rows[i]==guard);
        if(i<1||(i-1)%3||i>=3*n)require(columns[i]==guard);
    }
    return pairs;
}
void frozen(const char* path,int workers) {
    context="frozen original words/order";
    std::ifstream stream(path,std::ios::binary);require(bool(stream));
    char magic[8];stream.read(magic,8);require(std::string(magic,8)=="SXXCIRC1");
    const auto records=read<std::uint64_t>(stream);
    for(std::uint64_t i=0;i<records;++i) {
        CirclePerimeterRequest q;
        q.row=Int(read<std::int64_t>(stream));q.column=Int(read<std::int64_t>(stream));q.radius=Int(read<std::int64_t>(stream));
        q.method=read<std::uint64_t>(stream)?CirclePerimeterMethod::andres:CirclePerimeterMethod::bresenham;
        q.clipped=bool(read<std::uint64_t>(stream));q.height=Int(read<std::int64_t>(stream));q.width=Int(read<std::int64_t>(stream));
        const auto n=read<std::uint64_t>(stream);std::vector<Int> expected(2*n);
        for(auto& value:expected)value=Int(read<std::int64_t>(stream));
        CirclePerimeter prepared;require(prepare_circle_perimeter(q,prepared));
        require(circle_perimeter_count(prepared,workers)==n);
        require(output(prepared,workers)==expected);++fixture_records;
    }
    require(stream.peek()==std::char_traits<char>::eof());
}
void controls(int workers) {
    context="symmetry, stable clipping and worker invariants";
    for(auto method:{CirclePerimeterMethod::bresenham,CirclePerimeterMethod::andres}) {
        for(Int radius=-3;radius<=128;++radius) {
            CirclePerimeterRequest q{7,-3,radius,method,false,0,0};CirclePerimeter whole;
            require(prepare_circle_perimeter(q,whole));const auto uncut=output(whole,workers);
            require(uncut==output(whole,1));
            if(radius<0)require(uncut.empty());
            else {
                require(uncut.size()%16==0);
                for(std::size_t i=0;i<uncut.size();i+=16) {
                    const auto r=uncut[i]-q.row,c=uncut[i+1]-q.column;
                    const Int rr[]{r,-r,r,-r,c,-c,c,-c};const Int cc[]{c,c,-c,-c,r,r,-r,-r};
                    for(unsigned j=0;j<8;++j){require(uncut[i+2*j]==rr[j]+q.row);require(uncut[i+2*j+1]==cc[j]+q.column);}
                }
                if(radius==0){require(uncut.size()==16);for(std::size_t i=0;i<16;i+=2){require(uncut[i]==q.row);require(uncut[i+1]==q.column);}}
            }
            for(auto bounds:{std::pair<Int,Int>{0,9},{9,0},{-1,9},{9,-1},{8,9},{50,60}}) {
                auto clipped_q=q;clipped_q.clipped=true;clipped_q.height=bounds.first;clipped_q.width=bounds.second;
                CirclePerimeter clipped;require(prepare_circle_perimeter(clipped_q,clipped));
                std::vector<Int> expected;
                for(std::size_t i=0;i<uncut.size();i+=2)if(uncut[i]>=0&&uncut[i]<bounds.first&&uncut[i+1]>=0&&uncut[i+1]<bounds.second){expected.push_back(uncut[i]);expected.push_back(uncut[i+1]);}
                require(output(clipped,workers)==expected);
            }
        }
        // Long frozen benchmark domain must not inherit the tiny reference quota.
        CirclePerimeter prepared;CirclePerimeterRequest q{0,0,32768,method,false,0,0};
        require(prepare_circle_perimeter(q,prepared));require(circle_perimeter_count(prepared,workers)>50000);
        require(output(prepared,workers)==output(prepared,1));
    }
}
void boundary() {
    context="single boundary rejection and empty buffers";
    constexpr auto high=std::numeric_limits<Int>::max(),low=std::numeric_limits<Int>::min();
    CirclePerimeter sentinel{{1,2,3,CirclePerimeterMethod::andres,false,0,0},19};
    for(auto q:{CirclePerimeterRequest{high,0,1},CirclePerimeterRequest{low,0,1},CirclePerimeterRequest{0,high,1},CirclePerimeterRequest{0,low,1},CirclePerimeterRequest{0,0,high},CirclePerimeterRequest{0,0,low},CirclePerimeterRequest{0,0,-1,static_cast<CirclePerimeterMethod>(88)}}) {
        auto prepared=sentinel;require(!prepare_circle_perimeter(q,prepared));
        require(prepared.max_points==19&&prepared.request.row==1&&prepared.request.method==CirclePerimeterMethod::andres);
    }
    CirclePerimeter prepared;require(prepare_circle_perimeter({high,low,0},prepared));
    require(circle_perimeter_count(prepared,1)==8);require(output(prepared,1).size()==16);
    require(prepare_circle_perimeter({0,0,-1},prepared));require(circle_perimeter_count(prepared,4)==0);
    circle_perimeter_fill(prepared,{nullptr,nullptr,1,1},4);
}
void floating_state() {
    context="integer kernel preserves floating state";
    const auto saved_round=std::fegetround();const auto saved_flags=std::fetestexcept(FE_ALL_EXCEPT);
    for(auto rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        require(std::fesetround(rounding)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        CirclePerimeter prepared;require(prepare_circle_perimeter({4,5,7},prepared));
        (void)output(prepared,4);
        require(std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
    std::fesetround(saved_round);std::feclearexcept(FE_ALL_EXCEPT);if(saved_flags)std::feraiseexcept(saved_flags);
}
}
int main(int argc,char** argv) {
    if(argc>2){std::cerr<<"expected optional frozen fixture path\n";return 2;}
    boundary();
    for(int workers:{1,4}){if(argc==2)frozen(argv[1],workers);controls(workers);}
    floating_state();
    std::cout<<"circle frozen records="<<fixture_records<<"; native controls="<<checks<<"; workers=1,4; serial recurrence; no FP arithmetic\n";
}
