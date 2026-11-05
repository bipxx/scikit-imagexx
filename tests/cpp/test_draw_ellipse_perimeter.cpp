#include <skimagexx/draw_ellipse_perimeter.hpp>
#include <skimagexx/image.hpp>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx::detail;
using Int=std::intptr_t;
static std::size_t checks=0;
static void check(bool v){++checks;if(!v)throw std::runtime_error("ellipseperimetercontrol"+std::to_string(checks));}
template<class T> static T read(std::ifstream& f){T value{};f.read(reinterpret_cast<char*>(&value),sizeof(value));if(!f)throw std::runtime_error("fixturetruncated");return value;}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    if(std::strcmp(argv[1],"--point-order-only")==0) {
        const auto previous=skimagexx::get_num_threads();check(previous==1);
        for(int workers:{1,4}) {
            skimagexx::set_num_threads(workers);const int snapshot=skimagexx::get_num_threads();
            for(double angle:{0.,.3,-.5,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
                EllipsePerimeterRequest q;q.row=3;q.column=4;q.orientation=angle;q.clipped=true;
                q.height=std::numeric_limits<double>::quiet_NaN();q.width=0;
                EllipsePerimeter point;std::feclearexcept(FE_ALL_EXCEPT);
                check(prepare_ellipse_perimeter(q,point));check(point.scalar_result&&ellipse_perimeter_count(point)==1);
                Int rows[]{99,99,99},columns[]{99,99,99};
                ellipse_perimeter_fill(point,{rows+1,columns+1,1,1},snapshot);
                check(rows[1]==3&&columns[1]==4);check(rows[0]==99&&rows[2]==99&&columns[0]==99&&columns[2]==99);
                check(!(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW)));
            }
        }
        skimagexx::set_num_threads(previous);check(skimagexx::get_num_threads()==previous);
        std::cout<<"{\"point_order_controls\":"<<checks<<",\"workers\":[1,4],\"cases\":12,\"default_restored\":true}"<<std::endl;
        return 0;
    }
    const auto old=skimagexx::get_num_threads();check(old==1);
    std::ifstream f(argv[1],std::ios::binary);const auto records=read<std::uint32_t>(f);check(records==46);
    for(unsigned k=0;k<records;++k){
        EllipsePerimeterRequest q;
        q.row=read<Int>(f);q.column=read<Int>(f);q.row_radius=read<Int>(f);q.column_radius=read<Int>(f);
        q.orientation=read<double>(f);q.clipped=read<std::uint8_t>(f)!=0;q.height=read<double>(f);q.width=read<double>(f);
        const auto scalar=read<std::uint8_t>(f)!=0;const auto count=read<std::uint64_t>(f);
        std::vector<Int> rows(count),columns(count);
        f.read(reinterpret_cast<char*>(rows.data()),count*sizeof(Int));f.read(reinterpret_cast<char*>(columns.data()),count*sizeof(Int));check(bool(f));
        EllipsePerimeter p;std::feclearexcept(FE_ALL_EXCEPT);
        check(prepare_ellipse_perimeter(q,p));check(ellipse_perimeter_count(p)==count&&p.scalar_result==scalar);
        check(p.rows==rows&&p.columns==columns);check(!(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW)));
        for(int workers:{1,4}){
            skimagexx::set_num_threads(workers);const int snapshot=skimagexx::get_num_threads();
            std::vector<Int> r(count+2,9999),c(count+2,9999);
            std::feraiseexcept(FE_DIVBYZERO);const int flags=std::fetestexcept(FE_ALL_EXCEPT);
            ellipse_perimeter_fill(p,{r.data()+1,c.data()+1,1,1},snapshot);
            check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
            for(std::size_t i=0;i<count;++i){check(r[i+1]==rows[i]);check(c[i+1]==columns[i]);}
            check(r.front()==9999&&r.back()==9999&&c.front()==9999&&c.back()==9999);
            std::vector<Int> rg(3*count+2,9999),cg(3*count+2,9999);
            const auto offset=count?1+3*(count-1):1;
            ellipse_perimeter_fill(p,{rg.data()+1,cg.data()+offset,3,-3},snapshot);
            for(std::size_t i=0;i<count;++i){check(rg[1+3*i]==rows[i]);check(cg[offset-3*i]==columns[i]);}
            for(std::size_t i=0;i<rg.size();++i)if(i%3!=1||i==rg.size()-1){check(rg[i]==9999);check(cg[i]==9999);}
        }
        auto copy=p;if(!copy.rows.empty()){copy.rows[0]=999;check(p.rows==rows);}
    }
    EllipsePerimeter large;EllipsePerimeterRequest q;q.row_radius=q.column_radius=32768;
    check(prepare_ellipse_perimeter(q,large));const auto n=ellipse_perimeter_count(large);
    check(2*n>=262144&&n%4==0);check(!large.scalar_result);
    for(std::size_t i=0;i<n;i+=4){
        check(large.rows[i]==large.rows[i+1]&&large.rows[i+2]==-large.rows[i]&&large.rows[i+3]==large.rows[i+2]);
        check(large.columns[i]==-large.columns[i+1]&&large.columns[i+1]==large.columns[i+2]&&large.columns[i+3]==large.columns[i]);
        const auto square=large.rows[i]*large.rows[i]+large.columns[i]*large.columns[i];
        check(std::abs(square-Int(32768)*32768)<=3*32768+2);
    }
    for(int workers:{1,4}){
        skimagexx::set_num_threads(workers);const int snapshot=skimagexx::get_num_threads();
        std::vector<Int> rows(n),columns(n);ellipse_perimeter_fill(large,{rows.data(),columns.data(),1,1},snapshot);
        check(rows==large.rows&&columns==large.columns);
    }
    const auto saved_rows=large.rows,saved_columns=large.columns;
    for(auto bad:{-1,32769}){q.row_radius=bad;check(!prepare_ellipse_perimeter(q,large));check(large.rows==saved_rows&&large.columns==saved_columns);}
    q={};q.row_radius=2;q.column_radius=4;q.orientation=.3;check(!prepare_ellipse_perimeter(q,large));
    q={};q.row_radius=1;q.column_radius=1;q.row=1;q.orientation=.3;check(!prepare_ellipse_perimeter(q,large));
    q={};q.row_radius=1;q.column_radius=1;q.orientation=std::numeric_limits<double>::infinity();check(!prepare_ellipse_perimeter(q,large));
    q={};q.row_radius=1;q.column_radius=1;q.row=std::numeric_limits<Int>::max();check(!prepare_ellipse_perimeter(q,large));
    q={};q.row_radius=1;q.column_radius=1;q.clipped=true;q.height=std::numeric_limits<double>::quiet_NaN();check(!prepare_ellipse_perimeter(q,large));
    check(large.rows==saved_rows&&large.columns==saved_columns);
    q={};q.row=3;q.column=4;q.clipped=true;q.height=q.width=0;
    EllipsePerimeter point;check(prepare_ellipse_perimeter(q,point)&&point.scalar_result&&point.rows==std::vector<Int>{3}&&point.columns==std::vector<Int>{4});
    for(double angle:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
        q.orientation=angle;check(prepare_ellipse_perimeter(q,point)&&point.scalar_result&&point.rows==std::vector<Int>{3}&&point.columns==std::vector<Int>{4});
    }
    q.orientation=0;
    q.row_radius=q.column_radius=1;EllipsePerimeter empty;
    check(prepare_ellipse_perimeter(q,empty)&&!empty.scalar_result&&ellipse_perimeter_count(empty)==0);
    ellipse_perimeter_fill(empty,{nullptr,nullptr,1,1},1);
    skimagexx::set_num_threads(old);check(skimagexx::get_num_threads()==old);
    std::cout<<"{\"records\":"<<records<<",\"controls\":"<<checks<<",\"large_points\":"<<n<<",\"coordinate_words\":"<<2*n<<",\"workers\":[1,4],\"default_restored\":true}"<<std::endl;
}
