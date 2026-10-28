#include "resample_detail.hpp"
#include "execution.hpp"
#include "profile_coordinates_golden.hpp"
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
using skimagexx::detail::profile_coordinates_into;
namespace {
std::size_t checks=0;std::ofstream journal;
void check(bool value){++checks;if(!value)std::fprintf(stderr,"coordinate check %zu failed\n",checks);assert(value);}
double word(std::uint64_t bits){double value;std::memcpy(&value,&bits,8);return value;}
void goldens(int workers){
    for(const auto& saved:coordinate_cases){
        std::vector<double> rows(saved.length),cols(saved.length);
        std::memcpy(rows.data(),coordinate_words+saved.row,rows.size()*8);
        std::memcpy(cols.data(),coordinate_words+saved.col,cols.size()*8);
        const auto before_rows=rows,before_cols=cols;
        const auto count=2*saved.length*saved.width;
        std::vector<double> output(count+2,-777.);
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        const bool accepted=profile_coordinates_into(rows.data(),cols.data(),saved.length,saved.width,
            word(saved.row_width),word(saved.col_width),output.data()+1,workers);
        if(!accepted)std::fprintf(stderr,"refused golden %s\n",saved.id);
        check(accepted);check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
        check(output.front()==-777.&&output.back()==-777.);
        check(std::memcmp(output.data()+1,coordinate_words+saved.result,count*8)==0);
        check(std::memcmp(rows.data(),before_rows.data(),rows.size()*8)==0);
        check(std::memcmp(cols.data(),before_cols.data(),cols.size()*8)==0);
        journal.write(reinterpret_cast<const char*>(output.data()+1),std::streamsize(count*8));
    }
}
void tails_and_workers(int workers){
    // Exact dyadic/integer geometry independent of the implementation formula.
    for(std::size_t width:{1u,2u,3u,4u,5u,7u,8u,9u,15u,16u,17u,31u,32u,33u,65u}){
        const std::size_t length=width==33?8193:17;
        std::vector<double> rows(length),cols(length),output(2*length*width+2,-777.);
        for(std::size_t i=0;i<length;++i){rows[i]=double(i*2);cols[i]=double(i*3);}
        const double radius=double(width-1)/2.;
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        check(profile_coordinates_into(rows.data(),cols.data(),length,width,radius,-radius,output.data()+1,workers));
        check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
        check(output.front()==-777.&&output.back()==-777.);
        for(std::size_t i=0;i<length;++i)for(std::size_t j=0;j<width;++j){
            check(output[1+i*width+j]==rows[i]-radius+double(j));
            check(output[1+(length+i)*width+j]==cols[i]+radius-double(j));
        }
        journal.write(reinterpret_cast<const char*>(output.data()+1),std::streamsize((output.size()-2)*8));
    }
}
void refusal(int workers){
    double rows[2]={0.,1.},cols[2]={0.,1.};std::vector<double> output(134,-777.);
    auto rejected=[&](double radius){
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        check(!profile_coordinates_into(rows,cols,2,33,radius,0.,output.data()+1,workers));
        check(std::fetestexcept(FE_ALL_EXCEPT)==FE_DIVBYZERO);
        check(output.front()==-777.&&output.back()==-777.);
    };
    for(auto bits:{UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
                   UINT64_C(0x7ff8000000000001),UINT64_C(0x7ff0000000000001)}){
        rows[1]=word(bits);rejected(1.);rows[1]=1.;rejected(word(bits));
    }
    // The source zero-step branch is required, but an actual underflow asks
    // the literal NumPy route to produce its own warning/error/callback.
    rejected(std::numeric_limits<double>::denorm_min());
    rejected(std::numeric_limits<double>::max());
    auto rounding=std::fegetround();check(std::fesetround(FE_DOWNWARD)==0);
    rejected(1.);check(std::fegetround()==FE_DOWNWARD);check(std::fesetround(rounding)==0);
    rows[1]=1.;check(!profile_coordinates_into(rows,cols,1,UINT64_C(9007199254740993),0.,0.,output.data()+1,workers));
}
void invalid_arguments(){
    double input=0.,output=0.;
    for(int workers:{0,1025}){
        bool threw=false;try{profile_coordinates_into(&input,&input,1,1,0.,0.,&output,workers);}
        catch(const std::invalid_argument&){threw=true;}check(threw);
    }
    for(int which=0;which<4;++which){
        bool threw=false;try{profile_coordinates_into(which==0?nullptr:&input,&input,which==1?0:1,which==2?0:1,0.,0.,which==3?nullptr:&output,1);}
        catch(const std::invalid_argument&){threw=true;}check(threw);
    }
    bool threw=false;try{profile_coordinates_into(&input,&input,std::numeric_limits<std::size_t>::max(),33,0.,0.,&output,1);}
    catch(const std::overflow_error&){threw=true;}check(threw);
#ifndef _OPENMP
    threw=false;try{profile_coordinates_into(&input,&input,1,1,0.,0.,&output,4);}
    catch(const std::invalid_argument&){threw=true;}check(threw);
#endif
}
}
int main(int argc,char** argv){
    check(argc==2);journal.open(argv[1],std::ios::binary);check(bool(journal));check(get_num_threads()==1);
    for(int workers:{1,4}){
#ifndef _OPENMP
        if(workers!=1)continue;
#endif
        set_num_threads(workers);goldens(workers);tails_and_workers(workers);refusal(workers);
    }
    set_num_threads(1);invalid_arguments();check(get_num_threads()==1);
    std::cout<<checks<<" profile coordinate controls PASS avx2="<<has_avx2()<<" workers="<<get_num_threads()<<"\n";
}
