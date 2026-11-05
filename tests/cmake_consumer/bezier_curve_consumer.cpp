// Installed SDK component scaffold: only installed headers and exported target.
#include <skimagexx/bezier_curve.hpp>
#include <skimagexx/image.hpp>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace skimagexx::detail;
template<class T>static T read(std::ifstream& f){T value{};f.read(reinterpret_cast<char*>(&value),sizeof(value));if(!f)throw std::runtime_error("Bezier SDK fixture truncated");return value;}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    const auto available=skimagexx::has_avx2();const char* disabled=std::getenv("SKIMAGEXX_DISABLE_SIMD");
    if(disabled&&std::strcmp(disabled,"1")==0&&available)throw std::runtime_error("disabled dispatch");
    if(std::string(argv[2])!="auto"&&available!=(std::string(argv[2])=="1"))throw std::runtime_error("dispatch mismatch");
    const auto old=skimagexx::get_num_threads();if(old!=1)throw std::runtime_error("default budget");
    std::size_t checks=0;
    std::ifstream f(argv[1],std::ios::binary);const auto records=read<std::uint32_t>(f);
    if(records!=28)throw std::runtime_error("Bezier SDK original count");
    for(std::uint32_t k=0;k<records;++k) {
        BezierCurveRequest saved;
        saved.r0=read<std::intptr_t>(f);saved.c0=read<std::intptr_t>(f);saved.r1=read<std::intptr_t>(f);saved.c1=read<std::intptr_t>(f);saved.r2=read<std::intptr_t>(f);saved.c2=read<std::intptr_t>(f);
        saved.weight=read<double>(f);saved.clipped=read<std::uint8_t>(f)!=0;saved.height=read<double>(f);saved.width=read<double>(f);
        const auto n=read<std::uint64_t>(f);if(n>2048)throw std::runtime_error("Bezier SDK original bound");
        std::vector<std::intptr_t> r(n),c(n);
        f.read(reinterpret_cast<char*>(r.data()),n*sizeof(std::intptr_t));f.read(reinterpret_cast<char*>(c.data()),n*sizeof(std::intptr_t));if(!f)throw std::runtime_error("Bezier SDK original words");
        BezierCurve prepared;const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        if(!prepare_bezier_curve(saved,prepared)||prepared.rows!=r||prepared.columns!=c||std::fetestexcept(FE_ALL_EXCEPT)!=flags)throw std::runtime_error("Bezier SDK saved prepare");
        for(int workers:{1,4}) {
            skimagexx::set_num_threads(workers);const auto snapshot=skimagexx::get_num_threads();
            std::vector<std::intptr_t> rr(n),cc(n);
            bezier_curve_fill(prepared,{rr.data(),cc.data(),1,1},snapshot);
            if(rr!=r||cc!=c||skimagexx::get_num_threads()!=workers)throw std::runtime_error("Bezier SDK saved fill");
            checks+=2*n+3;
        }
    }
    if(f.peek()!=std::ifstream::traits_type::eof())throw std::runtime_error("Bezier SDK fixture tail");
    BezierCurve p;BezierCurveRequest q{0,0,65536,65536,131072,131072,1.};
    if(!prepare_bezier_curve(q,p)||bezier_curve_count(p)!=131073)throw std::runtime_error("analytic line");
    for(int workers:{1,4}) {
        skimagexx::set_num_threads(workers);const auto snapshot=skimagexx::get_num_threads();
        std::vector<std::intptr_t> rows(p.rows.size()),columns(p.rows.size());
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        bezier_curve_fill(p,{rows.data(),columns.data(),1,1},snapshot);
        if(std::fetestexcept(FE_ALL_EXCEPT)!=flags)throw std::runtime_error("FP state");
        for(std::size_t i=0;i<rows.size();++i){if(rows[i]!=std::intptr_t(i)||columns[i]!=std::intptr_t(i))throw std::runtime_error("analytic words");checks+=2;}
    }
    skimagexx::set_num_threads(old);if(skimagexx::get_num_threads()!=old)throw std::runtime_error("restore budget");
    std::cout<<"Bezier installed controls="<<checks<<" saved_records="<<records<<" available="<<available<<" default="<<old<<'\n';
}
