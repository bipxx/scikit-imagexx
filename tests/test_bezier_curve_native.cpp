// Source-analytic controls plus saved-only fixture loader; never runs originals.
#include <skimagexx/bezier_curve.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
using namespace skimagexx::detail;
using Int=std::intptr_t;
static std::size_t controls=0;
static void check(bool value){++controls;if(!value)throw std::runtime_error("Bezier control "+std::to_string(controls));}
template<class T>static T read(std::ifstream& f){T value{};f.read(reinterpret_cast<char*>(&value),sizeof(value));if(!f)throw std::runtime_error("Bezier fixture truncated");return value;}
static void outputs(const BezierCurve& p,const std::vector<Int>& expected_r,const std::vector<Int>& expected_c) {
    const auto n=bezier_curve_count(p);
    check(n==expected_r.size()&&n==expected_c.size());
    for(int workers:{1,4}) {
        skimagexx::set_num_threads(workers);const int snapshot=skimagexx::get_num_threads();
        std::vector<Int> rows(3*n+2,777),columns(3*n+2,777);
        const auto offset=n?1+3*(n-1):1;
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        bezier_curve_fill(p,{rows.data()+1,columns.data()+offset,3,-3},snapshot);
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
        for(std::size_t i=0;i<n;++i){check(rows[1+3*i]==expected_r[i]);check(columns[offset-3*i]==expected_c[i]);}
        for(std::size_t i=0;i<rows.size();++i)if(i%3!=1||i==rows.size()-1){check(rows[i]==777);check(columns[i]==777);}
        check(skimagexx::get_num_threads()==workers);
    }
}
int main(int argc,char** argv) {
    const auto old=skimagexx::get_num_threads();check(old==1);
    if(argc!=2)throw std::invalid_argument("saved Bezier fixture path required; no implicit capture");
    std::ifstream f(argv[1],std::ios::binary);check(bool(f));
    const auto records=read<std::uint32_t>(f);
    check(records==28); // Exact original-success records; fixture hash frozen separately.
    for(std::uint32_t k=0;k<records;++k) {
        BezierCurveRequest q;
        q.r0=read<Int>(f);q.c0=read<Int>(f);q.r1=read<Int>(f);q.c1=read<Int>(f);q.r2=read<Int>(f);q.c2=read<Int>(f);
        q.weight=read<double>(f);q.clipped=read<std::uint8_t>(f)!=0;q.height=read<double>(f);q.width=read<double>(f);
        const auto n=read<std::uint64_t>(f);check(n<=2048);
        std::vector<Int> rows(n),columns(n);
        f.read(reinterpret_cast<char*>(rows.data()),n*sizeof(Int));f.read(reinterpret_cast<char*>(columns.data()),n*sizeof(Int));check(bool(f));
        BezierCurve p;std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
        const auto flags=std::fetestexcept(FE_ALL_EXCEPT);check(prepare_bezier_curve(q,p));check(std::fetestexcept(FE_ALL_EXCEPT)==flags);
        check(p.rows==rows&&p.columns==columns);outputs(p,rows,columns);
    }
    check(f.peek()==std::ifstream::traits_type::eof());
    // Bounded analytic straight diagonal crossing independent-fill threshold.
    BezierCurveRequest q{0,0,65536,65536,131072,131072,1.};BezierCurve p;
    check(prepare_bezier_curve(q,p));check(bezier_curve_count(p)==131073);
    for(std::size_t i=0;i<p.rows.size();++i){check(p.rows[i]==Int(i));check(p.columns[i]==Int(i));}
    outputs(p,p.rows,p.columns);
    const auto before_r=p.rows,before_c=p.columns;
    q.weight=-1;check(!prepare_bezier_curve(q,p));check(p.rows==before_r&&p.columns==before_c);
    q.weight=1;q.r0=(Int(1)<<20)+1;check(!prepare_bezier_curve(q,p));check(p.rows==before_r&&p.columns==before_c);
    q={0,0,1,4,6,6,1.};
    const auto rounding=std::fegetround();check(std::fesetround(FE_DOWNWARD)==0);
    check(!prepare_bezier_curve(q,p));check(std::fegetround()==FE_DOWNWARD);
    check(std::fesetround(rounding)==0);check(p.rows==before_r&&p.columns==before_c);
#if defined(__SSE__) || defined(_M_X64)
    const auto csr=_mm_getcsr();
    for(unsigned mask:{1u<<6,1u<<15,1u<<13}) {
        _mm_setcsr(csr|mask);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        check(!prepare_bezier_curve(q,p));check(_mm_getcsr()==(csr|mask));
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);_mm_setcsr(csr);
        check(p.rows==before_r&&p.columns==before_c);
    }
#endif
    skimagexx::set_num_threads(old);check(skimagexx::get_num_threads()==old);
    std::cout<<"Bezier native controls="<<controls<<" saved_records="<<records<<" workers=1,4 default="<<old<<'\n';
}
