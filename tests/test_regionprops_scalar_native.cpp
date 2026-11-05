#include <skimagexx/regionprops_state.hpp>
#include <skimagexx/image.hpp>
#include <cfenv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool ok){++checks;if(!ok)throw std::runtime_error("RegionProperties scalar control failed at "+std::to_string(checks));}
template<class T>T read(std::ifstream& f){T value{};f.read(reinterpret_cast<char*>(&value),sizeof(value));require(bool(f));return value;}
double value(std::uint64_t bits){double x;std::memcpy(&x,&bits,8);return x;}
struct Row{std::uint8_t op;std::uint64_t first,second,expected;unsigned flags;};
template<class F>void reject(F call){bool refused=false;try{call();}catch(const std::invalid_argument&){refused=true;}require(refused);}
void scalar_entry_controls(){
 std::fenv_t original;require(std::fegetenv(&original)==0);
 const auto initial_round=std::fegetround(),initial_flags=std::fetestexcept(FE_ALL_EXCEPT);
 const auto workers=skimagexx::get_num_threads();
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 const auto initial_csr=_mm_getcsr();
 #endif
 require(std::fesetround(FE_TONEAREST)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_OVERFLOW|FE_DIVBYZERO);
 const auto nearest_flags=std::fetestexcept(FE_ALL_EXCEPT);
 const auto scaled=regionprops_scaled_count_f64(3,2.);const auto ratio=regionprops_extent_f64(6.,2.);
 require(scaled.bits==UINT64_C(0x4018000000000000)&&scaled.fp_status==0);
 require(ratio.bits==UINT64_C(0x4008000000000000)&&ratio.fp_status==0);
 require(std::fegetround()==FE_TONEAREST&&std::fetestexcept(FE_ALL_EXCEPT)==nearest_flags);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 const auto nearest_csr=_mm_getcsr();
 for(unsigned direction:{0x2000u,0x4000u,0x6000u}){
  const auto requested=(nearest_csr&~0x6000u)|direction;_mm_setcsr(requested);
  const auto rounding=std::fegetround(),flags=std::fetestexcept(FE_ALL_EXCEPT);
  // glibc may still report x87-nearest; UCRT may report the SSE direction.
  // Either reporting contract must be preserved by refusal before arithmetic.
  reject([]{regionprops_scaled_count_f64(1,1.);});
  require(_mm_getcsr()==requested&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
  reject([]{regionprops_extent_f64(1.,1.);});
  require(_mm_getcsr()==requested&&std::fegetround()==rounding&&std::fetestexcept(FE_ALL_EXCEPT)==flags);
  require(skimagexx::get_num_threads()==workers);
 }
 _mm_setcsr(nearest_csr);
 #endif
 require(std::fesetenv(&original)==0);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 _mm_setcsr(initial_csr);require(_mm_getcsr()==initial_csr);
 #endif
 require(std::fegetround()==initial_round&&std::fetestexcept(FE_ALL_EXCEPT)==initial_flags&&skimagexx::get_num_threads()==workers);
}
}
int main(int argc,char** argv){
 require(argc==2);const auto initial_workers=skimagexx::get_num_threads();
 for(int workers:{1,4}){skimagexx::set_num_threads(workers);scalar_entry_controls();}
 skimagexx::set_num_threads(initial_workers);
 if(std::string(argv[1])=="--fp-entry-only"){std::cout<<"PASS scalar-FP-entry checks="<<checks<<" workers=1,4 restored="<<skimagexx::get_num_threads()<<" AVX2="<<skimagexx::has_avx2()<<"\n";return 0;}
 std::ifstream f(argv[1],std::ios::binary);char magic[8]{};f.read(magic,8);require(std::memcmp(magic,"RPS64V1",8)==0);
 const auto n=read<std::uint32_t>(f);std::vector<Row> rows;
 for(std::size_t i=0;i<n;++i){Row r;r.op=read<std::uint8_t>(f);r.first=read<std::uint64_t>(f);r.second=read<std::uint64_t>(f);r.expected=read<std::uint64_t>(f);r.flags=read<std::uint32_t>(f);rows.push_back(r);}require(f.peek()==std::ifstream::traits_type::eof());
 const int old_workers=skimagexx::get_num_threads();std::fenv_t original;require(std::fegetenv(&original)==0);
 for(int workers:{1,4}){
  skimagexx::set_num_threads(workers);
  for(const auto& r:rows){
   std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_OVERFLOW|FE_DIVBYZERO);const int before=std::fetestexcept(FE_ALL_EXCEPT);
   const auto out=r.op==0?regionprops_scaled_count_f64(std::size_t(r.first),value(r.second)):regionprops_extent_f64(value(r.first),value(r.second));
   require(out.bits==r.expected);require(out.fp_status==r.flags);require(std::fetestexcept(FE_ALL_EXCEPT)==before);require(std::fegetround()==FE_TONEAREST);require(skimagexx::get_num_threads()==workers);
  }
 }
 // Existing population primitive is the sole area reduction; include noncanonical
 // nonzero bytes, empty masks and guard bytes without duplicating its loop.
 const std::uint8_t mask[]={91,0,1,0,255,1,0,91};require(regionprops_mask_population(mask+1,6)==3);require(mask[0]==91&&mask[7]==91);require(regionprops_mask_population(nullptr,0)==0);
 for(auto word:{UINT64_C(0x8000000000000000),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff0000000000001),UINT64_C(1)}){
  std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_OVERFLOW);const int before=std::fetestexcept(FE_ALL_EXCEPT);
  reject([&]{regionprops_scaled_count_f64(1,value(word));});require(std::fetestexcept(FE_ALL_EXCEPT)==before);
  reject([&]{regionprops_extent_f64(value(word),1.);});require(std::fetestexcept(FE_ALL_EXCEPT)==before);
 }
 reject([]{regionprops_scaled_count_f64(std::size_t(UINT64_C(9007199254740993)),1.);});
 reject([]{regionprops_scaled_count_f64(1,0x1p257);});reject([]{regionprops_extent_f64(0x1p310,1.);});
 std::fesetround(FE_DOWNWARD);reject([]{regionprops_scaled_count_f64(1,1.);});reject([]{regionprops_extent_f64(1.,1.);});require(std::fegetround()==FE_DOWNWARD);std::fesetround(FE_TONEAREST);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 const auto mxcsr=_mm_getcsr();_mm_setcsr(mxcsr|(1u<<15)|(1u<<6));reject([]{regionprops_scaled_count_f64(1,1.);});reject([]{regionprops_extent_f64(1.,1.);});require(_mm_getcsr()==(mxcsr|(1u<<15)|(1u<<6)));_mm_setcsr(mxcsr);
 #endif
 require(std::fesetenv(&original)==0);skimagexx::set_num_threads(old_workers);require(skimagexx::get_num_threads()==old_workers);
 std::cout<<"PASS records="<<n<<" checks="<<checks<<" workers=1,4 restored="<<old_workers<<" serial-f64 exact-bits/fenv\n";
}
