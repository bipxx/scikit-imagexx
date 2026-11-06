#include <skimagexx/draw_set_color.hpp>
#include <skimagexx/image.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
std::size_t integer_stage_controls(){
 using namespace skimagexx;using namespace skimagexx::detail;const auto previous=get_num_threads();std::size_t checks=0;
 for(int workers:{1,4}){
  set_num_threads(workers);double image=-0.,color=0.,alpha=-1.;std::intptr_t zero=0;
  SetColorView v{&image,{1,1,1},{8,8,0},&zero,&zero,1,&color,&alpha};v.integer_product=true;
  assert(draw_set_color(v,workers)&&image==0&&!std::signbit(image));++checks;
  image=-0.;v.integer_product=false;assert(draw_set_color(v,workers)&&std::signbit(image));++checks;
  v.integer_product=true;image=.5;color=16.;alpha=-16.;assert(draw_set_color(v,workers)&&image==-247.5);++checks;
  image=-0.;color=.5;alpha=-1.;const auto before=image;
  std::fenv_t saved;std::fegetenv(&saved);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
  assert(!draw_set_color(v,workers)&&std::signbit(image)==std::signbit(before)&&std::fetestexcept(FE_ALL_EXCEPT)==flags);std::fesetenv(&saved);++checks;
  color=0.;alpha=.5;assert(!draw_set_color(v,workers)&&std::signbit(image));++checks;
  const std::size_t n=262145;std::vector<double> pixels(n,-0.),alphas(n,-1.);std::vector<std::intptr_t> rows(n,0),cols(n);for(std::size_t i=0;i<n;++i)cols[i]=std::intptr_t(i);
  SetColorView large{pixels.data(),{1,n,1},{std::ptrdiff_t(n*8),8,0},rows.data(),cols.data(),n,&color,alphas.data()};large.integer_product=true;
  assert(draw_set_color(large,workers));for(double value:pixels){assert(value==0&&!std::signbit(value));++checks;}
 }
 set_num_threads(previous);assert(get_num_threads()==1);return checks;
}
std::size_t float32_controls(){
 using namespace skimagexx;using namespace skimagexx::detail;const auto previous=get_num_threads();std::size_t checks=0;
 for(int workers:{1,4}){
  set_num_threads(workers);float pixel=.5f;double color=.25,alpha=.5;std::intptr_t zero=0;
  SetColorView v{&pixel,{1,1,1},{4,4,0},&zero,&zero,1,&color,&alpha};v.image_float32=true;
  assert(draw_set_color(v,workers)&&pixel==.375f);++checks;
  pixel=-0.f;color=-1.;alpha=0.;v.integer_product=true;assert(draw_set_color(v,workers)&&!std::signbit(pixel));++checks;
  pixel=-0.f;v.integer_product=false;assert(draw_set_color(v,workers)&&std::signbit(pixel));++checks;
  // Inexact f64 stages precede refused f32underflow cast: preservecallerflags.
  pixel=0.f;color=std::nextafter(std::ldexp(1.,-128),1.);alpha=color;
  std::fenv_t saved;std::fegetenv(&saved);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
  assert(!draw_set_color(v,workers)&&pixel==0.f&&std::fetestexcept(FE_ALL_EXCEPT)==flags);std::fesetenv(&saved);++checks;
  std::uint32_t snan=0x7f800001u;std::memcpy(&pixel,&snan,4);color=.25;alpha=.5;std::fegetenv(&saved);const auto before=std::fetestexcept(FE_ALL_EXCEPT);assert(!draw_set_color(v,workers)&&std::fetestexcept(FE_ALL_EXCEPT)==before);std::fesetenv(&saved);++checks;
  const std::size_t n=262145;std::vector<float> pixels(n,.5f);std::vector<double> alphas(n,.5);std::vector<std::intptr_t> rows(n,0),cols(n);for(std::size_t i=0;i<n;++i)cols[i]=std::intptr_t(i);
  SetColorView large{pixels.data(),{1,n,1},{std::ptrdiff_t(n*4),4,0},rows.data(),cols.data(),n,&color,alphas.data()};large.image_float32=true;
  assert(draw_set_color(large,workers));for(float x:pixels){assert(x==.375f);++checks;}
 }
 set_num_threads(previous);assert(get_num_threads()==1);return checks;
}
int main(int argc,char**argv){
 const auto f32_checks=float32_controls();
 if(argc==2&&std::string(argv[1])=="--float32-only"){std::cout<<f32_checks<<" float32controls PASS default1 restored\n";return 0;}

 const auto integer_checks=integer_stage_controls();
 if(argc==2&&std::string(argv[1])=="--integer-stage-only"){std::cout<<integer_checks<<" integerstage controls PASS default1 restored\n";return 0;}

 using namespace skimagexx;using namespace skimagexx::detail;
 const auto old=get_num_threads();std::size_t checks=integer_checks+f32_checks;
 for(int workers:{1,4}){
  set_num_threads(workers);
  double image[4]={.5,.75,.25,1.};std::intptr_t rr[3]={0,0,1},cc[3]={0,0,1};double color=.25,alpha[3]={.25,.5,1.};
  SetColorView v{image,{2,2,1},{16,8,0},rr,cc,3,&color,alpha};
  assert(draw_set_color(v,get_num_threads()));assert(image[0]==.375&&image[3]==.25);checks+=3;
  const auto initial=image[0];color=std::nan("");assert(!draw_set_color(v,workers)&&image[0]==initial);++checks;color=.25;
  const std::size_t n=262145;std::vector<double> large(n,.5),a(n,.5);std::vector<std::intptr_t> r(n,0),c(n);for(std::size_t i=0;i<n;++i)c[i]=std::intptr_t(i);
  SetColorView big{large.data(),{1,n,1},{std::ptrdiff_t(n*8),8,0},r.data(),c.data(),n,&color,a.data()};
  assert(draw_set_color(big,workers));for(double x:large){assert(x==.375);++checks;}
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
  const auto csr=_mm_getcsr();_mm_setcsr((csr&~(3u<<13))|(1u<<13));assert(!draw_set_color(v,workers));assert(_mm_getcsr()==((csr&~(3u<<13))|(1u<<13)));_mm_setcsr(csr);checks+=2;
#endif
 }
 set_num_threads(old);assert(get_num_threads()==1);std::cout<<checks<<" controls PASS default1 restored\n";
}
