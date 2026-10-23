#include "skimagexx/hessian.hpp"
#include "skimagexx/image.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <iostream>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("Hessian native check failed");}
template<class T> void check(std::size_t h,std::size_t w){
 const auto count=h*w;std::vector<T> image(count),a(count),b(count),c(count),pa(count),pb(count),pc(count);
 for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x)image[y*w+x]=T(y*y+x*x+y*x);
 void* out[]={a.data(),b.data(),c.data()};const auto type=sizeof(T)==4 ? PixelType::float32 : PixelType::float64;
 set_num_threads(1);require(hessian_matrix_into(image.data(),type,out,{h,w},{{}},false));
 for(std::size_t y=2;y+2<h;++y)for(std::size_t x=2;x+2<w;++x){const auto i=y*w+x;require(a[i]==T(2) && b[i]==T(1) && c[i]==T(2));}
 void* parallel[]={pa.data(),pb.data(),pc.data()};
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(hessian_matrix_into(image.data(),type,parallel,{h,w},{{}},false));require(a==pa && b==pb && c==pc);
 const HessianPass first{{0,{-.5,0,.5},GaussianBorder::nearest,0},true},second{{1,{-.5,0,.5},GaussianBorder::nearest,0},true};
 require(hessian_matrix_into(image.data(),type,out,{h,w},{{first},{second}},true));
 for(std::size_t y=2;y+2<h;++y)for(std::size_t x=2;x+2<w;++x){const auto i=y*w+x;require(a[i]==T(2) && b[i]==T(1) && c[i]==T(2));}
 image[0]=T(NAN);require(!hessian_matrix_into(image.data(),type,out,{h,w},{{}},false));
}
template<class T> void volume(){
 const std::size_t h=7,w=9,d=11,n=h*w*d;std::vector<T> input(n);
 std::vector<std::vector<T>> result(6,std::vector<T>(n));std::vector<void*> outputs;for(auto& field:result)outputs.push_back(field.data());
 for(std::size_t a=0;a<h;++a)for(std::size_t b=0;b<w;++b)for(std::size_t c=0;c<d;++c)input[(a*w+b)*d+c]=T(a*a+b*b+c*c);
 require(hessian_matrix_into(input.data(),sizeof(T)==4 ? PixelType::float32 : PixelType::float64,outputs.data(),{h,w,d},{{}},false));
 for(std::size_t a=2;a+2<h;++a)for(std::size_t b=2;b+2<w;++b)for(std::size_t c=2;c+2<d;++c){const auto i=(a*w+b)*d+c;
  for(std::size_t f=0;f<6;++f)require(result[f][i]==T(f==0 || f==3 || f==5 ? 2 : 0));}
}
void tiny_derivative(){
 std::vector<double> input(19,-1.),output(19);void* fields[]={output.data()};
 const HessianPass zero{{0,{-0.,0.,0.},GaussianBorder::reflect,0},true};
 require(hessian_matrix_into(input.data(),PixelType::float64,fields,{19},{{zero}},true));
 for(auto x:output)require(x==0. && !std::signbit(x));
 const HessianPass tiny{{0,{-1e-40,0.,1e-40},GaussianBorder::reflect,0},true};
 require(hessian_matrix_into(input.data(),PixelType::float64,fields,{19},{{tiny}},true));
 for(auto x:output)require(std::abs(x+4e-80)<1e-94);
}
template<class T> void repeated(){
 const std::size_t h=513,w=519,n=h*w;std::vector<T> input(n),g0(n),g1(n),expected0(n),expected1(n),expected2(n),a(n),b(n),c(n),serial0,serial1,serial2;
 for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x)input[y*w+x]=y<100 ? T(-1) : y<200 ? T(-0.) : y<350 ? T(2) : T(0);
 auto pass=[&](const std::vector<T>& from,std::vector<T>& to,int axis){
  for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){const auto lo=axis==0 ? (y ? y-1 : 0)*w+x : y*w+(x ? x-1 : 0),hi=axis==0 ? (y+1<h ? y+1 : y)*w+x : y*w+(x+1<w ? x+1 : x);
   double sum=double(from[y*w+x])*0.;sum+=(double(from[lo])-double(from[hi]))*-.5;to[y*w+x]=T(sum);}
 };
 pass(input,g0,0);pass(input,g1,1);pass(g0,expected0,0);pass(g0,expected1,1);pass(g1,expected2,1);
 const HessianPass first{{0,{-.5,0.,.5},GaussianBorder::nearest,0},true},second{{1,{-.5,0.,.5},GaussianBorder::nearest,0},true};void* outputs[]={a.data(),b.data(),c.data()};
 set_num_threads(1);require(hessian_matrix_into(input.data(),sizeof(T)==4 ? PixelType::float32 : PixelType::float64,outputs,{h,w},{{first},{second}},true));
 auto same=[&](const std::vector<T>& x,const std::vector<T>& y){for(std::size_t i=0;i<n;++i)require(x[i]==y[i] && (x[i]!=T(0) || std::signbit(x[i])==std::signbit(y[i])));};
 same(a,expected0);same(b,expected1);same(c,expected2);serial0=a;serial1=b;serial2=c;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(hessian_matrix_into(input.data(),sizeof(T)==4 ? PixelType::float32 : PixelType::float64,outputs,{h,w},{{first},{second}},true));same(a,serial0);same(b,serial1);same(c,serial2);
}
void zero_sigma(){
 std::vector<std::uint8_t> input(19*23,128);std::vector<double> a(input.size()),b(input.size()),c(input.size());void* outputs[]={a.data(),b.data(),c.data()};
 require(hessian_matrix_into(input.data(),PixelType::uint8,outputs,{19,23},{{},{}},true));
 for(std::size_t i=0;i<input.size();++i)require(a[i]==128.*(1./255.) && b[i]==a[i] && c[i]==a[i]);
 std::vector<float> invalid(input.size(),0);invalid[0]=NAN;std::vector<float> fa(input.size()),fb(input.size()),fc(input.size());void* floats[]={fa.data(),fb.data(),fc.data()};
 require(!hessian_matrix_into(invalid.data(),PixelType::float32,floats,{19,23},{{},{}},true));
}
void padded_constant_precision(){
 std::vector<float> input{.11f,.23f,.37f,.41f,.53f,.67f,.79f},scratch(input.size()),expected(input.size()),actual(input.size());
 auto pass=[&](const std::vector<float>& from,std::vector<float>& to){for(std::size_t x=0;x<from.size();++x){const double left=x ? double(from[x-1]) : -.3,right=x+1<from.size() ? double(from[x+1]) : -.3;double sum=double(from[x])*.5;sum+=(left+right)*.25;to[x]=float(sum);}};
 pass(input,scratch);pass(scratch,expected);void* outputs[]={actual.data()};
 const HessianPass smoothing{{0,{.25,.5,.25},GaussianBorder::constant,-.3},false};
 require(hessian_matrix_into(input.data(),PixelType::float32,outputs,{input.size()},{{smoothing}},true));require(actual==expected);
}
int main(){padded_constant_precision();zero_sigma();repeated<float>();repeated<double>();tiny_derivative();volume<float>();volume<double>();check<float>(13,17);check<double>(13,17);check<float>(513,519);check<double>(513,519);
 require(hessian_matrix_into(nullptr,PixelType::float64,nullptr,{0,3},{{},{}},true));set_num_threads(1);std::cout<<"Hessian native checks passed\n";
}
