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
int main(){tiny_derivative();volume<float>();volume<double>();check<float>(13,17);check<double>(13,17);check<float>(513,519);check<double>(513,519);
 require(hessian_matrix_into(nullptr,PixelType::float64,nullptr,{0,3},{{},{}},true));set_num_threads(1);std::cout<<"Hessian native checks passed\n";
}
