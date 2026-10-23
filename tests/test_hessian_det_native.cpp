#include "skimagexx/hessian_det.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("Hessian determinant native check failed");}
template<class T> void exact(std::size_t n,std::size_t count){
 std::vector<std::vector<T>> fields(n*(n+1)/2,std::vector<T>(count));std::vector<const void*> pointers;
 double expected=1.;std::size_t field=0;
 for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){
  const T value=r==c ? T(r%2 ? -double(r+2) : double(r+2)) : T(0);
  if(r==c)expected*=value;std::fill(fields[field].begin(),fields[field].end(),value);pointers.push_back(fields[field++].data());
 }
 std::vector<T> out(count),serial;set_num_threads(1);
 require(hessian_det_from_fields_into(pointers.data(),out.data(),count,n,sizeof(T)==4));
 for(auto value:out)require(std::abs(double(value)-expected)<std::abs(expected)*(sizeof(T)==4 ? 1e-6 : 1e-13));serial=out;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(hessian_det_from_fields_into(pointers.data(),out.data(),count,n,sizeof(T)==4));require(out==serial);
 if(n==2){std::fill(fields[0].begin(),fields[0].end(),T(0));std::fill(fields[1].begin(),fields[1].end(),T(1));std::fill(fields[2].begin(),fields[2].end(),T(0));
  require(hessian_det_from_fields_into(pointers.data(),out.data(),count,n,sizeof(T)==4));for(auto value:out)require(value==T(-1));}
 for(auto& f:fields)std::fill(f.begin(),f.end(),T(1));
 require(hessian_det_from_fields_into(pointers.data(),out.data(),count,n,sizeof(T)==4));for(auto value:out)require(value==T(0) && !std::signbit(value));
 fields[0][0]=T(NAN);require(!hessian_det_from_fields_into(pointers.data(),out.data(),count,n,sizeof(T)==4));
}
int main(){
 for(std::size_t n=2;n<=5;++n){exact<float>(n,19);exact<double>(n,19);}exact<float>(3,266247);exact<double>(4,266247);
 std::vector<double> image(20),out(20);for(int i=0;i<20;++i)image[i]=i;
 require(hessian_det_approx_into(image.data(),PixelType::float64,out.data(),4,5,3));require(std::abs(out[0]-5./9)<1e-15);
 require(hessian_det_approx_into(image.data(),PixelType::float64,out.data(),4,5,4));require(out[0]==.703125);
 require(hessian_det_approx_into(image.data(),PixelType::float64,out.data(),4,5,0));for(auto value:out)require(std::isnan(value));
 require(hessian_det_approx_into(nullptr,PixelType::float64,nullptr,0,7,3));require(hessian_det_from_fields_into(nullptr,nullptr,0,2,false));
 bool error=false;try{hessian_det_from_fields_into(nullptr,nullptr,std::numeric_limits<std::size_t>::max(),2,false);}catch(const std::overflow_error&){error=true;}require(error);
 set_num_threads(1);std::cout<<"Hessian determinant native checks passed\n";
}
