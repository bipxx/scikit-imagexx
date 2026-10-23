#include "skimagexx/shape_index.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace skimagexx;
void require(bool value){if(!value)throw std::runtime_error("native shape-index check failed");}
template<class T> void run(std::size_t count){
 std::vector<T> a(count),b(count),c(count),out(count),serial;
 for(std::size_t i=0;i<count;++i){a[i]=T(double(int(i%101)-50)/13);b[i]=0;c[i]=T(double(int(i%71)-35)/17);}
 const void* fields[]={a.data(),b.data(),c.data()};set_num_threads(1);require(shape_index_from_hessian_into(fields,out.data(),count,sizeof(T)==4));
 const T factor=T(2/0x1.921fb54442d18p+1);
 for(std::size_t i=0;i<count;++i){const T mean=T(T(a[i]+c[i])/T(2)),half=T(T(a[i]-c[i])/T(2)),root=std::sqrt(T(half*half));
  const T upper=T(mean+root),lower=T(mean-root),ratio=T(T(lower+upper)/T(lower-upper)),expected=T(factor*T(std::atan(ratio)));
  if(std::isnan(expected))require(std::isnan(out[i]));
  else{require(std::abs(double(out[i])-double(expected))<(sizeof(T)==4 ? 3e-7 : 3e-14));if(expected==T(0))require(std::signbit(out[i])==std::signbit(expected));}
 }
 serial=out;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(shape_index_from_hessian_into(fields,out.data(),count,sizeof(T)==4));
 for(std::size_t i=0;i<count;++i)require(out[i]==serial[i] || (std::isnan(out[i]) && std::isnan(serial[i])));
 a[0]=T(NAN);require(!shape_index_from_hessian_into(fields,out.data(),count,sizeof(T)==4));
}
int main(){run<float>(19);run<double>(19);run<float>(266247);run<double>(266247);require(shape_index_from_hessian_into(nullptr,nullptr,0,false));
 bool overflow=false;try{shape_index_from_hessian_into(nullptr,nullptr,std::numeric_limits<std::size_t>::max(),false);}catch(const std::overflow_error&){overflow=true;}require(overflow);
 set_num_threads(1);std::cout<<"Shape-index native checks passed\n";
}
