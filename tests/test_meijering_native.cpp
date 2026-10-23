#include "skimagexx/meijering.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("Meijering native check failed");}
template<class T> void check(std::size_t n,std::size_t count){
 std::vector<std::vector<T>> fields(n*(n+1)/2,std::vector<T>(count));std::vector<const void*> pointers;
 std::size_t f=0;for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){for(std::size_t i=0;i<count;++i)fields[f][i]=r==c ? T(double(n-r)*double(i%17+1)) : T(0);pointers.push_back(fields[f++].data());}
 std::vector<T> output(count),previous(count,T(.25)),serial;const bool single=sizeof(T)==4;
 set_num_threads(1);require(meijering_accumulate_into(pointers.data(),previous.data(),output.data(),count,n,single,0.));
 for(std::size_t i=0;i<count;++i){const T expected=std::max(T(.25),T(T(n*T(i%17+1))/T(n*17)));require(std::abs(double(output[i])-double(expected))<(single ? 1e-6 : 1e-13));}serial=output;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(meijering_accumulate_into(pointers.data(),previous.data(),output.data(),count,n,single,0.));require(output==serial);
 previous[0]=T(INFINITY);require(!meijering_accumulate_into(pointers.data(),previous.data(),output.data(),count,n,single,0.));previous[0]=T(.25);
 if(n==2){std::fill(fields[0].begin(),fields[0].end(),T(3));std::fill(fields[1].begin(),fields[1].end(),T(0));std::fill(fields[2].begin(),fields[2].end(),T(-3));
  require(meijering_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,0.));for(auto x:output)require(x==T(1));
  std::fill(fields[2].begin(),fields[2].end(),T(-4));require(meijering_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,0.));for(auto x:output)require(x==T(0));}
 fields[0][0]=T(NAN);require(!meijering_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,0.));
}
int main(){for(std::size_t n=1;n<=5;++n){check<float>(n,19);check<double>(n,19);}check<float>(3,266247);check<double>(3,266247);
 require(meijering_accumulate_into(nullptr,nullptr,nullptr,0,1,false,0.));
 bool overflow=false;try{meijering_accumulate_into(nullptr,nullptr,nullptr,std::numeric_limits<std::size_t>::max(),5,false,0.);}catch(const std::overflow_error&){overflow=true;}require(overflow);
 set_num_threads(1);std::cout<<"Meijering native checks passed\n";
}
