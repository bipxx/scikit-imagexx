#include "skimagexx/sato.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace skimagexx;
void require(bool value){if(!value)throw std::runtime_error("Sato native check failed");}
template<class T> void check(std::size_t count,std::size_t n){
 std::vector<std::vector<T>> fields(n*(n+1)/2,std::vector<T>(count));std::vector<const void*> pointers;
 std::size_t f=0;for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){std::fill(fields[f].begin(),fields[f].end(),r==c ? T(r==0 ? 9 : r==1 ? 4 : 1) : T(0));pointers.push_back(fields[f++].data());}
 std::vector<T> result(count),prior(count,T(1)),serial;const bool single=sizeof(T)==4;
 set_num_threads(1);require(sato_accumulate_into(pointers.data(),prior.data(),result.data(),count,n,single,single,single,single,2.));
 const double expected=n==2 ? 18. : 12.;for(auto value:result)require(std::abs(double(value)-expected)<1e-12);serial=result;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(sato_accumulate_into(pointers.data(),prior.data(),result.data(),count,n,single,single,single,single,2.));require(result==serial);
 std::fill(prior.begin(),prior.end(),T(25));require(sato_accumulate_into(pointers.data(),prior.data(),result.data(),count,n,single,single,single,single,2.));for(auto value:result)require(value==T(25));
 for(auto& field:fields)std::fill(field.begin(),field.end(),T(1));
 if(n==3){require(sato_accumulate_into(pointers.data(),nullptr,result.data(),count,n,single,true,single,single,1.));for(auto value:result)require(value==T(0) && !std::signbit(value));}
 fields[0][0]=T(NAN);require(!sato_accumulate_into(pointers.data(),nullptr,result.data(),count,n,single,true,single,single,1.));
}
int main(){
 check<float>(19,2);check<double>(19,2);check<float>(19,3);check<double>(19,3);check<float>(266247,3);check<double>(266247,3);
 const std::uint64_t input[]={0,1,UINT64_MAX};double converted[3];ridge_prepare_into(input,PixelType::uint64,converted,3,false,true);
 require(converted[0]==0. && std::signbit(converted[0]) && converted[1]==-1. && converted[2]==-0x1p64);
 require(sato_accumulate_into(nullptr,nullptr,nullptr,0,3,false,false,false,false,1.));
 bool overflow=false;try{sato_accumulate_into(nullptr,nullptr,nullptr,std::numeric_limits<std::size_t>::max(),3,false,false,false,false,1.);}catch(const std::overflow_error&){overflow=true;}require(overflow);
 set_num_threads(1);std::cout<<"Sato native checks passed\n";
}
