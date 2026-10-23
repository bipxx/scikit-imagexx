#include "skimagexx/frangi.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
using namespace skimagexx;
void require(bool ok){if(!ok)throw std::runtime_error("Frangi native check failed");}
double analytic(std::size_t n,double first,double second,double third,double gamma_den){
 const double ra=second/third,rb=first/std::sqrt(second*third);
 if(n==2)return std::exp(-std::pow(first/second,2)/.5)*(1-std::exp(-(first*first+second*second)/gamma_den));
 return (1-std::exp(-ra*ra/.5))*std::exp(-rb*rb/.5)*(1-std::exp(-(first*first+second*second+third*third)/gamma_den));
}
template<class T> bool fixed_eigen(const void* const*,void* destination,std::size_t count,std::size_t n,bool single,void* context){
 require(single==(sizeof(T)==4));++*static_cast<int*>(context);auto* output=static_cast<T*>(destination);
 for(std::size_t i=0;i<count;++i){output[i]=T(4);output[count+i]=T(n==2 ? .2 : 1);if(n==3)output[2*count+i]=T(.2);}return true;
}
template<class T> void check(std::size_t count,std::size_t n){
 std::vector<std::vector<T>> fields(n*(n+1)/2,std::vector<T>(count));std::vector<const void*> pointers;
 std::size_t field=0;
 for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){std::fill(fields[field].begin(),fields[field].end(),r==c ? T(r==0 ? .1 : r==1 ? 2 : 3) : T(0));pointers.push_back(fields[field++].data());}
 std::vector<T> output(count),serial,previous(count,T(1));const bool single=sizeof(T)==4;
 FrangiParameters p;p.automatic_gamma=false;p.gamma_den=2;p.alpha_single=p.beta_single=p.gamma_single=single;
 set_num_threads(1);require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));
 const double expected=analytic(n,.1,2,3,2),tol=single ? 3e-6 : 2e-13;
 for(auto value:output)require(std::abs(double(value)-expected)<tol);serial=output;
#ifdef _OPENMP
 set_num_threads(4);
#endif
 require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));require(output==serial);
 require(frangi_accumulate_into(pointers.data(),previous.data(),output.data(),count,n,single,p));for(auto value:output)require(value==T(1));
 p.automatic_gamma=true;require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));require(!p.automatic_gamma && p.gamma_den>0);const auto denominator=p.gamma_den;
 for(auto& values:fields)for(auto& value:values)value=T(value*2);
 require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));require(p.gamma_den==denominator);
 int calls=0;const FrangiEigenProvider provider{&fixed_eigen<T>,&calls};p.eigen_provider=&provider;p.gamma_den=2;
 require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));require(calls==1);
 const double supplied=analytic(n,.2,n==2 ? 4 : 1,4,2);for(auto value:output)require(std::abs(double(value)-supplied)<tol);
 T repair[]={T(0),T(-0.),T(-1),T(.2),T(NAN)};ridge_hessian_finish_into(repair,5,single);require(repair[0]==1 && repair[1]==1 && repair[2]==1 && repair[3]==T(.2) && std::isnan(repair[4]));
}
int main(){check<float>(19,2);check<double>(19,2);check<float>(19,3);check<double>(19,3);check<float>(266247,3);check<double>(266247,3);
 FrangiParameters p;require(frangi_accumulate_into(nullptr,nullptr,nullptr,0,2,false,p));
 bool overflow=false;try{frangi_accumulate_into(nullptr,nullptr,nullptr,std::numeric_limits<std::size_t>::max(),3,false,p);}catch(const std::overflow_error&){overflow=true;}require(overflow);
 set_num_threads(1);std::cout<<"Frangi native checks passed\n";
}
