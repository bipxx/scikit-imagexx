#include "skimagexx/frangi.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <atomic>
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
void constant_exp(const double* input,double* output,std::size_t count,void* context){
 require(input!=output);static_cast<std::atomic<int>*>(context)->fetch_add(1);
 for(std::size_t i=0;i<count;++i){require(std::isfinite(input[i]));output[i]=.5;}
}
struct BoundaryContext {std::size_t count;int calls=0;};
bool boundary_eigen(const void* const*,void* destination,std::size_t count,std::size_t n,bool single,void*){
 require(n==3 && !single);auto* out=static_cast<double*>(destination);
 for(std::size_t i=0;i<count;++i){out[i]=.02;out[count+i]=1e-10;out[2*count+i]=0.;}return true;
}
void boundary_exp(const double* input,double* output,std::size_t count,void* context){
 auto& state=*static_cast<BoundaryContext*>(context);require(count==state.count && input!=output);++state.calls;
 for(std::size_t i=0;i<count;++i)output[i]=input[i]<0 && input[i]>-0x1p-54 ? std::nextafter(1.,0.) : std::exp(input[i]);
}
void boundary_check(){
 const std::size_t count=19;std::vector<double> zero(count),output(count);const void* fields[]={zero.data(),zero.data(),zero.data(),zero.data(),zero.data(),zero.data()};
 BoundaryContext context{count};const FrangiEigenProvider eigen{&boundary_eigen,nullptr};const FrangiExpProvider exponential{&boundary_exp,&context};
 FrangiParameters p;p.automatic_gamma=false;p.gamma_den=450;p.eigen_provider=&eigen;p.exp_provider=&exponential;
 require(frangi_accumulate_into(fields,nullptr,output.data(),count,3,false,p));require(context.calls==3);
 for(auto value:output)require(value>0 && value<1e-12);const auto original=output;
 ridge_hessian_finish_into(output.data(),count,false);require(output==original);
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
 if(!single){std::atomic<int> exp_calls{0};const FrangiExpProvider exponential{&constant_exp,&exp_calls};p.exp_provider=&exponential;
  require(frangi_accumulate_into(pointers.data(),nullptr,output.data(),count,n,single,p));require(exp_calls>0);
  for(auto value:output)require(value==(n==2 ? .25 : .125));p.exp_provider=nullptr;}
 T repair[]={T(0),T(-0.),T(-1),T(.2),T(NAN)};ridge_hessian_finish_into(repair,5,single);require(repair[0]==1 && repair[1]==1 && repair[2]==1 && repair[3]==T(.2) && std::isnan(repair[4]));
}
int main(){boundary_check();check<float>(19,2);check<double>(19,2);check<float>(19,3);check<double>(19,3);check<float>(266247,3);check<double>(266247,3);
 FrangiParameters p;require(frangi_accumulate_into(nullptr,nullptr,nullptr,0,2,false,p));
 bool overflow=false;try{frangi_accumulate_into(nullptr,nullptr,nullptr,std::numeric_limits<std::size_t>::max(),3,false,p);}catch(const std::overflow_error&){overflow=true;}require(overflow);
 set_num_threads(1);std::cout<<"Frangi native checks passed\n";
}
