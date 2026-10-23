#include "skimagexx/projective.hpp"
#include "skimagexx/image.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cfenv>
#include <limits>
namespace {void require(bool value){if(!value)throw std::runtime_error("projective native assertion failed");}}
int main(){
 using namespace skimagexx;
 for(int workers:{1,4}){
#ifndef _OPENMP
 if(workers>1)continue;
#endif
 set_num_threads(workers);
 for(std::size_t d:{2,3,4})for(std::size_t count:{0,1,3,4,5,67,150001}){
  std::vector<double> points(count*d),matrix((d+1)*(d+1),0.),out(count*d+2,-333.);
  for(std::size_t i=0;i<points.size();++i)points[i]=double(i%17)/10.;
  for(std::size_t i=0;i<=d;++i)matrix[i*(d+1)+i]=1.;
  for(std::size_t i=0;i<d;++i)matrix[i*(d+1)+d]=.3;
  projective_apply_into(points.data(),std::ptrdiff_t(d*8),8,matrix.data(),std::ptrdiff_t((d+1)*8),8,out.data()+1,count,d);
  require(out.front()==-333.&&out.back()==-333.);
  for(std::size_t i=0;i<count*d;++i)require(std::abs(out[i+1]-(points[i]+.3))<1e-15);
 }
 }
 bool caught=false;try{projective_apply_into(nullptr,16,8,nullptr,24,8,nullptr,1,2);}catch(const std::exception&){caught=true;}require(caught);
 double points[2]={2.,1.},matrix[9]={1.,0.,0.,0.,1.,0.,std::numeric_limits<double>::max(),0.,1.},output[2];
 require(projective_apply_checked_into(points,16,8,matrix,24,8,output,1,2));
 require(std::isfinite(output[0]) && std::isfinite(output[1]));
 matrix[6]=0.;std::feraiseexcept(FE_DIVBYZERO);
 require(!projective_apply_checked_into(points,16,8,matrix,24,8,output,1,2));
 require(std::fetestexcept(FE_DIVBYZERO)!=0);std::feclearexcept(FE_ALL_EXCEPT);
 set_num_threads(1);std::cout<<"native projective coordinates, tails, bounds and workers passed\n";
}
