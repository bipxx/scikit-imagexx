#include <skimagexx/flood_fill.hpp>
#include <skimagexx/image.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using I=std::ptrdiff_t;
I index(I k,const std::vector<I>& shape,const std::vector<I>& strides){I p=0;
 for(std::size_t j=shape.size();j-->0;){auto c=k%shape[j];k/=shape[j];p+=c*strides[j];}return p;}
template<class T>void check(const char* id,std::vector<I> shape,std::vector<I> strides,std::vector<I> seed,
 std::vector<I> fshape,std::vector<std::uint8_t> fp,bool fortran,std::vector<std::uint8_t> raw,
 std::vector<std::uint8_t> gold,bool fill,double value,bool tol,T low,T high){
 I count=1,lo=0,hi=0;for(std::size_t a=0;a<shape.size();++a){count*=shape[a];auto span=(shape[a]?shape[a]-1:0)*strides[a];lo+=std::min(I(0),span);hi+=std::max(I(0),span);}
 std::vector<std::uint8_t> backing(std::size_t(hi-lo+sizeof(T)),0);auto* p=backing.data()-lo;
 for(I k=0;k<count;++k)std::memcpy(p+index(k,shape,strides),raw.data()+k*sizeof(T),sizeof(T));
 skimagexx::FloodGeometry g{shape,strides,seed,fshape,fp,fortran};
 auto mask=skimagexx::flood(reinterpret_cast<const T*>(p),g,tol,low,high);
 if(fill){auto out=skimagexx::flood_fill(reinterpret_cast<const T*>(p),g,static_cast<T>(value),tol,low,high);
  if(out.size()*sizeof(T)!=gold.size()||(!gold.empty()&&std::memcmp(out.data(),gold.data(),gold.size())))throw std::runtime_error(std::string(id)+" owning fill");
  skimagexx::flood_fill_into(reinterpret_cast<T*>(p),g,static_cast<T>(value),tol,low,high);
  for(I k=0;k<count;++k)if(std::memcmp(p+index(k,shape,strides),gold.data()+k*sizeof(T),sizeof(T)))throw std::runtime_error(std::string(id)+" borrowed fill");
 }else for(I k=0;k<count;++k)if((mask.storage[mask.origin+index(k,shape,mask.strides)]==1)!=bool(gold[k]))throw std::runtime_error(std::string(id)+" mask");
 if(!fill)for(I k=0;k<count;++k)if(std::memcmp(p+index(k,shape,strides),raw.data()+k*sizeof(T),sizeof(T)))throw std::runtime_error(std::string(id)+" mutated input");
}
int main(){
 if(skimagexx::get_num_threads()!=1)throw std::runtime_error("SDK default worker count");

#include "fixtures/flood_fill_sdk_original_v1.inc"
 skimagexx::FloodGeometry g{{2,2},{2,1},{0,0},{3,3},std::vector<std::uint8_t>(9,1),false};
 std::uint8_t input[4]={1,1,2,2};
 auto serial=skimagexx::flood_fill(input,g,std::uint8_t(9));
 skimagexx::set_num_threads(4);auto optional=skimagexx::flood_fill(input,g,std::uint8_t(9));
 if(serial!=optional||skimagexx::get_num_threads()!=4)throw std::runtime_error("ordered serial worker policy");
 skimagexx::set_num_threads(1);
 auto invalid=g;invalid.seed[0]=2;
 try{skimagexx::flood(input,invalid);throw std::runtime_error("invalid seed accepted");}catch(const std::out_of_range&){}
 invalid=g;invalid.footprint.clear();
 try{skimagexx::flood(input,invalid);throw std::runtime_error("invalid footprint accepted");}catch(const std::invalid_argument&){}
 invalid=g;invalid.shape={std::numeric_limits<I>::max(),2};
 try{skimagexx::flood(input,invalid);throw std::runtime_error("overflow shape accepted");}catch(const std::overflow_error&){}
 std::cout<<"79 saved complete numeric SDK return cases; owning and borrowed fill\n";
}
