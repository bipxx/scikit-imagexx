#include "skimagexx/find_boundaries.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
using namespace skimagexx;using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("boundary signed-layout control failed");}
std::vector<std::ptrdiff_t> strides(const std::vector<std::size_t>& s,std::size_t bytes,bool f){std::vector<std::ptrdiff_t> r(s.size());if(f){for(std::size_t d=0;d<s.size();++d){r[d]=bytes;bytes*=s[d];}}else for(auto d=s.size();d-->0;){r[d]=bytes;bytes*=s[d];}return r;}
std::vector<std::size_t> position(std::size_t i,const std::vector<std::size_t>& s){std::vector<std::size_t> p(s.size());for(auto d=s.size();d-->0;){p[d]=i%s[d];i/=s[d];}return p;}
std::ptrdiff_t offset(const std::vector<std::size_t>& p,const std::vector<std::ptrdiff_t>& s){std::ptrdiff_t r=0;for(std::size_t d=0;d<p.size();++d)r+=std::ptrdiff_t(p[d])*s[d];return r;}
template<class T>PixelType tag(){if constexpr(sizeof(T)==1)return std::is_signed_v<T>?PixelType::int8:PixelType::uint8;if constexpr(sizeof(T)==2)return std::is_signed_v<T>?PixelType::int16:PixelType::uint16;if constexpr(sizeof(T)==4)return std::is_signed_v<T>?PixelType::int32:PixelType::uint32;return std::is_signed_v<T>?PixelType::int64:PixelType::uint64;}
template<class T>void exercise(const std::vector<std::size_t>& shape,bool f,bool reverse,bool boolean,bool broadcast=false){
 const auto n=std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<>());auto is=strides(shape,2*sizeof(T),f),os=strides(shape,2,!f);std::vector<unsigned char> input(2*n*sizeof(T)+32,0x9d);auto*in=input.data()+9;
 if(broadcast)is[0]=0;
 if(reverse){in+=std::ptrdiff_t(shape.back()-1)*is.back();is.back()=-is.back();}
 for(std::size_t i=0;i<n;++i){T v=boolean?T(i%3==0?0:i%3==1?2:255):T(std::is_signed_v<T>?int((i*13+i/7)%7)-3:int((i*13+i/7)%7));auto p=position(i,shape);std::memcpy(in+offset(p,is),&v,sizeof(T));}
 const auto saved=input;ComponentsView view{in,boolean?PixelType::boolean:tag<T>(),false,false,shape,is};
 auto value=[&](const std::vector<std::size_t>& p){T v;std::memcpy(&v,in+offset(p,is),sizeof(T));if(boolean)v=v!=0;return v;};
 std::size_t neighbor_count=1;for(auto ignored:shape){(void)ignored;neighbor_count*=3;}
 for(auto mode:{FindBoundaryMode::thick,FindBoundaryMode::inner,FindBoundaryMode::outer}){
  if(sizeof(T)==8 && mode==FindBoundaryMode::outer)continue;
  for(unsigned connectivity=1;connectivity<=shape.size();++connectivity)for(int bg:{0,2,-1}){
   if(n>262144 && (mode!=FindBoundaryMode::outer || connectivity!=shape.size() || bg!=0))continue;
   const auto bits=std::uint64_t(bg);FindBoundaryParameters parameters{mode,connectivity,bits,true};std::vector<unsigned char> truth(n);
   for(std::size_t i=0;i<n;++i){const auto p=position(i,shape);T minimum=std::numeric_limits<T>::max(),maximum=std::numeric_limits<T>::lowest(),full_max=maximum,inverted_min=minimum;
    for(std::size_t word=0;word<neighbor_count;++word){auto code=word;unsigned distance=0;auto q=p;for(std::size_t d=0;d<shape.size();++d){int delta=int(code%3)-1;code/=3;distance+=delta!=0;if(delta<0&&q[d])--q[d];if(delta>0&&q[d]+1<shape[d])++q[d];}T v=value(q);if(distance<=connectivity){minimum=std::min(minimum,v);maximum=std::max(maximum,v);}full_max=std::max(full_max,v);inverted_min=std::min(inverted_min,std::uint64_t(v)==bits?std::numeric_limits<T>::max():v);}
    const bool background=std::uint64_t(value(p))==bits;truth[i]=(minimum!=maximum)&&(mode==FindBoundaryMode::thick||(mode==FindBoundaryMode::inner?!background:background||full_max!=inverted_min));
   }
   for(int budget:{1,4,0}){set_num_threads(budget==0?4:1);std::vector<unsigned char> output(2*n+32,0xa7);auto*data=output.data()+8+std::ptrdiff_t(shape[0]-1)*os[0];auto output_strides=os;output_strides[0]=-output_strides[0];
    const auto flags=std::fetestexcept(FE_ALL_EXCEPT),rounding=std::fegetround();require(find_boundaries_into(view,parameters,data,output_strides,budget));require(std::fetestexcept(FE_ALL_EXCEPT)==flags&&std::fegetround()==rounding);
    std::vector<bool> written(output.size());for(std::size_t i=0;i<n;++i){auto at=8+std::ptrdiff_t(shape[0]-1)*os[0]+offset(position(i,shape),output_strides);require(output[at]==truth[i]);written[at]=true;}
    for(std::size_t i=0;i<output.size();++i)if(!written[i])require(output[i]==0xa7);require(input==saved);require(get_num_threads()==(budget==0?4:1));
   }
  }
 }
}
template<class T>void widths(){for(auto shape:{std::vector<std::size_t>{5},std::vector<std::size_t>{5,7},std::vector<std::size_t>{4,5,3},std::vector<std::size_t>{3,3,3,3}})for(bool f:{false,true})exercise<T>(shape,f,true,false);}
}
void floating_environment(){
 std::uint16_t input[64];std::iota(input,input+64,std::uint16_t{0});std::uint8_t output[64];
 ComponentsView view{input,PixelType::uint16,false,false,{8,8},{2,16}};FindBoundaryParameters p{FindBoundaryMode::inner,2,0,true};
 std::fenv_t controller;require(std::fegetenv(&controller)==0);
 for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
  require(std::fesetround(rounding)==0);require(std::feclearexcept(FE_ALL_EXCEPT)==0);require(std::feraiseexcept(FE_DIVBYZERO|FE_INEXACT)==0);
  require(find_boundaries_into(view,p,output,{8,1},1));require(std::fetestexcept(FE_ALL_EXCEPT)==(FE_DIVBYZERO|FE_INEXACT));require(std::fegetround()==rounding);
 }
 require(std::fesetenv(&controller)==0);
}
int main(){set_num_threads(1);require(get_num_threads()==1);widths<std::uint8_t>();widths<std::int8_t>();widths<std::uint16_t>();widths<std::int16_t>();widths<std::uint32_t>();widths<std::int32_t>();widths<std::uint64_t>();widths<std::int64_t>();exercise<std::uint8_t>({5,7},false,true,true);exercise<std::uint16_t>({3,5},true,false,false,true);floating_environment();
 // Partition boundaries fall inside physical F spans; output uses independent
 // reversed C order. This crosses the actual dispatcher threshold.
 exercise<std::uint16_t>({257,1021},true,true,false);set_num_threads(1);require(get_num_threads()==1);std::cout<<"PASS layout_controls="<<checks<<" global_restored="<<get_num_threads()<<"\n";}
