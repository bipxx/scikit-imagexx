#include "skimagexx/find_boundaries.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
using namespace skimagexx;
using namespace skimagexx::detail;
std::size_t controls=0;
void require(bool b){++controls;if(!b)throw std::runtime_error("find_boundaries analytic control failed");}
std::size_t count(const std::vector<std::size_t>& s){return std::accumulate(s.begin(),s.end(),std::size_t{1},std::multiplies<std::size_t>());}
std::vector<std::ptrdiff_t> cstrides(const std::vector<std::size_t>& s,std::size_t item=1){std::vector<std::ptrdiff_t> r(s.size());for(auto d=s.size();d-->0;){r[d]=std::ptrdiff_t(item);item*=s[d];}return r;}
template<class T>PixelType tag();
#define TAG(T,N) template<>PixelType tag<T>(){return PixelType::N;}
TAG(std::uint8_t,uint8) TAG(std::int8_t,int8) TAG(std::uint16_t,uint16) TAG(std::int16_t,int16) TAG(std::uint32_t,uint32) TAG(std::int32_t,int32) TAG(std::uint64_t,uint64) TAG(std::int64_t,int64)
#undef TAG
template<class T>void fixture(std::vector<T> labels,std::vector<std::size_t> shape,
 FindBoundaryMode mode,std::vector<std::uint8_t> expected,unsigned connectivity=1,
 std::uint64_t background=0,bool boolean=false) {
 ComponentsView v{labels.data(),boolean?PixelType::boolean:tag<T>(),false,false,shape,cstrides(shape,sizeof(T))};
 FindBoundaryParameters p{mode,connectivity,background,true};std::vector<std::size_t> outshape;
 require(find_boundaries_shape(v,mode,outshape));require(count(outshape)==expected.size());
 const auto original=labels;
 for(int workers:{1,4,0})for(bool reversed:{false,true}) {
  std::vector<std::uint8_t> storage(expected.size()+16,0xad);auto strides=cstrides(outshape);auto* out=storage.data()+8;
  if(reversed){out+=expected.size()-1;for(auto& s:strides)s=-s;}
  const auto flags=std::fetestexcept(FE_ALL_EXCEPT);const auto rounding=std::fegetround();
  require(find_boundaries_into(v,p,out,strides,workers));
  require(std::fetestexcept(FE_ALL_EXCEPT)==flags);require(std::fegetround()==rounding);
  for(std::size_t i=0;i<expected.size();++i)require(out[reversed?-std::ptrdiff_t(i):std::ptrdiff_t(i)]==expected[i]);
  require(std::all_of(storage.begin(),storage.begin()+8,[](auto a){return a==0xad;}));
  require(std::all_of(storage.end()-8,storage.end(),[](auto a){return a==0xad;}));require(labels==original);
 }
 // Transpose physical storage of all rank2 inputs/output; source logical order unchanged.
 if(shape.size()==2) {
  std::vector<T> f(labels.size());for(std::size_t y=0;y<shape[0];++y)for(std::size_t x=0;x<shape[1];++x)f[x*shape[0]+y]=labels[y*shape[1]+x];
  v.data=f.data();v.strides={std::ptrdiff_t(sizeof(T)),std::ptrdiff_t(shape[0]*sizeof(T))};
  std::vector<std::uint8_t> out(expected.size(),0xad);std::vector<std::ptrdiff_t> strides{1,std::ptrdiff_t(outshape[0])};
  require(find_boundaries_into(v,p,out.data(),strides,4));
  for(std::size_t y=0;y<outshape[0];++y)for(std::size_t x=0;x<outshape[1];++x)require(out[x*outshape[0]+y]==expected[y*outshape[1]+x]);
 }
}
template<class T>void widths(){
 fixture<T>({0,1,1,2},{4},FindBoundaryMode::thick,{1,1,1,1});
 fixture<T>({0,1,1,2},{4},FindBoundaryMode::inner,{0,1,1,1});
 if(sizeof(T)<8)fixture<T>({0,1,1,2},{4},FindBoundaryMode::outer,{1,0,1,1});
 fixture<T>({0,1,1,2},{4},FindBoundaryMode::subpixel,{0,1,0,0,0,1,0});
 fixture<T>({std::numeric_limits<T>::max(),1},{2},FindBoundaryMode::subpixel,{0,0,0});
}
void small_exhaustive(){
 // Independent binary edge-pair oracle: connectivity is Manhattan<=c.
 for(unsigned pattern=0;pattern<16;++pattern)for(unsigned connectivity:{1u,2u})for(auto mode:{FindBoundaryMode::thick,FindBoundaryMode::inner,FindBoundaryMode::outer}) {
  std::vector<std::uint8_t> a(4),truth(4);for(unsigned i=0;i<4;++i)a[i]=(pattern>>i)&1;
  for(int y=0;y<2;++y)for(int x=0;x<2;++x){bool different=false;
   for(int v=0;v<2;++v)for(int u=0;u<2;++u)if(unsigned(std::abs(v-y)+std::abs(u-x))<=connectivity)different|=a[v*2+u]!=a[y*2+x];
   auto center=a[y*2+x];truth[y*2+x]=different&&(mode==FindBoundaryMode::thick||(mode==FindBoundaryMode::inner?center!=0:center==0));
  }
  fixture<std::uint8_t>(a,{2,2},mode,truth,connectivity);
 }
}
void refusals(){
 std::array<std::uint64_t,2> a{{0,(std::uint64_t{1}<<53)+1}};std::array<std::uint8_t,8> out;out.fill(0xad);
 ComponentsView v{a.data(),PixelType::uint64,false,false,{2},{8}};FindBoundaryParameters p;
 auto refuse=[&]{require(!find_boundaries_into(v,p,out.data(),{1},1));require(std::all_of(out.begin(),out.end(),[](auto x){return x==0xad;}));};
 refuse();a[1]=1;p.mode=FindBoundaryMode::outer;refuse();
 p.mode=FindBoundaryMode::thick;v.swap=true;refuse();v.swap=false;v.half=true;refuse();v.half=false;
 v.type=PixelType::float64;refuse();v.type=PixelType::uint64;
 p.connectivity=0;refuse();p.connectivity=1;
 require(!find_boundaries_into(v,p,out.data(),{1},-1));require(!find_boundaries_into(v,p,out.data(),{1},1025));
 v.strides={std::numeric_limits<std::ptrdiff_t>::min()};refuse();v.strides={8};
 v.shape={0};std::vector<std::size_t> derived{99};require(!find_boundaries_shape(v,p.mode,derived));require(derived==std::vector<std::size_t>{99});
 v.shape={std::numeric_limits<std::size_t>::max()};refuse();v.shape={2};
 require(!find_boundaries_into(v,p,out.data(),{0},1));require(!find_boundaries_into(v,p,reinterpret_cast<std::uint8_t*>(a.data()),{1},1));
 require(std::all_of(out.begin(),out.end(),[](auto x){return x==0xad;}));
}
void signed_stride_storage(){
 std::vector<std::uint8_t> input(1+4*sizeof(std::int32_t),0);
 const std::int32_t values[4]={0,0,0,1};
 for(std::size_t y=0;y<2;++y)for(std::size_t x=0;x<2;++x)
  std::memcpy(input.data()+1+(y*2+1-x)*sizeof(std::int32_t),values+y*2+x,sizeof(std::int32_t));
 const auto saved=input;
 ComponentsView view{input.data()+1+sizeof(std::int32_t),PixelType::int32,false,false,{2,2},{8,-4}};
 std::array<std::uint8_t,12> output;output.fill(0xad);FindBoundaryParameters p;
 require(find_boundaries_into(view,p,output.data()+2,{4,2},4));
 for(std::size_t i=0;i<4;++i)require(output[2+2*i]==(i!=0));
 for(std::size_t i=0;i<output.size();++i)if(i<2 || i>8 || (i&1))require(output[i]==0xad);
 require(input==saved);
 // Broadcast storage is a legitimate borrowed array; singleton/zero strides
 // do not imply writable or overlapping output storage.
 std::array<std::uint8_t,2> labels{{0,1}};view={labels.data(),PixelType::uint8,false,false,{2,2},{0,1}};
 std::array<std::uint8_t,4> broadcast{};require(find_boundaries_into(view,p,broadcast.data(),{2,1},1));
 require(broadcast==std::array<std::uint8_t,4>{{1,1,1,1}});
 fixture<std::int32_t>({-1,-1,2},{3},FindBoundaryMode::inner,{0,0,1},1,std::uint64_t(-1));
}
int main(int argc,char** argv){
 if(argc==2 && std::strcmp(argv[1],"--storage-only")==0){signed_stride_storage();std::cout<<"PASS additional storage controls="<<controls<<"\n";return 0;}
 signed_stride_storage();
 set_num_threads(1);require(get_num_threads()==1);
 widths<std::uint8_t>();widths<std::int8_t>();widths<std::uint16_t>();widths<std::int16_t>();widths<std::uint32_t>();widths<std::int32_t>();widths<std::uint64_t>();widths<std::int64_t>();
 fixture<std::uint8_t>({7,2,2},{3},FindBoundaryMode::outer,{1,1,0},1,7);
 fixture<std::uint8_t>({255,1,1},{3},FindBoundaryMode::outer,{1,1,0},1,255);
 fixture<std::uint8_t>({0,2,255},{3},FindBoundaryMode::thick,{1,1,0},1,0,true);
 fixture<std::uint8_t>({0,2,255},{3},FindBoundaryMode::outer,{1,0,0},1,0,true);
 fixture<std::uint8_t>({0,2,255},{3},FindBoundaryMode::subpixel,{0,1,0,0,0},1,0,true);
 fixture<std::uint8_t>({0,0,0,1},{2,2},FindBoundaryMode::subpixel,{0,0,0,0,1,1,0,1,0});
 small_exhaustive();refusals();
 for(std::size_t rank=1;rank<=8;++rank) {
  std::vector<std::size_t> shape(rank,2);std::vector<std::uint16_t> zero(std::size_t{1}<<rank);
  std::vector<std::size_t> expanded(rank,3);
  fixture<std::uint16_t>(zero,shape,FindBoundaryMode::thick,std::vector<std::uint8_t>(zero.size()),unsigned(rank));
  fixture<std::uint16_t>(zero,shape,FindBoundaryMode::subpixel,std::vector<std::uint8_t>(count(expanded)),1);
 }
 // Cross the existing dispatcher threshold; current global budget is sampled per call.
 const std::size_t n=300001;std::vector<std::uint8_t> a(n),out(n);for(std::size_t i=0;i<n;++i)a[i]=i&1;
 ComponentsView v{a.data(),PixelType::uint8,false,false,{n},{1}};FindBoundaryParameters p;
 for(int budget:{4,1}){set_num_threads(budget);require(find_boundaries_into(v,p,out.data(),{1},0));for(auto x:out)require(x==1);require(get_num_threads()==budget);}
 set_num_threads(1);require(get_num_threads()==1);
 std::cout<<"PASS analytic controls="<<controls<<" global_restored="<<get_num_threads()<<"\n";
}
