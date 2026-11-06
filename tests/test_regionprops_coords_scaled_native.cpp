#include <skimagexx/regionprops_state.hpp>
#include <skimagexx/image.hpp>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON
using namespace skimagexx;
using namespace skimagexx::detail;
namespace {
std::size_t checks=0,records=0;
void require(bool yes){++checks;if(!yes)throw std::runtime_error("coords_scaled native control "+std::to_string(checks));}
std::uint64_t bits(double v){std::uint64_t r;std::memcpy(&r,&v,8);return r;}
double value(std::uint64_t r){double v;std::memcpy(&v,&r,8);return v;}
struct Gold{const char* id;std::size_t ndim,count;std::array<std::size_t,3> shape;std::array<std::intptr_t,3> origin;std::array<double,3> spacing,offset;std::vector<std::uint8_t> mask;std::vector<std::uint64_t> expected;bool f64;};
const Gold golds[]={
{"regionprops-coords-scaled/plain2d-c-off",2,3, {2,2,1}, {1,1,0}, {1.0,1.0,0.0}, {0.0,0.0,0.0}, {1,1,1,0}, {UINT64_C(0x3ff0000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000)}, true},
{"regionprops-coords-scaled/plain2d-f-on",2,3, {2,2,1}, {1,1,0}, {1.0,1.0,0.0}, {0.0,0.0,0.0}, {1,1,1,0}, {UINT64_C(0x3ff0000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x3ff0000000000000)}, true},
{"regionprops-coords-scaled/negative-anisotropic2d",2,3, {2,2,1}, {1,1,0}, {-0.5,2.0,0.0}, {-3.0,4.0,0.0}, {1,1,1,0}, {UINT64_C(0xc00c000000000000),UINT64_C(0xc00c000000000000),UINT64_C(0xc010000000000000),UINT64_C(0x4018000000000000),UINT64_C(0x4020000000000000),UINT64_C(0x4018000000000000)}, true},
{"regionprops-coords-scaled/scalar-spacing2d",2,3, {2,2,1}, {1,1,0}, {2.0,2.0,0.0}, {1.0,-2.0,0.0}, {1,1,1,0}, {UINT64_C(0x4008000000000000),UINT64_C(0x4008000000000000),UINT64_C(0x4014000000000000),UINT64_C(0x0000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x0000000000000000)}, true},
{"regionprops-coords-scaled/fractional3d",3,4, {2,2,2}, {0,0,0}, {0.3,0.5,1.25}, {0.0,-1.0,2.0}, {1,0,0,1,0,1,0,1}, {UINT64_C(0x0000000000000000),UINT64_C(0x0000000000000000),UINT64_C(0x3fd3333333333333),UINT64_C(0x3fd3333333333333),UINT64_C(0xbff0000000000000),UINT64_C(0xbfe0000000000000),UINT64_C(0xbff0000000000000),UINT64_C(0xbfe0000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x400a000000000000),UINT64_C(0x400a000000000000),UINT64_C(0x400a000000000000)}, true},
{"regionprops-coords-scaled/integer-spacing3d",3,4, {2,2,2}, {0,0,0}, {1.0,2.0,3.0}, {1.0,0.0,-2.0}, {1,0,0,1,0,1,0,1}, {UINT64_C(0x0000000000000001),UINT64_C(0x0000000000000001),UINT64_C(0x0000000000000002),UINT64_C(0x0000000000000002),UINT64_C(0x0000000000000000),UINT64_C(0x0000000000000002),UINT64_C(0x0000000000000000),UINT64_C(0x0000000000000002),UINT64_C(0xfffffffffffffffe),UINT64_C(0x0000000000000001),UINT64_C(0x0000000000000001),UINT64_C(0x0000000000000001)}, false},
{"regionprops-coords-scaled/float32-spacing-offset2d",2,3, {2,2,1}, {1,1,0}, {0.5,2.0,0.0}, {0.25,-0.5,0.0}, {1,1,1,0}, {UINT64_C(0x3fe8000000000000),UINT64_C(0x3fe8000000000000),UINT64_C(0x3ff4000000000000),UINT64_C(0x3ff8000000000000),UINT64_C(0x400c000000000000),UINT64_C(0x3ff8000000000000)}, true},
{"regionprops-coords-scaled/pixel-mutation-unprimed2d",2,2, {2,2,1}, {1,1,0}, {0.5,2.0,0.0}, {1.0,-2.0,0.0}, {0,1,1,0}, {UINT64_C(0x3ff8000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x0000000000000000)}, true},
{"regionprops-coords-scaled/primed-pixel-mutation2d",2,3, {2,2,1}, {1,1,0}, {0.5,2.0,0.0}, {1.0,-2.0,0.0}, {1,1,1,0}, {UINT64_C(0x3ff8000000000000),UINT64_C(0x3ff8000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x0000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x0000000000000000)}, true},
{"regionprops-coords-scaled/cached-mask-edit2d",2,2, {2,2,1}, {1,1,0}, {0.5,2.0,0.0}, {1.0,-2.0,0.0}, {0,1,1,0}, {UINT64_C(0x3ff8000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x0000000000000000)}, true},
{"regionprops-coords-scaled/primed-public-slice-shift2d",2,3, {2,2,1}, {2,2,0}, {0.5,2.0,0.0}, {1.0,-2.0,0.0}, {1,1,1,0}, {UINT64_C(0x4000000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4004000000000000),UINT64_C(0x4000000000000000),UINT64_C(0x4010000000000000),UINT64_C(0x4000000000000000)}, true},
{"regionprops-coords-scaled/empty-current-mask2d",2,0, {2,2,1}, {1,1,0}, {0.5,2.0,0.0}, {1.0,-2.0,0.0}, {0,0,0,0}, {}, true}
};
void saved(int workers){
 for(const auto& g:golds)for(int layout=0;layout<3;++layout){
  ++records;const auto d=g.ndim,n=g.count;std::array<std::ptrdiff_t,3> stride{};std::size_t product=1;
  if(layout==1){for(std::size_t a=0;a<d;++a){stride[a]=std::ptrdiff_t(product);product*=g.shape[a];}}
  else{for(auto a=d;a-->0;){stride[a]=std::ptrdiff_t(product);product*=g.shape[a];}}
  std::vector<std::uint8_t> storage(product+2,0x97);auto* mask=storage.data()+1;
  if(layout==2){mask+=product-1;for(std::size_t a=0;a<d;++a)stride[a]=-stride[a];}
  std::vector<std::intptr_t> local;
  for(std::size_t z=0;z<(d==3?g.shape[0]:1);++z)for(std::size_t r=0;r<g.shape[d-2];++r)for(std::size_t c=0;c<g.shape[d-1];++c){
   const auto flat=(z*g.shape[d-2]+r)*g.shape[d-1]+c;
   const auto off=d==2?std::ptrdiff_t(r)*stride[0]+std::ptrdiff_t(c)*stride[1]:std::ptrdiff_t(z)*stride[0]+std::ptrdiff_t(r)*stride[1]+std::ptrdiff_t(c)*stride[2];
   mask[off]=g.mask[flat];if(g.mask[flat]){if(d==3)local.push_back(std::intptr_t(z));local.push_back(std::intptr_t(r));local.push_back(std::intptr_t(c));}
  }
  const auto before=storage;std::vector<std::intptr_t> index(n*d+2,-987);auto* ip=index.data()+1;
  require(regionprops_coords_scaled_local_into(mask,d,g.shape,stride,n,ip,workers));
  for(std::size_t a=0;a<d;++a)for(std::size_t p=0;p<n;++p)require(ip[a*n+p]==local[p*d+a]);
  require(regionprops_coords_scaled_origin_into(ip,n,d,g.shape,g.origin,workers));
  for(std::size_t a=0;a<d;++a)for(std::size_t p=0;p<n;++p)require(ip[a*n+p]==local[p*d+a]+g.origin[a]);
  if(g.f64){std::vector<double> output(n*d+2,-777.);auto* op=output.data()+1;
   require(regionprops_coords_scaled_multiply_f64_into(ip,n,d,g.spacing,op,workers));
   require(regionprops_coords_scaled_add_f64_into(op,n,d,g.offset,workers));
   for(std::size_t p=0;p<n*d;++p)require(bits(op[p])==g.expected[p]);require(output.front()==-777.&&output.back()==-777.);
  }
  require(index.front()==-987&&index.back()==-987);require(storage==before);
 }
}
void refusal(){
 std::array<std::size_t,3> shape{2,2,1};std::array<std::intptr_t,3> origin{1,1,0};std::array<double,3> scale{.5,2.,0.};
 std::intptr_t index[]={0,0,1,0,1,0};double output[6]={-11,-11,-11,-11,-11,-11};
 for(auto bad:{std::numeric_limits<std::intptr_t>::min(),std::intptr_t(-1),std::intptr_t(2),std::numeric_limits<std::intptr_t>::max()}){auto old=index[4];index[4]=bad;const std::vector<std::intptr_t> before(index,index+6);require(!regionprops_coords_scaled_origin_into(index,3,2,shape,origin,1));require(std::equal(before.begin(),before.end(),index));index[4]=old;}
 auto bad_origin=origin;bad_origin[0]=std::numeric_limits<std::intptr_t>::max();require(!regionprops_coords_scaled_origin_into(index,3,2,shape,bad_origin,1));
 require(!regionprops_coords_scaled_origin_into(index,3,2,shape,origin,0));
 const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
 for(auto word:{UINT64_C(0x7ff0000000000001),UINT64_C(0x7ff0000000000000),UINT64_C(1),UINT64_C(0x4ff0000000000001)}){auto s=scale;s[1]=value(word);require(!regionprops_coords_scaled_multiply_f64_into(index,3,2,s,output,1));require(std::fetestexcept(FE_ALL_EXCEPT)==flags);for(auto v:output)require(v==-11.);require(!regionprops_coords_scaled_add_f64_into(output,3,2,s,1));}
 index[0]=2*regionprops_coords_scaled_max_origin+1;require(!regionprops_coords_scaled_multiply_f64_into(index,3,2,scale,output,1));index[0]=0;
 require(!regionprops_coords_scaled_multiply_f64_into(index,3,2,scale,reinterpret_cast<double*>(index),1));
 output[5]=value(UINT64_C(0x7ff0000000000001));auto before=bits(output[5]);require(!regionprops_coords_scaled_add_f64_into(output,3,2,scale,1));require(bits(output[5])==before);
 require(regionprops_coords_scaled_local_into(nullptr,2,{0,2,1},{2,1,0},0,nullptr,1));require(regionprops_coords_scaled_origin_into(nullptr,0,2,{0,2,1},origin,1));require(regionprops_coords_scaled_multiply_f64_into(nullptr,0,2,scale,nullptr,1));require(regionprops_coords_scaled_add_f64_into(nullptr,0,2,scale,1));
 std::uint8_t mask[4]={1,1,1,0};require(!regionprops_coords_scaled_local_into(mask,2,shape,{std::numeric_limits<std::ptrdiff_t>::min(),1,0},3,index,1));
 require(!regionprops_coords_scaled_local_into(mask,2,shape,{2,1,0},3,reinterpret_cast<std::intptr_t*>(mask),1));
}
void environment(){
 std::fenv_t saved;require(std::fegetenv(&saved)==0);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 const auto oldcsr=_mm_getcsr();
 #endif
 require(std::fesetround(FE_TONEAREST)==0);std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO|FE_OVERFLOW);
 const auto flags=std::fetestexcept(FE_ALL_EXCEPT);std::intptr_t in[]={1,2};double out[]={-9.,-9.};
 require(regionprops_coords_scaled_multiply_f64_into(in,1,2,{.3,.5,0},out,1));require(std::fetestexcept(FE_ALL_EXCEPT)==flags);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 const auto nearest=_mm_getcsr();
 for(auto change:{0x2000u,0x4000u,0x6000u,0x8000u,0x40u}){const auto requested=nearest|change;_mm_setcsr(requested);double untouched[2]={-8.,-8.};require(!regionprops_coords_scaled_multiply_f64_into(in,1,2,{.3,.5,0},untouched,1));require(untouched[0]==-8.&&untouched[1]==-8.);require(_mm_getcsr()==requested);require(!regionprops_coords_scaled_add_f64_into(untouched,1,2,{1.,1.,0},1));require(_mm_getcsr()==requested);}
 _mm_setcsr(nearest);
 #endif
 require(std::fesetround(FE_DOWNWARD)==0);require(!regionprops_coords_scaled_add_f64_into(out,1,2,{1.,1.,0},1));require(std::fegetround()==FE_DOWNWARD);require(std::fesetenv(&saved)==0);
 #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
 _mm_setcsr(oldcsr);
 #endif
}
void large(int workers){
 const std::size_t n=131073;std::vector<std::uint8_t> mask(n,1);std::vector<std::intptr_t> index(n*2);std::vector<double> output(n*2);
 require(regionprops_coords_scaled_local_into(mask.data(),2,{1,n,1},{std::ptrdiff_t(n),1,0},n,index.data(),workers));
 require(regionprops_coords_scaled_origin_into(index.data(),n,2,{1,n,1},{1,-3,0},workers));
 #ifdef _OPENMP
 std::vector<unsigned> worker_before(std::size_t(workers),0);
 #pragma omp parallel num_threads(workers)
 {
  #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
  const auto id=std::size_t(omp_get_thread_num());if(id){_mm_setcsr((_mm_getcsr()&~0x6000u)|0x2000u);worker_before[id]=_mm_getcsr();}
  #endif
 }
 #endif
 require(regionprops_coords_scaled_multiply_f64_into(index.data(),n,2,{.3,.5,0},output.data(),workers));require(regionprops_coords_scaled_add_f64_into(output.data(),n,2,{.125,-.25,0},workers));
 #ifdef _OPENMP
 #pragma omp parallel num_threads(workers)
 {
  #if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
  const auto id=std::size_t(omp_get_thread_num());if(id){if(_mm_getcsr()!=worker_before[id])std::abort();_mm_setcsr(_mm_getcsr()&~0x6000u);}
  #endif
 }
 #endif
 for(std::size_t p=0;p<n;++p){require(index[p]==1&&index[n+p]==std::intptr_t(p)-3);volatile double x=double(index[p])*.3;volatile double y=double(index[n+p])*.5;require(bits(output[p])==bits(x+.125));require(bits(output[n+p])==bits(y-.25));}
}
void old_coords(){
 // Saved regionprops-state/2d-cache-True: canonical C logical points below.
 std::int64_t labels[]={0,3,3,0,1,0,3,0,1,1,0,0};RegionPropsState state;ComponentsView view{labels,PixelType::int64,false,false,{3,4},{32,8}};require(prepare_regionprops_state(view,state));
 std::uint8_t mask[]={1,0,1,1};std::intptr_t expected[]={1,0,2,0,2,1};
 for(int layout=0;layout<3;++layout){std::intptr_t output[8];std::fill(output,output+8,-87);auto* start=output+1;std::ptrdiff_t step=16,dim=8;if(layout==1){step=8;dim=24;}if(layout==2){start=output+6;step=-16;dim=-8;}
 regionprops_coords_strided_into(state,0,mask,{2,1,0},start,step,dim);
 for(std::size_t p=0;p<3;++p)for(std::size_t a=0;a<2;++a){std::intptr_t got;std::memcpy(&got,reinterpret_cast<char*>(start)+std::ptrdiff_t(p)*step+std::ptrdiff_t(a)*dim,sizeof(got));require(got==expected[p*2+a]);}require(output[0]==-87&&output[7]==-87);}
}
}
int main(int argc,char** argv){try{if(argc==2 && std::string(argv[1])=="--old-coords-only"){old_coords();std::cout<<"saved_old_coords_checks="<<checks<<"\n";return 0;}const auto original=get_num_threads();set_num_threads(1);saved(1);large(1);bool four=true;try{set_num_threads(4);}catch(const std::invalid_argument&){four=false;require(get_num_threads()==1);}if(four){saved(4);large(4);}set_num_threads(1);require(get_num_threads()==1);saved(1);refusal();environment();old_coords();set_num_threads(original);require(get_num_threads()==original);std::cout<<"records="<<records<<" checks="<<checks<<" workers4="<<four<<" default_restore=1\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
