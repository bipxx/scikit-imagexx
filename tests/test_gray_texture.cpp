#include <skimagexx/gray_texture.hpp>
#include "fixtures/gray_texture_saved_v1.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cmath>
using namespace skimagexx;
namespace {
GrayNumericType numeric(const std::string& s){const auto t=s.substr(1);if(t=="b1")return GrayNumericType::boolean;
#define T(S,E) if(t==S)return GrayNumericType::E;
T("i1",i8)T("u1",u8)T("i2",i16)T("u2",u16)T("i4",i32)T("u4",u32)T("i8",i64)T("u8",u64)T("f2",f16)T("f4",f32)T("f8",f64)T("c8",c64)T("c16",c128)
#undef T
throw std::invalid_argument("saved POD dtype");}
GrayIntegerType integer(const std::string&s){auto n=numeric(s);if(n==GrayNumericType::boolean)return GrayIntegerType::u8;return static_cast<GrayIntegerType>(static_cast<int>(n)-1);}
GrayProperty property(const std::string&s){
#define P(S,E) if(s==S)return GrayProperty::E;
P("contrast",contrast)P("dissimilarity",dissimilarity)P("homogeneity",homogeneity)P("ASM",ASM)P("energy",energy)P("mean",mean)P("variance",variance)P("std",stddev)P("entropy",entropy)P("correlation",correlation)
#undef P
throw std::invalid_argument("saved property");}
struct Storage{std::vector<unsigned char>v;std::ptrdiff_t origin;};
Storage storage(const gray_saved::Case&c){std::ptrdiff_t lo=0,hi=0;const auto width=std::stoul(std::string(c.dtype).substr(2));for(std::size_t a=0;a<c.shape.size();++a){auto end=std::ptrdiff_t(c.shape[a]-1)*c.strides[a];lo+=std::min(end,std::ptrdiff_t(0));hi+=std::max(end,std::ptrdiff_t(0));}Storage out{std::vector<unsigned char>(std::size_t(hi-lo)+width,0xAA),-lo};std::size_t count=c.input.size()/width;for(std::size_t k=0;k<count;++k){std::size_t t=k;auto offset=out.origin;for(std::size_t a=c.shape.size();a-->0;){offset+=std::ptrdiff_t(t%c.shape[a])*c.strides[a];t/=c.shape[a];}std::memcpy(out.v.data()+offset,c.input.data()+k*width,width);}return out;}
template<class T>bool words(const std::vector<T>&v,const std::vector<std::uint64_t>&runs){std::size_t k=0;for(std::size_t r=0;r<runs.size();r+=2)for(std::uint64_t i=0;i<runs[r];++i){if(k>=v.size())return false;std::uint64_t actual=0;std::memcpy(&actual,&v[k++],sizeof(T));if(actual!=runs[r+1])return false;}return k==v.size();}
}
int main(){assert(get_num_threads()==1);std::size_t checked=0;for(int workers:{0,1,4})for(const auto&c:gray_saved::cases()){auto backing=storage(c);const auto old=backing.v;const void*p=backing.v.data()+backing.origin;bool ok=false;
if(c.matrix){auto out=graycomatrix(GrayImageView{p,c.shape[0],c.shape[1],c.strides[0],c.strides[1],integer(c.dtype),c.dtype[0]=='>'},c.distances,c.angles,c.levels,c.symmetric,c.normed,workers);assert(out.levels==c.levels&&out.distances==c.distances.size()&&out.angles==c.angles.size());ok=c.normed?words(out.probabilities,c.runs):words(out.counts,c.runs);}
else{auto out=graycoprops(GrayProbabilityView{p,c.shape[0],c.shape[2],c.shape[3],{c.strides[0],c.strides[1],c.strides[2],c.strides[3]},numeric(c.dtype),c.dtype[0]=='>'},property(c.prop),workers);assert(out.distances==c.shape[2]&&out.angles==c.shape[3]);ok=words(out.values,c.runs);}
if(!ok){std::cerr<<"SAVED_WORD_MISMATCH "<<c.id<<" workers="<<workers<<'\n';return 1;}assert(backing.v==old);++checked;}
std::uint8_t image[4]{0,1,1,0};GrayImageView view{image,2,2,2,1,GrayIntegerType::u8};std::size_t refused=0;
try{graycomatrix(view,{std::numeric_limits<double>::infinity()},{0},2);}catch(const std::invalid_argument&){++refused;}
try{graycomatrix(view,{1},{0},2,false,false,-1);}catch(const std::invalid_argument&){++refused;}
view.writeable=false;try{graycomatrix(view,{1},{0},2);}catch(const std::invalid_argument&){++refused;}
view.writeable=true;try{graycomatrix(view,{1},{0},std::numeric_limits<std::size_t>::max());}catch(const std::overflow_error&){++refused;}
assert(refused==4);
// Above the existing parallel threshold: symmetry writes independent cells.
std::vector<std::uint8_t> large(257*257);for(std::size_t k=0;k<large.size();++k)large[k]=std::uint8_t(k%16);
GrayImageView big{large.data(),257,257,257,1,GrayIntegerType::u8};auto one=graycomatrix(big,{1,2},{0,1.5707963267948966},256,true,true,1);auto four=graycomatrix(big,{1,2},{0,1.5707963267948966},256,true,true,4);assert(one.probabilities.size()==256*256*4&&one.probabilities==four.probabilities);assert(one.counts.empty()&&four.counts.empty());
set_num_threads(4);auto out=graycomatrix(view,{1},{0},2);assert(out.counts.size()==4);set_num_threads(1);assert(get_num_threads()==1);std::cout<<"PASS saved_returns="<<checked<<" distinct=63 budgets=0,1,4 controls=6 owning_operations=2\n";}
