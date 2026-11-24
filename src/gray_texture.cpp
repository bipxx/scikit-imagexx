// Source order: scikit-image0.26 feature/texture.py and feature/_texture.pyx.
// BSD-3-Clause; see third_party/scikit-image-LICENSE.txt.
#include <skimagexx/gray_texture.hpp>
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__GLIBC__) && defined(__GNUC__) && defined(__x86_64__)
// Both retained NumPy wheels bind this scalar math ABI. The host's default
// GLIBC_2.29 log differs on exceptional inputs; this is a generic primitive
// symbol binding for all values, never a NaN/result rewrite.
extern "C" double gray_texture_numpy_log(double);
__asm__(".symver gray_texture_numpy_log,log@GLIBC_2.2.5");
#endif
namespace skimagexx { namespace {
double source_log(double x){
#if defined(__GLIBC__) && defined(__GNUC__) && defined(__x86_64__)
return gray_texture_numpy_log(x);
#else
return std::log(x); // Exact other-platform math ABI remains unassessed.
#endif
}

std::size_t product(std::size_t a,std::size_t b) {if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("gray texture allocation overflow");return a*b;}
template<class T>T load(const char* p,bool swap){T v;unsigned char bytes[sizeof(T)];std::memcpy(bytes,p,sizeof(T));if(swap)std::reverse(bytes,bytes+sizeof(T));std::memcpy(&v,bytes,sizeof(T));return v;}
std::uint64_t integer(GrayImageView v,std::size_t r,std::size_t c){const auto* p=static_cast<const char*>(v.data)+std::ptrdiff_t(r)*v.row_stride+std::ptrdiff_t(c)*v.column_stride;
#define LOAD_S(K,T) case GrayIntegerType::K:{auto x=load<T>(p,v.swapped);if(x<0)throw std::invalid_argument("Negative-valued images are not supported.");return static_cast<std::uint64_t>(x);}
#define LOAD_U(K,T) case GrayIntegerType::K:return static_cast<std::uint64_t>(load<T>(p,v.swapped));
switch(v.type){LOAD_S(i8,std::int8_t) LOAD_U(u8,std::uint8_t) LOAD_S(i16,std::int16_t) LOAD_U(u16,std::uint16_t) LOAD_S(i32,std::int32_t) LOAD_U(u32,std::uint32_t) LOAD_S(i64,std::int64_t) LOAD_U(u64,std::uint64_t)}
#undef LOAD_S
#undef LOAD_U
throw std::invalid_argument("gray texture integer type");}
// NumPy contiguous float sum tree (same8lane/128leaf graph as metrics.cpp).
template<class F>double pairwise(std::size_t n,const F& get,std::size_t begin=0){if(n<8){double s=-0.;for(std::size_t i=0;i<n;++i)s+=get(begin+i);return s;}if(n>128){auto h=n/2;h-=h%8;return pairwise(h,get,begin)+pairwise(n-h,get,begin+h);}double v[8];for(std::size_t j=0;j<8;++j)v[j]=get(begin+j);std::size_t i=8;for(;i+8<=n;i+=8)for(std::size_t j=0;j<8;++j)v[j]+=get(begin+i+j);double s=((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7]));for(;i<n;++i)s+=get(begin+i);return s;}
std::ptrdiff_t rounded(double x){if(!std::isfinite(x)||x<=double(std::numeric_limits<std::ptrdiff_t>::min())+1.||x>=double(std::numeric_limits<std::ptrdiff_t>::max())-1.)throw std::invalid_argument("graycomatrix unsafe nonfinite/wide offset");return static_cast<std::ptrdiff_t>(x>=0.?x+.5:x-.5);}
void source_clear();
int budget(int workers){if(workers<0)throw std::invalid_argument("workers must be nonnegative");return workers?workers:get_num_threads();}
}
GrayMatrix graycomatrix(GrayImageView image,const std::vector<double>& distances,const std::vector<double>& angles,std::size_t levels,bool symmetric,bool normed,int workers){
 const int captured=budget(workers);if(!image.writeable)throw std::invalid_argument("buffer source array is read-only");if(image.swapped&&image.type!=GrayIntegerType::i8&&image.type!=GrayIntegerType::u8)throw std::invalid_argument("Big-endian buffer not supported on little-endian compiler");if(!image.data||!image.rows||!image.columns||!levels||distances.empty()||angles.empty())throw std::invalid_argument("graycomatrix empty shape");
 if(image.rows>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())||image.columns>std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))throw std::overflow_error("graycomatrix image extent");
 const auto planes=product(distances.size(),angles.size()),cells=product(levels,levels);GrayMatrix result{levels,distances.size(),angles.size(),normed,{},{}};result.counts.assign(product(cells,planes),0);
 // Validate once before independent plane processing; uint64 labels never cast
 // through signed storage. No source-image bound fixed to retained tiny cases.
 for(std::size_t r=0;r<image.rows;++r)for(std::size_t c=0;c<image.columns;++c)if(integer(image,r,c)>=levels)throw std::invalid_argument("The maximum grayscale value in the image should be smaller than the number of levels.");
 for(std::size_t a=0;a<angles.size();++a)for(std::size_t d=0;d<distances.size();++d){const double angle=angles[a],distance=distances[d];const auto dr=rounded(std::sin(angle)*distance),dc=rounded(std::cos(angle)*distance);const auto p=d*angles.size()+a;const auto rows=std::ptrdiff_t(image.rows),cols=std::ptrdiff_t(image.columns);if(dr>=rows||dr<=-rows||dc>=cols||dc<=-cols)continue;const auto r0=std::max(std::ptrdiff_t(0),-dr),r1=std::min(rows,rows-dr),c0=std::max(std::ptrdiff_t(0),-dc),c1=std::min(cols,cols-dc);for(auto r=r0;r<r1;++r)for(auto c=c0;c<c1;++c){const auto i=integer(image,std::size_t(r),std::size_t(c)),j=integer(image,std::size_t(r+dr),std::size_t(c+dc));++result.counts[(i*levels+j)*planes+p];}}
 if(symmetric){auto copy=result.counts;detail::partition(result.counts.size(),result.counts.size()>=detail::parallel_min?captured:1,[&](auto begin,auto end,auto){for(auto k=begin;k<end;++k){const auto cell=k/planes,i=cell/levels,j=cell%levels;result.counts[k]=copy[k]+copy[(j*levels+i)*planes+k%planes];}});}
 if(normed){source_clear();result.probabilities.resize(result.counts.size());for(std::size_t k=0;k<result.counts.size();++k)result.probabilities[k]=double(result.counts[k]);for(std::size_t p=0;p<planes;++p){double sum=0.;if(planes==1)sum=pairwise(cells,[&](auto i){return result.probabilities[i];});else for(std::size_t i=0;i<cells;++i)sum+=result.probabilities[i*planes+p];if(sum==0.)sum=1.;for(std::size_t i=0;i<cells;++i)result.probabilities[i*planes+p]/=sum;}result.counts.clear();}
 return result;
}
// NumPy npy_clear_floatstatus_barrier, pinned1.24.4/2.5.4 ieee754.c.src:
// source ufunc boundaries clear only four reportable exceptions, not precision
// or denormal operands. This precedes operations, never overwrites final state.
namespace {
void source_clear(){constexpr int flags=FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID;if(std::fetestexcept(flags))std::feclearexcept(flags);}
unsigned source_flags(){const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);return (f&FE_INVALID?1u:0u)|(f&FE_DIVBYZERO?2u:0u)|(f&FE_OVERFLOW?4u:0u)|(f&FE_UNDERFLOW?8u:0u);}
double ordered_multiply(double left,double right){
#if defined(__GNUC__) && defined(__x86_64__)
// Preserve source np.multiply operand order, including opposite-signed NaNs.
__asm__ volatile("mulsd %1,%0" : "+x"(left) : "x"(right));return left;
#else
return left*right;
#endif
}
}
// Fixed four-axis GLCM temporary metadata, not a general array/type engine.
// NPY_KEEPORDER's stable insertion comparison, pinned nditer_constr.c. Conflicting
// operands keep C-order; broadcast dimensions contribute zero strides.
namespace {
struct GrayLayout {std::array<std::size_t,4>shape;std::array<std::ptrdiff_t,4>strides;};
std::array<int,4> axis_order(GrayLayout a,const GrayLayout* b=nullptr){std::array<int,4>perm{3,2,1,0};for(int i=1;i<4;++i){const int x=perm[i];int pos=i;for(int j=i-1;j>=0;--j){bool ambiguous=true,swap=false;const int y=perm[j];for(const auto* v:std::array<const GrayLayout*,2>{&a,b})if(v&&v->shape[x]>1&&v->shape[y]>1&&v->strides[x]&&v->strides[y]){if(std::abs(v->strides[y])<=std::abs(v->strides[x]))swap=false;else if(ambiguous)swap=true;ambiguous=false;}if(!ambiguous){if(swap)pos=j;else break;}}for(int j=i;j>pos;--j)perm[j]=perm[j-1];perm[pos]=x;}return perm;}
GrayLayout binary_layout(GrayLayout a,GrayLayout b){GrayLayout out{};for(int k=0;k<4;++k)out.shape[k]=std::max(a.shape[k],b.shape[k]);auto order=axis_order(a,&b);std::ptrdiff_t stride=8;for(int k:order){out.strides[k]=stride;stride*=std::ptrdiff_t(out.shape[k]);}return out;}
GrayLayout reduced_layout(GrayLayout a){GrayLayout out{{1,1,a.shape[2],a.shape[3]},{0,0,0,0}};std::ptrdiff_t stride=8;for(int k:axis_order(a))if(k>=2){out.strides[k]=stride;stride*=std::ptrdiff_t(out.shape[k]);}return out;}
GrayLayout packed_layout(GrayLayout a){std::ptrdiff_t stride=8;for(int k:axis_order(a)){a.strides[k]=stride;stride*=std::ptrdiff_t(a.shape[k]);}return a;}
double probability_value(const char* p,GrayNumericType type,bool swapped){
 switch(type){
 case GrayNumericType::boolean:return load<std::uint8_t>(p,false)!=0;
 case GrayNumericType::i8:return double(load<std::int8_t>(p,false));case GrayNumericType::u8:return double(load<std::uint8_t>(p,false));
 case GrayNumericType::i16:return double(load<std::int16_t>(p,swapped));case GrayNumericType::u16:return double(load<std::uint16_t>(p,swapped));
 case GrayNumericType::i32:return double(load<std::int32_t>(p,swapped));case GrayNumericType::u32:return double(load<std::uint32_t>(p,swapped));
 case GrayNumericType::i64:return double(load<std::int64_t>(p,swapped));case GrayNumericType::u64:return double(load<std::uint64_t>(p,swapped));
 case GrayNumericType::f16:return double(detail::decode_half(load<std::uint16_t>(p,swapped)));
 case GrayNumericType::f32:case GrayNumericType::c64:return double(load<float>(p,swapped));
 case GrayNumericType::f64:case GrayNumericType::c128:return load<double>(p,swapped);
 }throw std::invalid_argument("graycoprops POD input type");}
struct GrayData {std::vector<double>v;GrayLayout layout;};
}
GrayProperties graycoprops(GrayProbabilityView source,GrayProperty property,int workers,GrayPolicyCallback policy,void* context){
 const int captured=budget(workers);(void)captured;const auto L=source.levels,planes=product(source.distances,source.angles),cells=product(L,L),length=product(cells,planes);
 if(!source.data||!L||!source.distances||!source.angles)throw std::invalid_argument("graycoprops empty shape");
 GrayData P{std::vector<double>(length),packed_layout(GrayLayout{{L,L,source.distances,source.angles},{source.strides[0],source.strides[1],source.strides[2],source.strides[3]}})};
 if((source.type==GrayNumericType::c64||source.type==GrayNumericType::c128)&&policy)policy("complex-cast",0,context);
 for(std::size_t i=0;i<L;++i)for(std::size_t j=0;j<L;++j)for(std::size_t d=0;d<source.distances;++d)for(std::size_t a=0;a<source.angles;++a){double value;const auto offset=std::ptrdiff_t(i)*source.strides[0]+std::ptrdiff_t(j)*source.strides[1]+std::ptrdiff_t(d)*source.strides[2]+std::ptrdiff_t(a)*source.strides[3];value=probability_value(reinterpret_cast<const char*>(source.data)+offset,source.type,source.swapped);P.v[(i*L+j)*planes+d*source.angles+a]=value;}
 auto notify=[&](const char* op){auto flags=source_flags();if(flags&&policy)policy(op,flags,context);};
 auto reduction=[&](const GrayData& x){source_clear();GrayData out{std::vector<double>(planes),reduced_layout(x.layout)};auto order=axis_order(x.layout);int fast_reduction=-1;std::size_t chunk=1;for(int axis:order)if(x.layout.shape[axis]>1){if(axis<2){if(fast_reduction<0)fast_reduction=axis;chunk*=L;}else break;}if(fast_reduction<0){for(int axis:order)if(axis<2){fast_reduction=axis;break;}}
  for(std::size_t p=0;p<planes;++p){auto get=[&](std::size_t k){if(fast_reduction==0)k=(k%L)*L+k/L;return x.v[k*planes+p];};double sum=0.;for(std::size_t begin=0;begin<cells;begin+=chunk)sum+=pairwise(chunk,[&](auto i){return get(begin+i);});out.v[p]=sum;}notify("reduce");return out;};
 auto transform=[&](const char* op,std::size_t n,GrayLayout layout,auto get){source_clear();GrayData out{std::vector<double>(n),layout};for(std::size_t k=0;k<n;++k)out.v[k]=get(k);notify(op);return out;};
 auto totals=reduction(P);for(auto& t:totals.v)if(t==0.)t=1.;
 P=transform("divide",length,P.layout,[&](auto k){return P.v[k]/totals.v[k%planes];});
 if(property==GrayProperty::invalid)throw std::domain_error("graycoprops invalid property");
 const GrayLayout I{{L,1,1,1},{8,8,8,8}},J{{1,L,1,1},{std::ptrdiff_t(L*8),8,8,8}};
 auto coordinate=[&](bool second){return transform("multiply",length,binary_layout(second?J:I,P.layout),[&](auto k){return ordered_multiply(double(second?(k/planes)%L:(k/planes)/L),P.v[k]);});};
 auto mean=[&](bool second){return reduction(coordinate(second));};
 auto differences=[&](const GrayData& means,bool second){return transform("subtract",length,binary_layout(second?J:I,means.layout),[&](auto k){return double(second?(k/planes)%L:(k/planes)/L)-means.v[k%planes];});};
 auto variance=[&](const GrayData& delta){auto square=transform("square",length,delta.layout,[&](auto k){return delta.v[k]*delta.v[k];});auto weighted=transform("multiply",length,binary_layout(P.layout,square.layout),[&](auto k){return ordered_multiply(P.v[k],square.v[k]);});return reduction(weighted);};
 auto root=[&](const GrayData& x){return transform("sqrt",planes,x.layout,[&](auto k){return std::sqrt(x.v[k]);});};
 GrayData answer;
 if(property==GrayProperty::mean)answer=mean(false);
 else if(property==GrayProperty::variance||property==GrayProperty::stddev){auto means=mean(false);auto delta=differences(means,false);answer=variance(delta);if(property==GrayProperty::stddev)answer=root(answer);}
 else if(property==GrayProperty::correlation){auto mi=mean(false),mj=mean(true);auto di=differences(mi,false),dj=differences(mj,true);auto si=root(variance(di)),sj=root(variance(dj));auto cross=transform("multiply",length,binary_layout(di.layout,dj.layout),[&](auto k){return ordered_multiply(di.v[k],dj.v[k]);});auto weighted=transform("multiply",length,binary_layout(P.layout,cross.layout),[&](auto k){return ordered_multiply(P.v[k],cross.v[k]);});auto cov=reduction(weighted);std::vector<unsigned char>regular(planes);source_clear();for(std::size_t k=0;k<planes;++k)regular[k]=!(si.v[k]<1e-15||sj.v[k]<1e-15);auto denominator=transform("multiply",planes,binary_layout(si.layout,sj.layout),[&](auto k){return regular[k]?ordered_multiply(si.v[k],sj.v[k]):1.;});answer=transform("divide",planes,GrayLayout{{1,1,source.distances,source.angles},{0,0,std::ptrdiff_t(source.angles*8),8}},[&](auto k){return regular[k]?cov.v[k]/denominator.v[k]:1.;});}
 else if(property==GrayProperty::entropy){auto logarithm=transform("log",length,P.layout,[&](auto k){return P.v[k]!=0.?source_log(P.v[k]):0.;});auto negative=transform("negative",length,logarithm.layout,[&](auto k){return -logarithm.v[k];});auto weighted=transform("multiply",length,binary_layout(P.layout,negative.layout),[&](auto k){return ordered_multiply(P.v[k],negative.v[k]);});answer=reduction(weighted);}
 else{GrayLayout weights{{L,L,1,1},{std::ptrdiff_t(L*8),8,0,0}};auto layout=property==GrayProperty::ASM||property==GrayProperty::energy?P.layout:binary_layout(P.layout,weights);auto weighted=transform("multiply",length,layout,[&](auto k){const double v=P.v[k],delta=double((k/planes)/L)-double((k/planes)%L);switch(property){case GrayProperty::contrast:return ordered_multiply(v,delta*delta);case GrayProperty::dissimilarity:return ordered_multiply(v,std::abs(delta));case GrayProperty::homogeneity:return ordered_multiply(v,1./(1.+delta*delta));case GrayProperty::ASM:case GrayProperty::energy:return v*v;default:throw std::invalid_argument("graycoprops property");}});answer=reduction(weighted);if(property==GrayProperty::energy)answer=root(answer);}
 const bool fortran=answer.layout.strides[2]<answer.layout.strides[3];
 return GrayProperties{source.distances,source.angles,std::move(answer.v),fortran};
}
}
