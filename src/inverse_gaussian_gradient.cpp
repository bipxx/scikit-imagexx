// scikit-image0.26 morphsnakes / SciPy ndimage source order, BSD-3-Clause.
#include <skimagexx/inverse_gaussian_gradient.hpp>
#include <skimagexx/image.hpp>
#include "execution.hpp"
#include "resize_cast.hpp"
#include "hessian_detail.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace skimagexx { namespace {
std::size_t count(const std::vector<std::size_t>& shape) {
    if(shape.size()>32)throw std::invalid_argument("IGG rank exceeds32");
    std::size_t n=1;for(auto d:shape){if(d&&n>std::size_t(PTRDIFF_MAX)/d)throw std::overflow_error("IGG shape overflow");n*=d;}return n;
}
int budget(int requested){const int n=requested?requested:get_num_threads();if(n<1||n>1024)throw std::invalid_argument("IGG worker budget");return n;}
// npy_clear_floatstatus_barrier clears only the four policy exceptions when
// present (retained core disassembly: feclearexcept argument0x1d). Inexact is
// not a NumPy policy exception and must not be cleared as FE_ALL_EXCEPT did.
void source_clear(){constexpr int mask=FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW;if(std::fetestexcept(mask))std::feclearexcept(mask);}
template<class T> struct complex_type:std::false_type{};
template<class T> struct complex_type<std::complex<T>>:std::true_type{};
template<class T> using Wide=std::conditional_t<complex_type<T>::value,std::complex<double>,double>;
template<class T> T cast(Wide<T> value,bool boolean=false) {
    if constexpr(complex_type<T>::value)return T(value);
    // ndimage's NPY_BOOL line buffer stores an npy_bool byte via raw cast;
    // canonical Boolean truth conversion happens in subsequent NumPy ufuncs.
    else if(boolean)return detail::resize_raw_cast<T>(value);
    else return detail::resize_raw_cast<T>(value);
}
template<class T> T product(T a,T b,bool boolean) {
    if constexpr(std::is_integral_v<T>){if(boolean)return T(bool(a)&&bool(b));using U=std::make_unsigned_t<T>;const U word=U(std::uint64_t(U(a))*std::uint64_t(U(b)));T out;std::memcpy(&out,&word,sizeof(T));return out;}
    else return a*b;
}
template<class T> T add(T a,T b,bool boolean) {
    if constexpr(std::is_integral_v<T>){if(boolean)return T(bool(a)||bool(b));using U=std::make_unsigned_t<T>;const U word=U(std::uint64_t(U(a))+std::uint64_t(U(b)));T out;std::memcpy(&out,&word,sizeof(T));return out;}
    else return a+b;
}
// Hessian's source symmetry-first/antisymmetry decision and center/pair order,
// extended to the original input dtype. Each output is independent; no SIMD
// or parallel reassociation of its ordered correlation is permitted.
template<class T> void correlate(const std::vector<T>& input,std::vector<T>& output,
 const std::vector<std::size_t>& shape,const InverseGaussianPass& pass,bool boolean,int workers) {
    const auto axis=pass.axis;if(axis>=shape.size()||pass.weights.empty()||pass.weights.size()%2!=1)throw std::invalid_argument("IGG derivative pass");
    const auto radius=pass.weights.size()/2,length=shape[axis];if(!input.size())return;
    bool symmetric=true,antisymmetric=true;
    for(std::size_t j=0;j<radius;++j){symmetric&=std::abs(pass.weights[j]-pass.weights[2*radius-j])<=std::numeric_limits<double>::epsilon();antisymmetric&=std::abs(pass.weights[j]+pass.weights[2*radius-j])<=std::numeric_limits<double>::epsilon();}
    std::size_t stride=1;for(auto a=axis+1;a<shape.size();++a)stride*=shape[a];
#ifdef SKIMAGEXX_AVX2
    if constexpr(std::is_floating_point_v<T>){
        if(stride==1&&(symmetric||antisymmetric)&&has_avx2()){
            detail::partition(input.size()/length,input.size()>=detail::parallel_min?workers:1,[&](auto begin,auto end,auto){
                std::vector<double> padded(length+2*radius);std::vector<const void*> left(radius),right(radius);
                for(std::size_t j=0;j<radius;++j){left[j]=padded.data()+j;right[j]=padded.data()+2*radius-j;}
                for(auto row=begin;row<end;++row){const auto* from=input.data()+row*length;auto* to=output.data()+row*length;
                    for(std::size_t i=0;i<length;++i)padded[radius+i]=double(from[i]);
                    for(std::size_t j=0;j<radius;++j){padded[j]=double(from[0]);padded[radius+length+j]=double(from[length-1]);}
                    auto done=simd::hessian_convolve(left.data(),right.data(),padded.data()+radius,to,length,radius,pass.weights.data(),!symmetric,0.,false,std::is_same_v<T,float>);
                    for(;done<length;++done){double sum=padded[radius+done]*pass.weights[radius];for(std::size_t j=0;j<radius;++j){const double a=padded[done+j],b=padded[done+2*radius-j];sum+=(symmetric?a+b:a-b)*pass.weights[j];}to[done]=T(sum);}
                }
            });return;
        }
    }
#endif
    detail::partition(input.size(),input.size()>=detail::parallel_min?workers:1,[&](auto begin,auto end,auto){
        for(auto i=begin;i<end;++i){const auto x=(i/stride)%length,base=i-x*stride;Wide<T> sum=Wide<T>(input[i])*pass.weights[radius];
            if(symmetric||antisymmetric){for(std::size_t j=0;j<radius;++j){const auto distance=radius-j;const auto lo=x<distance?0:x-distance,hi=std::min(length-1,x+distance);const Wide<T> a=input[base+lo*stride],b=input[base+hi*stride];sum+=(symmetric?a+b:a-b)*pass.weights[j];}}
            else{sum=Wide<T>(input[base+std::min(length-1,x+radius)*stride])*pass.weights.back();for(std::size_t j=0;j+1<pass.weights.size();++j){const auto position=std::clamp(std::ptrdiff_t(x)+std::ptrdiff_t(j)-std::ptrdiff_t(radius),std::ptrdiff_t(0),std::ptrdiff_t(length-1));sum+=Wide<T>(input[base+std::size_t(position)*stride])*pass.weights[j];}}
            output[i]=cast<T>(sum,boolean);
        }
    });
}
template<class T> void magnitude(const void* raw,void* out,const std::vector<std::size_t>& shape,
 const InverseGaussianPasses& plans,bool boolean,int workers) {
    const auto n=count(shape);if(n&&(!raw||!out))throw std::invalid_argument("IGG null storage");
    const auto* input=static_cast<const T*>(raw);if(shape.empty()){std::memcpy(out,raw,sizeof(T));return;}
    if(plans.size()!=shape.size())throw std::invalid_argument("IGG one derivative per axis");
    std::vector<T> total(n),current(n),next(n);
    for(std::size_t derivative=0;derivative<shape.size();++derivative){if(n)std::copy_n(input,n,current.begin());
        for(const auto& pass:plans[derivative]){correlate(current,next,shape,pass,boolean,workers);current.swap(next);}
        // These are the source ufunc stage boundaries, not posthoc flag masks.
        source_clear();
        for(std::size_t i=0;i<n;++i)current[i]=product(current[i],current[i],boolean);
        if(!derivative)total=current;
        else{source_clear();for(std::size_t i=0;i<n;++i)total[i]=add(total[i],current[i],boolean);}
    }
    source_clear();
    for(std::size_t i=0;i<n;++i){if constexpr(complex_type<T>::value)total[i]=std::sqrt(total[i]);else total[i]=cast<T>(std::sqrt(double(total[i])),boolean);}
    if(n)std::memcpy(out,total.data(),n*sizeof(T));
}
std::size_t broadcast_offset(std::size_t i,const std::vector<std::size_t>& source,const std::vector<std::size_t>& target){
    std::size_t out=0,stride=1;for(auto a=target.size();a-->0;){const auto x=i%target[a];i/=target[a];if(a>=target.size()-source.size()){const auto d=source[a-(target.size()-source.size())];if(d!=1&&d!=target[a])throw std::invalid_argument("IGG incompatible broadcast");if(d!=1)out+=x*stride;stride*=d;}}return out;
}
template<class T> std::complex<T> source_divide(std::complex<T> left,std::complex<T> right){
    const T ar=left.real(),ai=left.imag(),br=right.real(),bi=right.imag();
    // NumPy CFLOAT/CDOUBLE divide scales after forming the ratio; libstdc++'s
    // generic complex quotient uses a different finite rounding association.
    if(std::abs(br)>=std::abs(bi)){
        if(br==T(0)&&bi==T(0))return {ar/std::abs(br),ai/std::abs(bi)};
        const T ratio=bi/br,scale=T(1)/(br+bi*ratio);
        return {(ar+ai*ratio)*scale,(ai-ar*ratio)*scale};
    }
    const T ratio=br/bi,scale=T(1)/(bi+br*ratio);
    return {(ar*ratio+ai)*scale,(ai*ratio-ar)*scale};
}
template<class T> unsigned finish(const void* g,const void* a,void* o,const std::vector<std::size_t>& gs,const std::vector<std::size_t>& as,const std::vector<std::size_t>& os,unsigned stage,bool boolean=false) {
    const auto n=count(os);if(gs.size()>os.size()||as.size()>os.size()||stage>3)throw std::invalid_argument("IGG finish plan");
    const auto* gradient=static_cast<const T*>(g);const auto* alpha=static_cast<const T*>(a);auto* output=static_cast<T*>(o);
    source_clear();
    for(std::size_t i=0;i<n;++i){const T value=gradient[stage==0?broadcast_offset(i,gs,os):i];
        if(stage==0)output[i]=product(alpha[broadcast_offset(i,as,os)],value,boolean);
        else if(stage==1)output[i]=T(1)+value;
        else if(stage==2)output[i]=std::sqrt(value);
        else {if constexpr(complex_type<T>::value)output[i]=source_divide(T(1),value);else output[i]=T(1)/value;}
    }
    const auto flags=std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW);
    return(flags&FE_INVALID?1u:0u)|(flags&FE_DIVBYZERO?2u:0u)|(flags&FE_OVERFLOW?4u:0u)|(flags&FE_UNDERFLOW?8u:0u);
}
// Same NumPy pairwise reduction graph already used by metrics.cpp.
double coefficient_sum(const double* values,std::size_t n){
    if(n>128){auto half=n/2;half-=half%8;return coefficient_sum(values,half)+coefficient_sum(values+half,n-half);}
    if(n<8){double sum=-0.;for(std::size_t i=0;i<n;++i)sum+=values[i];return sum;}
    double lanes[8];for(unsigned j=0;j<8;++j)lanes[j]=values[j];std::size_t i=8;
    for(;i+8<=n;i+=8)for(unsigned j=0;j<8;++j)lanes[j]+=values[i+j];
    double sum=((lanes[0]+lanes[1])+(lanes[2]+lanes[3]))+((lanes[4]+lanes[5])+(lanes[6]+lanes[7]));
    for(;i<n;++i)sum+=values[i];return sum;
}
InverseGaussianPasses coefficients(const std::vector<std::size_t>& shape,const std::vector<double>& sigma){
    if(sigma.size()!=1&&sigma.size()!=shape.size())throw std::invalid_argument("IGG sigma rank");InverseGaussianPasses plans(shape.size());
    for(std::size_t derivative=0;derivative<shape.size();++derivative)for(std::size_t axis=0;axis<shape.size();++axis){const double sd=sigma.size()==1?sigma[0]:sigma[axis];if(!(sd>1e-15))continue;if(!std::isfinite(sd)||sd>double(PTRDIFF_MAX/8))throw std::invalid_argument("IGG finite radius required");const auto r=std::size_t(4*sd+.5);std::vector<double> weights(2*r+1);const double variance=sd*sd;for(std::size_t j=0;j<weights.size();++j){const auto x=double(j)-double(r);weights[j]=std::exp((-.5/variance)*(x*x));}const double sum=coefficient_sum(weights.data(),weights.size());for(auto& w:weights)w/=sum;if(axis==derivative){for(std::size_t j=0;j<weights.size();++j){const double x=double(j)-double(r);weights[j]=(0.+x*(-1./variance))*weights[j];}}std::reverse(weights.begin(),weights.end());plans[derivative].push_back({axis,std::move(weights)});}
    return plans;
}
}
void inverse_gaussian_gradient_magnitude(const void* a,void* b,PixelType type,bool complex,
 const std::vector<std::size_t>& shape,const InverseGaussianPasses& plans,int requested){
    const int workers=budget(requested);
    if(complex){if(type==PixelType::float32)magnitude<std::complex<float>>(a,b,shape,plans,false,workers);else if(type==PixelType::float64)magnitude<std::complex<double>>(a,b,shape,plans,false,workers);else throw std::invalid_argument("IGG complex dtype");return;}
    switch(type){
#define IGG_TYPE(E,T,B) case PixelType::E:magnitude<T>(a,b,shape,plans,B,workers);break
    IGG_TYPE(boolean,std::uint8_t,true);IGG_TYPE(uint8,std::uint8_t,false);IGG_TYPE(int8,std::int8_t,false);IGG_TYPE(uint16,std::uint16_t,false);IGG_TYPE(int16,std::int16_t,false);IGG_TYPE(uint32,std::uint32_t,false);IGG_TYPE(int32,std::int32_t,false);IGG_TYPE(uint64,std::uint64_t,false);IGG_TYPE(int64,std::int64_t,false);IGG_TYPE(float32,float,false);IGG_TYPE(float64,double,false);
#undef IGG_TYPE
    }
}
unsigned inverse_gaussian_finish(const void* g,const void* a,void* o,PixelType type,bool complex,const std::vector<std::size_t>& gs,const std::vector<std::size_t>& as,const std::vector<std::size_t>& os,unsigned stage,int requested){
    budget(requested);if(complex){if(type==PixelType::float32)return finish<std::complex<float>>(g,a,o,gs,as,os,stage);if(type==PixelType::float64)return finish<std::complex<double>>(g,a,o,gs,as,os,stage);}else{
        if(type==PixelType::float32)return finish<float>(g,a,o,gs,as,os,stage);if(type==PixelType::float64)return finish<double>(g,a,o,gs,as,os,stage);
        if(stage!=0)throw std::invalid_argument("IGG integral finish supports multiply only");
        switch(type){
#define IGG_FINISH(E,T,B) case PixelType::E:return finish<T>(g,a,o,gs,as,os,stage,B)
        IGG_FINISH(boolean,std::uint8_t,true);IGG_FINISH(uint8,std::uint8_t,false);IGG_FINISH(int8,std::int8_t,false);IGG_FINISH(uint16,std::uint16_t,false);IGG_FINISH(int16,std::int16_t,false);IGG_FINISH(uint32,std::uint32_t,false);IGG_FINISH(int32,std::int32_t,false);IGG_FINISH(uint64,std::uint64_t,false);IGG_FINISH(int64,std::int64_t,false);
#undef IGG_FINISH
        default:break;
        }
    }throw std::invalid_argument("IGG finish dtype");
}
template<class T> InverseGaussianResult<T> inverse_gaussian_gradient(const T* image,const std::vector<std::size_t>& shape,const std::vector<double>& sigma,const T* alpha,const std::vector<std::size_t>& alpha_shape,int requested){
    const int workers=budget(requested);auto plans=coefficients(shape,sigma);std::vector<T> gradient(count(shape));const auto type=std::is_same_v<T,float>?PixelType::float32:PixelType::float64;
    inverse_gaussian_gradient_magnitude(image,gradient.data(),type,false,shape,plans,workers);
    InverseGaussianResult<T> result;result.captured_workers=workers;result.shape.resize(std::max(shape.size(),alpha_shape.size()),1);
    for(std::size_t i=0;i<result.shape.size();++i){const auto axis=result.shape.size()-1-i;const auto a=i<shape.size()?shape[shape.size()-1-i]:1,b=i<alpha_shape.size()?alpha_shape[alpha_shape.size()-1-i]:1;if(a!=b&&a!=1&&b!=1)throw std::invalid_argument("IGG alpha broadcast");result.shape[axis]=a==1?b:a;}
    if(count(alpha_shape)&&!alpha)throw std::invalid_argument("IGG alpha storage");
    result.values.resize(count(result.shape));
    inverse_gaussian_finish(gradient.data(),alpha,result.values.data(),type,false,shape,alpha_shape,result.shape,0,workers);
    for(unsigned stage=1;stage<4;++stage)inverse_gaussian_finish(result.values.data(),alpha,result.values.data(),type,false,result.shape,alpha_shape,result.shape,stage,workers);return result;
}
template<class T> std::vector<T> inverse_gaussian_gradient(const T* image,const std::vector<std::size_t>& shape,const std::vector<double>& sigma,T alpha,int requested){return inverse_gaussian_gradient(image,shape,sigma,&alpha,{},requested).values;}
template<class T> InverseGaussianResult<T> inverse_gaussian_gradient(const CropView& image,const std::vector<double>& sigma,const T* alpha,const std::vector<std::size_t>& alpha_shape,int requested){
    const int workers=budget(requested);
    if(image.itemsize!=sizeof(T)||image.shape.size()!=image.strides.size())throw std::invalid_argument("IGG typed input view");
    std::vector<std::size_t> shape;for(auto d:image.shape){if(d<0)throw std::invalid_argument("IGG negative extent");shape.push_back(std::size_t(d));}
    const auto n=count(shape);std::vector<T> dense(n);std::vector<CropSlice> axes;std::vector<std::ptrdiff_t> strides(shape.size());std::size_t stride=sizeof(T);
    for(auto d:image.shape)axes.push_back({0,d});
    for(auto a=shape.size();a-->0;){strides[a]=std::ptrdiff_t(stride);stride*=shape[a];}
    crop_copy_into(image,axes,dense.data(),strides);
    return inverse_gaussian_gradient(dense.data(),shape,sigma,alpha,alpha_shape,workers);
}
template std::vector<float> inverse_gaussian_gradient(const float*,const std::vector<std::size_t>&,const std::vector<double>&,float,int);
template std::vector<double> inverse_gaussian_gradient(const double*,const std::vector<std::size_t>&,const std::vector<double>&,double,int);
template InverseGaussianResult<float> inverse_gaussian_gradient(const float*,const std::vector<std::size_t>&,const std::vector<double>&,const float*,const std::vector<std::size_t>&,int);
template InverseGaussianResult<double> inverse_gaussian_gradient(const double*,const std::vector<std::size_t>&,const std::vector<double>&,const double*,const std::vector<std::size_t>&,int);
template InverseGaussianResult<float> inverse_gaussian_gradient(const CropView&,const std::vector<double>&,const float*,const std::vector<std::size_t>&,int);
template InverseGaussianResult<double> inverse_gaussian_gradient(const CropView&,const std::vector<double>&,const double*,const std::vector<std::size_t>&,int);
}
