// Source operation order: scikit-image0.26 feature.texture.draw_multiblock_lbp.
// BSD-3-Clause: third_party/scikit-image-LICENSE.txt. No original forwarding.
#include <skimagexx/draw_multiblock_lbp.hpp>
#include "execution.hpp"
#include "half.hpp"
#include <cfenv>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if !defined(SKIMAGEXX_LBP_SCALAR) && (defined(__SSE2__) || defined(_M_X64))
#include <emmintrin.h>
#define LBP_SSE2 1
#endif
#pragma STDC FENV_ACCESS ON
namespace skimagexx { namespace {
std::size_t count(const std::vector<std::ptrdiff_t>& shape) {
    std::size_t n=1;for(auto s:shape){if(s<0)throw std::invalid_argument("negative LBP dimension");if(s&&n>std::size_t(PTRDIFF_MAX)/std::size_t(s))throw std::overflow_error("LBP shape overflow");n*=std::size_t(s);}return n;
}
std::ptrdiff_t add(std::ptrdiff_t a,std::ptrdiff_t b){if((b>0&&a>PTRDIFF_MAX-b)||(b<0&&a<PTRDIFF_MIN-b))throw std::overflow_error("LBP index overflow");return a+b;}
std::ptrdiff_t offset(std::ptrdiff_t n,int sign){if(sign==0)return 0;if(sign<0){if(n==PTRDIFF_MIN)throw std::overflow_error("LBP offset overflow");return -n;}return n;}
std::ptrdiff_t clip(std::ptrdiff_t x,std::ptrdiff_t n){if(x<0)x=add(x,n);return std::max(std::ptrdiff_t(0),std::min(x,n));}
template<class T> std::vector<T> pack(const MultiblockLbpInput& in){
    if(in.image.itemsize!=sizeof(T))throw std::invalid_argument("LBP input itemsize mismatch");
    const auto n=count(in.image.shape);std::vector<T> a(n);std::vector<CropSlice> slices;std::vector<std::ptrdiff_t> strides(in.image.shape.size());std::ptrdiff_t stride=sizeof(T);
    for(std::size_t k=in.image.shape.size();k-->0;){strides[k]=stride;if(in.image.shape[k]&&stride>PTRDIFF_MAX/in.image.shape[k])throw std::overflow_error("LBP copy span overflow");stride*=in.image.shape[k];}
    for(auto s:in.image.shape)slices.push_back({0,s});crop_copy_into(in.image,slices,a.data(),strides);
    if(in.byte_swapped&&sizeof(T)>1)for(auto& value:a){auto* p=reinterpret_cast<unsigned char*>(&value);std::reverse(p,p+sizeof(T));}
    return a;
}
template<class T> std::vector<double> normalized(const MultiblockLbpInput& in){
    auto input=pack<T>(in);std::vector<double> out(input.size());
    // Existing converter's small-call path is serial and takes no global read.
    // Keep each call below parallel_min so the operation owns exactly one
    // captured worker budget, including large integer inputs.
    constexpr std::size_t chunk=detail::parallel_min-1;
    for(std::size_t k=0;k<input.size();k+=std::min(chunk,input.size()-k)){auto n=std::min(chunk,input.size()-k);convert_into(input.data()+k,out.data()+k,n,in.type,PixelType::float64);}return out;
}
void scale64(const double* a,double factor,double* out,std::size_t n,bool simd){
    std::size_t k=0;
#ifdef LBP_SSE2
    if(simd){const auto f=_mm_set1_pd(factor);for(;k+2<=n;k+=2)_mm_storeu_pd(out+k,_mm_mul_pd(_mm_loadu_pd(a+k),f));}
#else
    (void)simd;
#endif
    for(;k<n;++k)out[k]=a[k]*factor;
}
void sum64(const double* a,const double* b,double* out,std::size_t n,bool simd){
    std::size_t k=0;
#ifdef LBP_SSE2
    if(simd)for(;k+2<=n;k+=2)_mm_storeu_pd(out+k,_mm_add_pd(_mm_loadu_pd(a+k),_mm_loadu_pd(b+k)));
#else
    (void)simd;
#endif
    for(;k<n;++k)out[k]=a[k]+b[k];
}
std::size_t color_index(std::size_t flat,const std::vector<std::ptrdiff_t>& patch,const MultiblockLbpColor& color){
    if(color.shape.size()>patch.size())throw std::invalid_argument("LBP color cannot broadcast");std::size_t index=0,stride=1;
    for(std::size_t k=patch.size();k-->0;){auto dim=std::size_t(patch[k]);auto coord=dim?flat%dim:0;if(dim)flat/=dim;
        if(k+color.shape.size()>=patch.size()){auto s=color.shape[k+color.shape.size()-patch.size()];if(s!=1&&s!=patch[k])throw std::invalid_argument("LBP color cannot broadcast");if(s!=1)index+=coord*stride;stride*=std::size_t(s);}}
    return index;
}
void validate_color(const MultiblockLbpColor& c){if(count(c.shape)!=c.values.size())throw std::invalid_argument("LBP color storage mismatch");}
}
MultiblockLbpImage draw_multiblock_lbp(const MultiblockLbpInput& in,const MultiblockLbpParameters& p){
    MultiblockLbpImage result;result.captured_workers=get_num_threads();
    validate_color(p.greater);validate_color(p.less);const auto n=count(in.image.shape);result.shape=in.image.shape;
    const bool gray=result.shape.size()<3;if(gray)result.shape.push_back(3);if(result.shape.size()<2&&!p.prepare_only)throw std::invalid_argument("LBP output needs two axes");const auto total=count(result.shape);
    std::vector<double> converted;result.is_float32=in.type==PixelType::float32;
    if(in.half){auto words=pack<std::uint16_t>(in);result.is_float32=true;result.float32.resize(n);for(std::size_t k=0;k<n;++k)result.float32[k]=detail::decode_half(words[k]);}
    else switch(in.type){
    case PixelType::float32:result.float32=pack<float>(in);break;
    case PixelType::float64:converted=pack<double>(in);break;
    case PixelType::boolean:{auto words=pack<std::uint8_t>(in);converted.resize(n);for(std::size_t k=0;k<n;++k)converted[k]=words[k]!=0?1.:0.;break;}
    case PixelType::uint8:converted=normalized<std::uint8_t>(in);break;case PixelType::int8:converted=normalized<std::int8_t>(in);break;
    case PixelType::uint16:converted=normalized<std::uint16_t>(in);break;case PixelType::int16:converted=normalized<std::int16_t>(in);break;
    case PixelType::uint32:converted=normalized<std::uint32_t>(in);break;case PixelType::int32:converted=normalized<std::int32_t>(in);break;
    case PixelType::uint64:converted=normalized<std::uint64_t>(in);break;case PixelType::int64:converted=normalized<std::int64_t>(in);break;
    default:throw std::invalid_argument("LBP pixel type unsupported");}
    if(result.is_float32){if(gray){auto old=std::move(result.float32);result.float32.resize(total);for(std::size_t k=0;k<total;++k)result.float32[k]=old[k/3];}}
    else{result.float64.resize(total);for(std::size_t k=0;k<total;++k)result.float64[k]=converted[gray?k/3:k];}
    result.is_half=in.half;
    if(p.prepare_only){if(result.is_half){result.float16.resize(total);for(std::size_t k=0;k<total;++k)result.float16[k]=detail::encode_half(result.float32[k]);}return result;}
    bool simd=p.simd&&std::getenv("SKIMAGEXX_DISABLE_SIMD")==nullptr;
#ifdef LBP_SSE2
    result.sse2_enabled=simd;
#else
    result.sse2_enabled=false;simd=false;
#endif
    const int offsets[8][2]={{-1,-1},{-1,0},{-1,1},{0,1},{1,1},{1,0},{1,-1},{0,-1}};
    const auto tail=count(std::vector<std::ptrdiff_t>(result.shape.begin()+2,result.shape.end()));
    // Ordered serial phases intentionally retain same-thread rounding/sticky
    // semantics. No extra worker snapshot or speculative parallel FP reduction.
    for(int block=0;block<8;++block){
        const auto rr=add(add(p.r,p.height),offset(p.height,offsets[block][0])),cc=add(add(p.c,p.width),offset(p.width,offsets[block][1]));
        const auto r0=clip(rr,result.shape[0]),r1=std::max(r0,clip(add(rr,p.height),result.shape[0]));const auto c0=clip(cc,result.shape[1]),c1=std::max(c0,clip(add(cc,p.width),result.shape[1]));
        auto shape=result.shape;shape[0]=r1-r0;shape[1]=c1-c0;const auto size=count(shape);std::vector<double> first(size),values(size),rhs(size),sum(size);std::size_t k=0;
        for(auto r=r0;r<r1;++r)for(auto c=c0;c<c1;++c)for(std::size_t ch=0;ch<tail;++ch){const auto index=(std::size_t(r)*std::size_t(result.shape[1])+std::size_t(c))*tail+ch;first[k++]=result.is_float32?double(result.float32[index]):result.float64[index];}
        const double complement=p.explicit_complement?p.complement:1.0-p.alpha;
        if(result.is_float32&&(p.first_product_float32||p.first_product_half)){const float factor=p.first_product_half?detail::decode_half(detail::encode_half(float(complement))):float(complement);for(k=0;k<size;++k){volatile float product=float(first[k])*factor;values[k]=p.first_product_half?double(detail::decode_half(detail::encode_half(product))):double(product);}}
        else scale64(first.data(),complement,values.data(),size,simd);
        const auto& color=(std::uint8_t(p.code)&(1u<<(7-block)))?p.greater:p.less;std::vector<double> colors(color.values.size());scale64(color.values.data(),p.alpha,colors.data(),colors.size(),simd);
        // Even empty slices evaluate the color product and broadcasting.
        (void)color_index(0,shape,color);for(k=0;k<size;++k)rhs[k]=colors[color_index(k,shape,color)];sum64(values.data(),rhs.data(),sum.data(),size,simd);k=0;
        for(auto r=r0;r<r1;++r)for(auto c=c0;c<c1;++c)for(std::size_t ch=0;ch<tail;++ch){const auto index=(std::size_t(r)*std::size_t(result.shape[1])+std::size_t(c))*tail+ch;if(result.is_float32){auto value=float(sum[k++]);result.float32[index]=result.is_half?detail::decode_half(detail::encode_half(value)):value;}else result.float64[index]=sum[k++];}
    }
    if(result.is_half){result.float16.resize(total);for(std::size_t k=0;k<total;++k)result.float16[k]=detail::encode_half(result.float32[k]);}
    return result;
}
namespace {
std::ptrdiff_t byte_index(std::size_t flat,const std::vector<std::ptrdiff_t>& shape,const CropView& v){
    if(v.shape.size()>shape.size())throw std::invalid_argument("LBP broadcast rank");
    std::ptrdiff_t offset=0;
    for(std::size_t k=shape.size();k-->0;){const auto n=std::size_t(shape[k]);const auto coordinate=n?flat%n:0;if(n)flat/=n;
        if(k+v.shape.size()>=shape.size()){auto a=k+v.shape.size()-shape.size();if(v.shape[a]!=1&&v.shape[a]!=shape[k])throw std::invalid_argument("LBP broadcast shape");if(v.shape[a]!=1)offset+=std::ptrdiff_t(coordinate)*v.strides[a];}}
    return offset;
}
double read_word(const MultiblockLbpFloatView& v,std::ptrdiff_t offset){
    unsigned char b[8];const auto n=v.image.itemsize;if(n!=2&&n!=4&&n!=8)throw std::invalid_argument("LBP floating width");
    std::memcpy(b,static_cast<const unsigned char*>(v.image.data)+offset,n);if(v.swapped)std::reverse(b,b+n);
    if(n==2){std::uint16_t x;std::memcpy(&x,b,2);return detail::decode_half(x);}if(n==4){float x;std::memcpy(&x,b,4);return x;}double x;std::memcpy(&x,b,8);return x;
}
void write_word(const MultiblockLbpFloatView& v,std::ptrdiff_t offset,double value){
    unsigned char b[8];const auto n=v.image.itemsize;
    if(n==2){auto x=detail::encode_half(float(value));std::memcpy(b,&x,2);}else if(n==4){float x=float(value);std::memcpy(b,&x,4);}else if(n==8)std::memcpy(b,&value,8);else throw std::invalid_argument("LBP floating width");
    if(v.swapped)std::reverse(b,b+n);std::memcpy(const_cast<unsigned char*>(static_cast<const unsigned char*>(v.image.data))+offset,b,n);
}
}
unsigned multiblock_lbp_binary(const MultiblockLbpFloatView& a,const MultiblockLbpFloatView& b,const MultiblockLbpFloatView& out,bool plus){
    const auto n=count(out.image.shape);
    // NumPy ufuncs clear arithmetic status immediately before their loop and
    // report the loop's actual status. This is operation semantics, not a
    // comparison mask or caller-state restoration. Never restore flags here.
    std::feclearexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
    std::size_t k=0;
#ifdef LBP_SSE2
    auto packed=[](const MultiblockLbpFloatView& v){std::ptrdiff_t stride=std::ptrdiff_t(v.image.itemsize);for(std::size_t d=v.image.shape.size();d-->0;){if(v.image.shape[d]>1&&v.image.strides[d]!=stride)return false;stride*=v.image.shape[d];}return !v.swapped;};
    const bool scalar_a=a.image.shape.empty(),scalar_b=b.image.shape.empty();
    if(std::getenv("SKIMAGEXX_DISABLE_SIMD")==nullptr&&packed(a)&&packed(b)&&packed(out)&&a.image.itemsize==out.image.itemsize&&b.image.itemsize==out.image.itemsize&&(scalar_a||a.image.shape==out.image.shape)&&(scalar_b||b.image.shape==out.image.shape)){
        if(out.image.itemsize==8){const auto* x=static_cast<const double*>(a.image.data);const auto* y=static_cast<const double*>(b.image.data);auto* z=const_cast<double*>(static_cast<const double*>(out.image.data));for(;k+2<=n;k+=2){const auto xx=scalar_a?_mm_set1_pd(read_word(a,0)):_mm_loadu_pd(x+k),yy=scalar_b?_mm_set1_pd(read_word(b,0)):_mm_loadu_pd(y+k);_mm_storeu_pd(z+k,plus?_mm_add_pd(xx,yy):_mm_mul_pd(xx,yy));}}
        else if(out.image.itemsize==4){const auto* x=static_cast<const float*>(a.image.data);const auto* y=static_cast<const float*>(b.image.data);auto* z=const_cast<float*>(static_cast<const float*>(out.image.data));for(;k+4<=n;k+=4){const auto xx=scalar_a?_mm_set1_ps(float(read_word(a,0))):_mm_loadu_ps(x+k),yy=scalar_b?_mm_set1_ps(float(read_word(b,0))):_mm_loadu_ps(y+k);_mm_storeu_ps(z+k,plus?_mm_add_ps(xx,yy):_mm_mul_ps(xx,yy));}}
    }
#endif
    for(;k<n;++k){double x=read_word(a,byte_index(k,out.image.shape,a.image)),y=read_word(b,byte_index(k,out.image.shape,b.image));double z;
        if(out.image.itemsize<=4){volatile float xx=float(x),yy=float(y);volatile float zz=plus?xx+yy:xx*yy;z=zz;}
        else{volatile double xx=x,yy=y;volatile double zz=plus?xx+yy:xx*yy;z=zz;}
        write_word(out,byte_index(k,out.image.shape,out.image),z);
    }
    const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
    return unsigned(bool(f&FE_DIVBYZERO))|unsigned(bool(f&FE_OVERFLOW))*2|unsigned(bool(f&FE_UNDERFLOW))*4|unsigned(bool(f&FE_INVALID))*8;
}
void multiblock_lbp_assign(const MultiblockLbpFloatView& out,const MultiblockLbpFloatView& a){
    const auto n=count(out.image.shape);for(std::size_t k=0;k<n;++k)write_word(out,byte_index(k,out.image.shape,out.image),read_word(a,byte_index(k,out.image.shape,a.image)));
}
}
