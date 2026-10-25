#include "skimagexx/h_extrema.hpp"
#include "skimagexx/image.hpp"
#include "half.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
namespace {
struct Type {std::size_t bytes;std::uint64_t mask,sign;};
Type integer_type(ReconstructionType type) {
    std::size_t bytes=0;bool signed_value=false;
    switch(type) {
    case ReconstructionType::uint8:bytes=1;break;
    case ReconstructionType::int8:bytes=1;signed_value=true;break;
    case ReconstructionType::uint16:bytes=2;break;
    case ReconstructionType::int16:bytes=2;signed_value=true;break;
    case ReconstructionType::uint32:bytes=4;break;
    case ReconstructionType::int32:bytes=4;signed_value=true;break;
    case ReconstructionType::uint64:bytes=8;break;
    case ReconstructionType::int64:bytes=8;signed_value=true;break;
    default:throw std::invalid_argument("H-extrema integer stage requires an integer dtype");
    }
    const auto mask=bytes==8 ? ~std::uint64_t(0) : (std::uint64_t(1)<<(8*bytes))-1;
    return {bytes,mask,signed_value ? std::uint64_t(1)<<(8*bytes-1) : 0};
}
std::size_t width(ReconstructionType type) {
    return type==ReconstructionType::float64 ? 8 : integer_type(type).bytes;
}
std::size_t multiply(std::size_t a,std::size_t b) {
    if(b && a>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/b)
        throw std::overflow_error("H-extrema dimensions overflow");
    return a*b;
}
std::size_t add(std::size_t a,std::size_t b) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(b>limit || a>limit-b)throw std::overflow_error("H-extrema byte extent overflows");
    return a+b;
}
struct Range {std::uintptr_t begin=0,end=0;};
Range range(const void* data,std::size_t bytes) {
    if(!bytes)return {};
    const auto pointer=reinterpret_cast<std::uintptr_t>(data);
    if(!data || bytes>std::numeric_limits<std::uintptr_t>::max()-pointer)
        throw std::invalid_argument("H-extrema invalid memory span");
    return {pointer,pointer+bytes};
}
bool overlaps(Range a,Range b) {
    return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end;
}
template<class T> Range metadata(ReconstructionSpan<T> values) {
    if(values.size && reinterpret_cast<std::uintptr_t>(values.data)%alignof(T))
        throw std::invalid_argument("H-extrema unaligned metadata");
    return range(values.data,multiply(values.size,sizeof(T)));
}
struct Validated {std::size_t samples;Range extent;};
Validated validate(const ReconstructionView& view) {
    if(view.shape.size!=view.strides.size)throw std::invalid_argument("H-extrema shape/strides rank mismatch");
    metadata(view.shape);metadata(view.strides);
    std::size_t samples=1,negative=0,positive=0;
    for(std::size_t axis=0;axis<view.shape.size;++axis) {
        const auto length=view.shape.data[axis];
        if(!length)throw std::invalid_argument("H-extrema requires nonempty input");
        samples=multiply(samples,length);
        const auto stride=view.strides.data[axis];
        if(length==1)continue;
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())
            throw std::overflow_error("H-extrema byte stride overflows");
        const auto extent=multiply(length-1,std::size_t(stride<0 ? -stride : stride));
        if(stride<0)negative=add(negative,extent);else positive=add(positive,extent);
    }
    const auto item=width(view.type);positive=add(positive,item);
    const auto pointer=reinterpret_cast<std::uintptr_t>(view.data);
    const auto backing=range(view.allocation,view.allocation_bytes);
    if(!view.data || pointer<negative || positive>std::numeric_limits<std::uintptr_t>::max()-pointer)
        throw std::invalid_argument("H-extrema invalid input extent");
    const Range extent{pointer-negative,pointer+positive};
    if(extent.begin<backing.begin || extent.end>backing.end)
        throw std::invalid_argument("H-extrema view outside owning allocation");
    return {samples,extent};
}
Range output_range(ReconstructionSpan<unsigned char> output,std::size_t needed,
                   const ReconstructionView& first,const Validated& a,
                   const ReconstructionView* second=nullptr,const Validated* b=nullptr) {
    if(output.size<needed)throw std::invalid_argument("H-extrema output span too small");
    const auto out=range(output.data,needed);
    if(overlaps(out,range(&first,sizeof(first))) || overlaps(out,a.extent) ||
       overlaps(out,metadata(first.shape)) || overlaps(out,metadata(first.strides)))
        throw std::invalid_argument("H-extrema output overlaps input or metadata");
    if(second && (overlaps(out,range(second,sizeof(*second))) || overlaps(out,b->extent) ||
                  overlaps(out,metadata(second->shape)) || overlaps(out,metadata(second->strides))))
        throw std::invalid_argument("H-extrema output overlaps input or metadata");
    return out;
}
const unsigned char* address(const ReconstructionView& view,std::size_t index) {
    std::ptrdiff_t offset=0;
    for(std::size_t axis=view.shape.size;axis-->0;) {
        const auto coordinate=index%view.shape.data[axis];index/=view.shape.data[axis];
        offset+=std::ptrdiff_t(coordinate)*view.strides.data[axis];
    }
    return static_cast<const unsigned char*>(view.data)+offset;
}
std::uint64_t load(const ReconstructionView& view,std::size_t index,std::size_t bytes) {
    const auto* pointer=address(view,index);std::uint64_t value=0;
    if(bytes==1)value=*pointer;
    else if(bytes==2){std::uint16_t word;std::memcpy(&word,pointer,2);value=word;}
    else if(bytes==4){std::uint32_t word;std::memcpy(&word,pointer,4);value=word;}
    else std::memcpy(&value,pointer,8);
    if(view.swapped && bytes>1) {
        auto input=value;value=0;
        for(std::size_t i=0;i<bytes;++i){value=(value<<8)|(input&255);input>>=8;}
    }
    return value;
}
void store(unsigned char* output,std::uint64_t value,std::size_t bytes) {
    if(bytes==1)*output=static_cast<unsigned char>(value);
    else if(bytes==2){const auto word=std::uint16_t(value);std::memcpy(output,&word,2);}
    else if(bytes==4){const auto word=std::uint32_t(value);std::memcpy(output,&word,4);}
    else std::memcpy(output,&value,8);
}
void method_check(ReconstructionMethod method) {
    if(method!=ReconstructionMethod::dilation && method!=ReconstructionMethod::erosion)
        throw std::invalid_argument("H-extrema invalid reconstruction method");
}
struct Environment {
    fenv_t saved{};
    Environment() {
        if(std::fegetenv(&saved))throw std::domain_error("H-extrema FP environment unavailable");
    }
    ~Environment(){std::fesetenv(&saved);}
    void check() {
        if(std::fegetround()!=FE_TONEAREST)throw std::domain_error("H-extrema requires nearest rounding");
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        if(_mm_getcsr()&((1u<<15)|(1u<<6)))throw std::domain_error("H-extrema requires gradual underflow");
#endif
        if(std::feholdexcept(&saved))throw std::domain_error("H-extrema FP environment unavailable");
    }
};
double integer_double(std::uint64_t bits,Type type) {
    if(!type.sign)return double(bits);
    if(bits&type.sign)bits|=~type.mask;
    std::int64_t value;std::memcpy(&value,&bits,sizeof(value));return double(value);
}
double floating(const ReconstructionView& view,std::size_t index) {
    const auto bits=load(view,index,8);double value;std::memcpy(&value,&bits,8);return value;
}
}

std::size_t h_extrema_validate_view(const ReconstructionView& view) {return validate(view).samples;}
HExtremaRange h_extrema_integer_range(const ReconstructionView& view) {
    const auto type=integer_type(view.type);const auto valid=validate(view);
    auto low=load(view,0,type.bytes)^type.sign,high=low;
    for(std::size_t i=1;i<valid.samples;++i) {
        const auto value=load(view,i,type.bytes)^type.sign;
        low=std::min(low,value);high=std::max(high,value);
    }
    return {(high-low)&type.mask,valid.samples};
}
HExtremaInfo h_extrema_integer_seed_into(const ReconstructionView& view,std::uint64_t height,
    ReconstructionMethod method,ReconstructionSpan<unsigned char> output) {
    const auto type=integer_type(view.type);const auto valid=validate(view);method_check(method);
    if(height>type.mask)throw std::invalid_argument("H-extrema height exceeds integer dtype span");
    output_range(output,multiply(valid.samples,type.bytes),view,valid);
    int workers=1;
#ifdef _OPENMP
    if(valid.samples>=65536)workers=get_num_threads();
#pragma omp parallel for schedule(static) num_threads(workers) if(workers>1)
#endif
    for(std::ptrdiff_t index=0;index<static_cast<std::ptrdiff_t>(valid.samples);++index) {
        const auto i=std::size_t(index);const auto key=load(view,i,type.bytes)^type.sign;
        const auto shifted=method==ReconstructionMethod::dilation
            ? (height>key ? 0 : key-height)
            : (height>type.mask-key ? type.mask : key+height);
        store(output.data+i*type.bytes,shifted^type.sign,type.bytes);
    }
    return {workers,false};
}
HExtremaInfo h_extrema_integer_mask_into(const ReconstructionView& image,
    const ReconstructionView& reconstructed,double height,ReconstructionMethod method,
    ReconstructionSpan<unsigned char> output) {
    const auto type=integer_type(image.type);const auto a=validate(image);
    if(reconstructed.type!=ReconstructionType::float64)
        throw std::invalid_argument("H-extrema integer reconstruction must be float64");
    const auto b=validate(reconstructed);method_check(method);
    if(image.shape.size!=reconstructed.shape.size || a.samples!=b.samples)
        throw std::invalid_argument("H-extrema reconstruction shape mismatch");
    for(std::size_t axis=0;axis<image.shape.size;++axis)
        if(image.shape.data[axis]!=reconstructed.shape.data[axis])
            throw std::invalid_argument("H-extrema reconstruction shape mismatch");
    output_range(output,a.samples,image,a,&reconstructed,&b);
    Environment environment;environment.check();
    if(!std::isfinite(height) || height<0)throw std::domain_error("H-extrema requires finite nonnegative height");
    // A direct C++ caller must not get a partially written mask on rejection.
    for(std::size_t i=0;i<b.samples;++i)
        if(!std::isfinite(floating(reconstructed,i)))
            throw std::domain_error("H-extrema reconstruction must be finite");
    for(std::size_t i=0;i<a.samples;++i) {
        const auto value=integer_double(load(image,i,type.bytes),type);
        const auto rec=floating(reconstructed,i);
        const double residue=method==ReconstructionMethod::dilation ? value-rec : rec-value;
        output.data[i]=static_cast<unsigned char>(residue>=height);
    }
    return {1,false};
}

namespace {
std::size_t real_width(ReconstructionType type) {
    if(type==ReconstructionType::boolean)return 1;
    if(type==ReconstructionType::float16)return 2;
    if(type==ReconstructionType::float32)return 4;
    return width(type);
}
// Kept separate from the integer-v1 validator: its dtype rejection and error
// behavior must not change when real-stage input support is added.
Validated validate_real(const ReconstructionView& view) {
    if(view.shape.size!=view.strides.size)throw std::invalid_argument("H-extrema shape/strides rank mismatch");
    metadata(view.shape);metadata(view.strides);
    std::size_t samples=1,negative=0,positive=0;
    for(std::size_t axis=0;axis<view.shape.size;++axis) {
        const auto length=view.shape.data[axis];
        if(!length)throw std::invalid_argument("H-extrema requires nonempty input");
        samples=multiply(samples,length);
        const auto stride=view.strides.data[axis];
        if(length==1)continue;
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())
            throw std::overflow_error("H-extrema byte stride overflows");
        const auto extent=multiply(length-1,std::size_t(stride<0 ? -stride : stride));
        if(stride<0)negative=add(negative,extent);else positive=add(positive,extent);
    }
    positive=add(positive,real_width(view.type));
    const auto pointer=reinterpret_cast<std::uintptr_t>(view.data);
    const auto backing=range(view.allocation,view.allocation_bytes);
    if(!view.data || pointer<negative || positive>std::numeric_limits<std::uintptr_t>::max()-pointer)
        throw std::invalid_argument("H-extrema invalid input extent");
    const Range extent{pointer-negative,pointer+positive};
    if(extent.begin<backing.begin || extent.end>backing.end)
        throw std::invalid_argument("H-extrema view outside owning allocation");
    return {samples,extent};
}
double real_value(const ReconstructionView& view,std::size_t index) {
    if(view.type==ReconstructionType::boolean)return load(view,index,1)!=0 ? 1. : 0.;
    if(view.type==ReconstructionType::float16)
        return double(detail::decode_half(std::uint16_t(load(view,index,2))));
    if(view.type==ReconstructionType::float32) {
        const auto bits=std::uint32_t(load(view,index,4));float value;
        std::memcpy(&value,&bits,sizeof(value));return double(value);
    }
    if(view.type==ReconstructionType::float64)return floating(view,index);
    const auto type=integer_type(view.type);
    return integer_double(load(view,index,type.bytes),type);
}
bool exact_float32(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const auto exponent=unsigned((bits>>52)&2047);
    const auto fraction=bits&0xfffffffffffffULL;
    if(exponent==2047)return false;
    if(!exponent)return fraction==0; // F64 subnormals cannot be nonzero F32.
    const int power=int(exponent)-1023;
    if(power>127 || power<-149)return false;
    const auto significand=fraction|(std::uint64_t(1)<<52);
    const unsigned discard=power>=-126 ? 29u : unsigned(-power-97);
    return (significand&((std::uint64_t(1)<<discard)-1))==0;
}
}

std::size_t h_extrema_validate_real_view(const ReconstructionView& view) {
    return validate_real(view).samples;
}
HExtremaRealInfo h_extrema_real_mask_into(const ReconstructionView& image,
    const ReconstructionView& reconstructed,double height,ReconstructionMethod method,
    ReconstructionSpan<unsigned char> output) {
    const auto a=validate_real(image),b=validate_real(reconstructed);method_check(method);
    const bool single=image.type==ReconstructionType::float16 || image.type==ReconstructionType::float32;
    if(reconstructed.type!=(single ? ReconstructionType::float32 : ReconstructionType::float64))
        throw std::invalid_argument("H-extrema real reconstruction precision mismatch");
    if(image.shape.size!=reconstructed.shape.size || a.samples!=b.samples)
        throw std::invalid_argument("H-extrema reconstruction shape mismatch");
    for(std::size_t axis=0;axis<image.shape.size;++axis)
        if(image.shape.data[axis]!=reconstructed.shape.data[axis])
            throw std::invalid_argument("H-extrema reconstruction shape mismatch");
    output_range(output,a.samples,image,a,&reconstructed,&b);
    Environment environment;
    if(std::fegetround()!=FE_TONEAREST)return {};
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)))return {};
#endif
    environment.check();
    if(!std::isfinite(height) || height<0 || (single && !exact_float32(height)))return {};
    const double bound=single ? double(std::numeric_limits<float>::max())/2. : std::numeric_limits<double>::max()/2.;
    // Admission is input-defined and completes before the first store. No
    // candidate/oracle mask or approximate threshold selects this branch.
    for(std::size_t i=0;i<a.samples;++i) {
        const auto value=real_value(image,i),rec=real_value(reconstructed,i);
        if(!std::isfinite(value) || !std::isfinite(rec) || std::abs(value)>bound || std::abs(rec)>bound)return {};
    }
    if(single) {
        const float threshold=static_cast<float>(height);
        for(std::size_t i=0;i<a.samples;++i) {
            const float value=static_cast<float>(real_value(image,i));
            const float rec=static_cast<float>(real_value(reconstructed,i));
            // Force one source F32 rounding even on excess-precision targets.
            volatile float residue=method==ReconstructionMethod::dilation ? value-rec : rec-value;
            output.data[i]=static_cast<unsigned char>(residue>=threshold);
        }
    } else {
        for(std::size_t i=0;i<a.samples;++i) {
            const double value=real_value(image,i),rec=real_value(reconstructed,i);
            volatile double residue=method==ReconstructionMethod::dilation ? value-rec : rec-value;
            output.data[i]=static_cast<unsigned char>(residue>=height);
        }
    }
    return {true,{1,false}};
}
}
