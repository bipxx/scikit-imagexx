#include "skimagexx/reconstruction.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace skimagexx {
#ifdef SKIMAGEXX_AVX2
namespace detail {
bool reconstruction_order_u8(const std::uint8_t*,const std::uint8_t*,std::size_t,bool);
bool reconstruction_order_u16(const std::uint16_t*,const std::uint16_t*,std::size_t,bool);
void reconstruction_copy_u8_f64(const std::uint8_t*,double*,std::size_t);
void reconstruction_copy_u16_f64(const std::uint16_t*,double*,std::size_t);
}
#endif
namespace {
constexpr auto absent=std::numeric_limits<std::size_t>::max();
std::size_t mul(std::size_t a,std::size_t b) {
    if(b && a>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/b)throw std::overflow_error("reconstruction dimensions overflow");
    return a*b;
}
std::size_t add(std::size_t a,std::size_t b) {
    if(a>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())-b)throw std::overflow_error("reconstruction dimensions overflow");
    return a+b;
}
std::size_t width(ReconstructionType t) {
    switch(t) {
    case ReconstructionType::boolean:case ReconstructionType::uint8:case ReconstructionType::int8:return 1;
    case ReconstructionType::uint16:case ReconstructionType::int16:case ReconstructionType::float16:return 2;
    case ReconstructionType::uint32:case ReconstructionType::int32:case ReconstructionType::float32:return 4;
    case ReconstructionType::uint64:case ReconstructionType::int64:case ReconstructionType::float64:return 8;
    }
    throw std::invalid_argument("invalid reconstruction dtype");
}
struct Range {std::uintptr_t begin=0,end=0;};
Range range(const void* p,std::size_t bytes) {
    if(!bytes)return {};
    auto a=reinterpret_cast<std::uintptr_t>(p);
    if(!p || a>std::numeric_limits<std::uintptr_t>::max()-bytes)throw std::invalid_argument("invalid reconstruction span");
    return {a,a+bytes};
}
bool overlap(Range a,Range b){return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end;}
template<class T> Range span(ReconstructionSpan<T> a,std::size_t used) {
    if(a.size<used)throw std::invalid_argument("reconstruction workspace too small");
    if(used && reinterpret_cast<std::uintptr_t>(a.data)%alignof(T))throw std::invalid_argument("unaligned reconstruction workspace");
    return range(a.data,mul(used,sizeof(T)));
}
std::size_t elements(const ReconstructionView& v) {
    if(v.shape.size!=v.strides.size)throw std::invalid_argument("reconstruction shape/strides rank mismatch");
    span(v.shape,v.shape.size);span(v.strides,v.strides.size);
    std::size_t n=1;
    for(std::size_t d=0;d<v.shape.size;++d) {
        if(!v.shape.data[d])throw std::invalid_argument("reconstruction requires nonempty input");
        n=mul(n,v.shape.data[d]);
    }
    return n;
}
Range extent(const ReconstructionView& v) {
    elements(v);const auto item=width(v.type);std::size_t lo=0,hi=0;
    for(std::size_t d=0;d<v.shape.size;++d) {
        auto stride=v.strides.data[d];auto count=v.shape.data[d]-1;
        if(!count)continue;
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("reconstruction strides overflow");
        auto step=mul(count,std::size_t(stride<0 ? -stride : stride));
        if(stride<0)lo=add(lo,step);else hi=add(hi,step);
    }
    auto p=reinterpret_cast<std::uintptr_t>(v.data);auto backing=range(v.allocation,v.allocation_bytes);
    if(!v.data || p<lo || p>std::numeric_limits<std::uintptr_t>::max()-hi-item)throw std::invalid_argument("invalid reconstruction byte extent");
    Range r{p-lo,p+hi+item};
    if(r.begin<backing.begin || r.end>backing.end)throw std::invalid_argument("reconstruction view outside allocation");
    return r;
}
const unsigned char* address(const ReconstructionView& v,std::size_t index) {
    std::ptrdiff_t offset=0;
    for(std::size_t d=v.shape.size;d-->0;) {auto c=index%v.shape.data[d];index/=v.shape.data[d];offset+=std::ptrdiff_t(c)*v.strides.data[d];}
    return static_cast<const unsigned char*>(v.data)+offset;
}
std::uint64_t bits(const ReconstructionView& v,std::size_t i) {
    const auto* p=address(v,i);auto n=width(v.type);std::uint64_t u=0;
    if(n==1)u=*p;
    else if(n==2){std::uint16_t x;std::memcpy(&x,p,2);u=x;}
    else if(n==4){std::uint32_t x;std::memcpy(&x,p,4);u=x;}
    else std::memcpy(&u,p,8);
    if(v.swapped && n>1) {auto x=u;u=0;for(std::size_t k=0;k<n;++k){u=(u<<8)|(x&255);x>>=8;}}
    return u;
}
struct Number {bool integral=true,signed_value=false;unsigned integer_bits=0,float_bits=0;std::uint64_t u=0;std::int64_t s=0;double f=0;};
Number number(const ReconstructionView& v,std::size_t i) {
    auto u=bits(v,i);Number n;
    switch(v.type) {
    case ReconstructionType::boolean:n.u=u!=0;n.integer_bits=1;break;
    case ReconstructionType::uint8:n.u=u;n.integer_bits=8;break;
    case ReconstructionType::uint16:n.u=u;n.integer_bits=16;break;
    case ReconstructionType::uint32:n.u=u;n.integer_bits=32;break;
    case ReconstructionType::uint64:n.u=u;n.integer_bits=64;break;
    case ReconstructionType::int8:n.s=static_cast<std::int8_t>(u);n.signed_value=true;n.integer_bits=7;break;
    case ReconstructionType::int16:n.s=static_cast<std::int16_t>(u);n.signed_value=true;n.integer_bits=15;break;
    case ReconstructionType::int32:n.s=static_cast<std::int32_t>(u);n.signed_value=true;n.integer_bits=31;break;
    case ReconstructionType::int64:std::memcpy(&n.s,&u,8);n.signed_value=true;n.integer_bits=63;break;
    case ReconstructionType::float16: {
        n.integral=false;n.float_bits=16;auto e=(u>>10)&31,frac=u&1023;
        n.f=e==31 ? (frac ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity()) : e ? std::ldexp(double(1024+frac),int(e)-25) : std::ldexp(double(frac),-24);
        if(u&32768)n.f=-n.f;break;
    }
    case ReconstructionType::float32:{std::uint32_t x=std::uint32_t(u);float f;std::memcpy(&f,&x,4);n.integral=false;n.float_bits=32;n.f=f;break;}
    case ReconstructionType::float64:n.integral=false;n.float_bits=64;std::memcpy(&n.f,&u,8);break;
    }
    return n;
}
template<class T> T cast(const Number& n){return n.integral ? (n.signed_value ? T(n.s) : T(n.u)) : T(n.f);}
bool greater(const Number& a,const Number& b) {
    if(a.integral && b.integral) {
        if(a.signed_value && b.signed_value)return a.s>b.s;
        if(!a.signed_value && !b.signed_value)return a.u>b.u;
        if(a.signed_value)return a.s>=0 && std::uint64_t(a.s)>b.u;
        return b.s<0 || a.u>std::uint64_t(b.s);
    }
    unsigned p=std::max(a.float_bits,b.float_bits),integer=std::max(a.integer_bits,b.integer_bits);
    if(p==16 && integer>11)p=32;
    if(p<=32 && integer>24)p=64;
    return p<=32 ? cast<float>(a)>cast<float>(b) : cast<double>(a)>cast<double>(b);
}
bool contiguous(const ReconstructionView& v) {
    if(v.swapped)return false;
    auto step=width(v.type);
    for(std::size_t d=v.shape.size;d-->0;){if(v.shape.data[d]>1 && v.strides.data[d]!=std::ptrdiff_t(step))return false;step=mul(step,v.shape.data[d]);}
    return true;
}
bool byte_type(ReconstructionType t){return t==ReconstructionType::uint8 || t==ReconstructionType::boolean;}
struct Environment {
    fenv_t saved{};
    Environment(){if(std::fegetenv(&saved))throw std::domain_error("reconstruction FP environment unavailable");}
    ~Environment(){std::fesetenv(&saved);}
    void check() {
        if(std::fegetround()!=FE_TONEAREST)throw std::domain_error("reconstruction requires nearest rounding");
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        if(_mm_getcsr()&((1u<<15)|(1u<<6)))throw std::domain_error("reconstruction requires gradual underflow");
#endif
        if(std::feholdexcept(&saved))throw std::domain_error("reconstruction FP environment unavailable");
    }
};
template<class T> T canonical(T v,ReconstructionZero z) {
    if(v!=T(0) || !z.supplied)return v;
    using U=typename std::conditional<sizeof(T)==4,std::uint32_t,std::uint64_t>::type;
    U u=z.negative ? U(1)<<(sizeof(T)*8-1) : U(0);T out;std::memcpy(&out,&u,sizeof(T));return out;
}
template<class T> std::uint64_t key(T v,bool erosion) {
    using U=typename std::conditional<sizeof(T)==4,std::uint32_t,std::uint64_t>::type;
    U u;std::memcpy(&u,&v,sizeof(T));const U sign=U(1)<<(sizeof(T)*8-1);u=(u&sign) ? ~u : u^sign;
    if(erosion)u=~u;return u;
}
unsigned bit_width(std::uint64_t u) {unsigned n=0;if(u>>32){u>>=32;n+=32;}if(u>>16){u>>=16;n+=16;}if(u>>8){u>>=8;n+=8;}if(u>>4){u>>=4;n+=4;}if(u>>2){u>>=2;n+=2;}if(u>>1){u>>=1;++n;}return n+unsigned(u!=0);}
template<class T> struct Queue {
    T* out;const std::size_t* map;std::size_t* nodes;std::size_t* prev;std::size_t* heads;
    std::size_t count,head_count;std::uint64_t last=0;ReconstructionQueue kind;bool erosion;
    T value(std::size_t n)const{return out[map[n]];}
    std::uint64_t priority(std::size_t n)const{return key(value(n),erosion);}
    std::size_t bucket(std::size_t n)const {
        if(kind==ReconstructionQueue::bucket8 || kind==ReconstructionQueue::bucket16) {
            auto u=std::size_t(value(n));return erosion ? head_count-1-u : u;
        }
        return bit_width(last^priority(n));
    }
    bool better(std::size_t a,std::size_t b)const {auto x=priority(a),y=priority(b);return x>y || (x==y && a<b);}
    void exchange(std::size_t a,std::size_t b){std::swap(nodes[a],nodes[b]);prev[nodes[a]]=a;prev[nodes[b]]=b;}
    void down(std::size_t i){for(;;){auto c=2*i+1;if(c>=count)return;if(c+1<count && better(nodes[c+1],nodes[c]))++c;if(!better(nodes[c],nodes[i]))return;exchange(i,c);i=c;}}
    void link(std::size_t n){auto b=bucket(n);nodes[n]=heads[b];prev[n]=absent;if(heads[b]!=absent)prev[heads[b]]=n;heads[b]=n;}
    void unlink(std::size_t n){auto b=bucket(n),p=prev[n],q=nodes[n];if(p==absent)heads[b]=q;else nodes[p]=q;if(q!=absent)prev[q]=p;nodes[n]=prev[n]=absent;}
    void initialize() {
        if(kind==ReconstructionQueue::indexed_heap){for(std::size_t i=0;i<count;++i){nodes[i]=i;prev[i]=i;}for(auto i=count/2;i-->0;)down(i);return;}
        std::fill(heads,heads+head_count,absent);
        for(std::size_t i=0;i<count;++i)last=std::max(last,(kind==ReconstructionQueue::bucket8 || kind==ReconstructionQueue::bucket16) ? std::uint64_t(bucket(i)) : priority(i));
        for(std::size_t i=0;i<count;++i)link(i);
    }
    std::size_t pop() {
        if(kind==ReconstructionQueue::indexed_heap){auto n=nodes[0];--count;prev[n]=absent;if(count){nodes[0]=nodes[count];prev[nodes[0]]=0;down(0);}return n;}
        std::size_t b=0;
        if(kind==ReconstructionQueue::bucket8 || kind==ReconstructionQueue::bucket16){b=std::size_t(last);while(heads[b]==absent)--b;last=b;}
        else if(heads[0]==absent) {
            b=1;while(heads[b]==absent)++b;
            std::uint64_t next_last=0;for(auto n=heads[b];n!=absent;n=nodes[n])next_last=std::max(next_last,priority(n));
            auto n=heads[b];heads[b]=absent;last=next_last;
            while(n!=absent){auto following=nodes[n];link(n);n=following;}b=0;
        }
        auto n=heads[b];unlink(n);--count;return n;
    }
    void raise(std::size_t n,T v) {
        if(kind==ReconstructionQueue::indexed_heap){auto i=prev[n];out[map[n]]=v;while(i && better(nodes[i],nodes[(i-1)/2])){auto p=(i-1)/2;exchange(i,p);i=p;}return;}
        unlink(n);out[map[n]]=v;link(n);
    }
};
}

bool reconstruction_ordered(const ReconstructionView& seed,const ReconstructionView& mask,ReconstructionMethod method) {
    Environment environment;environment.check();auto n=elements(seed);elements(mask);extent(seed);extent(mask);
    if(seed.shape.size!=mask.shape.size)throw std::invalid_argument("reconstruction seed/mask shape mismatch");
    for(std::size_t d=0;d<seed.shape.size;++d)if(seed.shape.data[d]!=mask.shape.data[d])throw std::invalid_argument("reconstruction seed/mask shape mismatch");
    if(method!=ReconstructionMethod::dilation && method!=ReconstructionMethod::erosion)throw std::invalid_argument("invalid reconstruction method");
    for(std::size_t i=0;i<n;++i){auto a=number(seed,i),b=number(mask,i);if(method==ReconstructionMethod::dilation ? greater(a,b) : greater(b,a))return false;}
    return true;
}

bool reconstruction_negative_zero(const ReconstructionView& seed,const ReconstructionView& mask) {
    Environment environment;environment.check();
    for(const auto* v:{&seed,&mask}) {
        const auto n=elements(*v);extent(*v);
        if(v->type!=ReconstructionType::float16 && v->type!=ReconstructionType::float32 && v->type!=ReconstructionType::float64)continue;
        for(std::size_t i=0;i<n;++i){auto x=number(*v,i).f;if(x==0 && std::signbit(x))return true;}
    }
    return false;
}

ReconstructionRequirements reconstruction_requirements(const ReconstructionView& seed,const ReconstructionView& mask,const ReconstructionFootprint& f,ReconstructionQueue requested) {
    auto n=elements(seed);elements(mask);elements(f.values);extent(seed);extent(mask);extent(f.values);
    if(seed.shape.size!=mask.shape.size || seed.shape.size!=f.values.shape.size || f.values.type!=ReconstructionType::boolean || f.offset.size!=seed.shape.size)throw std::invalid_argument("reconstruction rank/footprint mismatch");
    span(f.offset,f.offset.size);std::size_t padded=1,neighbors=1;
    for(std::size_t d=0;d<seed.shape.size;++d) {
        if(seed.shape.data[d]!=mask.shape.data[d])throw std::invalid_argument("reconstruction seed/mask shape mismatch");
        if(f.offset.data[d]>=f.values.shape.data[d])throw std::invalid_argument("reconstruction offset outside footprint");
        padded=mul(padded,add(seed.shape.data[d],f.values.shape.data[d]-1));neighbors=mul(neighbors,f.values.shape.data[d]);
    }
    ReconstructionRequirements r;r.samples=n;r.padded_samples=padded;r.neighbor_capacity=neighbors;
    r.output_type=(mask.type==ReconstructionType::float16 || mask.type==ReconstructionType::float32) ? ReconstructionType::float32 : ReconstructionType::float64;
    r.output_bytes=mul(padded,width(r.output_type));r.mask_bytes=mul(n,width(r.output_type));
    const bool b8=byte_type(seed.type) && byte_type(mask.type),b16=seed.type==ReconstructionType::uint16 && mask.type==ReconstructionType::uint16;
    if(requested==ReconstructionQueue::automatic)requested=b8 ? ReconstructionQueue::bucket8 : b16 && n>=4096 ? ReconstructionQueue::bucket16 : r.output_type==ReconstructionType::float32 ? ReconstructionQueue::radix32 : ReconstructionQueue::radix64;
    if((requested==ReconstructionQueue::bucket8 && !b8) || (requested==ReconstructionQueue::bucket16 && !b16) || (requested==ReconstructionQueue::radix32 && r.output_type!=ReconstructionType::float32) || (requested==ReconstructionQueue::radix64 && r.output_type!=ReconstructionType::float64))throw std::invalid_argument("incompatible reconstruction queue");
    switch(requested){case ReconstructionQueue::bucket8:r.bucket_heads=256;break;case ReconstructionQueue::bucket16:r.bucket_heads=65536;break;case ReconstructionQueue::radix32:r.bucket_heads=33;break;case ReconstructionQueue::radix64:r.bucket_heads=65;break;case ReconstructionQueue::indexed_heap:break;default:throw std::invalid_argument("invalid reconstruction queue");}
    r.queue=requested;return r;
}
namespace {
template<class T> ReconstructionInfo execute(const ReconstructionView& seed,const ReconstructionView& mask,const ReconstructionFootprint& f,ReconstructionMethod method,const ReconstructionBuffers& b,ReconstructionZero z,const ReconstructionRequirements& r,int limit) {
    Environment environment;environment.check();const bool erosion=method==ReconstructionMethod::erosion;
    std::array<Range,8> writable={span(b.output,r.output_bytes),span(b.mask,r.mask_bytes),span(b.to_padded,r.samples),span(b.from_padded,r.padded_samples),span(b.queue,r.samples),span(b.position,r.samples),span(b.heads,r.bucket_heads),span(b.neighbors,r.neighbor_capacity)};
    if(reinterpret_cast<std::uintptr_t>(b.output.data)%alignof(T) || reinterpret_cast<std::uintptr_t>(b.mask.data)%alignof(T))throw std::invalid_argument("unaligned reconstruction floating workspace");
    std::array<Range,10> readonly={extent(seed),extent(mask),extent(f.values),span(seed.shape,seed.shape.size),span(seed.strides,seed.strides.size),span(mask.shape,mask.shape.size),span(mask.strides,mask.strides.size),span(f.values.shape,f.values.shape.size),span(f.values.strides,f.values.strides.size),span(f.offset,f.offset.size)};
    for(std::size_t i=0;i<writable.size();++i){for(auto a:readonly)if(overlap(writable[i],a))throw std::invalid_argument("reconstruction workspace/input overlap");for(std::size_t j=0;j<i;++j)if(overlap(writable[i],writable[j]))throw std::invalid_argument("reconstruction workspace overlap");}
    bool integer_fast=false;
#ifdef SKIMAGEXX_AVX2
    if(std::is_same<T,double>::value && has_avx2() && seed.type==mask.type && contiguous(seed) && contiguous(mask)) {
        bool ordered=true;
        if(seed.type==ReconstructionType::uint8) {integer_fast=true;ordered=detail::reconstruction_order_u8(static_cast<const std::uint8_t*>(seed.data),static_cast<const std::uint8_t*>(mask.data),r.samples,erosion);}
        else if(seed.type==ReconstructionType::uint16 && reinterpret_cast<std::uintptr_t>(seed.data)%2==0 && reinterpret_cast<std::uintptr_t>(mask.data)%2==0) {integer_fast=true;ordered=detail::reconstruction_order_u16(static_cast<const std::uint16_t*>(seed.data),static_cast<const std::uint16_t*>(mask.data),r.samples,erosion);}
        if(!ordered)throw std::invalid_argument(erosion ? "Intensity of seed image must be greater than that of the mask image for reconstruction by erosion." : "Intensity of seed image must be less than that of the mask image for reconstruction by dilation.");
    }
#endif
    bool negative_zero=false;T pad=0;
    if(!integer_fast)for(std::size_t i=0;i<r.samples;++i) {
        const auto a=number(seed,i),c=number(mask,i);
        if((!a.integral && !std::isfinite(a.f)) || (!c.integral && !std::isfinite(c.f)))throw std::domain_error("nonfinite reconstruction input requires source provider");
        if(erosion ? greater(c,a) : greater(a,c))throw std::invalid_argument(erosion ? "Intensity of seed image must be greater than that of the mask image for reconstruction by erosion." : "Intensity of seed image must be less than that of the mask image for reconstruction by dilation.");
        T x=cast<T>(a),y=cast<T>(c);
        if(!std::isfinite(x) || !std::isfinite(y))throw std::domain_error("reconstruction conversion exceeds finite output domain");
        negative_zero|=(x==T(0) && std::signbit(x)) || (y==T(0) && std::signbit(y));
        x=canonical(x,z);if(!i || (erosion ? x>pad : x<pad))pad=x;
    }
    if(integer_fast) {
        for(std::size_t i=0;i<r.samples;++i) {T x=seed.type==ReconstructionType::uint8 ? T(static_cast<const std::uint8_t*>(seed.data)[i]) : T(static_cast<const std::uint16_t*>(seed.data)[i]);if(!i || (erosion ? x>pad : x<pad))pad=x;}
    }
    if(!negative_zero && z.supplied && z.negative)throw std::invalid_argument("negative zero representative is not present");
    if(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW))throw std::domain_error("reconstruction conversion requires FP provider");
    if(negative_zero && !z.supplied)throw std::domain_error("reconstruction signed zero requires source representative");
    // ALL guards above precede the first numerical/workspace store.
    auto* out=reinterpret_cast<T*>(b.output.data);auto* bounds=reinterpret_cast<T*>(b.mask.data);
    std::size_t neighbors=0;
    for(std::size_t i=0;i<r.neighbor_capacity;++i) {
        if(!bits(f.values,i))continue;
        auto index=i;std::ptrdiff_t delta=0;std::size_t stride=1;bool center=true;
        for(std::size_t d=seed.shape.size;d-->0;) {auto c=index%f.values.shape.data[d];index/=f.values.shape.data[d];auto off=std::ptrdiff_t(c)-std::ptrdiff_t(f.offset.data[d]);center&=off==0;delta+=off*std::ptrdiff_t(stride);stride=mul(stride,seed.shape.data[d]+f.values.shape.data[d]-1);}
        if(!center)b.neighbors.data[neighbors++]=delta;
    }
    int team=1;
#ifdef _OPENMP
    if(r.padded_samples>=16384)team=std::min(limit,int(std::min<std::size_t>(r.padded_samples,32768)));
#else
    (void)limit;
#endif
    const bool direct_copy=!negative_zero && seed.type==r.output_type && mask.type==r.output_type && contiguous(seed) && contiguous(mask);
    const auto columns=seed.shape.size ? seed.shape.data[seed.shape.size-1] : std::size_t(1);
    const auto rows=r.samples/columns;
    auto fill=[&](std::size_t i){out[i]=pad;b.from_padded.data[i]=absent;};
    auto prepare=[&](std::size_t i){auto index=i;std::size_t dest=0,stride=1;for(std::size_t d=seed.shape.size;d-->0;){auto c=index%seed.shape.data[d];index/=seed.shape.data[d];dest+=(c+f.offset.data[d])*stride;stride*=seed.shape.data[d]+f.values.shape.data[d]-1;}b.to_padded.data[i]=dest;b.from_padded.data[dest]=i;};
    auto copy_row=[&](std::size_t row) {
        const auto first=row*columns;auto* destination=out+b.to_padded.data[first];auto* boundary=bounds+first;
        if(direct_copy){std::memcpy(destination,static_cast<const unsigned char*>(seed.data)+first*sizeof(T),columns*sizeof(T));std::memcpy(boundary,static_cast<const unsigned char*>(mask.data)+first*sizeof(T),columns*sizeof(T));return;}
#ifdef SKIMAGEXX_AVX2
        if constexpr(std::is_same<T,double>::value) {
            if(seed.type==ReconstructionType::uint8){detail::reconstruction_copy_u8_f64(static_cast<const std::uint8_t*>(seed.data)+first,destination,columns);detail::reconstruction_copy_u8_f64(static_cast<const std::uint8_t*>(mask.data)+first,boundary,columns);}
            else {detail::reconstruction_copy_u16_f64(static_cast<const std::uint16_t*>(seed.data)+first,destination,columns);detail::reconstruction_copy_u16_f64(static_cast<const std::uint16_t*>(mask.data)+first,boundary,columns);}
        }
#endif
    };
    int actual=1;
#ifdef _OPENMP
    if(team>1) {
#pragma omp parallel num_threads(team)
        {
#pragma omp single
            actual=omp_get_num_threads();
#pragma omp for schedule(static)
            for(std::ptrdiff_t i=0;i<std::ptrdiff_t(r.padded_samples);++i)fill(std::size_t(i));
#pragma omp for schedule(static)
            for(std::ptrdiff_t i=0;i<std::ptrdiff_t(r.samples);++i)prepare(std::size_t(i));
            if(integer_fast || direct_copy) {
#pragma omp for schedule(static)
                for(std::ptrdiff_t row=0;row<std::ptrdiff_t(rows);++row)copy_row(std::size_t(row));
            }
        }
    } else
#endif
    {for(std::size_t i=0;i<r.padded_samples;++i)fill(i);for(std::size_t i=0;i<r.samples;++i)prepare(i);if(integer_fast || direct_copy)for(std::size_t row=0;row<rows;++row)copy_row(row);}
    // Potentially rounding conversions stay on the checked caller thread.
    // Worker stages perform only index stores, bit copies, and exact small-
    // unsigned-to-double conversions; stale worker FP modes cannot affect bits.
    if(!integer_fast && !direct_copy)for(std::size_t i=0;i<r.samples;++i){out[b.to_padded.data[i]]=canonical(cast<T>(number(seed,i)),z);bounds[i]=canonical(cast<T>(number(mask,i)),z);}
    Queue<T> q{out,b.to_padded.data,b.queue.data,b.position.data,b.heads.data,r.samples,r.bucket_heads,0,r.queue,erosion};q.initialize();
    const auto floor=key(pad,erosion);
    while(q.count) {
        auto node=q.pop();const T current=out[b.to_padded.data[node]];
        if(key(current,erosion)==floor)break;
        for(std::size_t k=0;k<neighbors;++k) {
            auto p=std::ptrdiff_t(b.to_padded.data[node])+b.neighbors.data[k];auto neighbor=b.from_padded.data[std::size_t(p)];if(neighbor==absent)continue;
            T candidate=erosion ? std::max(current,bounds[neighbor]) : std::min(current,bounds[neighbor]);T previous=out[b.to_padded.data[neighbor]];
            if(erosion ? candidate<previous : candidate>previous)q.raise(neighbor,candidate);
        }
    }
    const bool simd_used=integer_fast && (r.samples>=(seed.type==ReconstructionType::uint8 ? 32u : 16u) || columns>=8);
    return {r.queue,actual,simd_used,false};
}
}
ReconstructionInfo reconstruction_into(const ReconstructionView& seed,const ReconstructionView& mask,const ReconstructionFootprint& f,ReconstructionMethod method,const ReconstructionBuffers& buffers,ReconstructionZero zero,ReconstructionQueue requested) {
    if(method!=ReconstructionMethod::dilation && method!=ReconstructionMethod::erosion)throw std::invalid_argument("invalid reconstruction method");
    const int limit=get_num_threads();const auto r=reconstruction_requirements(seed,mask,f,requested);
    return r.output_type==ReconstructionType::float32 ? execute<float>(seed,mask,f,method,buffers,zero,r,limit) : execute<double>(seed,mask,f,method,buffers,zero,r,limit);
}
}
