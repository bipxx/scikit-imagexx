// Integer finite Radon source order: scikit-image0.26 finite_radon_transform.py.
// BSD-3-Clause, third_party/scikit-image-LICENSE.txt. No original forwarding.
#include <skimagexx/finite_radon.hpp>
#include <skimagexx/image.hpp>
#include "execution.hpp"
#include "half.hpp"
#include <array>
#include <cmath>
#include <cfenv>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif
#if !defined(SKIMAGEXX_FINITE_RADON_SCALAR) && (defined(__SSE2__) || defined(_M_X64))
#define FRT_SSE2 1
#endif
namespace skimagexx { namespace {
bool signed_type(PixelType t){return t==PixelType::int8||t==PixelType::int16||t==PixelType::int32||t==PixelType::int64;}
std::size_t extent(const FiniteRadonInput& a,bool inverse){
    const auto& v=a.image;
    if(v.shape.size()!=2||v.strides.size()!=2||v.shape[0]<0||v.shape[1]<0||v.shape[1]>4096||v.shape[0]!=v.shape[1]+(inverse?1:0))throw std::invalid_argument(inverse?"Input must be an (n+1) row x n column, 2-D array":"Input must be a square, 2-D array");
    const auto n=std::size_t(v.shape[1]);if(n>4096)throw std::length_error("finite Radon assessed allocation frontier4096");
    const auto width=a.half?2:pixel_size(a.type)*(a.complex?2:1);
    if(v.itemsize!=width)throw std::invalid_argument("finite Radon POD width mismatch");
    if(v.shape[0]&&n&&!v.data)throw std::invalid_argument("finite Radon null storage");
    return n;
}
std::uint64_t word(const FiniteRadonInput& a,std::size_t r,std::size_t c){
    const auto& v=a.image;const auto* p=static_cast<const unsigned char*>(v.data)+std::ptrdiff_t(r)*v.strides[0]+std::ptrdiff_t(c)*v.strides[1];
    std::uint64_t x=0;unsigned char b[8];std::memcpy(b,p,v.itemsize);if(a.swapped)std::reverse(b,b+v.itemsize);std::memcpy(&x,b,v.itemsize);
    if(a.type==PixelType::boolean)return x!=0;
    if(signed_type(a.type)&&v.itemsize<8&&(x&(std::uint64_t(1)<<(v.itemsize*8-1))))x|=~((std::uint64_t(1)<<(v.itemsize*8))-1);
    return x;
}
void accumulate(std::uint32_t* out,const std::uint32_t* in,std::size_t n,bool simd){
    std::size_t j=0;
#ifdef FRT_SSE2
    if(simd)for(;j+4<=n;j+=4){const auto x=_mm_loadu_si128(reinterpret_cast<const __m128i*>(in+j)),y=_mm_loadu_si128(reinterpret_cast<const __m128i*>(out+j));_mm_storeu_si128(reinterpret_cast<__m128i*>(out+j),_mm_add_epi32(x,y));}
#else
    (void)simd;
#endif
    for(;j<n;++j)out[j]+=in[j];
}
std::vector<std::uint32_t> packed(const FiniteRadonInput& a,std::size_t rows,std::size_t n){
    std::vector<std::uint32_t> x(rows*n);for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<n;++c)x[r*n+c]=std::uint32_t(word(a,r,c));return x;
}
void projections(const std::vector<std::uint32_t>& x,std::size_t n,std::uint32_t* out,bool inverse,int workers){
    // Each projection is independent. Shift computed once per row; two spans,
    // no per-element modulus, rolled buffer, allocation or shared reduction.
    const bool simd=std::getenv("SKIMAGEXX_DISABLE_SIMD")==nullptr;
    detail::partition(n,n*n*n>=detail::parallel_min?workers:1,[&](std::size_t lo,std::size_t hi,std::size_t){
        for(std::size_t m=lo;m<hi;++m){auto* dst=out+m*n;
            for(std::size_t r=0;r<n;++r){const auto shift=(m*r)%n;const auto start=inverse&&shift?n-shift:shift;const auto* src=x.data()+r*n;accumulate(dst,src+start,n-start,simd);accumulate(dst+n-start,src,start,simd);}
        }
    });
}
int captured(int workers){if(workers<0)throw std::invalid_argument("finite Radon workers must be nonnegative");return workers?workers:get_num_threads();}
bool floating(const FiniteRadonInput& a){return a.half||a.type==PixelType::float32||a.type==PixelType::float64;}
template<class T>T real_word(const FiniteRadonInput& a,std::size_t r,std::size_t c,unsigned part){
    const auto* p=static_cast<const unsigned char*>(a.image.data)+std::ptrdiff_t(r)*a.image.strides[0]+std::ptrdiff_t(c)*a.image.strides[1];
    const auto width=a.half?2:sizeof(T);unsigned char raw[8];std::memcpy(raw,p+part*width,width);if(a.swapped)std::reverse(raw,raw+width);
    if(a.half){std::uint16_t x;std::memcpy(&x,raw,2);return detail::decode_half(x);}T value;std::memcpy(&value,raw,sizeof(T));return value;
}
// NumPy's stable8-lane pairwise association (complex uses4 complex lanes).
// Derived from numpy/core/src/umath/loops_utils.h.src, BSD-3-Clause;
// see third_party/numpy-LICENSE.txt. No provider reduction is executed.
template<class T>T pair_sum(const T* a,std::size_t n,std::size_t step,bool complex){
    const std::size_t lanes=complex?4:8,block=complex?64:128;
    if(n<lanes){T s=T(-0.0);for(std::size_t k=0;k<n;++k)s=T(s+a[k*step]);return s;}
    if(n<=block){std::array<T,8> r{};for(std::size_t k=0;k<lanes;++k)r[k]=a[k*step];std::size_t k=lanes;
        for(;k+lanes<=n;k+=lanes)for(std::size_t j=0;j<lanes;++j)r[j]=T(r[j]+a[(k+j)*step]);
        T s=complex?T(T(r[0]+r[1])+T(r[2]+r[3])):T(T(T(r[0]+r[1])+T(r[2]+r[3]))+T(T(r[4]+r[5])+T(r[6]+r[7])));
        for(;k<n;++k)s=T(s+a[k*step]);return s;
    }
    auto first=n/2;first-=first%lanes;return T(pair_sum(a,first,step,complex)+pair_sum(a+first*step,n-first,step,complex));
}
template<class T>std::uint32_t cast_uint32(T value){
    // Scalar hardware truncation reproduces NumPy's uint32 cast, including
    // nonfinite invalid status and low-word wrap, without C++ undefined casts.
#if defined(__SSE2__) || defined(_M_X64)
    if constexpr(sizeof(T)==4)return std::uint32_t(_mm_cvttss_si64(_mm_set_ss(value)));
    else return std::uint32_t(_mm_cvttsd_si64(_mm_set_sd(value)));
#else
    if(!std::isfinite(value)||value<T(-9223372036854775808.0)||value>=T(9223372036854775808.0))throw std::domain_error("finite Radon non-x64 cast frontier");
    return std::uint32_t(std::int64_t(value));
#endif
}
// Retained NumPy1.24.4/2.5.4 aligned contiguous double->uint32 cast:
// compiler-generated four-word groups subtract2^31 for ordered high lanes,
// truncate to signed32, then xor the high bit. A0..3word tail truncates64.
// Reproduce each wide lane with scalar SSE operations even in disabled/scalar
// builds: source cast grouping is observable numeric/FP semantics, not an
// optional optimization. Complex/float/half cast domains retain their path.
std::uint32_t contiguous_double_word(double value) {
#if defined(__SSE2__) || defined(_M_X64)
    const auto x=_mm_set_sd(value),boundary=_mm_set_sd(2147483648.0);
    const auto high=_mm_cmple_sd(boundary,x);
    const bool upper=(_mm_movemask_pd(high)&1)!=0;
    const auto adjusted=_mm_sub_sd(x,_mm_set_sd(upper?2147483648.0:0.0));
    return std::uint32_t(_mm_cvttsd_si32(adjusted))^(upper?0x80000000u:0u);
#else
    if(!std::isfinite(value)||value<0||value>=4294967296.0)throw std::domain_error("finite Radon non-x64 contiguous cast frontier");
    return std::uint32_t(value);
#endif
}

template<class T>FiniteRadonTransform project_float(const FiniteRadonInput& a,std::size_t n,bool inverse,int budget,FiniteRadonNotice notify,void* context){
    FiniteRadonTransform out;out.rows=n+(inverse?0:1);out.columns=n;out.captured_workers=budget;out.values.resize(out.rows*n);out.cast_status.resize(out.rows);if(!n)return out;
    const unsigned components=a.complex?2:1;std::vector<T> work(n*n*components),temp(n*components),sums(n*components);
    for(std::size_t r=0;r<n;++r)for(std::size_t c=0;c<n;++c)for(unsigned p=0;p<components;++p)work[(r*n+c)*components+p]=real_word<T>(a,r,c,p);
    // Floating reductions/warnings remain caller-thread ordered. Repeated
    // circular shifts use two contiguous spans and one reusable row buffer.
    for(std::size_t m=0;m<out.rows;++m){
        if(m>0&&m<n)for(std::size_t r=1;r<n;++r){const auto shift=inverse?n-r:r;const auto first=(n-shift)*components;auto* row=work.data()+r*n*components;
            std::copy(row+shift*components,row+n*components,temp.data());std::copy(row,row+shift*components,temp.data()+first);std::copy(temp.begin(),temp.end(),row);}
        for(std::size_t c=0;c<n;++c){
            T value;
            if(m==n)value=T(T(0)+pair_sum(work.data()+c*n*components,n,components,a.complex));
            else{value=T(0);for(std::size_t r=0;r<n;++r)value=T(value+work[(r*n+c)*components]);}
            if(a.complex){volatile T imag=T(0);if(m==n)imag=T(T(0)+pair_sum(work.data()+c*n*components+1,n,components,true));else for(std::size_t r=0;r<n;++r)imag=T(imag+work[(r*n+c)*components+1]);(void)imag;}
            if(a.half)value=detail::decode_half(detail::encode_half(float(value)));
            sums[c*components]=value;
        }
        if(a.complex){out.cast_status[m]|=1;if(notify)notify(context,1);}
        // NumPy array_assign_array.c prepares floating cast status at this
        // exact phase, clearing only when one of its four error flags is set.
        // This is source operation semantics, never an after-call FP mask.
        constexpr int cast_errors=FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID;
        if(std::fetestexcept(cast_errors))std::feclearexcept(cast_errors);
        const auto wide_end=(sizeof(T)==8&&!a.complex)?n-n%4:0;
        for(std::size_t c=0;c<n;++c){const auto value=sums[c*components];
            const bool wide=c<wide_end;
            out.values[m*n+c]=wide?contiguous_double_word(double(value)):cast_uint32(value);
        }
        // Like NumPy's post-cast status barrier, read actual hardware invalid.
        // A fractional value just below signed32 minimum can truncate to a
        // valid endpoint; classify the operation, not a guessed value range.
        if(std::fetestexcept(FE_INVALID))out.cast_status[m]|=2;
        if((out.cast_status[m]&2)&&notify)notify(context,2);
    }
    return out;
}
}
FiniteRadonTransform frt2(const FiniteRadonInput& a,int workers,FiniteRadonNotice notice,void* context){
    const int budget=captured(workers);const auto n=extent(a,false);FiniteRadonTransform out;out.rows=n+1;out.columns=n;out.captured_workers=budget;out.values.resize((n+1)*n);if(!n)return out;
    if(floating(a))return a.type==PixelType::float64?project_float<double>(a,n,false,budget,notice,context):project_float<float>(a,n,false,budget,notice,context);
    auto x=packed(a,n,n);projections(x,n,out.values.data(),false,budget);
    for(std::size_t r=0;r<n;++r)for(std::size_t c=0;c<n;++c)out.values[n*n+r]+=x[r*n+c];return out;
}
FiniteRadonTransform finite_radon_inverse_prefix(const FiniteRadonInput& a,int workers,FiniteRadonNotice notice,void* context){
    const int budget=captured(workers);const auto n=extent(a,true);if(!n)throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    if(floating(a))return a.type==PixelType::float64?project_float<double>(a,n,true,budget,notice,context):project_float<float>(a,n,true,budget,notice,context);
    FiniteRadonTransform out;out.rows=out.columns=n;out.captured_workers=budget;out.values.resize(n*n);auto x=packed(a,n,n);projections(x,n,out.values.data(),true,budget);return out;
}
InverseFiniteRadonTransform ifrt2(const FiniteRadonInput& a,const FiniteRadonPlan& plan,int workers,FiniteRadonNotice notice,void* context){
    const int budget=captured(workers);const auto n=extent(a,true);if(!n)throw std::out_of_range("index 0 is out of bounds for axis 0 with size 0");
    if(floating(a)){(void)finite_radon_inverse_prefix(a,budget,notice,context);throw FiniteRadonCastingError();}
    if(signed_type(a.type))throw FiniteRadonCastingError();
    if(plan.accumulator_bits!=32&&plan.accumulator_bits!=64)throw std::invalid_argument("finite Radon accumulator must be32or64bits");
    if((plan.signed_sum&&a.type!=PixelType::boolean)||(plan.accumulator_bits==32&&a.image.itemsize==8))throw std::invalid_argument("finite Radon source accumulator plan mismatch");
    InverseFiniteRadonTransform out;out.rows=out.columns=n;out.captured_workers=budget;out.values.resize(n*n);
    auto x=packed(a,n,n);std::vector<std::uint32_t> sums(n*n);projections(x,n,sums.data(),true,budget);
    std::uint64_t total=0;for(std::size_t c=0;c<n;++c)total+=word(a,0,c);if(plan.accumulator_bits==32)total=std::uint32_t(total);
    const bool narrow=plan.weak_scalar?total<=std::numeric_limits<std::uint32_t>::max():plan.accumulator_bits==32&&!plan.signed_sum;
    // Integer projection work may run in parallel; all float64 conversion and
    // division stay on caller thread in C order, preserving actual FP status.
    for(std::size_t r=0;r<n;++r){const auto last=std::uint32_t(word(a,n,r));for(std::size_t c=0;c<n;++c){const auto v=std::uint32_t(sums[r*n+c]+last);double value;
        if(narrow)value=double(std::uint32_t(v-std::uint32_t(total)));
        else if(plan.signed_sum&&!plan.weak_scalar)value=double(std::int64_t(v)-std::int64_t(total));
        else value=double(std::uint64_t(v)-total);
        out.values[r*n+c]=value/double(n);
    }}return out;
}
}
