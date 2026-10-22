#include "simd.hpp"
#include "li_detail.hpp"
#include <immintrin.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

namespace skimagexx::simd {
namespace {
template<std::size_t Lanes> struct alignas(32) Order {
    std::array<std::array<int,8>,std::size_t{1}<<Lanes> indices{};
    std::array<unsigned char,std::size_t{1}<<Lanes> counts{};
};
template<std::size_t Lanes> constexpr Order<Lanes> make_order() {
    Order<Lanes> out;
    for(std::size_t mask=0;mask<out.indices.size();++mask) {
        std::size_t at=0;
        for(int selected=1;selected>=0;--selected)for(std::size_t lane=0;lane<Lanes;++lane)if(bool(mask&(std::size_t{1}<<lane))==bool(selected)) {
            for(std::size_t word=0;word<8/Lanes;++word)out.indices[mask][at++]=int(lane*(8/Lanes)+word);
            if(selected)++out.counts[mask];
        }
    }
    return out;
}
template<class T> struct Vector;
#define LI_VECTOR(TYPE,REG,SFX,LANES) \
template<>struct Vector<TYPE>{ \
    using R=REG;static constexpr std::size_t lanes=LANES; \
    static R set(TYPE x){return _mm256_set1_##SFX(x);} \
    static R load(const TYPE* x){return _mm256_loadu_##SFX(x);} \
    static void store(TYPE* x,R v){_mm256_storeu_##SFX(x,v);} \
    static R add(R a,R b){return _mm256_add_##SFX(a,b);} \
    static R sub(R a,R b){return _mm256_sub_##SFX(a,b);} \
    static R minimum(R a,R b){return _mm256_min_##SFX(a,b);} \
    static R maximum(R a,R b){return _mm256_max_##SFX(a,b);} \
    static R abs(R x){return _mm256_andnot_##SFX(set(TYPE(-0.)),x);} \
    static R le(R a,R b){return _mm256_cmp_##SFX(a,b,_CMP_LE_OQ);} \
    static R eq(R a,R b){return _mm256_cmp_##SFX(a,b,_CMP_EQ_OQ);} \
    static R valid(R x){return _mm256_cmp_##SFX(x,x,_CMP_ORD_Q);} \
    static R choose(R no,R yes,R mask){return _mm256_blendv_##SFX(no,yes,mask);} \
    static unsigned bits(R x){return unsigned(_mm256_movemask_##SFX(x));} \
};
LI_VECTOR(float,__m256,ps,8)
LI_VECTOR(double,__m256d,pd,4)
#undef LI_VECTOR

template<class T> std::size_t scan(const T* input,std::size_t n,T first,detail::LiScan& out) {
    using V=Vector<T>;static constexpr auto order=make_order<V::lanes>();
    const auto inf=V::set(std::numeric_limits<T>::infinity()),negative_inf=V::set(-std::numeric_limits<T>::infinity());
    const auto limit=V::set(std::numeric_limits<T>::max()),initial=V::set(first);
    auto low=inf,high=negative_inf;std::size_t i=0;
    for(;i+V::lanes<=n;i+=V::lanes) {
        const auto value=V::load(input+i),finite=V::le(V::abs(value),limit);
        const auto valid=V::bits(V::valid(value)),mask=V::bits(finite);
        out.valid+=order.counts[valid];out.finite+=order.counts[mask];
        if(valid&~V::bits(V::eq(value,initial)))out.constant=false;
        low=V::minimum(low,V::choose(inf,value,finite));high=V::maximum(high,V::choose(negative_inf,value,finite));
    }
    T lo[V::lanes],hi[V::lanes];V::store(lo,low);V::store(hi,high);
    for(std::size_t j=0;j<V::lanes;++j){out.low=std::min(out.low,double(lo[j]));out.high=std::max(out.high,double(hi[j]));}
    return i;
}
template<class T> std::size_t shift(const T* input,T* output,std::size_t n,T low) {
    using V=Vector<T>;const auto offset=V::set(low);std::size_t i=0;
    for(;i+V::lanes<=n;i+=V::lanes)V::store(output+i,V::sub(V::load(input+i),offset));
    return i;
}
template<class T> double sum(const T* input,std::size_t n) {
    if(n<8){T total=-T(0);for(std::size_t i=0;i<n;++i)total=T(total+input[i]);return double(total);}
    using V=Vector<T>;auto first=V::load(input);typename V::R second{};
    if constexpr(std::is_same_v<T,double>)second=V::load(input+4);
    std::size_t i=8;
    for(;i+8<=n;i+=8) {
        first=V::add(first,V::load(input+i));
        if constexpr(std::is_same_v<T,double>)second=V::add(second,V::load(input+i+4));
    }
    T r[8];V::store(r,first);if constexpr(std::is_same_v<T,double>)V::store(r+4,second);
    T total=T(T(T(r[0]+r[1])+T(r[2]+r[3]))+T(T(r[4]+r[5])+T(r[6]+r[7])));
    for(;i<n;++i)total=T(total+input[i]);return double(total);
}
template<class T> std::size_t count(const T* input,std::size_t n,T pivot,std::size_t& out) {
    using V=Vector<T>;static constexpr auto order=make_order<V::lanes>();const auto threshold=V::set(pivot);std::size_t i=0;
    for(;i+V::lanes<=n;i+=V::lanes)out+=order.counts[V::bits(V::le(V::load(input+i),threshold))];
    return i;
}
template<class T> std::size_t partition(const T* input,std::size_t n,T pivot,T* low,T* high,std::size_t low_cap,std::size_t high_cap,std::size_t& left,std::size_t& right) {
    using V=Vector<T>;static constexpr auto order=make_order<V::lanes>();
    constexpr auto all=unsigned((std::size_t{1}<<V::lanes)-1);const auto threshold=V::set(pivot);std::size_t i=0;
    auto store=[](T* output,__m256i packed,std::size_t available,std::size_t selected) {
        if(available>=V::lanes)_mm256_storeu_si256(reinterpret_cast<__m256i*>(output),packed);
        else {alignas(32) T lanes[V::lanes];_mm256_store_si256(reinterpret_cast<__m256i*>(lanes),packed);std::memcpy(output,lanes,selected*sizeof(T));}
    };
    for(;i+V::lanes<=n;i+=V::lanes) {
        const auto value=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input+i));
        const auto mask=V::bits(V::le(V::load(input+i),threshold)),other=all^mask;
        const auto packed_low=_mm256_permutevar8x32_epi32(value,_mm256_load_si256(reinterpret_cast<const __m256i*>(order.indices[mask].data())));
        const auto packed_high=_mm256_permutevar8x32_epi32(value,_mm256_load_si256(reinterpret_cast<const __m256i*>(order.indices[other].data())));
        store(low+left,packed_low,low_cap-left,order.counts[mask]);store(high+right,packed_high,high_cap-right,order.counts[other]);
        left+=order.counts[mask];right+=order.counts[other];
    }
    return i;
}
}
std::size_t li_scan(const void* input,std::size_t n,bool single,double first,detail::LiScan& out) {
    return single ? scan(static_cast<const float*>(input),n,float(first),out) : scan(static_cast<const double*>(input),n,first,out);
}
std::size_t li_shift(const void* input,void* output,std::size_t n,bool single,double low) {
    return single ? shift(static_cast<const float*>(input),static_cast<float*>(output),n,float(low)) : shift(static_cast<const double*>(input),static_cast<double*>(output),n,low);
}
double li_sum(const void* input,std::size_t n,bool single) {return single ? sum(static_cast<const float*>(input),n) : sum(static_cast<const double*>(input),n);}
std::size_t li_count(const void* input,std::size_t n,bool single,double pivot,std::size_t& out) {return single ? count(static_cast<const float*>(input),n,float(pivot),out) : count(static_cast<const double*>(input),n,pivot,out);}
std::size_t li_partition(const void* input,std::size_t n,bool single,double pivot,void* low,void* high,std::size_t low_cap,std::size_t high_cap,std::size_t& left,std::size_t& right) {
    return single ? partition(static_cast<const float*>(input),n,float(pivot),static_cast<float*>(low),static_cast<float*>(high),low_cap,high_cap,left,right) : partition(static_cast<const double*>(input),n,pivot,static_cast<double*>(low),static_cast<double*>(high),low_cap,high_cap,left,right);
}
}
