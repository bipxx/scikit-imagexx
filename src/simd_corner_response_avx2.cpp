#include "corner_response_detail.hpp"
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <type_traits>

namespace skimagexx::simd { namespace {
template<class T> struct Vec;
#define VECTOR(T,S,V,W) template<> struct Vec<T>{using Type=V;static constexpr std::size_t width=W; \
static V load(const T* p){return _mm256_loadu_##S(p);} static void store(T* p,V x){_mm256_storeu_##S(p,x);} \
static V value(T x){return _mm256_set1_##S(x);} static V add(V a,V b){return _mm256_add_##S(a,b);} \
static V sub(V a,V b){return _mm256_sub_##S(a,b);} static V mul(V a,V b){return _mm256_mul_##S(a,b);} \
static V div(V a,V b){return _mm256_div_##S(a,b);} static V root(V x){return _mm256_sqrt_##S(x);} \
static V equal(V a,V b){return _mm256_cmp_##S(a,b,_CMP_EQ_OQ);} \
static V different(V a,V b){return _mm256_cmp_##S(a,b,_CMP_NEQ_OQ);} \
static V bounded(V x,T limit){return _mm256_cmp_##S(_mm256_andnot_##S(value(T(-0.)),x),value(limit),_CMP_LE_OQ);} \
static V mask(V a,V b){return _mm256_and_##S(a,b);} static V blend(V a,V b,V mask){return _mm256_blendv_##S(a,b,mask);} \
static bool all(V x){return _mm256_movemask_##S(x)==((1<<W)-1);} static bool any(V x){return _mm256_movemask_##S(x)!=0;} };
VECTOR(float,ps,__m256,8)
VECTOR(double,pd,__m256d,4)
#undef VECTOR

template<class T> std::size_t run(const void* const* fields,void* destination,void* other,std::size_t begin,std::size_t end,bool wide,CornerResponse method,double parameter,bool& valid){
    using V=Vec<T>;const auto zero=V::value(T(0)),one=V::value(T(1)),two=V::value(T(2)),four=V::value(T(4));
    auto bound=T(detail::corner_bound<T>());if(double(bound)>detail::corner_bound<T>())bound=std::nextafter(bound,-std::numeric_limits<T>::infinity());
    for(;begin+V::width<=end;begin+=V::width){
        const auto a=V::load(static_cast<const T*>(fields[0])+begin),b=V::load(static_cast<const T*>(fields[1])+begin),c=V::load(static_cast<const T*>(fields[2])+begin);
        if(!V::all(V::mask(V::mask(V::bounded(a,bound),V::bounded(b,bound)),V::bounded(c,bound)))){valid=false;return begin;}
        const auto trace=V::add(a,c),determinant=V::sub(V::mul(a,c),V::mul(b,b));
        if constexpr(std::is_same_v<T,float>){
            if(wide){
                const auto base=method==CornerResponse::harris ? V::mul(trace,trace) : V::mul(two,determinant);
                for(int half=0;half<2;++half){
                    const auto narrow=[&](auto x){return half ? _mm256_extractf128_ps(x,1) : _mm256_castps256_ps128(x);};
                    const auto d=_mm256_cvtps_pd(narrow(determinant)),t=_mm256_cvtps_pd(narrow(trace)),part=_mm256_cvtps_pd(narrow(base));
                    __m256d result;
                    if(method==CornerResponse::harris)result=_mm256_sub_pd(d,_mm256_mul_pd(_mm256_set1_pd(parameter),part));
                    else{
                        const auto denominator=_mm256_add_pd(t,_mm256_set1_pd(parameter));
                        if(!Vec<double>::all(Vec<double>::bounded(denominator,std::numeric_limits<double>::max())) || Vec<double>::any(Vec<double>::equal(denominator,Vec<double>::value(0)))){valid=false;return begin;}
                        result=_mm256_div_pd(part,denominator);
                    }
                    if(!Vec<double>::all(Vec<double>::bounded(result,std::numeric_limits<double>::max()))){valid=false;return begin;}
                    Vec<double>::store(static_cast<double*>(destination)+begin+4*half,result);
                }
                continue;
            }
        }
        typename V::Type result;
        if(method==CornerResponse::shi_tomasi){
            const auto difference=V::sub(a,c);
            const auto root=V::root(V::add(V::mul(difference,difference),V::mul(four,V::mul(b,b))));
            result=V::div(V::sub(trace,root),two);
        }else if(method==CornerResponse::harris){
            result=V::sub(determinant,V::mul(V::value(T(parameter)),V::mul(trace,trace)));
        }else if(method==CornerResponse::noble){
            const auto denominator=V::add(trace,V::value(T(parameter)));
            if(!V::all(V::bounded(denominator,std::numeric_limits<T>::max())) || V::any(V::equal(denominator,zero))){valid=false;return begin;}
            result=V::div(V::mul(two,determinant),denominator);
        }else{
            const auto active=V::different(trace,zero),square=V::mul(trace,trace);
            if(V::any(V::mask(active,V::equal(square,zero)))){valid=false;return begin;}
            result=V::blend(zero,V::div(determinant,V::blend(one,trace,active)),active);
            const auto q=V::blend(zero,V::div(V::mul(four,determinant),V::blend(one,square,active)),active);
            if(!V::all(V::bounded(q,std::numeric_limits<T>::max()))){valid=false;return begin;}
            V::store(static_cast<T*>(other)+begin,q);
        }
        if(!V::all(V::bounded(result,std::numeric_limits<T>::max()))){valid=false;return begin;}
        V::store(static_cast<T*>(destination)+begin,result);
    }
    return begin;
}
}}
namespace skimagexx::simd {
std::size_t corner_response(const void* const* fields,void* output,void* second,std::size_t begin,std::size_t end,bool single,bool wide,CornerResponse method,double parameter,bool& valid){
    return single ? run<float>(fields,output,second,begin,end,wide,method,parameter,valid)
                  : run<double>(fields,output,second,begin,end,false,method,parameter,valid);
}
}
