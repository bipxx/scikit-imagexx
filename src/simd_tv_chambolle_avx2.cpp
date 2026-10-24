#include "skimagexx/tv_chambolle.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace skimagexx::simd {
namespace {
template<class T> struct Vec;
template<> struct Vec<float> {
    using V=__m256;static constexpr std::size_t lanes=8;
    static V load(const void* p){return _mm256_loadu_ps(static_cast<const float*>(p));}
    static void store(void* p,V v){_mm256_storeu_ps(static_cast<float*>(p),v);}
    static V zero(){return _mm256_setzero_ps();}
    static V set(float x){return _mm256_set1_ps(x);}
    static V add(V a,V b){return _mm256_add_ps(a,b);}
    static V sub(V a,V b){return _mm256_sub_ps(a,b);}
    static V mul(V a,V b){return _mm256_mul_ps(a,b);}
    static V div(V a,V b){return _mm256_div_ps(a,b);}
    static V root(V a){return _mm256_sqrt_ps(a);}
    static V neg(V a){return _mm256_xor_ps(a,_mm256_set1_ps(-0.0f));}
};
template<> struct Vec<double> {
    using V=__m256d;static constexpr std::size_t lanes=4;
    static V load(const void* p){return _mm256_loadu_pd(static_cast<const double*>(p));}
    static void store(void* p,V v){_mm256_storeu_pd(static_cast<double*>(p),v);}
    static V zero(){return _mm256_setzero_pd();}
    static V set(double x){return _mm256_set1_pd(x);}
    static V add(V a,V b){return _mm256_add_pd(a,b);}
    static V sub(V a,V b){return _mm256_sub_pd(a,b);}
    static V mul(V a,V b){return _mm256_mul_pd(a,b);}
    static V div(V a,V b){return _mm256_div_pd(a,b);}
    static V root(V a){return _mm256_sqrt_pd(a);}
    static V neg(V a){return _mm256_xor_pd(a,_mm256_set1_pd(-0.0));}
};
template<class T> const unsigned char* at(const void* p,std::size_t i){return static_cast<const unsigned char*>(p)+i*sizeof(T);}
template<class T> unsigned char* at(void* p,std::size_t i){return static_cast<unsigned char*>(p)+i*sizeof(T);}
template<class T> T read(const void* p,std::size_t i){T v;std::memcpy(&v,at<T>(p,i),sizeof(T));return v;}
template<class T> void write(void* p,std::size_t i,T v){std::memcpy(at<T>(p,i),&v,sizeof(T));}
template<class T> std::size_t divergence(const void* p,void* d,std::size_t n,std::size_t rank,std::size_t begin,std::size_t length,const std::size_t* shape,const std::size_t* steps) {
    using X=Vec<T>;std::size_t done=0;
    while(done+X::lanes<=length) {
        const auto index=begin+done;
        // Only vectorize a span whose predecessor-existence condition is
        // constant in every axis. Boundary leftovers use exact scalar loads;
        // never speculatively read before plane0 to mask the access afterward.
        std::size_t span=length-done;
        for(std::size_t a=0;a<rank;++a) {
            const auto position=(index/steps[a])%shape[a];
            const auto boundary=position ? (shape[a]-position)*steps[a]-index%steps[a]:steps[a]-index%steps[a];
            span=std::min(span,boundary);
        }
        if(span<X::lanes) {
            T value=T(0);
            for(std::size_t a=0;a<rank;++a)value=T(value+read<T>(p,a*n+index));
            value=T(-value);
            for(std::size_t a=0;a<rank;++a)if((index/steps[a])%shape[a])value=T(value+read<T>(p,a*n+index-steps[a]));
            write<T>(d,done,value);++done;continue;
        }
        const auto stop=done+span-span%X::lanes;
        for(;done<stop;done+=X::lanes) {
            const auto i=begin+done;auto value=X::zero();
            for(std::size_t a=0;a<rank;++a)value=X::add(value,X::load(at<T>(p,a*n+i)));
            value=X::neg(value);
            for(std::size_t a=0;a<rank;++a)if((i/steps[a])%shape[a])value=X::add(value,X::load(at<T>(p,a*n+i-steps[a])));
            X::store(at<T>(d,done),value);
        }
    }
    return done;
}
template<class T> std::size_t gradient(const void* out,void* g,std::size_t step,std::size_t length) {
    using X=Vec<T>;std::size_t i=0;
    for(;i+X::lanes<=length;i+=X::lanes)X::store(at<T>(g,i),X::sub(X::load(at<T>(out,i+step)),X::load(at<T>(out,i))));
    return i;
}
template<class T> std::size_t norm(const void* g,void* output,std::size_t n,std::size_t rank,std::size_t length) {
    using X=Vec<T>;std::size_t i=0;
    for(;i+X::lanes<=length;i+=X::lanes) {
        auto value=X::zero();
        for(std::size_t a=0;a<rank;++a){const auto x=X::load(at<T>(g,a*n+i));value=X::add(value,X::mul(x,x));}
        X::store(at<T>(output,i),X::root(value));
    }
    return i;
}
template<class T> std::size_t dual(void* p,const void* g,const void* denominator,std::size_t n,std::size_t rank,std::size_t begin,std::size_t length,double tau_work) {
    using X=Vec<T>;const auto tau=X::set(T(tau_work));std::size_t i=0;
    for(;i+X::lanes<=length;i+=X::lanes) {
        const auto divisor=X::load(at<T>(denominator,i));
        for(std::size_t a=0;a<rank;++a) {
            const auto element=a*n+begin+i;
            const auto product=X::mul(tau,X::load(at<T>(g,element)));
            const auto difference=X::sub(X::load(at<T>(p,element)),product);
            X::store(at<T>(p,element),X::div(difference,divisor));
        }
    }
    return i;
}
}
std::size_t tv_divergence(const void* p,void* d,std::size_t n,std::size_t rank,std::size_t begin,std::size_t length,const std::size_t* shape,const std::size_t* steps,bool single) {
    return single ? divergence<float>(p,d,n,rank,begin,length,shape,steps):divergence<double>(p,d,n,rank,begin,length,shape,steps);
}
std::size_t tv_gradient(const void* out,void* g,std::size_t step,std::size_t length,bool single) {
    return single ? gradient<float>(out,g,step,length):gradient<double>(out,g,step,length);
}
std::size_t tv_norm(const void* g,void* norm_out,std::size_t n,std::size_t rank,std::size_t length,bool single) {
    return single ? norm<float>(g,norm_out,n,rank,length):norm<double>(g,norm_out,n,rank,length);
}
std::size_t tv_dual(void* p,const void* g,const void* denominator,std::size_t n,std::size_t rank,std::size_t begin,std::size_t length,double tau_work,bool single) {
    return single ? dual<float>(p,g,denominator,n,rank,begin,length,tau_work):dual<double>(p,g,denominator,n,rank,begin,length,tau_work);
}
}
