#include "simd.hpp"
#include <immintrin.h>
#include <type_traits>
namespace skimagexx::simd {
void adaptive_mean4(const double* input,double* output,std::size_t length,std::size_t window) {
    auto total=_mm256_setzero_pd();const auto divisor=_mm256_set1_pd(double(window));
    for(std::size_t k=0;k<window;++k)total=_mm256_add_pd(total,_mm256_loadu_pd(input+4*k));
    _mm256_storeu_pd(output,_mm256_div_pd(total,divisor));
    for(std::size_t x=1;x<length;++x){const auto difference=_mm256_sub_pd(_mm256_loadu_pd(input+4*(x+window-1)),_mm256_loadu_pd(input+4*(x-1)));total=_mm256_add_pd(total,difference);_mm256_storeu_pd(output+4*x,_mm256_div_pd(total,divisor));}
}
namespace {
template<class T> struct V;
template<>struct V<float>{using R=__m256;static constexpr std::size_t lanes=8;static R load(const float* p){return _mm256_loadu_ps(p);}static void store(float* p,R x){_mm256_storeu_ps(p,x);}static R low(R a,R b){return _mm256_min_ps(a,b);}static R high(R a,R b){return _mm256_max_ps(a,b);}};
template<>struct V<double>{using R=__m256d;static constexpr std::size_t lanes=4;static R load(const double* p){return _mm256_loadu_pd(p);}static void store(double* p,R x){_mm256_storeu_pd(p,x);}static R low(R a,R b){return _mm256_min_pd(a,b);}static R high(R a,R b){return _mm256_max_pd(a,b);}};
template<int Phase,int K,class M> void network(typename M::R (&values)[9]) {
    if constexpr(Phase<9) {
        if constexpr(K<8){const auto a=values[K],b=values[K+1];values[K]=M::low(a,b);values[K+1]=M::high(a,b);network<Phase,K+2,M>(values);}
        else network<Phase+1,(Phase+1)%2,M>(values);
    }
}
template<int K,class T> void load9(const T* input,std::size_t pitch,typename V<T>::R (&values)[9]) {
    if constexpr(K<9){values[K]=V<T>::load(input+std::size_t(K/3)*pitch+std::size_t(K%3));load9<K+1>(input,pitch,values);}
}
template<class T> std::size_t median9(const T* input,T* output,std::size_t count,std::size_t pitch) {
    using M=V<T>;std::size_t x=0;
    for(;x+M::lanes<=count;x+=M::lanes){typename M::R values[9];load9<0>(input+x,pitch,values);
        // Fixed odd/even sorting network: independent pixels occupy SIMD lanes.
        network<0,0,M>(values);
        M::store(output+x,values[4]);
    }return x;
}
}
std::size_t adaptive_median9(const void* input,void* output,std::size_t count,std::size_t pitch,bool single) {
    return single ? median9(static_cast<const float*>(input),static_cast<float*>(output),count,pitch) : median9(static_cast<const double*>(input),static_cast<double*>(output),count,pitch);
}
std::size_t adaptive_offset(const void* input,void* output,std::size_t count,bool source_single,bool target_single,double value) {
    std::size_t i=0;
    if(target_single && source_single){auto* dst=static_cast<float*>(output);const auto* src=static_cast<const float*>(input);const auto scalar=_mm256_set1_ps(float(value));for(;i+8<=count;i+=8)_mm256_storeu_ps(dst+i,_mm256_sub_ps(_mm256_loadu_ps(src+i),scalar));}
    else if(!target_single){auto* dst=static_cast<double*>(output);const auto scalar=_mm256_set1_pd(value);for(;i+4<=count;i+=4){const auto x=source_single ? _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(input)+i)) : _mm256_loadu_pd(static_cast<const double*>(input)+i);_mm256_storeu_pd(dst+i,_mm256_sub_pd(x,scalar));}}
    return i;
}
}
