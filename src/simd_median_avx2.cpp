#include "simd.hpp"
#include <immintrin.h>
namespace skimagexx::simd { namespace {
struct Floating {
    using V=__m256d;
    static V load(const double* p){return _mm256_loadu_pd(p);}
    static void save(double* p,V v){_mm256_storeu_pd(p,v);}
    static V low(V a,V b){return _mm256_min_pd(a,b);}
    static V high(V a,V b){return _mm256_max_pd(a,b);}
};
struct Integer {
    using V=__m256i;
    static V load(const std::int64_t* p){return _mm256_loadu_si256(reinterpret_cast<const V*>(p));}
    static void save(std::int64_t* p,V v){_mm256_storeu_si256(reinterpret_cast<V*>(p),v);}
    static V low(V a,V b){return _mm256_blendv_epi8(a,b,_mm256_cmpgt_epi64(a,b));}
    static V high(V a,V b){return _mm256_blendv_epi8(b,a,_mm256_cmpgt_epi64(a,b));}
};
template<int N,int Phase,int K,class M> void network(typename M::V (&v)[N]) {
    if constexpr(Phase<N){if constexpr(K+1<N){const auto a=v[K],b=v[K+1];v[K]=M::low(a,b);v[K+1]=M::high(a,b);network<N,Phase,K+2,M>(v);}else network<N,Phase+1,(Phase+1)%2,M>(v);}
}
template<int N,int K,class M,class T> void loads(const T* input,const std::size_t* offsets,typename M::V (&v)[N]) {
    if constexpr(K<N){v[K]=M::load(input+offsets[K]);loads<N,K+1,M>(input,offsets,v);}
}
template<int N,class M,class T> std::size_t sort(const T* input,T* output,std::size_t n,const std::size_t* offsets,std::size_t rank) {
    std::size_t i=0;for(;i+4<=n;i+=4){typename M::V v[N];loads<N,0,M>(input+i,offsets,v);network<N,0,0,M>(v);M::save(output+i,v[rank]);}return i;
}
template<class M,class T> std::size_t dispatch(const T* a,T* b,std::size_t n,const std::size_t* offsets,std::size_t taps,std::size_t rank) {
    switch(taps){
#define SORT(N) case N:return sort<N,M>(a,b,n,offsets,rank);
    SORT(1) SORT(2) SORT(3) SORT(4) SORT(5) SORT(6) SORT(7) SORT(8) SORT(9)
#undef SORT
    }return 0;
}
}
std::size_t median_small(const void* input,void* output,std::size_t n,const std::size_t* offsets,std::size_t taps,std::size_t rank,bool integer) {
    return integer ? dispatch<Integer>(static_cast<const std::int64_t*>(input),static_cast<std::int64_t*>(output),n,offsets,taps,rank) : dispatch<Floating>(static_cast<const double*>(input),static_cast<double*>(output),n,offsets,taps,rank);
}
}
