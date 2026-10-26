#include "coords_internal.hpp"
#include <immintrin.h>
#include <limits>
namespace skimagexx::simd {
namespace {
template<class T>struct Ops;
template<>struct Ops<float>{
    using V=__m256;static constexpr std::size_t width=8;
    static V zero(){return _mm256_setzero_ps();}static V one(){return _mm256_set1_ps(1);}
    static V broadcast(float x){return _mm256_set1_ps(x);}static V load(const float*p){return _mm256_loadu_ps(p);}
    static V gather(const float*p,const int*i){return _mm256_i32gather_ps(p,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(i)),4);}
    static V mul(V a,V b){return _mm256_mul_ps(a,b);}static V add(V a,V b){return _mm256_add_ps(a,b);}
    static void store(float*p,V v){_mm256_storeu_ps(p,v);}
};
template<>struct Ops<double>{
    using V=__m256d;static constexpr std::size_t width=4;
    static V zero(){return _mm256_setzero_pd();}static V one(){return _mm256_set1_pd(1);}
    static V broadcast(double x){return _mm256_set1_pd(x);}static V load(const double*p){return _mm256_loadu_pd(p);}
    static V gather(const double*p,const int*i){return _mm256_i32gather_pd(p,_mm_loadu_si128(reinterpret_cast<const __m128i*>(i)),8);}
    static V mul(V a,V b){return _mm256_mul_pd(a,b);}static V add(V a,V b){return _mm256_add_pd(a,b);}
    static void store(double*p,V v){_mm256_storeu_pd(p,v);}
};
template<class T>void reduce(const T* powers,std::size_t N,std::size_t D,std::size_t K,T*out,std::size_t begin,std::size_t end){
    using V=Ops<T>;auto cell=begin;
    if(K<=std::size_t(std::numeric_limits<int>::max()))for(;end-cell>=V::width;cell+=V::width){
        int exponent[3][V::width];bool same[3],contiguous[3];
        for(std::size_t lane=0;lane<V::width;++lane){auto index=cell+lane;for(auto axis=D;axis--;){exponent[axis][lane]=int(index%K);index/=K;}}
        for(std::size_t axis=0;axis<D;++axis){
            same[axis]=contiguous[axis]=true;
            for(std::size_t lane=1;lane<V::width;++lane){same[axis]&=exponent[axis][lane]==exponent[axis][0];contiguous[axis]&=exponent[axis][lane]==exponent[axis][0]+int(lane);}
        }
        auto sum=V::zero();
        for(std::size_t point=0;point<N;++point){
            auto product=V::one();
            for(std::size_t axis=0;axis<D;++axis){
                const auto row=powers+(point*D+axis)*K;
                const auto weight=same[axis]?V::broadcast(row[exponent[axis][0]]):contiguous[axis]?V::load(row+exponent[axis][0]):V::gather(row,exponent[axis]);
                product=V::mul(product,weight);
            }
            sum=V::add(sum,product);
        }
        V::store(out+cell,sum);
    }
    for(;cell<end;++cell)out[cell]=detail::coords_cell(powers,N,D,K,cell);
}
}
void moments_coords_reduce(const float*p,std::size_t N,std::size_t D,std::size_t K,float*out,std::size_t b,std::size_t e){reduce(p,N,D,K,out,b,e);}
void moments_coords_reduce(const double*p,std::size_t N,std::size_t D,std::size_t K,double*out,std::size_t b,std::size_t e){reduce(p,N,D,K,out,b,e);}
}
