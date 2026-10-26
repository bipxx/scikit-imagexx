#include "block_reduce_internal.hpp"
#include <immintrin.h>
#include <climits>

namespace skimagexx::simd {
namespace {
template<class T>struct Vector;
template<>struct Vector<float>{
    using R=__m256;static constexpr std::size_t width=8;
    static R fill(float v){return _mm256_set1_ps(v);}static R add(R a,R b){return _mm256_add_ps(a,b);}
    static R load(const float* p,int stride){
        if(stride==1)return _mm256_loadu_ps(p);
        return _mm256_i32gather_ps(p,_mm256_setr_epi32(0,stride,2*stride,3*stride,4*stride,5*stride,6*stride,7*stride),4);
    }
    static void store(float* p,R v){_mm256_storeu_ps(p,v);}
};
template<>struct Vector<double>{
    using R=__m256d;static constexpr std::size_t width=4;
    static R fill(double v){return _mm256_set1_pd(v);}static R add(R a,R b){return _mm256_add_pd(a,b);}
    static R load(const double* p,int stride){
        if(stride==1)return _mm256_loadu_pd(p);
        return _mm256_i32gather_pd(p,_mm_setr_epi32(0,stride,2*stride,3*stride),8);
    }
    static void store(double* p,R v){_mm256_storeu_pd(p,v);}
};
template<class V,class Get>typename V::R pairwise(std::size_t begin,std::size_t count,const Get& get){
    if(count<8){auto sum=V::fill(-0.0);for(std::size_t i=0;i<count;++i)sum=V::add(sum,get(begin+i));return sum;}
    if(count<=128){
        typename V::R lanes[8];for(std::size_t j=0;j<8;++j)lanes[j]=get(begin+j);
        std::size_t i=8;for(;i+8<=count;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=V::add(lanes[j],get(begin+i+j));
        auto sum=V::add(V::add(V::add(lanes[0],lanes[1]),V::add(lanes[2],lanes[3])),V::add(V::add(lanes[4],lanes[5]),V::add(lanes[6],lanes[7])));
        for(;i<count;++i)sum=V::add(sum,get(begin+i));return sum;
    }
    auto left=count/2;left-=left%8;const auto a=pairwise<V>(begin,left,get),b=pairwise<V>(begin+left,count-left,get);return V::add(a,b);
}
template<class T>std::size_t row(const T* input,T* output,std::size_t columns,std::size_t fy,std::size_t fx,std::size_t count){
    using V=Vector<T>;
    if(fx>std::size_t(INT_MAX)/(V::width-1))return 0;
    std::size_t i=0;
    for(;count-i>=V::width;i+=V::width){
        const auto get=[&](std::size_t position){return V::load(input+i*fx+(position/fx)*columns+position%fx,int(fx));};
        V::store(output+i,V::add(V::fill(0),pairwise<V>(0,fy*fx,get)));
    }
    return i;
}
}
std::size_t block_reduce_row(const float* a,float* out,std::size_t columns,std::size_t fy,std::size_t fx,std::size_t n){return row(a,out,columns,fy,fx,n);}
std::size_t block_reduce_row(const double* a,double* out,std::size_t columns,std::size_t fy,std::size_t fx,std::size_t n){return row(a,out,columns,fy,fx,n);}
}
