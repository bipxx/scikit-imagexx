#include "moments_internal.hpp"
#include <immintrin.h>
#include <algorithm>

namespace skimagexx::simd {
namespace {
template<class R> struct Ops;
template<>struct Ops<float> {
    using V=__m256;static constexpr std::size_t width=8;
    static V broadcast(float x){return _mm256_set1_ps(x);}
    static V zero(){return _mm256_setzero_ps();}static V load(const float*p){return _mm256_loadu_ps(p);}
    static void store(float*p,V v){_mm256_storeu_ps(p,v);}static V add(V a,V b){return _mm256_add_ps(a,b);}
    static V sub(V a,V b){return _mm256_sub_ps(a,b);}static V mul(V a,V b){return _mm256_mul_ps(a,b);}
};
template<>struct Ops<double> {
    using V=__m256d;static constexpr std::size_t width=4;
    static V broadcast(double x){return _mm256_set1_pd(x);}
    static V zero(){return _mm256_setzero_pd();}static V load(const double*p){return _mm256_loadu_pd(p);}
    static void store(double*p,V v){_mm256_storeu_pd(p,v);}static V add(V a,V b){return _mm256_add_pd(a,b);}
    static V sub(V a,V b){return _mm256_sub_pd(a,b);}static V mul(V a,V b){return _mm256_mul_pd(a,b);}
};
// Only complete worker-owned blocks: each input vector feeds K2/K3/K4 outputs.
template<std::size_t K,class T> void fused_project(const T*calc,const T*powers,
        std::size_t n,std::size_t inner,T*out,std::size_t begin,std::size_t end) {
    using V=Ops<T>;
    if constexpr(K==3 && std::is_same_v<T,double>) {
        if(n>=256 && inner>=512) {
            // Bounded3KiB column tile, each cell still accumulates ascending j.
            for(auto outer=begin/(K*inner);outer<end/(K*inner);++outer) {
                std::size_t rest=0;
                while(inner-rest>=V::width) {
                    const auto columns=std::min(std::size_t(128),inner-rest)/V::width*V::width;
                    alignas(32) double sums[3][128]{};
                    for(std::size_t j=0;j<n;++j) {
                        const auto w0=V::broadcast(powers[j*K]),w1=V::broadcast(powers[j*K+1]),w2=V::broadcast(powers[j*K+2]);
                        const auto* values=calc+(outer*n+j)*inner+rest;
                        for(std::size_t col=0;col<columns;col+=V::width) {
                            const auto a=V::load(values+col);
                            V::store(sums[0]+col,V::add(V::load(sums[0]+col),V::mul(a,w0)));
                            V::store(sums[1]+col,V::add(V::load(sums[1]+col),V::mul(a,w1)));
                            V::store(sums[2]+col,V::add(V::load(sums[2]+col),V::mul(a,w2)));
                        }
                    }
                    for(std::size_t k=0;k<K;++k)for(std::size_t col=0;col<columns;col+=V::width)
                        V::store(out+(outer*K+k)*inner+rest+col,V::load(sums[k]+col));
                    rest+=columns;
                }
                for(;rest<inner;++rest)for(std::size_t k=0;k<K;++k)
                    out[(outer*K+k)*inner+rest]=detail::moments_one(calc,powers,n,K,inner,(outer*K+k)*inner+rest);
            }
            return;
        }
    }
    for(auto outer=begin/(K*inner);outer<end/(K*inner);++outer) {
        std::size_t rest=0;
        if constexpr(std::is_same_v<T,double>) {
            for(;inner-rest>=8;rest+=8) {
                typename V::V first[K],second[K];
                for(std::size_t k=0;k<K;++k){first[k]=V::zero();second[k]=V::zero();}
                for(std::size_t j=0;j<n;++j) {
                    const auto values=calc+(outer*n+j)*inner+rest;
                    const auto a=V::load(values),b=V::load(values+4);
                    for(std::size_t k=0;k<K;++k) {
                        const auto weight=V::broadcast(powers[j*K+k]);
                        first[k]=V::add(first[k],V::mul(a,weight));
                        second[k]=V::add(second[k],V::mul(b,weight));
                    }
                }
                for(std::size_t k=0;k<K;++k) {
                    auto target=out+(outer*K+k)*inner+rest;
                    V::store(target,first[k]);V::store(target+4,second[k]);
                }
            }
        }
        for(;inner-rest>=V::width;rest+=V::width) {
            typename V::V sums[K];for(auto&sum:sums)sum=V::zero();
            for(std::size_t j=0;j<n;++j) {
                const auto values=V::load(calc+(outer*n+j)*inner+rest);
                for(std::size_t k=0;k<K;++k)
                    sums[k]=V::add(sums[k],V::mul(values,V::broadcast(powers[j*K+k])));
            }
            for(std::size_t k=0;k<K;++k)V::store(out+(outer*K+k)*inner+rest,sums[k]);
        }
        for(;rest<inner;++rest)for(std::size_t k=0;k<K;++k) {
            const auto index=(outer*K+k)*inner+rest;
            out[index]=detail::moments_one(calc,powers,n,K,inner,index);
        }
    }
}
template<class T> void project(const T*calc,const T*powers,std::size_t n,std::size_t K,
                               std::size_t inner,T*out,std::size_t begin,std::size_t end) {
    using R=typename detail::MomentsValue<T>::real;using V=Ops<R>;
    if constexpr(!detail::MomentsValue<T>::complex) {
        if((K==2 || K==3 || K==4) && inner>=V::width && begin%(K*inner)==0 && end%(K*inner)==0) {
            if(K==2)fused_project<2>(calc,powers,n,inner,out,begin,end);
            else if(K==3)fused_project<3>(calc,powers,n,inner,out,begin,end);
            else fused_project<4>(calc,powers,n,inner,out,begin,end);
            return;
        }
    }
    auto i=begin;
    for(;end-i>=V::width;i+=V::width) {
        if constexpr(!detail::MomentsValue<T>::complex) {
            const auto rest=i%inner,k=(i/inner)%K,outer=i/(inner*K);
            if(inner-rest>=V::width) {
                auto sum=V::zero();
                for(std::size_t j=0;j<n;++j)
                    sum=V::add(sum,V::mul(V::load(calc+(outer*n+j)*inner+rest),V::broadcast(powers[j*K+k])));
                V::store(out+i,sum);continue;
            }
            if(inner==1 && K-k>=V::width) {
                auto sum=V::zero();
                for(std::size_t j=0;j<n;++j)
                    sum=V::add(sum,V::mul(V::broadcast(calc[outer*n+j]),V::load(powers+j*K+k)));
                V::store(out+i,sum);continue;
            }
        }
        auto real=V::zero(),imag=V::zero();
        for(std::size_t j=0;j<n;++j) {
            R ar[V::width],br[V::width],ai[V::width]{},bi[V::width]{};
            for(std::size_t lane=0;lane<V::width;++lane) {
                const auto index=i+lane,rest=index%inner,k=(index/inner)%K,outer=index/(inner*K);
                const auto a=calc[(outer*n+j)*inner+rest],b=powers[j*K+k];
                if constexpr(detail::MomentsValue<T>::complex) {
                    ar[lane]=a.real();ai[lane]=a.imag();br[lane]=b.real();bi[lane]=b.imag();
                } else {ar[lane]=a;br[lane]=b;}
            }
            const auto a=V::load(ar),b=V::load(br);
            if constexpr(detail::MomentsValue<T>::complex) {
                const auto c=V::load(ai),d=V::load(bi);
                real=V::add(real,V::sub(V::mul(a,b),V::mul(c,d)));
                imag=V::add(imag,V::add(V::mul(a,d),V::mul(c,b)));
            } else real=V::add(real,V::mul(a,b));
        }
        R r[V::width],im[V::width];V::store(r,real);
        if constexpr(detail::MomentsValue<T>::complex)V::store(im,imag);
        for(std::size_t lane=0;lane<V::width;++lane) {
            if constexpr(detail::MomentsValue<T>::complex)out[i+lane]=T(r[lane],im[lane]);
            else out[i+lane]=r[lane];
        }
    }
    for(;i<end;++i)out[i]=detail::moments_one(calc,powers,n,K,inner,i);
}
}
#define SKIMAGEXX_MOMENTS_DEFINE(T) \
void moments_project(const T*a,const T*p,std::size_t n,std::size_t K,std::size_t inner, \
                     T*out,std::size_t begin,std::size_t end) { project(a,p,n,K,inner,out,begin,end); }
SKIMAGEXX_MOMENTS_DEFINE(float)
SKIMAGEXX_MOMENTS_DEFINE(double)
SKIMAGEXX_MOMENTS_DEFINE(std::complex<float>)
SKIMAGEXX_MOMENTS_DEFINE(std::complex<double>)
#undef SKIMAGEXX_MOMENTS_DEFINE
}

// Fixed default normalized moments: four independent double divisions per row.
#include "moments_normalized.hpp"

namespace skimagexx::simd {
void moments_normalize_default(const float* input, const double* denominator, float* out) {
    for (int row = 0; row != 4; ++row) {
        const auto divisor = _mm256_loadu_pd(denominator + row);
        const auto values = _mm256_cvtps_pd(_mm_loadu_ps(input + 4 * row));
        _mm_storeu_ps(out + 4 * row, _mm256_cvtpd_ps(_mm256_div_pd(values, divisor)));
    }
}

void moments_normalize_default(const double* input, const double* denominator, double* out) {
    for (int row = 0; row != 4; ++row) {
        const auto divisor = _mm256_loadu_pd(denominator + row);
        const auto values = _mm256_loadu_pd(input + 4 * row);
        _mm256_storeu_pd(out + 4 * row, _mm256_div_pd(values, divisor));
    }
}
}  // namespace skimagexx::simd
