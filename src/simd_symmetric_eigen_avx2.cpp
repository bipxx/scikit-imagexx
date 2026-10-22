#include "symmetric_eigen_detail.hpp"
#include <immintrin.h>
#include <vector>
#include <type_traits>

namespace skimagexx::simd { namespace {
__m256d absolute(__m256d value){return _mm256_andnot_pd(_mm256_set1_pd(-0.),value);}
__m256 absolute(__m256 value){return _mm256_andnot_ps(_mm256_set1_ps(-0.f),value);}
template<class T> __m256d load(const void* field,std::size_t i) {
    if constexpr(std::is_same_v<T,float>)return _mm256_cvtps_pd(_mm_loadu_ps(static_cast<const float*>(field)+i));
    else return _mm256_loadu_pd(static_cast<const double*>(field)+i);
}
template<class T> void store(T* destination,__m256d value) {
    if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(destination,_mm256_cvtpd_ps(value));
    else _mm256_storeu_pd(destination,value);
}
struct alignas(32) Lanes {__m256d value;double output[4];};
template<class T> std::size_t run(const void* const* fields,T* output,std::size_t count,std::size_t n,
    std::size_t begin,std::size_t end,bool pair,bool interleaved,bool& supported) {
    auto i=begin;const auto limit=detail::eigen_limit<T>(n,pair);
    if constexpr(std::is_same_v<T,float>){
        if(pair && n==2){
            const auto bound=_mm256_set1_ps(float(limit)),half=_mm256_set1_ps(.5f);
            for(;i+8<=end;i+=8){
                const auto a=_mm256_loadu_ps(static_cast<const float*>(fields[0])+i),b=_mm256_loadu_ps(static_cast<const float*>(fields[1])+i),c=_mm256_loadu_ps(static_cast<const float*>(fields[2])+i);
                auto finite=_mm256_and_ps(_mm256_cmp_ps(absolute(a),bound,_CMP_LE_OQ),_mm256_cmp_ps(absolute(b),bound,_CMP_LE_OQ));
                finite=_mm256_and_ps(finite,_mm256_cmp_ps(absolute(c),bound,_CMP_LE_OQ));
                if(_mm256_movemask_ps(finite)!=255){supported=false;return i;}
                const auto mean=_mm256_mul_ps(_mm256_add_ps(a,c),half),difference=_mm256_mul_ps(_mm256_sub_ps(a,c),half);
                const auto root=_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(b,b),_mm256_mul_ps(difference,difference)));
                const auto upper=_mm256_add_ps(mean,root),lower=_mm256_sub_ps(mean,root);
                if(interleaved){const auto lo=_mm256_unpacklo_ps(lower,upper),hi=_mm256_unpackhi_ps(lower,upper);
                    _mm256_storeu_ps(output+2*i,_mm256_permute2f128_ps(lo,hi,0x20));_mm256_storeu_ps(output+2*i+8,_mm256_permute2f128_ps(lo,hi,0x31));
                }else{_mm256_storeu_ps(output+i,upper);_mm256_storeu_ps(output+count+i,lower);}
            }
            return i;
        }
    }
    if(pair && n==2){
        const auto bound=_mm256_set1_pd(limit),half=_mm256_set1_pd(.5);
        for(;i+4<=end;i+=4){const auto a=load<T>(fields[0],i),b=load<T>(fields[1],i),c=load<T>(fields[2],i);
            auto finite=_mm256_and_pd(_mm256_cmp_pd(absolute(a),bound,_CMP_LE_OQ),_mm256_cmp_pd(absolute(b),bound,_CMP_LE_OQ));
            finite=_mm256_and_pd(finite,_mm256_cmp_pd(absolute(c),bound,_CMP_LE_OQ));
            if(_mm256_movemask_pd(finite)!=15){supported=false;return i;}
            const auto mean=_mm256_mul_pd(_mm256_add_pd(a,c),half),difference=_mm256_mul_pd(_mm256_sub_pd(a,c),half);
            const auto root=_mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(b,b),_mm256_mul_pd(difference,difference)));
            const auto upper=_mm256_add_pd(mean,root),lower=_mm256_sub_pd(mean,root);
            if(interleaved){const auto lo=_mm256_unpacklo_pd(lower,upper),hi=_mm256_unpackhi_pd(lower,upper);
                store(output+2*i,_mm256_permute2f128_pd(lo,hi,0x20));store(output+2*i+4,_mm256_permute2f128_pd(lo,hi,0x31));
            }else{store(output+i,upper);store(output+count+i,lower);}
        }
        return i;
    }
    const auto one=_mm256_set1_pd(1.),two=_mm256_set1_pd(2.),zero=_mm256_setzero_pd();
    const auto bound=_mm256_set1_pd(limit),tolerance=_mm256_set1_pd(detail::eigen_tolerance);
    std::vector<Lanes> matrix(n*n),values(n);
    for(;i+4<=end;i+=4){
        auto scale=zero;std::size_t field=0;
        for(std::size_t r=0;r<n;++r)for(std::size_t c=r;c<n;++c){const auto x=load<T>(fields[field++],i);
            if(_mm256_movemask_pd(_mm256_cmp_pd(absolute(x),bound,_CMP_LE_OQ))!=15){supported=false;return i;}
            matrix[r*n+c].value=matrix[c*n+r].value=x;scale=_mm256_max_pd(scale,absolute(x));
        }
        if(n==1){store(output+i,matrix[0].value);continue;}
        scale=_mm256_blendv_pd(scale,one,_mm256_cmp_pd(scale,zero,_CMP_EQ_OQ));
        for(auto& x:matrix)x.value=_mm256_div_pd(x.value,scale);
        bool converged=false;
        for(int sweep=0;sweep<48 && !converged;++sweep){
            for(std::size_t p=0;p<n;++p)for(std::size_t q=p+1;q<n;++q){
                const auto off=matrix[p*n+q].value;const auto active=_mm256_cmp_pd(absolute(off),tolerance,_CMP_GT_OQ);
                if(!_mm256_movemask_pd(active))continue;
                const auto denominator=_mm256_blendv_pd(one,_mm256_mul_pd(two,off),active);
                const auto theta=_mm256_div_pd(_mm256_sub_pd(matrix[q*n+q].value,matrix[p*n+p].value),denominator);
                const auto sign=_mm256_blendv_pd(one,_mm256_set1_pd(-1.),_mm256_cmp_pd(theta,zero,_CMP_LT_OQ));
                auto t=_mm256_div_pd(sign,_mm256_add_pd(absolute(theta),_mm256_sqrt_pd(_mm256_add_pd(one,_mm256_mul_pd(theta,theta)))));
                t=_mm256_and_pd(t,active);
                const auto c=_mm256_div_pd(one,_mm256_sqrt_pd(_mm256_add_pd(one,_mm256_mul_pd(t,t)))),s=_mm256_mul_pd(t,c);
                const auto shift=_mm256_mul_pd(t,off);
                matrix[p*n+p].value=_mm256_sub_pd(matrix[p*n+p].value,shift);matrix[q*n+q].value=_mm256_add_pd(matrix[q*n+q].value,shift);
                matrix[p*n+q].value=matrix[q*n+p].value=zero;
                for(std::size_t k=0;k<n;++k)if(k!=p && k!=q){const auto a=matrix[k*n+p].value,b=matrix[k*n+q].value;
                    matrix[k*n+p].value=matrix[p*n+k].value=_mm256_sub_pd(_mm256_mul_pd(c,a),_mm256_mul_pd(s,b));
                    matrix[k*n+q].value=matrix[q*n+k].value=_mm256_add_pd(_mm256_mul_pd(s,a),_mm256_mul_pd(c,b));
                }
            }
            auto maximum=zero;
            for(std::size_t r=0;r<n;++r)for(std::size_t c=r+1;c<n;++c)maximum=_mm256_max_pd(maximum,absolute(matrix[r*n+c].value));
            converged=_mm256_movemask_pd(_mm256_cmp_pd(maximum,tolerance,_CMP_LE_OQ))==15;
        }
        if(!converged){supported=false;return i;}
        for(std::size_t d=0;d<n;++d)values[d].value=matrix[(n-1-d)*n+n-1-d].value;
        for(std::size_t pass=0;pass<n;++pass)for(std::size_t a=0;a+1<n-pass;++a){
            const auto x=values[a].value,y=values[a+1].value;
            const auto swap=_mm256_cmp_pd(x,y,_CMP_LT_OQ);
            values[a].value=_mm256_blendv_pd(x,y,swap);
            values[a+1].value=_mm256_blendv_pd(y,x,swap);
        }
        if(interleaved){
            for(std::size_t d=0;d<n;++d)_mm256_store_pd(values[d].output,_mm256_mul_pd(values[d].value,scale));
            for(std::size_t lane=0;lane<4;++lane)for(std::size_t d=0;d<n;++d)output[(i+lane)*n+n-1-d]=T(values[d].output[lane]);
        }else for(std::size_t d=0;d<n;++d)store(output+d*count+i,_mm256_mul_pd(values[d].value,scale));
    }
    return i;
}
}}
namespace skimagexx::simd {
std::size_t symmetric_eigenvalues(const void* const* fields,void* output,std::size_t count,std::size_t dimension,
    std::size_t begin,std::size_t end,bool single,bool pair,bool interleaved,bool& supported) {
    return single ? run(fields,static_cast<float*>(output),count,dimension,begin,end,pair,interleaved,supported)
                  : run(fields,static_cast<double*>(output),count,dimension,begin,end,pair,interleaved,supported);
}
}
