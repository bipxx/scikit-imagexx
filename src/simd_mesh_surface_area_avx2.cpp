#include "mesh_surface_area.hpp"
#include <immintrin.h>
#include <limits>

namespace skimagexx::simd {
namespace {
template<class T>struct Vector;
template<>struct Vector<float>{
    using V=__m256;static constexpr std::size_t lanes=8;
    static V load(const float* p,std::size_t s){
        const auto stride=int(s);return _mm256_i32gather_ps(p,_mm256_setr_epi32(0,stride,2*stride,3*stride,4*stride,5*stride,6*stride,7*stride),4);
    }
    static V sub(V a,V b){return _mm256_sub_ps(a,b);}
    static V mul(V a,V b){return _mm256_mul_ps(a,b);}
    static V add(V a,V b){return _mm256_add_ps(a,b);}
    static V sqrt(V a){return _mm256_sqrt_ps(a);}
    static void store(float* p,V a){_mm256_storeu_ps(p,a);}
    static __m256i bad(V a){
        const auto value=_mm256_and_si256(_mm256_castps_si256(a),_mm256_set1_epi32(0x7fffffff));
        const auto mask=_mm256_set1_epi32(0x7f800000),exponent=_mm256_and_si256(value,mask),zero=_mm256_setzero_si256();
        return _mm256_andnot_si256(_mm256_cmpeq_epi32(value,zero),_mm256_or_si256(_mm256_cmpeq_epi32(exponent,zero),_mm256_cmpeq_epi32(exponent,mask)));
    }
};
template<>struct Vector<double>{
    using V=__m256d;static constexpr std::size_t lanes=4;
    static V load(const double* p,std::size_t s){
        const auto stride=static_cast<long long>(s);return _mm256_i64gather_pd(p,_mm256_setr_epi64x(0,stride,2*stride,3*stride),8);
    }
    static V sub(V a,V b){return _mm256_sub_pd(a,b);}
    static V mul(V a,V b){return _mm256_mul_pd(a,b);}
    static V add(V a,V b){return _mm256_add_pd(a,b);}
    static V sqrt(V a){return _mm256_sqrt_pd(a);}
    static void store(double* p,V a){_mm256_storeu_pd(p,a);}
    static __m256i bad(V a){
        const auto value=_mm256_and_si256(_mm256_castpd_si256(a),_mm256_set1_epi64x(INT64_C(0x7fffffffffffffff)));
        const auto mask=_mm256_set1_epi64x(INT64_C(0x7ff0000000000000)),exponent=_mm256_and_si256(value,mask),zero=_mm256_setzero_si256();
        return _mm256_andnot_si256(_mm256_cmpeq_epi64(value,zero),_mm256_or_si256(_mm256_cmpeq_epi64(exponent,zero),_mm256_cmpeq_epi64(exponent,mask)));
    }
};
template<class T>std::size_t norms(const T* data,std::size_t count,std::size_t stride,T* output,bool& accepted){
    using Op=Vector<T>;using V=typename Op::V;
    if constexpr(sizeof(T)==4)if(stride>std::size_t(std::numeric_limits<int>::max())/7)return 0;
    std::size_t i=0;
    for(;count-i>=Op::lanes;i+=Op::lanes){
        auto bad=_mm256_setzero_si256();
        auto check=[&](V value){bad=_mm256_or_si256(bad,Op::bad(value));return value;};
        V a[3],b[3],squares[3];
        for(std::size_t j=0;j<3;++j){
            const auto first=check(Op::load(data+i*stride+j,stride));
            a[j]=check(Op::sub(first,check(Op::load(data+i*stride+3+j,stride))));
            b[j]=check(Op::sub(first,check(Op::load(data+i*stride+6+j,stride))));
        }
        for(std::size_t j=0;j<3;++j){
            const auto k=(j+1)%3,l=(j+2)%3;
            const auto cross=check(Op::sub(check(Op::mul(a[k],b[l])),check(Op::mul(a[l],b[k]))));
            squares[j]=check(Op::mul(cross,cross));
        }
        const auto sum=check(Op::add(check(Op::add(squares[0],squares[1])),squares[2]));
        Op::store(output+i,check(Op::sqrt(sum)));
        accepted&=_mm256_testz_si256(bad,bad)!=0;
    }
    return i;
}
}
std::size_t mesh_face_norms(const void* data,ColocType type,std::size_t count,std::size_t stride,void* output,bool& accepted){
    if(type==ColocType::float32)return norms(static_cast<const float*>(data),count,stride,static_cast<float*>(output),accepted);
    return norms(static_cast<const double*>(data),count,stride,static_cast<double*>(output),accepted);
}
}
