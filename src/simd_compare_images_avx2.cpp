#include "compare_images.hpp"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace skimagexx::simd { namespace {
template<class T>bool finite(const T* p,std::size_t count) {
    using U=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,std::uint64_t>;
    const U exponent=std::is_same_v<T,float>?U(0x7f800000):U(0x7ff0000000000000ULL);
    const auto mask=std::is_same_v<T,float>?_mm256_set1_epi32(int(exponent)):_mm256_set1_epi64x(static_cast<long long>(exponent));
    constexpr std::size_t lanes=32/sizeof(T);std::size_t i=0;
    for(;count-i>=lanes;i+=lanes){
        auto value=_mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i)),mask);
        value=std::is_same_v<T,float>?_mm256_cmpeq_epi32(value,mask):_mm256_cmpeq_epi64(value,mask);
        if(_mm256_movemask_epi8(value))return false;
    }
    for(;i<count;++i){U bits;std::memcpy(&bits,p+i,sizeof(T));if((bits&exponent)==exponent)return false;}
    return true;
}
template<class T>__m256 load_float(const T* p){
    if constexpr(std::is_same_v<T,float>)return _mm256_loadu_ps(p);
    else return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(_mm256_loadu_pd(p))),_mm256_cvtpd_ps(_mm256_loadu_pd(p+4)),1);
}
template<class T>__m256d load_double(const T* p){
    if constexpr(std::is_same_v<T,double>)return _mm256_loadu_pd(p);
    else return _mm256_cvtps_pd(_mm_loadu_ps(p));
}
template<class A,class B,class O>std::size_t span(const A* first,const B* second,O* output,std::size_t count,detail::CompareMethod method){
    constexpr std::size_t lanes=32/sizeof(O);std::size_t i=0;
    for(;count-i>=lanes;i+=lanes){
        if constexpr(std::is_same_v<O,float>){
            auto value=load_float(first+i);
            if(method!=detail::CompareMethod::checkerboard){
                const auto other=load_float(second+i);
                value=method==detail::CompareMethod::blend?_mm256_add_ps(other,value):_mm256_sub_ps(other,value);
                value=method==detail::CompareMethod::blend?_mm256_mul_ps(value,_mm256_set1_ps(.5f)):_mm256_andnot_ps(_mm256_set1_ps(-0.0f),value);
            }
            _mm256_storeu_ps(output+i,value);
        }else{
            auto value=load_double(first+i);
            if(method!=detail::CompareMethod::checkerboard){
                const auto other=load_double(second+i);
                value=method==detail::CompareMethod::blend?_mm256_add_pd(other,value):_mm256_sub_pd(other,value);
                value=method==detail::CompareMethod::blend?_mm256_mul_pd(value,_mm256_set1_pd(.5)):_mm256_andnot_pd(_mm256_set1_pd(-0.0),value);
            }
            _mm256_storeu_pd(output+i,value);
        }
    }
    return i;
}
}
bool compare_finite(const void* p,std::size_t count,PixelType type){return type==PixelType::float32?finite(static_cast<const float*>(p),count):finite(static_cast<const double*>(p),count);}
std::size_t compare_span(const void* a,const void* b,void* o,std::size_t count,PixelType ta,PixelType tb,PixelType to,detail::CompareMethod method){
#define SELECT(A,B,O) return span(static_cast<const A*>(a),static_cast<const B*>(b),static_cast<O*>(o),count,method)
    if(to==PixelType::float32){
        if(ta==PixelType::float32){if(tb==PixelType::float32){SELECT(float,float,float);}else {SELECT(float,double,float);}}
        else {if(tb==PixelType::float32){SELECT(double,float,float);}else {SELECT(double,double,float);}}
    }else{
        if(ta==PixelType::float32){if(tb==PixelType::float32){SELECT(float,float,double);}else {SELECT(float,double,double);}}
        else {if(tb==PixelType::float32){SELECT(double,float,double);}else {SELECT(double,double,double);}}
    }
#undef SELECT
}
}
