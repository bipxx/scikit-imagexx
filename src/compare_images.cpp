#include "compare_images.hpp"
#include "execution.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif
namespace skimagexx::detail { namespace {
bool environment() {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
    if(_mm_getcsr()&0x8040)return false;
#endif
    return true;
}
class Capture {
    std::fenv_t saved_;std::atomic<bool>& refused_;bool active_;
public:
    explicit Capture(std::atomic<bool>& refused):refused_(refused),active_(std::feholdexcept(&saved_)==0) {
        if(!active_||!environment())refused_.store(true);
    }
    bool ready()const{return active_&&environment();}
    ~Capture(){if(active_){if(std::fetestexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW))refused_.store(true);std::fesetenv(&saved_);}}
};
template<class T>bool finite(const T* p,std::size_t count) {
    using U=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,std::uint64_t>;
    const U exponent=std::is_same_v<T,float>?U(0x7f800000):U(0x7ff0000000000000ULL);
    for(std::size_t i=0;i<count;++i){U bits;std::memcpy(&bits,p+i,sizeof(T));if((bits&exponent)==exponent)return false;}
    return true;
}
bool finite(CompareInput input,std::size_t count) {
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())return simd::compare_finite(input.data,count,input.type);
#endif
    return input.type==PixelType::float32?finite(static_cast<const float*>(input.data),count):finite(static_cast<const double*>(input.data),count);
}
template<class T>bool finite_value(T value) {
    using U=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,std::uint64_t>;
    const U exponent=std::is_same_v<T,float>?U(0x7f800000):U(0x7ff0000000000000ULL);
    U bits;std::memcpy(&bits,&value,sizeof(T));return (bits&exponent)!=exponent;
}
template<class A,class B,class O>bool span_typed(const A* first,const B* second,O* output,std::size_t count,CompareMethod method) {
    bool all_finite=true;
    for(std::size_t i=0;i<count;++i){
        if(method==CompareMethod::checkerboard){output[i]=O(first[i]);continue;}
        const A a=first[i];const B b=second[i];
        all_finite&=finite_value(a)&finite_value(b);
        const O left=O(a),right=O(b);
        const O middle=method==CompareMethod::blend?O(right+left):O(right-left);
        output[i]=method==CompareMethod::blend?O(O(.5)*middle):std::fabs(middle);
    }
    return all_finite;
}
bool span(CompareInput first,CompareInput second,void* output,std::size_t count,PixelType output_type,CompareMethod method) {
    std::size_t done=0;bool all_finite=true;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())done=simd::compare_span(first.data,second.data,output,count,first.type,second.type,output_type,method,all_finite);
#endif
    const auto* a=static_cast<const unsigned char*>(first.data)+done*(first.type==PixelType::float32?4:8);
    const auto* b=static_cast<const unsigned char*>(second.data)+done*(second.type==PixelType::float32?4:8);
    auto* o=static_cast<unsigned char*>(output)+done*(output_type==PixelType::float32?4:8);
    count-=done;
#define SELECT(A,B,O) all_finite&=span_typed(reinterpret_cast<const A*>(a),reinterpret_cast<const B*>(b),reinterpret_cast<O*>(o),count,method)
    if(output_type==PixelType::float32){
        if(first.type==PixelType::float32){if(second.type==PixelType::float32)SELECT(float,float,float);else SELECT(float,double,float);}
        else {if(second.type==PixelType::float32)SELECT(double,float,float);else SELECT(double,double,float);}
    }else{
        if(first.type==PixelType::float32){if(second.type==PixelType::float32)SELECT(float,float,double);else SELECT(float,double,double);}
        else {if(second.type==PixelType::float32)SELECT(double,float,double);else SELECT(double,double,double);}
    }
#undef SELECT
    return all_finite;
}
CompareInput offset(CompareInput input,std::size_t index){input.data=static_cast<const unsigned char*>(input.data)+index*(input.type==PixelType::float32?4:8);return input;}
}
bool compare_images_into(CompareInput first,CompareInput second,void* output,std::size_t count,
    CompareMethod method,const CompareGrid& grid,int workers) {
    if((first.type!=PixelType::float32&&first.type!=PixelType::float64)||(second.type!=PixelType::float32&&second.type!=PixelType::float64)||workers<1)return false;
    if(method!=CompareMethod::diff&&method!=CompareMethod::blend&&method!=CompareMethod::checkerboard)return false;
    if(count&&(!first.data||!second.data||!output))return false;
    if(count>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/8)return false;
#ifndef _OPENMP
    if(workers>1)return false;
#endif
    if(method==CompareMethod::checkerboard){
        if(grid.columns&&grid.rows>std::numeric_limits<std::size_t>::max()/grid.columns)return false;
        if(grid.rows*grid.columns!=count)return false;
        if(grid.row_tiles&&grid.column_tiles&&(grid.row_step==0||grid.column_step==0||grid.row_tiles>grid.rows/grid.row_step||grid.column_tiles>grid.columns/grid.column_step))return false;
    }
    if(!environment())return false;
    if(!count)return true;
    const auto output_type=method==CompareMethod::checkerboard?first.type:
        (first.type==PixelType::float64||second.type==PixelType::float64?PixelType::float64:PixelType::float32);
    const std::size_t output_size=output_type==PixelType::float32?4:8;
    std::atomic<bool> refused{false};
    // One held environment covers the serial admission scan and caller-thread
    // chunk; each OpenMP worker captures/restores its own established environment.
    {
        Capture caller(refused);if(!caller.ready())return false;
        if(method==CompareMethod::checkerboard&&(!finite(first,count)||!finite(second,count)))return false;
        const int team=count>=parallel_min?workers:1;
        const std::size_t units=method==CompareMethod::checkerboard?grid.rows:count;
        partition(units,team,[&](auto begin,auto end,auto){
            Capture worker(refused);if(!worker.ready())return;
            if(method!=CompareMethod::checkerboard){
                if(!span(offset(first,begin),offset(second,begin),static_cast<unsigned char*>(output)+begin*output_size,end-begin,output_type,method))refused.store(true);return;
            }
            const auto covered_rows=grid.row_tiles&&grid.column_tiles?grid.row_step*grid.row_tiles:0;
            for(auto row=begin;row<end;++row){
                std::size_t column=0;
                if(grid.row_tiles&&grid.column_tiles&&row<covered_rows){
                    const auto row_tile=row/grid.row_step;
                    for(std::size_t tile=0;tile<grid.column_tiles;++tile){
                        const auto index=row*grid.columns+column;
                        const auto chosen=((row_tile+tile)%2==0)?first:second;
                        // Checker assignment casts the chosen source directly to
                        // first dtype. The unused second span pointer stays valid.
                        span(offset(chosen,index),offset(chosen,index),static_cast<unsigned char*>(output)+index*output_size,grid.column_step,output_type,method);
                        column+=grid.column_step;
                    }
                }
                if(column<grid.columns){const auto index=row*grid.columns+column;span(offset(second,index),offset(second,index),static_cast<unsigned char*>(output)+index*output_size,grid.columns-column,output_type,method);}
            }
        });
    }
    return !refused.load();
}
}
