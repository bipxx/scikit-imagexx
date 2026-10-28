#include "compare_images.hpp"
#include "skimagexx/image.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
#include <xmmintrin.h>
#endif
using namespace skimagexx;using namespace skimagexx::detail;
namespace {
std::size_t checks=0;
void require(bool value){++checks;if(!value)throw std::runtime_error("compare control "+std::to_string(checks));}
template<class T>constexpr PixelType type(){return std::is_same_v<T,float>?PixelType::float32:PixelType::float64;}
template<class T>void raw(std::ofstream& stream,const T* p,std::size_t n){stream.write(reinterpret_cast<const char*>(p),std::streamsize(n*sizeof(T)));}
template<class A,class B,class O>void matrix(std::ofstream& stream,int workers){
    for(auto method:{CompareMethod::diff,CompareMethod::blend,CompareMethod::checkerboard}){
        if(method==CompareMethod::checkerboard&&!std::is_same_v<A,O>)continue;
        if(method!=CompareMethod::checkerboard&&std::is_same_v<O,float>&&(!std::is_same_v<A,float>||!std::is_same_v<B,float>))continue;
        for(std::size_t n:{1u,2u,3u,4u,7u,8u,9u,15u,16u,17u,31u,32u,33u,77u,263169u}){
            std::vector<A>a(n);std::vector<B>b(n);std::vector<O>output(n+2,O(-123)),expected(n);
            for(std::size_t i=0;i<n;++i){a[i]=A(int(i*37%251)-125)/A(256);b[i]=B(int(i*61%251)-125)/B(256);}
            CompareGrid grid{1,n,1,3,1,n/3};
            for(std::size_t i=0;i<n;++i){
                if(method==CompareMethod::checkerboard){const bool first=i<3*(n/3)&&((i/3)%2==0);expected[i]=first?O(a[i]):O(b[i]);}
                else {const O left=O(a[i]),right=O(b[i]);const O middle=method==CompareMethod::blend?O(right+left):O(right-left);expected[i]=method==CompareMethod::blend?O(O(.5)*middle):std::fabs(middle);}
            }
            require(compare_images_into({a.data(),type<A>()},{b.data(),type<B>()},output.data()+1,n,method,grid,workers));
            require(output.front()==O(-123)&&output.back()==O(-123));
            require(std::memcmp(output.data()+1,expected.data(),n*sizeof(O))==0);raw(stream,output.data()+1,n);
        }
    }
}
void fixtures(const char* path,std::ofstream& stream,int workers){
    std::ifstream input(path,std::ios::binary);require(bool(input));std::uint64_t records=0;input.read(reinterpret_cast<char*>(&records),8);require(records==36);
    for(std::uint64_t record=0;record<records;++record){
        std::uint64_t row[12];input.read(reinterpret_cast<char*>(row),sizeof(row));
        const auto n=std::size_t(row[4]);std::vector<std::uint64_t>a((n*row[0]+7)/8),b((n*row[1]+7)/8),expected((row[11]+7)/8),output(expected.size()+2,0x123456789abcdef0ULL);
        input.read(reinterpret_cast<char*>(a.data()),std::streamsize(n*row[0]));input.read(reinterpret_cast<char*>(b.data()),std::streamsize(n*row[1]));input.read(reinterpret_cast<char*>(expected.data()),std::streamsize(row[11]));require(bool(input));
        CompareGrid grid{std::size_t(row[5]),std::size_t(row[6]),std::size_t(row[7]),std::size_t(row[8]),std::size_t(row[9]),std::size_t(row[10])};
        require(compare_images_into({a.data(),row[0]==4?PixelType::float32:PixelType::float64},{b.data(),row[1]==4?PixelType::float32:PixelType::float64},output.data()+1,n,CompareMethod(row[3]),grid,workers));
        require(std::memcmp(output.data()+1,expected.data(),std::size_t(row[11]))==0);
        require(output.front()==0x123456789abcdef0ULL&&output.back()==0x123456789abcdef0ULL);raw(stream,reinterpret_cast<const char*>(output.data()+1),std::size_t(row[11]));
    }
    require(input.peek()==std::ifstream::traits_type::eof());
}
template<class T>void exceptional(){
    using U=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,std::uint64_t>;
    const U exponent=std::is_same_v<T,float>?U(0x7f800000):U(0x7ff0000000000000ULL);
    for(std::size_t n=1;n<=17;++n)for(std::size_t index=0;index<n;++index)for(U bits:{exponent,U(exponent|1),U(exponent|(U(1)<<(std::is_same_v<T,float>?22:51)))}){
        std::vector<T>a(n,T(.25)),b(n,T(.5)),output(n);std::memcpy(a.data()+index,&bits,sizeof(T));
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const int flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(!compare_images_into({a.data(),type<T>()},{b.data(),type<T>()},output.data(),n,CompareMethod::diff,{},1));require(std::fetestexcept(FE_ALL_EXCEPT)==flags);
        require(!compare_images_into({b.data(),type<T>()},{a.data(),type<T>()},output.data(),n,CompareMethod::blend,{},1));require(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
    for(std::size_t n:{1u,4u,8u,9u,17u}){
        std::vector<T>a(n,std::numeric_limits<T>::max()),output(n);
        std::feclearexcept(FE_ALL_EXCEPT);require(!compare_images_into({a.data(),type<T>()},{a.data(),type<T>()},output.data(),n,CompareMethod::blend,{},1));require(std::fetestexcept(FE_ALL_EXCEPT)==0);
        std::fill(a.begin(),a.end(),std::numeric_limits<T>::min());std::vector<T>b(n,std::numeric_limits<T>::denorm_min());
        require(!compare_images_into({a.data(),type<T>()},{b.data(),type<T>()},output.data(),n,CompareMethod::blend,{},1));require(std::fetestexcept(FE_ALL_EXCEPT)==0);
        std::fill(a.begin(),a.end(),T(-0.0));std::fill(b.begin(),b.end(),T(0.0));
        require(compare_images_into({a.data(),type<T>()},{b.data(),type<T>()},output.data(),n,CompareMethod::diff,{},1));for(T value:output)require(value==0&&!std::signbit(value));
    }
    std::vector<T>a(8,T(.25)),b(8,T(.5)),output(8);
    const auto original_round=std::fegetround();require(std::fesetround(FE_DOWNWARD)==0);
    require(!compare_images_into({a.data(),type<T>()},{b.data(),type<T>()},output.data(),8,CompareMethod::blend,{},1));require(std::fegetround()==FE_DOWNWARD);std::fesetround(original_round);
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_IX86_FP)
    const auto csr=_mm_getcsr();_mm_setcsr(csr|0x8040);require(!compare_images_into({a.data(),type<T>()},{b.data(),type<T>()},output.data(),8,CompareMethod::diff,{},1));require(_mm_getcsr()==(csr|0x8040));_mm_setcsr(csr);
#endif
}
void cast_controls(){
    float first[17],output[17];double second[17];
    for(std::size_t n:{1u,4u,8u,9u,17u}){
        std::fill_n(first,n,.25f);std::fill_n(second,n,.5);
        CompareGrid all_second{1,n,0,0,0,0};
        std::fill_n(second,n,std::numeric_limits<double>::max());std::feclearexcept(FE_ALL_EXCEPT);
        require(!compare_images_into({first,PixelType::float32},{second,PixelType::float64},output,n,CompareMethod::checkerboard,all_second,1));require(std::fetestexcept(FE_ALL_EXCEPT)==0);
        std::fill_n(second,n,std::numeric_limits<double>::min());
        require(!compare_images_into({first,PixelType::float32},{second,PixelType::float64},output,n,CompareMethod::checkerboard,all_second,1));require(std::fetestexcept(FE_ALL_EXCEPT)==0);
        for(std::size_t i=0;i<n;++i)second[i]=i%2?.5:std::numeric_limits<double>::max();
        require(compare_images_into({first,PixelType::float32},{second,PixelType::float64},output,n,CompareMethod::checkerboard,{1,n,1,1,1,n},1));
        for(std::size_t i=0;i<n;++i)require(output[i]==(i%2?.5f:.25f));
        for(std::size_t i=0;i<n;++i)second[i]=1.0+double(int(i%3)+1)/16777216.0;
        float expected[17];for(std::size_t i=0;i<n;++i)expected[i]=float(second[i]);
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        require(compare_images_into({first,PixelType::float32},{second,PixelType::float64},output,n,CompareMethod::checkerboard,all_second,1));
        require(std::fetestexcept(FE_ALL_EXCEPT)==flags);require(std::memcmp(output,expected,n*4)==0);
        double a[17],b[17],result[17];std::fill_n(a,n,std::numeric_limits<double>::max());std::fill_n(b,n,-std::numeric_limits<double>::max());
        require(!compare_images_into({a,PixelType::float64},{b,PixelType::float64},result,n,CompareMethod::diff,{},1));require(std::fetestexcept(FE_ALL_EXCEPT)==flags);
    }
}
void grids(std::ofstream& stream,int workers){
    const std::size_t h=513,w=515,n=h*w;std::vector<float>a(n),b(n),output(n),expected(n);
    for(std::size_t i=0;i<n;++i){a[i]=float(i%251)/256;b[i]=-a[i];}
    for(auto grid:std::vector<CompareGrid>{{h,w,64,64,8,8},{h,w,0,0,0,0},{h,w,1,1,h,w},{h,w,100,101,5,5}}){
        for(std::size_t y=0;y<h;++y)for(std::size_t x=0;x<w;++x){bool first=grid.row_tiles&&grid.column_tiles&&y<grid.row_step*grid.row_tiles&&x<grid.column_step*grid.column_tiles&&((y/grid.row_step+x/grid.column_step)%2==0);expected[y*w+x]=first?a[y*w+x]:b[y*w+x];}
        require(compare_images_into({a.data(),PixelType::float32},{b.data(),PixelType::float32},output.data(),n,CompareMethod::checkerboard,grid,workers));require(std::memcmp(output.data(),expected.data(),n*4)==0);raw(stream,output.data(),n);
    }
}
}
namespace fused_controls {
std::size_t checks=0;std::ofstream journal;
void check(bool value){if(!value)throw std::runtime_error("prospective fused control "+std::to_string(checks));++checks;}
template<class T>constexpr PixelType tag(){return std::is_same_v<T,float>?PixelType::float32:PixelType::float64;}
template<class T>void poison(T* p,unsigned pattern){
    using U=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,std::uint64_t>;
    constexpr unsigned mantissa=std::is_same_v<T,float>?23:52;
    constexpr U exponent=std::is_same_v<T,float>?U(0x7f800000):U(UINT64_C(0x7ff0000000000000));
    U bits=exponent;
    if(pattern%3==1)bits|=U(1);                    // signaling NaN
    if(pattern%3==2)bits|=U(1)<<(mantissa-1);      // quiet NaN
    if(pattern>=3)bits|=U(1)<<(sizeof(T)*8-1);     // both signs, including infinity
    std::memcpy(p,&bits,sizeof(T));
}
template<class A,class B,class O>void reject(std::size_t n,std::size_t position,unsigned slot,
    unsigned pattern,CompareMethod method,CompareGrid grid,int workers,int flags){
    std::vector<A>a(n,A(.25));std::vector<B>b(n,B(.5));std::vector<O>output(n+2,O(-123));
    if(slot==0||slot==2||slot==3)poison(a.data()+position,pattern);else poison(b.data()+position,pattern);
    if(slot==2)poison(b.data()+position,pattern);
    if(slot==3)poison(a.data(),pattern);
    if(slot==4)poison(b.data(),pattern);
    const auto saved_a=a;const auto saved_b=b;
    std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(flags);const int entered=std::fetestexcept(FE_ALL_EXCEPT);
    const bool accepted=compare_images_into({a.data(),tag<A>()},{b.data(),tag<B>()},output.data()+1,n,method,grid,workers);
    check(!accepted);check(std::fetestexcept(FE_ALL_EXCEPT)==entered);
    check(output.front()==O(-123)&&output.back()==O(-123));
    check(std::memcmp(a.data(),saved_a.data(),n*sizeof(A))==0);
    check(std::memcmp(b.data(),saved_b.data(),n*sizeof(B))==0);
    journal.write(reinterpret_cast<const char*>(&accepted),sizeof(accepted));journal.write(reinterpret_cast<const char*>(&entered),sizeof(entered));
}
template<class A,class B>void pair(int workers){
    using O=std::conditional_t<std::is_same_v<A,double>||std::is_same_v<B,double>,double,float>;
    for(std::size_t n=1;n<=33;++n){
        // All scalar tails modulo4/8 and several complete packet boundaries.
        for(std::size_t position=0;position<n;++position)for(unsigned slot:{0u,1u})
        for(unsigned pattern=0;pattern<6;++pattern)for(auto method:{CompareMethod::diff,CompareMethod::blend})
        for(int flags:{0,FE_DIVBYZERO,FE_INVALID|FE_OVERFLOW,FE_UNDERFLOW|FE_INEXACT})
            reject<A,B,O>(n,position,slot,pattern,method,{},workers,flags);
    }
    // Both operands and repeated hits must not cancel in the accumulated mask.
    for(unsigned slot:{2u,3u,4u})for(auto method:{CompareMethod::diff,CompareMethod::blend})
        reject<A,B,O>(33,16,slot,2,method,{},workers,FE_INVALID);
    // Actual parallel threshold and each worker chunk/tail. Quiet NaNs require
    // the finite mask even when arithmetic itself raises no exception.
    const std::size_t n=263169;
    for(std::size_t position:{std::size_t(0),n/4,n/4+1,n/2,n/2+1,3*n/4,3*n/4+1,n-1})
    for(unsigned slot:{0u,1u})for(auto method:{CompareMethod::diff,CompareMethod::blend})
        reject<A,B,O>(n,position,slot,2,method,{},workers,FE_DIVBYZERO);
    // Preserve the existing whole-input checker admission, even for an unused
    // nonfinite input. Checker output dtype remains the first input's dtype.
    for(std::size_t count:{1u,4u,8u,17u}){
        reject<A,B,A>(count,count-1,1,2,CompareMethod::checkerboard,{1,count,1,count,1,1},workers,FE_INVALID);
        reject<A,B,A>(count,0,0,0,CompareMethod::checkerboard,{1,count,0,0,0,0},workers,FE_DIVBYZERO);
    }
    // Zero-sign rules are source-provable and were not fully covered by the
    // old same-type diff-only exceptional cohort.
    for(unsigned signs=0;signs<4;++signs)for(auto method:{CompareMethod::diff,CompareMethod::blend}){
        std::vector<A>a(17,signs&1?A(-0.):A(0.));std::vector<B>b(17,signs&2?B(-0.):B(0.));std::vector<O>output(19,O(-123));
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        check(compare_images_into({a.data(),tag<A>()},{b.data(),tag<B>()},output.data()+1,17,method,{},workers));
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);check(output.front()==O(-123)&&output.back()==O(-123));
        for(std::size_t i=1;i<=17;++i){check(output[i]==O(0.));check(std::signbit(output[i])==(method==CompareMethod::blend&&signs==3));}
    }
    // Exponent-zero finite values must not be rejected as nonfinite. The
    // subtraction of equal representable subnormals is exactly positive zero.
    const double tiny=(std::is_same_v<A,float>||std::is_same_v<B,float>)?double(std::numeric_limits<float>::denorm_min()):std::numeric_limits<double>::denorm_min();
    for(std::size_t count:{1u,4u,8u,9u,17u,33u}){
        std::vector<A>a(count,A(tiny));std::vector<B>b(count,B(tiny));std::vector<O>output(count+2,O(-123));
        std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_INVALID);const auto flags=std::fetestexcept(FE_ALL_EXCEPT);
        check(compare_images_into({a.data(),tag<A>()},{b.data(),tag<B>()},output.data()+1,count,CompareMethod::diff,{},workers));
        check(std::fetestexcept(FE_ALL_EXCEPT)==flags);check(output.front()==O(-123)&&output.back()==O(-123));
        for(std::size_t i=1;i<=count;++i)check(output[i]==O(0.)&&!std::signbit(output[i]));
    }
}

void run(const std::string& path){
    journal.open(path+".fused-controls.bin",std::ios::binary);check(bool(journal));check(get_num_threads()==1);
    for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers!=1)continue;
#endif
            set_num_threads(workers);pair<float,float>(workers);pair<float,double>(workers);pair<double,float>(workers);pair<double,double>(workers);
        }
        set_num_threads(1);std::feclearexcept(FE_ALL_EXCEPT);check(get_num_threads()==1);
    std::cout<<checks<<" prospective fused controls PASS\n";
}
}
int main(int argc,char**argv){
    try{
        require(argc==2||argc==3);std::ofstream stream(argv[1],std::ios::binary);require(bool(stream));
        const auto initial_round=std::fegetround();std::fesetround(FE_TONEAREST);
        for(int workers:{1,4}){
#ifndef _OPENMP
            if(workers==4){double a=1,b=2,o=0;require(!compare_images_into({&a,PixelType::float64},{&b,PixelType::float64},&o,1,CompareMethod::diff,{},workers));continue;}
#endif
            matrix<float,float,float>(stream,workers);matrix<float,double,double>(stream,workers);matrix<double,float,double>(stream,workers);matrix<double,double,double>(stream,workers);matrix<float,double,float>(stream,workers);
            if(argc==3)fixtures(argv[2],stream,workers);grids(stream,workers);
        }
        exceptional<float>();exceptional<double>();cast_controls();
        double a=1,b=2,o=0;require(!compare_images_into({&a,PixelType::uint8},{&b,PixelType::float64},&o,1,CompareMethod::diff,{},1));require(!compare_images_into({&a,PixelType::float64},{&b,PixelType::float64},&o,1,CompareMethod::diff,{},0));require(!compare_images_into({nullptr,PixelType::float64},{&b,PixelType::float64},&o,1,CompareMethod::diff,{},1));
        require(!compare_images_into({&a,PixelType::float64},{&b,PixelType::float64},&o,1,CompareMethod::checkerboard,{1,1,0,1,1,1},1));
        require(compare_images_into({nullptr,PixelType::float64},{nullptr,PixelType::float32},nullptr,0,CompareMethod::diff,{},1));
        std::feclearexcept(FE_ALL_EXCEPT);std::fesetround(initial_round);require(get_num_threads()==1);
        fused_controls::run(argv[1]);
        std::cout<<checks<<" Compare controls PASS avx2="<<has_avx2()<<" workers=1\n";
    }catch(const std::exception& exc){std::cerr<<exc.what()<<"\n";return 1;}
}
