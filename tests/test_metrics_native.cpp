#include "skimagexx/metrics.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cfenv>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define METRIC_TEST_MXCSR 1
#endif
#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
bool metric_swapped_f32_range(const unsigned char*,std::size_t,MetricRawRange&);
}
#endif
using namespace skimagexx;
namespace {
void require(bool value){if(!value)throw std::runtime_error("metric native check failed");}
template<class T> MetricImageView view(const std::vector<T>& data,PixelType type) {
    return {data.data(),type,false,false,{data.size()},{std::ptrdiff_t(sizeof(T))}};
}
template<class T> void load_layouts(PixelType type,bool single) {
    for(const auto n:{std::size_t(4),std::size_t(5),std::size_t(7),std::size_t(8),std::size_t(9),std::size_t(127),std::size_t(128),std::size_t(129),std::size_t(257)})
    for(const int step:{1,-1,3,-3})for(const bool swap:{false,true}) {
        const auto stride=std::size_t(std::abs(step))*sizeof(T),bytes=(n-1)*stride+sizeof(T)+1;
        // Exactly one leading unaligned byte, no readable trailing padding.
        // ASan therefore catches widened narrow loads past the final lane.
        auto storage=std::make_unique<unsigned char[]>(bytes);
        std::memset(storage.get(),0xcd,bytes);
        std::vector<T> logical(n),zero(n,T(0));
        for(std::size_t i=0;i<n;++i) {
            if constexpr(std::is_floating_point_v<T>)logical[i]=T(double(int(i%37)-18)*.0625);
            else {using U=std::make_unsigned_t<T>;const U bits=U(std::uint64_t(0xd1b54a32d192ed03ULL)*(i+1));std::memcpy(&logical[i],&bits,sizeof(T));}
            const auto index=step>0 ? i:n-1-i;
            unsigned char encoded[sizeof(T)];std::memcpy(encoded,&logical[i],sizeof(T));
            for(std::size_t j=0;j<sizeof(T);++j)storage[1+index*stride+j]=encoded[swap ? sizeof(T)-1-j:j];
        }
        const std::vector<unsigned char> before(storage.get(),storage.get()+bytes);
        MetricImageView input{storage.get()+1+(step<0 ? (n-1)*stride:0),type,false,swap,{n},{std::ptrdiff_t(step)*std::ptrdiff_t(sizeof(T))}};
        for(auto mode:{MetricNormalization::none,MetricNormalization::euclidean,MetricNormalization::mean,MetricNormalization::min_max}) {
            const auto expected=metric_sums(view(logical,type),view(zero,type),single,mode,true);
            const auto actual=metric_sums(input,view(zero,type),single,mode);
            require(!std::memcmp(&actual.error,&expected.error,sizeof(double)));
            require(!std::memcmp(&actual.energy,&expected.energy,sizeof(double)));
            require(!std::memcmp(&actual.mean,&expected.mean,sizeof(double)));
            require(actual.low==expected.low && actual.high==expected.high && actual.valid==expected.valid && actual.nan==expected.nan);
        }
        require(!std::memcmp(storage.get(),before.data(),bytes));
    }
}
template<class T> void workers(PixelType type,bool single) {
    std::vector<T> a(257*1025),b(a.size());
    for(std::size_t i=0;i<a.size();++i) {
        const auto word=std::uint64_t{0x9e3779b97f4a7c15ULL}*(i+1),other=std::uint64_t{0xd1b54a32d192ed03ULL}*(i+3);
        if constexpr(std::is_floating_point_v<T>) {
            a[i]=T(double(word%1000003)/1000003.-.5);b[i]=T(double(other%1000033)/1000033.-.5);
        } else {
            using U=std::make_unsigned_t<T>;
            const U first=U(word),second=U(other);std::memcpy(&a[i],&first,sizeof(T));std::memcpy(&b[i],&second,sizeof(T));
        }
    }
    for(auto mode:{MetricNormalization::none,MetricNormalization::euclidean,MetricNormalization::mean,MetricNormalization::min_max}) {
        set_num_threads(1);const auto one=metric_sums(view(a,type),view(b,type),single,mode);
        #ifdef _OPENMP
        set_num_threads(4);
        #endif
        const auto four=metric_sums(view(a,type),view(b,type),single,mode);
        require(!std::memcmp(&one.error,&four.error,sizeof(double)));
        require(!std::memcmp(&one.energy,&four.energy,sizeof(double)));
        require(!std::memcmp(&one.mean,&four.mean,sizeof(double)));
        require(one.low==four.low && one.high==four.high && one.valid==four.valid && one.nan==four.nan);
    }
    // Materialized exceptional stages share the same global workers and
    // preserve their arrays/status independently of team size.
    using Work=std::conditional_t<std::is_same_v<T,float>,float,double>;
    std::vector<Work> first(a.size()),second(a.size());
    for(int operation=0;operation<4;++operation) {
        set_num_threads(1);
        const auto one=metric_stage_into(view(a,type),view(b,type),view(first,single ? PixelType::float32:PixelType::float64),operation,single);
#ifdef _OPENMP
        set_num_threads(4);
#endif
        const auto four=metric_stage_into(view(a,type),view(b,type),view(second,single ? PixelType::float32:PixelType::float64),operation,single);
        require(one==four && !std::memcmp(first.data(),second.data(),first.size()*sizeof(Work)));
    }
    set_num_threads(1);
}
template<class T> void reverse_rows(PixelType type,bool single) {
    for(const std::size_t width:{1u,3u,4u,5u,7u,8u,9u,31u,127u,128u,129u,513u}) {
        const auto n=3*width,bytes=n*sizeof(T);
        // One unaligned leading byte and no trailing readable padding.
        auto storage=std::make_unique<unsigned char[]>(bytes+1);
        std::vector<T> logical(n),other(n),actual(n),expected(n);
        for(std::size_t i=0;i<n;++i) {
            logical[i]=T(std::ldexp(double(int(i%67)-33)+.125,int(i%13)-6));
            other[i]=T(std::ldexp(double(int(i%31)-15)+.375,int(i%11)-5));
            std::memcpy(storage.get()+1+(n-1-i)*sizeof(T),&logical[i],sizeof(T));
        }
        MetricImageView input{storage.get()+1+(n-1)*sizeof(T),type,false,false,{3,width},
                              {-std::ptrdiff_t(width*sizeof(T)),-std::ptrdiff_t(sizeof(T))}};
        auto contiguous=view(logical,type),second=view(other,type);
        contiguous.shape=second.shape={3,width};contiguous.strides=second.strides={std::ptrdiff_t(width*sizeof(T)),std::ptrdiff_t(sizeof(T))};
        for(const auto block:{0u,32u,8192u})for(auto mode:{MetricNormalization::none,MetricNormalization::euclidean,MetricNormalization::mean,MetricNormalization::min_max}) {
            const auto wanted=metric_sums(contiguous,second,single,mode,true,false,block);
            const auto got=metric_sums(input,second,single,mode,false,false,block);
            require(!std::memcmp(&got.error,&wanted.error,8) && !std::memcmp(&got.energy,&wanted.energy,8) && !std::memcmp(&got.mean,&wanted.mean,8));
            require(!std::memcmp(&got.low,&wanted.low,8) && !std::memcmp(&got.high,&wanted.high,8));
        }
        auto out=view(actual,type),reference_out=view(expected,type);
        out.shape=reference_out.shape={3,width};out.strides=reference_out.strides=contiguous.strides;
        for(int operation=0;operation<4;++operation) {
            const auto flags=metric_stage_into(input,second,out,operation,single);
            const auto wanted_flags=metric_stage_into(contiguous,second,reference_out,operation,single);
            require(flags==wanted_flags && !std::memcmp(actual.data(),expected.data(),bytes));
        }
        for(std::size_t i=0;i<n;++i)require(!std::memcmp(storage.get()+1+(n-1-i)*sizeof(T),&logical[i],sizeof(T)));
    }
}
void half_load_layouts() {
    std::vector<std::uint16_t> encodings;
    for(unsigned word=0;word<65536;++word)if((word&0x7c00)!=0x7c00)encodings.push_back(std::uint16_t(word));
    for(const auto n:{std::size_t(4),std::size_t(5),std::size_t(7),std::size_t(8),std::size_t(9),std::size_t(127),std::size_t(128),std::size_t(129),std::size_t(257),encodings.size()}) {
        std::vector<std::uint16_t> logical(encodings.begin(),encodings.begin()+n),zero(n,0);
        auto first=view(logical,PixelType::uint16),second=view(zero,PixelType::uint16);first.half=second.half=true;
        for(auto mode:{MetricNormalization::none,MetricNormalization::euclidean,MetricNormalization::mean,MetricNormalization::min_max}) {
            const auto expected=metric_sums(first,second,true,mode,true);
            for(const int step:{1,-1,2,-2,3,-3})for(const bool swap:{false,true}) {
                const auto stride=std::size_t(std::abs(step))*2,bytes=(n-1)*stride+3;
                auto storage=std::make_unique<unsigned char[]>(bytes);std::memset(storage.get(),0xcd,bytes);
                for(std::size_t i=0;i<n;++i) {
                    unsigned char encoded[2];std::memcpy(encoded,&logical[i],2);
                    const auto index=step>0 ? i:n-1-i;
                    storage[1+index*stride]=encoded[swap ? 1:0];storage[2+index*stride]=encoded[swap ? 0:1];
                }
                const std::vector<unsigned char> before(storage.get(),storage.get()+bytes);
                auto input=first;input.data=storage.get()+1+(step<0 ? (n-1)*stride:0);input.swap=swap;input.strides[0]=step*2;
                const auto actual=metric_sums(input,second,true,mode);
                require(!std::memcmp(&expected.error,&actual.error,8) && !std::memcmp(&expected.energy,&actual.energy,8) && !std::memcmp(&expected.mean,&actual.mean,8));
                require(expected.low==actual.low && expected.high==actual.high && expected.valid==actual.valid && expected.nan==actual.nan);
                require(!std::memcmp(storage.get(),before.data(),bytes));
            }
        }
    }
}

void swapped_raw_range_contract() {
    std::fenv_t saved;require(std::fegetenv(&saved)==0);
#ifdef METRIC_TEST_MXCSR
    const auto saved_csr=_mm_getcsr();
#endif
    std::size_t checks=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())for(std::size_t count=0;count<8;++count) {
        MetricRawRange result{1,2,true};
        // The explicit small-count contract makes null storage valid here:
        // these calls must return without attempting any read.
        require(!simd::metric_swapped_f32_range(nullptr,count,result));
        require(result.low==1 && result.high==2 && result.has_nan);
        ++checks;
    }
#endif
    for(const int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO})
    for(const unsigned denormals:{0u,1u}) {
        require(std::fesetround(rounding)==0);
#ifdef METRIC_TEST_MXCSR
        _mm_setcsr((_mm_getcsr()&~0x8040u)|(denormals ? 0x8040u:0u));
#else
        if(denormals)continue;
#endif
        for(const std::size_t width:{1u,2u,3u,4u,5u,6u,7u,8u,9u,10u,15u,16u,17u,31u,32u,33u,127u,128u,129u,513u})
        for(unsigned pattern=0;pattern<10;++pattern)
        for(unsigned alignment=0;alignment<2;++alignment) {
            const auto n=3*width,bytes=n*4+alignment,control_bytes=(n-1)*8+4+alignment;
            // Exact allocation ends expose any vector or scalar tail overread
            // under ASan. The sparse control uses the unchanged scalar path.
            auto storage=std::make_unique<unsigned char[]>(bytes),control=std::make_unique<unsigned char[]>(control_bytes);
            std::memset(storage.get(),0xcd,bytes);std::memset(control.get(),0xcd,control_bytes);
            for(std::size_t i=0;i<n;++i) {
                std::uint32_t bits=(std::uint32_t(i%254)+1u)<<23;
                bits|=std::uint32_t(i*48271u)&0x007fffffu;
                if(i%3==0)bits|=0x80000000u;
                if(pattern==9)bits=0xbf800000u;
                if(pattern==8)bits=(i&1 ? 0x80000000u:0u)|std::uint32_t(i%0x007fffffu+1);
                if(pattern==7)bits=i&1 ? 0x80000000u:0u;
                if(pattern==6)bits=i&1 ? 0xffc00017u:0x7f800001u;
                if((i==0 || i==n/2 || i==n-1) && pattern>=1 && pattern<=5) {
                    const std::uint32_t special[]={0,0,0x00000001u,0x7f800000u,0x7fc00011u,0x7f800003u};
                    bits=special[pattern]|(i&1 ? 0x80000000u:0u);
                }
                unsigned char native[4];std::memcpy(native,&bits,4);
                for(std::size_t j=0;j<4;++j)storage[alignment+4*i+j]=control[alignment+8*i+j]=native[3-j];
            }
            const std::vector<unsigned char> before(storage.get(),storage.get()+bytes);
            for(unsigned layout=0;layout<5;++layout) {
                MetricImageView dense{storage.get()+alignment,PixelType::float32,false,true,{3,width},{std::ptrdiff_t(4*width),4}};
                if(layout==1)dense.strides={4,12};
                if(layout==2){dense.data=storage.get()+alignment+4*(n-1);dense.strides={-std::ptrdiff_t(4*width),-4};}
                if(layout==3){dense.data=storage.get()+alignment+4*(width-1);dense.strides={std::ptrdiff_t(4*width),-4};}
                if(layout==4){dense.shape={4,3,width};dense.strides={0,4,12};}
                auto sparse=dense;
                sparse.data=control.get()+alignment+2*(static_cast<const unsigned char*>(dense.data)-(storage.get()+alignment));
                for(auto& stride:sparse.strides)stride*=2;
                std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
#ifdef METRIC_TEST_MXCSR
                _mm_setcsr(_mm_getcsr()&~0x02u); // Clear nonstandard denormal status identically.
#endif
                const auto actual=metric_raw_range(dense);const int actual_flags=std::fetestexcept(FE_ALL_EXCEPT);
#ifdef METRIC_TEST_MXCSR
                const auto actual_csr=_mm_getcsr();
#endif
                std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
#ifdef METRIC_TEST_MXCSR
                _mm_setcsr(_mm_getcsr()&~0x02u);
#endif
                const auto expected=metric_raw_range(sparse);const int expected_flags=std::fetestexcept(FE_ALL_EXCEPT);
                require(actual.low==expected.low && actual.high==expected.high && actual.has_nan==expected.has_nan);
                require(actual_flags==expected_flags && std::fegetround()==rounding);
#ifdef METRIC_TEST_MXCSR
                require(actual_csr==_mm_getcsr());
#endif
                ++checks;
            }
#ifdef SKIMAGEXX_AVX2
            if(has_avx2()) {
                MetricRawRange result{0x0123456789abcdefULL,0xfedcba9876543210ULL,true};
                std::feclearexcept(FE_ALL_EXCEPT);std::feraiseexcept(FE_DIVBYZERO);
                const int flags=std::fetestexcept(FE_ALL_EXCEPT);
#ifdef METRIC_TEST_MXCSR
                const auto csr=_mm_getcsr();
#endif
                const bool admitted=simd::metric_swapped_f32_range(storage.get()+alignment,n,result);
                require(admitted==(n>=8 && (pattern==0 || pattern==9)));
                if(!admitted)require(result.low==0x0123456789abcdefULL && result.high==0xfedcba9876543210ULL && result.has_nan);
                else {
                    const MetricImageView sparse{control.get()+alignment,PixelType::float32,false,true,{n},{8}};
                    const auto expected=metric_raw_range(sparse);
                    require(result.low==expected.low && result.high==expected.high && result.has_nan==expected.has_nan);
                }
                require(flags==std::fetestexcept(FE_ALL_EXCEPT));
#ifdef METRIC_TEST_MXCSR
                require(csr==_mm_getcsr());
#endif
                ++checks;
            }
#endif
            require(!std::memcmp(storage.get(),before.data(),bytes));
        }
    }
    require(std::fesetenv(&saved)==0);
#ifdef METRIC_TEST_MXCSR
    _mm_setcsr(saved_csr);
#endif
    std::cout<<"Swapped float32 exact raw-range/FEnv checks passed: "<<checks<<"\n";
}
}
void leaf_order_regression() {
    for(const auto n:{1u,3u,4u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,255u,256u,257u,8193u}) {
        std::vector<double> a(n),b(n);
        for(unsigned i=0;i<n;++i) {
            a[i]=std::ldexp(double(int((i*7919u)%2003u)-1001)+.125,int(i%31)-15);
            b[i]=std::ldexp(double(int((i*541u)%997u)-498)+.375,int(i%23)-11);
        }
        for(const auto block:{0u,32u,8192u})for(const auto mode:{MetricNormalization::none,MetricNormalization::euclidean,MetricNormalization::mean,MetricNormalization::min_max}) {
            const auto expected=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,mode,true,false,block);
            const auto actual=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,mode,false,false,block);
            require(!std::memcmp(&actual.error,&expected.error,8) && !std::memcmp(&actual.energy,&expected.energy,8) && !std::memcmp(&actual.mean,&expected.mean,8));
            require(!std::memcmp(&actual.low,&expected.low,8) && !std::memcmp(&actual.high,&expected.high,8));
            require(actual.nonfinite==expected.nonfinite && actual.nan==expected.nan && actual.valid==expected.valid);
        }
    }
}
int main() {
    swapped_raw_range_contract();
    leaf_order_regression();
    reverse_rows<float>(PixelType::float32,true);reverse_rows<double>(PixelType::float64,false);
    require(get_num_threads()==1);
    const std::vector<double> a{1.,2.,3.,4.},b(4,0.);
    const auto error=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,MetricNormalization::none);
    require(error.error==30.);
    const auto energy=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,MetricNormalization::euclidean);
    require(energy.error==30. && energy.energy==30.);
    const auto mean=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,MetricNormalization::mean);
    require(mean.error==30. && mean.mean==10.);
    const auto range=metric_sums(view(a,PixelType::float64),view(b,PixelType::float64),false,MetricNormalization::min_max);
    require(range.low==1. && range.high==4.);
    const std::vector<std::uint64_t> wide{std::numeric_limits<std::uint64_t>::max(),(std::uint64_t{1}<<63)+1};
    const auto bounds=metric_raw_range(view(wide,PixelType::uint64));
    require(bounds.low==(std::uint64_t{1}<<63)+1 && bounds.high==std::numeric_limits<std::uint64_t>::max());
    const std::vector<std::uint16_t> half{0x3c00,0x4000,0x4200,0x4400},half_zero(4,0);
    auto half_a=view(half,PixelType::uint16),half_b=view(half_zero,PixelType::uint16);half_a.half=half_b.half=true;
    const auto half_result=metric_sums(half_a,half_b,true,MetricNormalization::euclidean);
    require(half_result.error==30. && half_result.energy==30.);
    half_a.data=half.data()+3;half_a.strides[0]=-2;
    require(metric_sums(half_a,half_b,true,MetricNormalization::none).error==30.);
    const std::vector<std::uint16_t> half_endian{0x003c,0x0040,0x0042,0x0044};
    half_a=view(half_endian,PixelType::uint16);half_a.half=half_a.swap=true;
    require(metric_sums(half_a,half_b,true,MetricNormalization::none).error==30.);
    const std::vector<std::uint16_t> half_small{1,1023,0x8001,0x83ff};
    half_a=view(half_small,PixelType::uint16);half_a.half=true;
    const float small=0x1p-24f,large=1023*0x1p-24f;
    require(metric_sums(half_a,half_b,true,MetricNormalization::none).error==2*double(float(small*small))+2*double(float(large*large)));
    const std::vector<std::uint8_t> boolean{0,2,128,255},bool_zero(4,0);
    require(metric_sums(view(boolean,PixelType::boolean),view(bool_zero,PixelType::boolean),false,MetricNormalization::none).error==3.);
    const float scalar_a=.3f,scalar_b=.7f,delta=float(scalar_a-scalar_b);
    const MetricImageView scalar_first{&scalar_a,PixelType::float32,false,false,{},{}},scalar_second{&scalar_b,PixelType::float32,false,false,{},{}};
    require(metric_sums(scalar_first,scalar_second,true,MetricNormalization::none,false,true).error==double(delta)*double(delta));
    const std::vector<float> negative_zero(37,-0.f),ones(37,1.f);
    require(!std::signbit(metric_sums(view(negative_zero,PixelType::float32),view(ones,PixelType::float32),true,MetricNormalization::mean).mean));
    std::vector<float> stage_a{std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity()},
                       stage_b{-std::numeric_limits<float>::max(),std::numeric_limits<float>::infinity()},delta_values(2),square_values(2);
    require(metric_stage_into(view(stage_a,PixelType::float32),view(stage_b,PixelType::float32),view(delta_values,PixelType::float32),1,true)==5);
    stage_a.assign(2,0);stage_b.assign(2,0);
    require(metric_stage_into(view(delta_values,PixelType::float32),view(delta_values,PixelType::float32),view(square_values,PixelType::float32),2,true)==0);
    require(std::isinf(square_values[0]) && std::isnan(square_values[1]));
    std::vector<float> finite_half;
    for(unsigned h=65536;h-->0;)if((h&0x7c00)!=0x7c00) {
        const auto word=std::uint16_t(h);const MetricImageView one_half{&word,PixelType::uint16,true,false,{1},{2}};
        finite_half.push_back(float(detail::metric_read(one_half,0)));
    }
    require(metric_sums(view(finite_half,PixelType::float32),view(finite_half,PixelType::float32),true,MetricNormalization::mean,false,false,8192).mean==3.9375);
    half_load_layouts();
    workers<float>(PixelType::float32,true);workers<double>(PixelType::float64,false);
    workers<std::uint8_t>(PixelType::uint8,false);workers<std::int8_t>(PixelType::int8,false);
    workers<std::uint16_t>(PixelType::uint16,false);workers<std::int16_t>(PixelType::int16,false);
    workers<std::uint32_t>(PixelType::uint32,false);workers<std::int32_t>(PixelType::int32,false);
    workers<std::uint64_t>(PixelType::uint64,false);workers<std::int64_t>(PixelType::int64,false);
    load_layouts<float>(PixelType::float32,true);load_layouts<double>(PixelType::float64,false);
    load_layouts<std::uint8_t>(PixelType::boolean,false);
    load_layouts<std::uint8_t>(PixelType::uint8,false);load_layouts<std::int8_t>(PixelType::int8,false);
    load_layouts<std::uint16_t>(PixelType::uint16,false);load_layouts<std::int16_t>(PixelType::int16,false);
    load_layouts<std::uint32_t>(PixelType::uint32,false);load_layouts<std::int32_t>(PixelType::int32,false);
    load_layouts<std::uint64_t>(PixelType::uint64,false);load_layouts<std::int64_t>(PixelType::int64,false);
    std::cout<<"Metric known values, full-width raw bounds and exact worker tree checks passed\n";
}
