#include "skimagexx/metrics.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
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
}
int main() {
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
