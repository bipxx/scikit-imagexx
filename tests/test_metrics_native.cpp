#include "skimagexx/metrics.hpp"
#include "skimagexx/image.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
using namespace skimagexx;
namespace {
void require(bool value){if(!value)throw std::runtime_error("metric native check failed");}
template<class T> MetricImageView view(const std::vector<T>& data,PixelType type) {
    return {data.data(),type,false,false,{data.size()},{std::ptrdiff_t(sizeof(T))}};
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
    set_num_threads(1);
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
    workers<float>(PixelType::float32,true);workers<double>(PixelType::float64,false);
    workers<std::uint8_t>(PixelType::uint8,false);workers<std::int8_t>(PixelType::int8,false);
    workers<std::uint16_t>(PixelType::uint16,false);workers<std::int16_t>(PixelType::int16,false);
    workers<std::uint32_t>(PixelType::uint32,false);workers<std::int32_t>(PixelType::int32,false);
    workers<std::uint64_t>(PixelType::uint64,false);workers<std::int64_t>(PixelType::int64,false);
    std::cout<<"Metric known values, full-width raw bounds and exact worker tree checks passed\n";
}
