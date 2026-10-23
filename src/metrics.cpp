#include "skimagexx/metrics.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx {
namespace {
template<class T> T read(const unsigned char* data,bool swap) {
    T result;
    if(swap && sizeof(T)>1) {unsigned char bytes[sizeof(T)];for(std::size_t i=0;i<sizeof(T);++i)bytes[i]=data[sizeof(T)-1-i];std::memcpy(&result,bytes,sizeof(T));}
    else std::memcpy(&result,data,sizeof(T));
    return result;
}
template<class T> T work_read(const MetricImageView& view,std::ptrdiff_t offset) {
    if constexpr(std::is_same_v<T,float>) {
        const auto* data=static_cast<const unsigned char*>(view.data)+offset;
        if(view.half)return detail::decode_half(read<std::uint16_t>(data,view.swap));
        if(view.type==PixelType::float32)return read<float>(data,view.swap);
    }
    return T(detail::metric_read(view,offset));
}
template<class T> bool signaling(T value) {
    using U=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
    U bits;std::memcpy(&bits,&value,sizeof(T));
    constexpr U fraction=[](){if constexpr(sizeof(T)==4)return U(0x007fffff);else return U(0x000fffffffffffffULL);}();
    constexpr U exponent=[](){if constexpr(sizeof(T)==4)return U(0x7f800000);else return U(0x7ff0000000000000ULL);}();
    constexpr U quiet=[](){if constexpr(sizeof(T)==4)return U(0x00400000);else return U(0x0008000000000000ULL);}();
    return (bits&exponent)==exponent && (bits&fraction)!=0 && (bits&quiet)==0;
}
std::size_t count(const MetricImageView& view) {
    const auto bytes=pixel_size(view.type);
    if(view.half && view.type!=PixelType::uint16)throw std::invalid_argument("half metric storage must be uint16");
    if(view.shape.size()!=view.strides.size())throw std::invalid_argument("one metric stride per axis required");
    if(std::find(view.shape.begin(),view.shape.end(),0)!=view.shape.end())return 0;
    std::size_t n=1;for(auto length:view.shape) {if(length>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes/n)throw std::overflow_error("metric dimensions overflow");n*=length;}
    if(n && !view.data)throw std::invalid_argument("nonnull metric input required");
    return n;
}
std::ptrdiff_t offset(const MetricImageView& view,std::size_t index) {
    std::ptrdiff_t result=0;
    for(std::size_t axis=view.shape.size();axis-->0;){result+=std::ptrdiff_t(index%view.shape[axis])*view.strides[axis];index/=view.shape[axis];}
    return result;
}
unsigned status() {const auto value=std::fetestexcept(FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);return (value&FE_OVERFLOW ? 1u:0u)|(value&FE_UNDERFLOW ? 2u:0u)|(value&FE_INVALID ? 4u:0u);}
template<class T,class F> T pairwise(std::size_t n,const F& get) {
    if(n<8){T sum=-T(0);for(std::size_t i=0;i<n;++i)sum=T(sum+T(get(i)));return sum;}
    T lanes[8];for(std::size_t i=0;i<8;++i)lanes[i]=T(get(i));std::size_t i=8;
    for(;i+8<=n;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=T(lanes[j]+T(get(i+j)));
    T sum=T(T(T(lanes[0]+lanes[1])+T(lanes[2]+lanes[3]))+T(T(lanes[4]+lanes[5])+T(lanes[6]+lanes[7])));
    for(;i<n;++i)sum=T(sum+T(get(i)));
    return sum;
}
template<class T> MetricSums merge(const MetricSums& a,const MetricSums& b,bool flags) {
    MetricSums result=a;
    if(flags)std::feclearexcept(FE_ALL_EXCEPT);
    result.error=a.error+b.error;
    result.error_reduce_flags|=b.error_reduce_flags;if(flags)result.error_reduce_flags|=status();
    if(flags)std::feclearexcept(FE_ALL_EXCEPT);
    result.energy=a.energy+b.energy;
    result.energy_reduce_flags|=b.energy_reduce_flags;if(flags)result.energy_reduce_flags|=status();
    if(flags)std::feclearexcept(FE_ALL_EXCEPT);
    result.mean=double(T(T(a.mean)+T(b.mean)));
    result.mean_reduce_flags|=b.mean_reduce_flags;if(flags)result.mean_reduce_flags|=status();
    result.subtract_flags|=b.subtract_flags;result.square_flags|=b.square_flags;result.energy_flags|=b.energy_flags;
    result.cast0_flags|=b.cast0_flags;result.cast1_flags|=b.cast1_flags;
    result.nonfinite|=b.nonfinite;result.nan|=b.nan;result.range_signaling|=b.range_signaling;
    if(!result.valid && b.valid){result.low=b.low;result.high=b.high;result.valid=true;}
    else if(b.valid){if(b.low<result.low)result.low=b.low;if(b.high>result.high)result.high=b.high;}
    return result;
}
template<class T> MetricSums leaf(const MetricImageView& a,const MetricImageView& b,std::size_t begin,std::size_t n,MetricNormalization mode,bool flags) {
    alignas(32) double error[128],energy[128],values[128];T precise[128];MetricSums result;
    const bool need_energy=mode==MetricNormalization::euclidean,need_mean=mode==MetricNormalization::mean,need_range=mode==MetricNormalization::min_max;
    if(!flags) {
        std::size_t done=0;
        while(done<n) {
            const auto index=begin+done;
            const auto width=a.shape.empty() ? std::size_t{1}:a.shape.back();
            const auto span=std::min(n-done,width-index%width);
            const auto first=offset(a,index),second=offset(b,index);
            const auto step0=a.shape.empty() ? 0:a.strides.back(),step1=b.shape.empty() ? 0:b.strides.back();
            std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())i=simd::metric_values(a,b,first,second,step0,step1,span,std::is_same_v<T,float>,mode,error+done,energy+done,values+done);
#endif
            for(;i<span;++i) {
                const T x=work_read<T>(a,first+std::ptrdiff_t(i)*step0),y=work_read<T>(b,second+std::ptrdiff_t(i)*step1);
                const T delta=T(x-y);error[done+i]=double(T(delta*delta));
                if(need_energy)energy[done+i]=double(T(x*x));
                values[done+i]=double(x);
            }
            done+=span;
        }
    } else {
        T second[128],difference[128];
        std::feclearexcept(FE_ALL_EXCEPT);
        for(std::size_t i=0;i<n;++i)precise[i]=work_read<T>(a,offset(a,begin+i));
        if constexpr(std::is_same_v<T,double>)result.cast0_flags=status();
        std::feclearexcept(FE_ALL_EXCEPT);
        for(std::size_t i=0;i<n;++i)second[i]=work_read<T>(b,offset(b,begin+i));
        if constexpr(std::is_same_v<T,double>)result.cast1_flags=status();
        for(std::size_t i=0;i<n;++i){values[i]=double(precise[i]);if(need_range)result.range_signaling|=signaling(precise[i]);}
        std::feclearexcept(FE_ALL_EXCEPT);
        for(std::size_t i=0;i<n;++i)difference[i]=T(precise[i]-second[i]);
        result.subtract_flags=status();std::feclearexcept(FE_ALL_EXCEPT);
        for(std::size_t i=0;i<n;++i)error[i]=double(T(difference[i]*difference[i]));
        result.square_flags=status();
        if(need_energy) {
            std::feclearexcept(FE_ALL_EXCEPT);for(std::size_t i=0;i<n;++i)energy[i]=double(T(precise[i]*precise[i]));result.energy_flags=status();
        }
    }
    for(std::size_t i=0;i<n;++i) {
        const auto value=values[i];result.nonfinite|=!std::isfinite(error[i]) || !std::isfinite(value);
        if(need_energy)result.nonfinite|=!std::isfinite(energy[i]);
        if(need_range) {
            if(std::isnan(value)){result.nan=true;continue;}
            if(!result.valid){result.low=result.high=value;result.valid=true;}
            else {if(value<result.low)result.low=value;if(value>result.high)result.high=value;}
        }
    }
    if(flags)std::feclearexcept(FE_ALL_EXCEPT);
    result.error=pairwise<double>(n,[&](auto i){return error[i];});if(flags)result.error_reduce_flags|=status();
    if(need_energy) {if(flags)std::feclearexcept(FE_ALL_EXCEPT);result.energy=pairwise<double>(n,[&](auto i){return energy[i];});if(flags)result.energy_reduce_flags|=status();}
    if(need_mean) {if(flags)std::feclearexcept(FE_ALL_EXCEPT);result.mean=double(pairwise<T>(n,[&](auto i){return flags ? precise[i]:T(values[i]);}));if(flags)result.mean_reduce_flags|=status();}
    return result;
}
template<class T> MetricSums tree(const MetricImageView& a,const MetricImageView& b,std::size_t begin,std::size_t n,MetricNormalization mode,bool flags) {
    if(n<=128)return leaf<T>(a,b,begin,n,mode,flags);
    auto half=n/2;half-=half%8;
    return merge<T>(tree<T>(a,b,begin,half,mode,flags),tree<T>(a,b,begin+half,n-half,mode,flags),flags);
}
template<class T> MetricSums run(const MetricImageView& a,const MetricImageView& b,std::size_t n,MetricNormalization mode,bool flags) {
    const auto workers=detail::active_workers(n);
    if(workers==1)return tree<T>(a,b,0,n,mode,flags);
    struct Node {std::size_t begin,n,left=0,right=0;MetricSums value;};
    std::vector<Node> nodes;std::vector<std::size_t> leaves;
    auto build=[&](auto&& self,std::size_t begin,std::size_t length,int budget)->std::size_t {
        const auto id=nodes.size();nodes.push_back({begin,length,0,0,{}});
        if(budget>1 && length>=8192) {
            auto half=length/2;half-=half%8;
            const auto left=self(self,begin,half,budget/2),right=self(self,begin+half,length-half,budget-budget/2);
            nodes[id].left=left;nodes[id].right=right;
        } else leaves.push_back(id);
        return id;
    };
    build(build,0,n,workers);
    detail::partition(leaves.size(),std::min(workers,int(leaves.size())),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){auto& node=nodes[leaves[i]];node.value=tree<T>(a,b,node.begin,node.n,mode,flags);}});
    for(std::size_t i=nodes.size();i-->0;)if(nodes[i].left)nodes[i].value=merge<T>(nodes[nodes[i].left].value,nodes[nodes[i].right].value,flags);
    return nodes[0].value;
}
template<class T,bool Boolean=false,bool Half=false> MetricRawRange raw_range(const MetricImageView& view,std::size_t n) {
    T low{},high{};double low_half=0,high_half=0;bool valid=false;MetricRawRange result;
    for(std::size_t i=0;i<n;++i) {
        T value=read<T>(static_cast<const unsigned char*>(view.data)+offset(view,i),view.swap);
        if constexpr(Boolean)value=T(value!=0);
        if constexpr(Half) {
            const double decoded=detail::decode_half(value);if(std::isnan(decoded)){result.has_nan=true;continue;}
            if(!valid){low=high=value;low_half=high_half=decoded;valid=true;}
            else {if(decoded<low_half){low=value;low_half=decoded;}if(decoded>high_half){high=value;high_half=decoded;}}
        } else {
            if constexpr(std::is_floating_point_v<T>)if(std::isnan(value)){result.has_nan=true;continue;}
            if(!valid){low=high=value;valid=true;}else {if(value<low)low=value;if(value>high)high=value;}
        }
    }
    std::memcpy(&result.low,&low,sizeof(T));std::memcpy(&result.high,&high,sizeof(T));return result;
}
}
namespace detail {
double metric_read(const MetricImageView& view,std::ptrdiff_t byte_offset) {
    const auto* source=static_cast<const unsigned char*>(view.data)+byte_offset;
    if(view.half) {
        const auto half=read<std::uint16_t>(source,view.swap);
        if((half&0x7c00)==0x7c00 && (half&0x03ff)) {
            const std::uint64_t bits=(std::uint64_t(half&0x8000)<<48)|0x7ff0000000000000ULL|(std::uint64_t(half&0x03ff)<<42);
            double result;std::memcpy(&result,&bits,8);return result;
        }
        return decode_half(half);
    }
    switch(view.type) {
        case PixelType::boolean:return read<std::uint8_t>(source,false)!=0 ? 1.:0.;
        #define READ(tag,T) case PixelType::tag:return double(read<T>(source,view.swap));
        READ(uint8,std::uint8_t) READ(int8,std::int8_t) READ(uint16,std::uint16_t) READ(int16,std::int16_t)
        READ(uint32,std::uint32_t) READ(int32,std::int32_t) READ(uint64,std::uint64_t) READ(int64,std::int64_t)
        READ(float32,float) READ(float64,double)
        #undef READ
        default:throw std::invalid_argument("unsupported metric type");
    }
}
}
MetricSums metric_sums(const MetricImageView& a,const MetricImageView& b,bool single,MetricNormalization mode,bool capture_underflow,bool scalar_power_double) {
    const auto n=count(a);count(b);if(a.shape!=b.shape)throw std::invalid_argument("metric shapes must match");
    if(int(mode)<0 || int(mode)>3)throw std::invalid_argument("invalid metric normalization");
    if(!n)return {};
    auto result=single ? run<float>(a,b,n,mode,false):run<double>(a,b,n,mode,false);
    if(capture_underflow || result.nonfinite || !std::isfinite(result.error) || !std::isfinite(result.energy) || !std::isfinite(result.mean))
        result=single ? run<float>(a,b,n,mode,true):run<double>(a,b,n,mode,true);
    if(scalar_power_double && single && a.shape.empty()) {
        // NumPy1.x promotes a 0-D float32 scalar ** Python-int operation to
        // double after the float32 subtraction. Array products and the
        // Euclidean denominator's multiplication retain float32 precision.
        const float x=work_read<float>(a,0),y=work_read<float>(b,0),delta=float(x-y);
        result.error=double(delta)*double(delta);result.square_flags=0;result.error_reduce_flags=0;
    }
    if(mode==MetricNormalization::mean && result.mean==0) {
        // NumPy's outer reduction starts with +0, even when every tree leaf
        // is -0. Keep exact scalar denominator signs on all compilers.
        const std::uint64_t zero=0;std::memcpy(&result.mean,&zero,sizeof(double));
    }
    return result;
}
MetricRawRange metric_raw_range(const MetricImageView& view) {
    const auto n=count(view);if(!n)throw std::invalid_argument("empty metric range");
    if(view.half)return raw_range<std::uint16_t,false,true>(view,n);
    switch(view.type) {
        case PixelType::boolean:return raw_range<std::uint8_t,true>(view,n);
        #define RANGE(tag,T) case PixelType::tag:return raw_range<T>(view,n);
        RANGE(uint8,std::uint8_t) RANGE(int8,std::int8_t) RANGE(uint16,std::uint16_t) RANGE(int16,std::int16_t)
        RANGE(uint32,std::uint32_t) RANGE(int32,std::int32_t) RANGE(uint64,std::uint64_t) RANGE(int64,std::int64_t)
        RANGE(float32,float) RANGE(float64,double)
        #undef RANGE
        default:throw std::invalid_argument("unsupported metric range type");
    }
}
}
