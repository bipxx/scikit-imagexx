#include "skimagexx/li.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include "li_detail.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace skimagexx {
namespace {
template<class T> bool constant_input(const T* input,std::size_t n) {
#ifdef SKIMAGEXX_AVX2
    if(has_avx2()) {
        if constexpr(std::is_same_v<T,std::uint8_t>)return simd::uniform(input,n);
        else return simd::li_uniform(input,n,std::is_same_v<T,float>);
    }
#endif
    return std::all_of(input,input+n,[first=input[0]](T value){return value==first;});
}
template<class T> T pairwise(const T* values,std::size_t n) {
    if(n<=128) {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())return T(simd::li_sum(values,n,std::is_same_v<T,float>));
#endif
        if(n<8){T sum=-T(0);for(std::size_t i=0;i<n;++i)sum=T(sum+values[i]);return sum;}
        T r[8];std::copy_n(values,8,r);std::size_t i=8;
        for(;i+8<=n;i+=8)for(int j=0;j<8;++j)r[j]=T(r[j]+values[i+std::size_t(j)]);
        T sum=T(T(T(r[0]+r[1])+T(r[2]+r[3]))+T(T(r[4]+r[5])+T(r[6]+r[7])));
        for(;i<n;++i)sum=T(sum+values[i]);return sum;
    }
    auto half=n/2;half-=half%8;
    return T(pairwise(values,half)+pairwise(values+half,n-half));
}
template<class T> T sum(const T* values,std::size_t n,int workers) {
    if(workers==1 || n<8192)return pairwise(values,n);
    // Cut the original reduction tree into a fixed frontier, then combine
    // its nodes in the same order. This also works with OpenMP2/MSVC and with
    // dynamically reduced teams; worker count cannot change rounding.
    struct Node {const T* data;std::size_t count,left=0,right=0;T value=0;};
    std::vector<Node> nodes;std::vector<std::size_t> leaves;
    auto build=[&](auto&& self,const T* data,std::size_t count,int budget)->std::size_t {
        const auto id=nodes.size();nodes.push_back({data,count,0,0,0});
        if(budget>1 && count>=8192) {
            auto half=count/2;half-=half%8;
            const auto left=self(self,data,half,budget/2),right=self(self,data+half,count-half,budget-budget/2);
            nodes[id].left=left;nodes[id].right=right;
        } else leaves.push_back(id);
        return id;
    };
    build(build,values,n,workers);
    detail::partition(leaves.size(),std::min(workers,int(leaves.size())),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){auto& node=nodes[leaves[i]];node.value=pairwise(node.data,node.count);}});
    for(std::size_t i=nodes.size();i-- >0;)if(nodes[i].left)nodes[i].value=T(nodes[nodes[i].left].value+nodes[nodes[i].right].value);
    return nodes[0].value;
}
template<class T> T mean(const T* values,std::size_t n,int workers,std::size_t block=0) {
    if(!n)return std::numeric_limits<T>::quiet_NaN();
    T total=0;
    if(block && n>block) {
        const auto chunks=n/block+(n%block!=0);std::vector<T> partial(chunks);
        detail::partition(chunks,std::min(workers,int(chunks)),[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i)partial[i]=pairwise(values+i*block,std::min(block,n-i*block));});
        for(const auto value:partial)total=T(total+value);
    } else total=sum(values,n,workers);
    return T(double(total)/double(n));
}
template<class T> detail::LiScan scan(const T* input,std::size_t n,T first) {
    detail::LiScan result;
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())i=simd::li_scan(input,n,std::is_same_v<T,float>,double(first),result);
#endif
    for(;i<n;++i) {
        const auto value=input[i];if(std::isnan(value))continue;
        ++result.valid;result.constant=result.constant && value==first;
        if(!std::isfinite(value))continue;
        ++result.finite;result.low=std::min(result.low,double(value));result.high=std::max(result.high,double(value));
    }
    return result;
}
template<class T> void shift(const T* input,T* output,std::size_t n,T low) {
    std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())i=simd::li_shift(input,output,n,std::is_same_v<T,float>,double(low));
#endif
    for(;i<n;++i)output[i]=T(input[i]-low);
}
template<class T> std::size_t low_count(const T* input,std::size_t n,T pivot) {
    std::size_t i=0,count=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())i=simd::li_count(input,n,std::is_same_v<T,float>,double(pivot),count);
#endif
    for(;i<n;++i)count+=input[i]<=pivot;return count;
}
template<class T> std::size_t compact(const T* input,std::size_t n,T pivot,T* low,T* high,std::size_t low_cap,std::size_t high_cap) {
    std::size_t i=0,left=0,right=0;
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())i=simd::li_partition(input,n,std::is_same_v<T,float>,double(pivot),low,high,low_cap,high_cap,left,right);
#else
    (void)low_cap;(void)high_cap;
#endif
    for(;i<n;++i) {const auto value=input[i];if(value<=pivot)low[left++]=value;else high[right++]=value;}
    return left;
}
template<class T> std::size_t partition_values(const std::vector<T>& values,T pivot,std::vector<T>& low,std::vector<T>& high,int workers) {
    const auto n=values.size();
    if(workers==1)return compact(values.data(),n,pivot,low.data(),high.data(),low.size(),high.size());
    struct Segment {std::size_t begin=0,end=0,left=0,low=0,high=0;};
    std::vector<Segment> segments(std::size_t(workers),Segment{});
    detail::partition(n,workers,[&](auto begin,auto end,auto id){segments[id]={begin,end,low_count(values.data()+begin,end-begin,pivot),0,0};});
    std::size_t left=0,right=0;
    for(auto& segment:segments) {segment.low=left;segment.high=right;left+=segment.left;right+=segment.end-segment.begin-segment.left;}
    // Process the recorded segments even if the second OpenMP team is smaller.
    detail::partition(segments.size(),workers,[&](auto begin,auto end,auto){
        for(auto j=begin;j<end;++j) {
            const auto& s=segments[j];if(s.begin==s.end)continue;
            compact(values.data()+s.begin,s.end-s.begin,pivot,low.data()+s.low,high.data()+s.high,s.left,s.end-s.begin-s.left);
        }
    });
    return left;
}
template<class T> double minimum_spacing(const std::vector<T>& values,int workers) {
    using Key=std::conditional_t<sizeof(T)==4,std::uint32_t,std::uint64_t>;
    const auto n=values.size();std::vector<Key> keys(n),scratch(n);
    detail::partition(n,workers,[&](auto begin,auto end,auto){for(auto i=begin;i<end;++i){Key bits;std::memcpy(&bits,&values[i],sizeof(T));keys[i]=values[i]==0 ? Key(0) : bits;}});
    struct Segment {std::size_t begin=0,end=0;};
    std::vector<Segment> segments(std::size_t(workers),Segment{});
    std::vector<std::array<std::size_t,256>> counts(std::size_t(workers),std::array<std::size_t,256>{});
    for(unsigned pass=0;pass<sizeof(Key);++pass) {
        for(auto& c:counts)c.fill(0);for(auto& s:segments)s={};
        detail::partition(n,workers,[&](auto begin,auto end,auto id){segments[id]={begin,end};for(auto i=begin;i<end;++i)++counts[id][(keys[i]>>(8*pass))&255];});
        std::size_t position=0;
        for(std::size_t bucket=0;bucket<256;++bucket)for(auto& c:counts){const auto length=c[bucket];c[bucket]=position;position+=length;}
        detail::partition(segments.size(),workers,[&](auto begin,auto end,auto){for(auto id=begin;id<end;++id){auto next=counts[id];for(auto i=segments[id].begin;i<segments[id].end;++i){const auto value=keys[i];scratch[next[(value>>(8*pass))&255]++]=value;}}});
        keys.swap(scratch);
    }
    T spacing=std::numeric_limits<T>::infinity(),previous=0;bool found=false,first=true;Key old=0;
    for(const auto key:keys) {
        if(!first && key==old)continue;
        T value;std::memcpy(&value,&key,sizeof(T));
        if(!first){spacing=std::min(spacing,T(value-previous));found=true;}
        previous=value;old=key;first=false;
    }
    return found ? double(spacing) : std::numeric_limits<double>::quiet_NaN();
}
template<class T> T logarithm(T value,const LiMath* math) {
    return math && math->log ? T(math->log(double(value),std::is_same_v<T,float>,math->context)) : std::log(value);
}
template<class T> LiResult floating_run(const std::vector<T>& values,T low,double tolerance,bool has_guess,double guess,const LiMath* math) {
    const int workers=detail::active_workers(values.size());
    const auto block=math ? math->reduction_block : 0;
    T next=has_guess ? T(guess) : mean(values.data(),values.size(),workers,block);
    if(!std::isfinite(next))return {0,false,32};
    LiResult result;std::vector<T> back(values.size()+8),fore(values.size()+8);
    if(math && math->iteration)math->iteration(double(T(next+low)),false,math->context);
    bool first=true;T current=0;std::size_t iterations=0;
    while((first ? std::abs(double(next)+2*tolerance) : double(std::abs(T(next-current))))>tolerance) {
        first=false;current=next;
        const auto left=partition_values(values,current,back,fore,workers),right=values.size()-left;
        const auto mean_fore=mean(fore.data(),right,workers,block),mean_back=mean(back.data(),left,workers,block);
        if(!right)result.status|=1;
        if(!std::isfinite(mean_back) || (!std::isfinite(mean_fore) && right))return {0,false,32};
        if(mean_back==0)break;
        const auto numerator=T(mean_back-mean_fore),denominator=T(logarithm(mean_back,math)-logarithm(mean_fore,math));
        next=T(numerator/denominator);result.updated=true;
        if(math && math->iteration)math->iteration(double(T(next+low)),true,math->context);
        if(math && math->poll && ((++iterations%1024)==0 || values.size()>=8192))math->poll(math->context);
    }
    result.value=double(T(next+low));return result;
}
}
struct LiImage::Impl {
    PixelType type;LiKind kind=LiKind::ready;double early=0,low=0,high=0;
    std::size_t count=0;
    std::array<double,257> histogram_count{},histogram_sum{};
    double initial=0;
    std::vector<float> single;
    std::vector<double> wide;
    template<class T> void prepare(const T* input,std::size_t n,std::vector<T>& values) {
        if(!std::isnan(input[0]) && constant_input(input,n)){kind=LiKind::constant;early=double(input[0]);return;}
        std::size_t first=0;while(first<n && std::isnan(input[first]))++first;
        if(first==n){kind=LiKind::empty;return;}
        const auto stats=scan(input,n,input[first]);
        if(stats.constant){kind=LiKind::constant;early=double(input[first]);return;}
        if(!stats.finite){kind=LiKind::infinities;return;}
        low=stats.low;high=double(T(T(stats.high)-T(low)));count=stats.finite;
        if(!std::isfinite(high)){kind=LiKind::unsupported;return;}
        values.resize(count);
        if(stats.finite==n)detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto){shift(input+begin,values.data()+begin,end-begin,T(low));});
        else {std::size_t j=0;for(std::size_t i=0;i<n;++i)if(std::isfinite(input[i]))values[j++]=T(input[i]-T(low));}
    }
};
LiImage::LiImage(const void* source,std::size_t n,PixelType type):impl_(new Impl) {
    auto& s=*impl_;s.type=type;
    if(n>std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double))throw std::overflow_error("Li image dimensions overflow");
    if(n && !source)throw std::invalid_argument("nonnull Li input required");
    if(!n){s.kind=LiKind::empty;return;}
    if(type==PixelType::boolean) {
        const auto* values=static_cast<const unsigned char*>(source);const bool first=values[0]!=0;
        s.kind=LiKind::constant;s.early=first;
        for(std::size_t i=1;i<n;++i)if((values[i]!=0)!=first){s.kind=LiKind::boolean_error;break;}
    } else if(type==PixelType::uint8 || type==PixelType::uint16) {
        std::vector<std::uint8_t> packed;
        const auto* values=static_cast<const std::uint8_t*>(source);
        if(type==PixelType::uint16) {
            // This bounded partition has exactly the existing byte histogram.
            // Check the sample bound before touching caller storage. No uint16
            // value outside 0..255 is admitted, including constant inputs.
            if(n>2147483647){s.kind=LiKind::unsupported;return;}
            const auto* wide=static_cast<const std::uint16_t*>(source);
            packed.resize(n);
            for(std::size_t i=0;i<n;++i) {
                if(wide[i]>255){s.kind=LiKind::unsupported;return;}
                packed[i]=std::uint8_t(wide[i]);
            }
            values=packed.data();s.type=PixelType::uint8;
        }
        if(constant_input(values,n)){s.kind=LiKind::constant;s.early=double(values[0]);return;}
        const auto counts=histogram_u8(values,1,n);
        std::size_t low=0,high=255;while(!counts[low])++low;while(!counts[high])--high;
        if(low==high){s.kind=LiKind::constant;s.early=double(low);return;}
        s.low=double(low);s.high=double(high-low);s.count=n;
        double sum=0;
        for(std::size_t i=low;i<=high;++i) {
            const auto at=i-low;const double weight=float(counts[i]);
            s.histogram_count[at+1]=s.histogram_count[at]+weight;
            s.histogram_sum[at+1]=s.histogram_sum[at]+weight*double(at);
            sum+=double(counts[i])*double(at);
        }
        s.initial=sum/double(n);
    } else if(type==PixelType::float32)s.prepare(static_cast<const float*>(source),n,s.single);
    else if(type==PixelType::float64)s.prepare(static_cast<const double*>(source),n,s.wide);
    else throw std::invalid_argument("Li input types are boolean, uint8, float32 and float64");
}
LiImage::~LiImage()=default;
LiKind LiImage::kind()const{return impl_->kind;}
double LiImage::value()const{return impl_->early;}
double LiImage::minimum()const{return impl_->low;}
double LiImage::maximum()const{return impl_->high;}
double LiImage::spacing()const {
    const auto& s=*impl_;
    if(s.kind!=LiKind::ready)throw std::invalid_argument("Li spacing requires prepared finite input");
    return s.type==PixelType::uint8 ? 1. : s.type==PixelType::float32 ? minimum_spacing(s.single,detail::active_workers(s.count)) : minimum_spacing(s.wide,detail::active_workers(s.count));
}
LiResult LiImage::run(double tolerance,bool has_guess,double guess,const LiMath* math)const {
    const auto& s=*impl_;
    if(s.kind!=LiKind::ready)throw std::invalid_argument("Li iteration requires prepared finite input");
    if(tolerance<0)throw std::invalid_argument("Li tolerance must be nonnegative");
    if(s.type==PixelType::float32)return floating_run(s.single,float(s.low),tolerance,has_guess,guess,math);
    if(s.type==PixelType::float64)return floating_run(s.wide,s.low,tolerance,has_guess,guess,math);
    double next=has_guess ? guess : s.initial,current=-2*tolerance;LiResult result;
    const auto bins=std::size_t(s.high)+1;
    std::size_t iterations=0;
    if(math && math->iteration)math->iteration(next+s.low,false,math->context);
    while(std::abs(next-current)>tolerance) {
        current=next;
        const auto split=current<0 ? std::size_t(0) : current>=s.high ? bins : std::size_t(std::floor(current))+1;
        const auto back_weight=s.histogram_count[split],fore_weight=s.histogram_count[bins]-back_weight;
        const auto mean_back=s.histogram_sum[split]/back_weight;
        const auto mean_fore=(s.histogram_sum[bins]-s.histogram_sum[split])/fore_weight;
        if(mean_back==0)break;
        next=(mean_back-mean_fore)/(logarithm(mean_back,math)-logarithm(mean_fore,math));result.updated=true;
        if(math && math->iteration)math->iteration(next+s.low,true,math->context);
        if(math && math->poll && (++iterations%1024)==0)math->poll(math->context);
    }
    result.value=next+s.low;return result;
}
}
