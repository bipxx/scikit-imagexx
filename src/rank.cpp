#include "skimagexx/rank.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#endif
namespace skimagexx {
namespace {
unsigned first(std::uint64_t word) {
#ifdef _MSC_VER
    unsigned long index;_BitScanForward64(&index,word);return unsigned(index);
#else
    return unsigned(__builtin_ctzll(word));
#endif
}
unsigned last(std::uint64_t word) {
#ifdef _MSC_VER
    unsigned long index;_BitScanReverse64(&index,word);return unsigned(index);
#else
    return 63u-unsigned(__builtin_clzll(word));
#endif
}
struct Run {std::ptrdiff_t plane,row,col;std::size_t count;};
struct Plan {std::vector<Run> initial,enter[3],leave[3];};
Plan plan(const RankView& v) {
    Plan result;
    const auto& f=v.footprint_shape;
    const auto active=[&](std::ptrdiff_t p,std::ptrdiff_t r,std::ptrdiff_t c){return p>=0 && r>=0 && c>=0 && std::size_t(p)<f[0] && std::size_t(r)<f[1] && std::size_t(c)<f[2] && v.footprint[(std::size_t(p)*f[1]+std::size_t(r))*f[2]+std::size_t(c)]!=0;};
    const auto append=[](std::vector<Run>& runs,std::ptrdiff_t p,std::ptrdiff_t r,std::ptrdiff_t c){
        if(!runs.empty() && runs.back().plane==p && runs.back().row==r && runs.back().col+std::ptrdiff_t(runs.back().count)==c)++runs.back().count;
        else runs.push_back({p,r,c,1});
    };
    for(std::size_t p=0;p<f[0];++p)for(std::size_t r=0;r<f[1];++r)for(std::size_t c=0;c<f[2];++c) {
        if(!active(p,r,c))continue;
        const auto pp=std::ptrdiff_t(p)-v.center[0],rr=std::ptrdiff_t(r)-v.center[1],cc=std::ptrdiff_t(c)-v.center[2];
        append(result.initial,pp,rr,cc);
        for(int axis=1;axis<3;++axis) {
            if(!active(std::ptrdiff_t(p),std::ptrdiff_t(r)+(axis==1),std::ptrdiff_t(c)+(axis==2)))append(result.enter[axis],pp,rr,cc);
            if(!active(std::ptrdiff_t(p),std::ptrdiff_t(r)-(axis==1),std::ptrdiff_t(c)-(axis==2)))append(result.leave[axis],pp,rr,cc);
        }
    }
    return result;
}
struct Histogram {
    std::size_t limit;
    std::vector<std::int64_t> counts;
    std::vector<std::uint64_t> bits,summary;
    std::uint64_t root=0;
    std::int64_t sum=0,pop=0;
    explicit Histogram(std::size_t bins,bool extrema):limit(bins),counts(extrema ? bins:0),bits(extrema ? (bins+63)/64:0),summary(extrema ? (bins+4095)/4096:0){}
    void add(std::uint32_t value,std::int64_t count) {
        pop+=count;sum+=std::int64_t(value)*count;
        if(value>=limit)throw std::invalid_argument("aliased rank image exceeded histogram bounds");
        if(counts.empty())return;
        const auto before=counts[value];counts[value]+=count;
        if((before==0)==(counts[value]==0))return;
        const auto word=value/64,group=word/64;const auto old=bits[word];
        if(counts[value])bits[word]|=std::uint64_t{1}<<(value%64);else bits[word]&=~(std::uint64_t{1}<<(value%64));
        if(bool(old)==bool(bits[word]))return;
        if(bits[word])summary[group]|=std::uint64_t{1}<<(word%64);else summary[group]&=~(std::uint64_t{1}<<(word%64));
        if(summary[group])root|=std::uint64_t{1}<<group;else root&=~(std::uint64_t{1}<<group);
    }
    std::int64_t integer(RankOperation op)const {
        if(op==RankOperation::pop)return pop;
        if(!pop)return 0.;
        if(op==RankOperation::sum)return sum;
        if(!root)return 0.;
        const auto group=op==RankOperation::minimum ? first(root):last(root);
        const auto word=group*64+(op==RankOperation::minimum ? first(summary[group]):last(summary[group]));
        return std::int64_t(word*64+(op==RankOperation::minimum ? first(bits[word]):last(bits[word])));
    }
};
template<class T> void update(Histogram& hist,const RankView& v,const std::vector<Run>& runs,std::ptrdiff_t p,std::ptrdiff_t r,std::ptrdiff_t c,int sign,int row_adjust=0,int col_adjust=0) {
    const auto* image=static_cast<const unsigned char*>(v.image);
    const auto read=[](const unsigned char* p){T value;std::memcpy(&value,p,sizeof(T));return value;};
    for(const auto& run:runs) {
        const auto pp=p+run.plane,rr=r+run.row+row_adjust,cc=c+run.col+col_adjust;
        if(pp<0 || rr<0 || std::size_t(pp)>=v.shape[0] || std::size_t(rr)>=v.shape[1])continue;
        const auto lo=std::max(std::ptrdiff_t{0},-cc),hi=std::min(std::ptrdiff_t(run.count),std::ptrdiff_t(v.shape[2])-cc);
        if(lo>=hi)continue;
        const auto index=(std::size_t(pp)*v.shape[1]+std::size_t(rr))*v.shape[2]+std::size_t(cc+lo),n=std::size_t(hi-lo);
        const auto* data=image+index*sizeof(T);
        if(!v.mask && hist.counts.empty() && !(v.uint16 && v.serial_alias)) {
            std::uint64_t sum=0;
#ifdef SKIMAGEXX_AVX2
            if(has_avx2())sum=simd::rank_sum(data,n,sizeof(T)==2);
            else
#endif
            for(std::size_t i=0;i<n;++i)sum+=read(data+i*sizeof(T));
            hist.pop+=std::int64_t(n)*sign;hist.sum+=std::int64_t(sum)*sign;continue;
        }
        std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
        if(!v.mask && has_avx2()) {
            constexpr auto width=32/sizeof(T);
            for(;i+width<=n;i+=width) {
                std::uint32_t value;
                if(simd::rank_uniform(data+i*sizeof(T),sizeof(T)==2,value))hist.add(value,std::int64_t(width)*sign);
                else for(std::size_t j=0;j<width;++j)hist.add(read(data+(i+j)*sizeof(T)),sign);
            }
        }
#endif
        for(;i<n;++i)if(!v.mask || v.mask[index+i])hist.add(read(data+i*sizeof(T)),sign);
    }
}
template<class T,class Value> void store(unsigned char* out,Value value) {
    T result;
    if constexpr(std::is_integral_v<T> && std::is_floating_point_v<Value>)result=T(std::int64_t(value));else result=T(value);
    std::memcpy(out,&result,sizeof(result));
}
template<class T> void process(const RankView& v,RankOperation op,const Plan& prepared,std::size_t begin,std::size_t end) {
    while(begin<end) {
        const auto p=begin/v.shape[1],start=begin%v.shape[1],finish=std::min(v.shape[1],start+end-begin);
        Histogram hist(v.bins,op==RankOperation::minimum || op==RankOperation::maximum);
        auto r=std::ptrdiff_t(start),c=std::ptrdiff_t{0};
        update<T>(hist,v,prepared.initial,p,r,c,1);
        const auto write=[&](){
            auto* out=static_cast<unsigned char*>(v.output)+std::ptrdiff_t(p)*v.output_strides[0]+r*v.output_strides[1]+c*v.output_strides[2];
            const auto emit=[&](auto value){switch(v.output_type){case PixelType::boolean:case PixelType::uint8:store<std::uint8_t>(out,value);break;case PixelType::uint16:store<std::uint16_t>(out,value);break;case PixelType::float32:store<float>(out,value);break;case PixelType::float64:store<double>(out,value);break;default:break;}};
            if(op==RankOperation::mean)emit(hist.pop ? double(hist.sum)/double(hist.pop):0.);
            else emit(hist.integer(op));
        };
        write();bool east=true;
        while(true) {
            if(east)for(c=1;c<std::ptrdiff_t(v.shape[2]);++c) {
                update<T>(hist,v,prepared.enter[2],p,r,c,1);update<T>(hist,v,prepared.leave[2],p,r,c,-1,0,-1);write();
            }
            else for(c=std::ptrdiff_t(v.shape[2])-2;c>=0;--c) {
                update<T>(hist,v,prepared.leave[2],p,r,c,1);update<T>(hist,v,prepared.enter[2],p,r,c,-1,0,1);write();
            }
            c=east ? std::ptrdiff_t(v.shape[2])-1:0;
            if(++r>=std::ptrdiff_t(finish))break;
            update<T>(hist,v,prepared.enter[1],p,r,c,1);update<T>(hist,v,prepared.leave[1],p,r,c,-1,-1,0);write();east=!east;
        }
        begin=p*v.shape[1]+finish;
    }
}
}
void rank_filter_into(const RankView& view,RankOperation operation) {
    if(int(operation)<0 || int(operation)>4)throw std::invalid_argument("invalid rank operation");
    for(int axis=0;axis<3;++axis)if(!view.shape[axis] || !view.footprint_shape[axis] || view.center[axis]<0 || std::size_t(view.center[axis])>=view.footprint_shape[axis])throw std::invalid_argument("invalid rank dimensions or center");
    if(!view.image || !view.output || !view.footprint || view.bins<1 || view.bins>65536)throw std::invalid_argument("invalid rank buffers or bins");
    std::size_t pixels=1,footprint_pixels=1,span=0;
    const auto bound=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    const auto itemsize=pixel_size(view.output_type);
    if(view.output_type!=PixelType::boolean && view.output_type!=PixelType::uint8 && view.output_type!=PixelType::uint16 && view.output_type!=PixelType::float32 && view.output_type!=PixelType::float64)throw std::invalid_argument("invalid rank output type");
    for(int axis=0;axis<3;++axis) {
        if(view.shape[axis]>bound/pixels || view.footprint_shape[axis]>bound/footprint_pixels)throw std::overflow_error("rank dimensions overflow");
        pixels*=view.shape[axis];footprint_pixels*=view.footprint_shape[axis];
        if(view.shape[axis]>1) {
            if(view.output_strides[axis]<=0 || std::size_t(view.output_strides[axis])>(bound-span)/(view.shape[axis]-1))throw std::overflow_error("rank output strides overflow");
            span+=(view.shape[axis]-1)*std::size_t(view.output_strides[axis]);
        }
    }
    if(pixels>bound/(view.uint16 ? 2:1) || span>bound-itemsize)throw std::overflow_error("rank buffer size overflow");
    const auto prepared=plan(view);const auto rows=view.shape[0]*view.shape[1];
    const auto workers=view.serial_alias ? 1:detail::active_workers(rows*view.shape[2]);
    const auto run=[&](auto begin,auto end){if(view.uint16)process<std::uint16_t>(view,operation,prepared,begin,end);else process<std::uint8_t>(view,operation,prepared,begin,end);};
    if(workers==1){run(std::size_t{0},rows);return;}
    // C++ exceptions must leave an OpenMP team through the calling thread.
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{run(begin,end);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
std::uint16_t rank_max_u16(const std::uint16_t* data,std::size_t count) {
#ifdef SKIMAGEXX_AVX2
    if(has_avx2())return simd::rank_max_u16(data,count);
#endif
    std::uint16_t result=0;for(std::size_t i=0;i<count;++i){std::uint16_t value;std::memcpy(&value,reinterpret_cast<const unsigned char*>(data)+2*i,2);result=std::max(result,value);}return result;
}
RankHalfConversion rank_half_u8(const std::uint16_t* source,std::uint8_t* output,std::size_t count) {
    if(!count)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    const auto workers=detail::active_workers(count);
    struct Bounds {float low=0,high=0;bool nan=false;};
    std::vector<Bounds> bounds(std::size_t(workers),Bounds{});
    const auto read=[&](std::size_t i){std::uint16_t word;std::memcpy(&word,reinterpret_cast<const unsigned char*>(source)+2*i,2);return word;};
    detail::partition(count,workers,[&](auto begin,auto end,auto id){
        auto& result=bounds[id];
        for(auto i=begin;i<end;++i){const auto value=detail::decode_half(read(i));if(std::isnan(value))result.nan=true;else{result.low=std::min(result.low,value);result.high=std::max(result.high,value);}}
    });
    Bounds range;for(const auto& part:bounds){range.nan|=part.nan;range.low=std::min(range.low,part.low);range.high=std::max(range.high,part.high);}
    if(!range.nan && (range.low<-1 || range.high>1))throw std::invalid_argument("Images of type float must be between -1 and 1.");
    std::vector<RankHalfConversion> statuses(std::size_t(workers),RankHalfConversion{});
    detail::partition(count,workers,[&](auto begin,auto end,auto id){
        auto& info=statuses[id];
        for(auto i=begin;i<end;++i) {
            const auto bits=read(i);const float value=detail::decode_half(bits);
            if(std::isnan(value)){info.nan=true;info.invalid_multiply|=(bits&0x0200)==0;output[i]=0;continue;}
            const auto half=detail::encode_half(value*255.f);const auto product=detail::decode_half(half);
            info.overflow|=std::isfinite(value) && std::isinf(product);
            // All integers through255 are exactly representable as half.
            // Explicit ties-to-even avoids depending on host rounding mode.
            if(product<=0)output[i]=0;
            else if(product>=255)output[i]=255;
            else {auto low=int(product);const auto fraction=product-float(low);if(fraction>.5f || (fraction==.5f && (low&1)))++low;output[i]=std::uint8_t(low);}
        }
    });
    RankHalfConversion info;for(const auto& part:statuses){info.nan|=part.nan;info.invalid_multiply|=part.invalid_multiply;info.overflow|=part.overflow;}return info;
}
}
