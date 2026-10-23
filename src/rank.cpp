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
    // Allocated only for equalize: bounded 64-bin block sums avoid scanning
    // every occupied uint16 bin for each center. Old operations pay no update.
    std::vector<std::int64_t> prefix_blocks;
    std::uint64_t root=0;
    std::int64_t sum=0,pop=0;
    explicit Histogram(std::size_t bins,bool extrema,bool prefix=false):limit(bins),counts(extrema ? bins:0),bits(extrema ? (bins+63)/64:0),summary(extrema ? (bins+4095)/4096:0),prefix_blocks(prefix ? (bins+63)/64:0){}
    template<bool Prefix> void add(std::uint32_t value,std::int64_t count) {
        pop+=count;sum+=std::int64_t(value)*count;
        if(value>=limit)throw std::invalid_argument("aliased rank image exceeded histogram bounds");
        if(counts.empty())return;
        const auto before=counts[value];counts[value]+=count;
        if constexpr(Prefix)prefix_blocks[value/64]+=count;
        if((before==0)==(counts[value]==0))return;
        const auto word=value/64,group=word/64;const auto old=bits[word];
        if(counts[value])bits[word]|=std::uint64_t{1}<<(value%64);else bits[word]&=~(std::uint64_t{1}<<(value%64));
        if(bool(old)==bool(bits[word]))return;
        if(bits[word])summary[group]|=std::uint64_t{1}<<(word%64);else summary[group]&=~(std::uint64_t{1}<<(word%64));
        if(summary[group])root|=std::uint64_t{1}<<group;else root&=~(std::uint64_t{1}<<group);
    }
    double equalize(std::uint32_t center)const {
        if(!pop)return 0.;
        std::int64_t cumulative=0;
        const auto block=center/64;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2()) {
            cumulative=simd::rank_count_prefix(prefix_blocks.data(),block);
            cumulative+=simd::rank_count_prefix(counts.data()+block*64,center%64+1);
        } else
#endif
        {
            for(std::size_t i=0;i<block;++i)cumulative+=prefix_blocks[i];
            for(std::size_t i=block*64;i<=center;++i)cumulative+=counts[i];
        }
        // Cython multiplies integers before converting for division.
        return double(std::int64_t(limit-1)*cumulative)/double(pop);
    }
    template<class F> void occupied(F visit)const {
        auto groups=root;
        while(groups) {
            const auto group=first(groups);groups&=groups-1;
            auto words=summary[group];
            while(words) {
                const auto word=group*64+first(words);words&=words-1;
                auto values=bits[word];
                while(values) {
                    const auto index=word*64+first(values);values&=values-1;
                    if(!visit(index,counts[index]))return;
                }
            }
        }
    }
    double entropy()const {
        double result=0.;
        if(pop)occupied([&](std::size_t,std::int64_t count){
            const double p=double(count)/double(pop);
            // Pinned Cython uses natural C log, ascending bins, this literal
            // and this operation order. Do not reassociate or use log2.
            if(p>0)result-=p*std::log(p)/0.6931471805599453;
            return true;
        });
        return result;
    }
    std::int64_t geometric_mean(const std::vector<double>& logarithms)const {
        if(!pop)return 0;
        double total=0.;
        occupied([&](std::size_t index,std::int64_t count){total+=double(count)*logarithms[index];return true;});
        const double value=std::exp(total/double(pop))-1.;
        // interpolation.pxd's round: ties away from zero, then C truncation.
        return std::int64_t(value>0 ? value+.5:value-.5);
    }
    std::int64_t noise(std::uint32_t center)const {
        if(counts[center])return 0;
        std::size_t lower=0,upper=limit-1;
        const auto word=center/64,group=word/64,bit=center%64;
        auto below=bit ? bits[word]&((std::uint64_t{1}<<bit)-1):0;
        if(below)lower=word*64+last(below);
        else {
            auto words=word%64 ? summary[group]&((std::uint64_t{1}<<(word%64))-1):0;
            auto g=group;
            if(!words) {
                const auto groups=group ? root&((std::uint64_t{1}<<group)-1):0;
                if(groups){g=last(groups);words=summary[g];}
            }
            if(words){const auto w=g*64+last(words);lower=w*64+last(bits[w]);}
        }
        auto above=bit==63 ? 0:bits[word]&(~std::uint64_t{0}<<(bit+1));
        if(above)upper=word*64+first(above);
        else {
            auto words=word%64==63 ? 0:summary[group]&(~std::uint64_t{0}<<(word%64+1));
            auto g=group;
            if(!words) {
                const auto groups=group==63 ? 0:root&(~std::uint64_t{0}<<(group+1));
                if(groups){g=first(groups);words=summary[g];}
            }
            if(words){const auto w=g*64+first(words);upper=w*64+first(bits[w]);}
        }
        return std::int64_t(std::min(std::size_t(center)-lower,upper-std::size_t(center)));
    }
    bool median(std::int64_t& selected)const {
        selected=0;
        if(!pop)return true;
        double remaining=double(pop)/2.;bool store=false;
        // Feedback counts may be negative: the CDF is not monotone. Visit
        // occupied bins in order and retain the first crossing, or no store.
        occupied([&](std::size_t index,std::int64_t count){remaining-=double(count);if(remaining<0){selected=std::int64_t(index);store=true;return false;}return true;});
        return store;
    }
    std::int64_t integer(RankOperation op)const {
        if(op==RankOperation::pop)return pop;
        if(!pop)return 0.;
        if(op==RankOperation::sum)return sum;
        if(!root)return 0.;
        if(op==RankOperation::gradient)return integer(RankOperation::maximum)-integer(RankOperation::minimum);
        if(op==RankOperation::median) {
            std::int64_t selected;median(selected);
            return selected;
        }
        if(op==RankOperation::modal) {
            std::int64_t maximum=0,selected=0;
            occupied([&](std::size_t index,std::int64_t count){if(count>maximum){maximum=count;selected=std::int64_t(index);}return true;});
            return selected;
        }
        const auto group=op==RankOperation::minimum ? first(root):last(root);
        const auto word=group*64+(op==RankOperation::minimum ? first(summary[group]):last(summary[group]));
        return std::int64_t(word*64+(op==RankOperation::minimum ? first(bits[word]):last(bits[word])));
    }
};
template<class T,bool Prefix> void update(Histogram& hist,const RankView& v,const std::vector<Run>& runs,std::ptrdiff_t p,std::ptrdiff_t r,std::ptrdiff_t c,int sign,int row_adjust=0,int col_adjust=0) {
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
                if(simd::rank_uniform(data+i*sizeof(T),sizeof(T)==2,value))hist.add<Prefix>(value,std::int64_t(width)*sign);
                else for(std::size_t j=0;j<width;++j)hist.add<Prefix>(read(data+(i+j)*sizeof(T)),sign);
            }
        }
#endif
        for(;i<n;++i)if(!v.mask || v.mask[index+i])hist.add<Prefix>(read(data+i*sizeof(T)),sign);
    }
}
template<class T,class Value> void store(unsigned char* out,Value value) {
    T result;
    if constexpr(std::is_integral_v<T> && std::is_floating_point_v<Value>)result=T(std::int64_t(value));else result=T(value);
    std::memcpy(out,&result,sizeof(result));
}
template<class T,bool Prefix=false> void process(const RankView& v,RankOperation op,const Plan& prepared,std::size_t begin,std::size_t end,const std::vector<double>& logarithms={},std::size_t depth=0,bool domain_only=false) {
    while(begin<end) {
        const auto p=begin/v.shape[1],start=begin%v.shape[1],finish=std::min(v.shape[1],start+end-begin);
        Histogram hist(v.bins,op!=RankOperation::mean && op!=RankOperation::sum && op!=RankOperation::pop && op!=RankOperation::threshold && op!=RankOperation::subtract_mean,Prefix);
        auto r=std::ptrdiff_t(start),c=std::ptrdiff_t{0};
        update<T,Prefix>(hist,v,prepared.initial,p,r,c,1);
        const auto write=[&](){
            auto* out=static_cast<unsigned char*>(v.output)+std::ptrdiff_t(p)*v.output_strides[0]+r*v.output_strides[1]+c*v.output_strides[2];
            const auto emit=[&](auto value){
                if(domain_only){const double truncated=std::trunc(double(value));const double maximum=v.output_type==PixelType::uint16 ? 65535.:255.;if(!std::isfinite(truncated) || truncated<0 || truncated>maximum)throw std::invalid_argument("subtract_mean unsigned conversion outside defined coverage");return;}
                switch(v.output_type){case PixelType::boolean:case PixelType::uint8:store<std::uint8_t>(out,value);break;case PixelType::uint16:store<std::uint16_t>(out,value);break;case PixelType::float32:store<float>(out,value);break;case PixelType::float64:store<double>(out,value);break;default:break;}};
            if(op==RankOperation::windowed_histogram) {
                const double scale=hist.pop ? 1./double(hist.pop):0.;
                const auto emit_vector=[&](auto tag){using O=decltype(tag);std::size_t i=0;
#ifdef SKIMAGEXX_AVX2
                    if constexpr(std::is_floating_point_v<O>)if(has_avx2() && hist.pop<=std::numeric_limits<std::int32_t>::max())i=simd::rank_histogram_normalized(hist.counts.data(),out,depth,scale,sizeof(O)==4);
#endif
                    for(;i<depth;++i)store<O>(out+i*sizeof(O),double(hist.counts[i])*scale);
                };
                switch(v.output_type){case PixelType::boolean:case PixelType::uint8:emit_vector(std::uint8_t{});break;case PixelType::uint16:emit_vector(std::uint16_t{});break;case PixelType::float32:emit_vector(float{});break;case PixelType::float64:emit_vector(double{});break;default:break;}
                return;
            }
            if(op==RankOperation::mean)emit(hist.pop ? double(hist.sum)/double(hist.pop):0.);
            else if(op==RankOperation::entropy)emit(hist.entropy());
            else if(op==RankOperation::geometric_mean)emit(hist.geometric_mean(logarithms));
            else if(op==RankOperation::median){std::int64_t selected;if(hist.median(selected))emit(selected);}
            else if(int(op)>=9) {
                T pixel;const auto index=(std::size_t(p)*v.shape[1]+std::size_t(r))*v.shape[2]+std::size_t(c);
                std::memcpy(&pixel,static_cast<const unsigned char*>(v.image)+index*sizeof(T),sizeof(T));
                if(pixel>=v.bins)throw std::invalid_argument("rank center exceeded histogram bounds");
                if(op==RankOperation::threshold)emit(hist.pop && double(pixel)>double(hist.sum)/double(hist.pop) ? std::int64_t{1}:std::int64_t{0});
                else if(op==RankOperation::noise_filter)emit(hist.noise(pixel));
                else if(op==RankOperation::subtract_mean)emit(hist.pop ? ((double(pixel)-double(hist.sum)/double(hist.pop))/2.+double(v.bins/2))-1.:0.);
                else if(op==RankOperation::equalize)emit(hist.equalize(pixel));
                else {
                    const auto low=hist.integer(RankOperation::minimum),high=hist.integer(RankOperation::maximum),g=std::int64_t(pixel);
                    if(op==RankOperation::autolevel) {
                        // Signed integer division truncates toward zero even
                        // for floating outputs. Integer stores wrap separately.
                        emit(high>low ? std::int64_t(v.bins-1)*(g-low)/(high-low):std::int64_t{0});
                    } else emit(hist.pop ? (high-g<g-low ? high:low):std::int64_t{0});
                }
            }
            else emit(hist.integer(op));
        };
        write();bool east=true;
        while(true) {
            if(east)for(c=1;c<std::ptrdiff_t(v.shape[2]);++c) {
                update<T,Prefix>(hist,v,prepared.enter[2],p,r,c,1);update<T,Prefix>(hist,v,prepared.leave[2],p,r,c,-1,0,-1);write();
            }
            else for(c=std::ptrdiff_t(v.shape[2])-2;c>=0;--c) {
                update<T,Prefix>(hist,v,prepared.leave[2],p,r,c,1);update<T,Prefix>(hist,v,prepared.enter[2],p,r,c,-1,0,1);write();
            }
            c=east ? std::ptrdiff_t(v.shape[2])-1:0;
            if(++r>=std::ptrdiff_t(finish))break;
            update<T,Prefix>(hist,v,prepared.enter[1],p,r,c,1);update<T,Prefix>(hist,v,prepared.leave[1],p,r,c,-1,-1,0);write();east=!east;
        }
        begin=p*v.shape[1]+finish;
    }
}
}
namespace {
void validate(const RankView& view,RankOperation operation) {
    if(int(operation)<0 || int(operation)>12)throw std::invalid_argument("invalid rank operation");
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
}
// Test exact destination element bytes against a contiguous read buffer.
// Bounding intervals cheaply reject the common disjoint case; row intervals
// avoid scanning each pixel for ordinary contiguous/pixel-vector output.
bool overlaps(const RankView& v,const void* input,std::size_t bytes,std::size_t pixel_bytes=0) {
    if(!input || !bytes)return false;
    const auto base=reinterpret_cast<std::uintptr_t>(v.output),low=reinterpret_cast<std::uintptr_t>(input);
    const auto maximum=std::numeric_limits<std::uintptr_t>::max();
    if(bytes>maximum-low)throw std::overflow_error("rank pointer range overflow");
    const auto high=low+bytes,item=pixel_bytes ? pixel_bytes:pixel_size(v.output_type);
    std::size_t span=item;
    for(int axis=0;axis<3;++axis)span+=(v.shape[axis]-1)*std::size_t(v.output_strides[axis]);
    if(span>maximum-base)throw std::overflow_error("rank pointer range overflow");
    if(base>=high || base+span<=low)return false;
    for(std::size_t p=0;p<v.shape[0];++p)for(std::size_t r=0;r<v.shape[1];++r) {
        const auto row=base+p*std::size_t(v.output_strides[0])+r*std::size_t(v.output_strides[1]);
        if(v.shape[2]==1 || std::size_t(v.output_strides[2])==item) {
            if(row<high && row+v.shape[2]*item>low)return true;
        } else {
            const auto step=std::size_t(v.output_strides[2]);
            // First possible store whose end is after low. This remains exact
            // for disjoint strided views sharing a bounding address interval.
            const auto c=low>=row+item ? (low-row-item)/step+1:0;
            if(c<v.shape[2] && row+c*step<high)return true;
        }
    }
    return false;
}
RankAliases aliases(const RankView& v) {
    const auto pixels=v.shape[0]*v.shape[1]*v.shape[2];
    return {overlaps(v,v.image,pixels*(v.uint16 ? 2:1)),overlaps(v,v.mask,pixels)};
}
// A sufficient injectivity certificate for arbitrary positive SDK strides.
// Uncertified layouts remain valid, but their stores must run serially.
bool independent_stores(const RankView& v,std::size_t pixel_bytes=0) {
    std::array<int,3> axes{0,1,2};
    std::sort(axes.begin(),axes.end(),[&](int a,int b){return v.output_strides[a]<v.output_strides[b];});
    auto span=pixel_bytes ? pixel_bytes:pixel_size(v.output_type);
    for(const auto axis:axes)if(v.shape[axis]>1) {
        const auto stride=std::size_t(v.output_strides[axis]);
        if(stride<span)return false;
        span+=(v.shape[axis]-1)*stride;
    }
    return true;
}
void run(const RankView& view,RankOperation operation) {
    const auto prepared=plan(view);const auto rows=view.shape[0]*view.shape[1];
    const auto workers=view.serial_alias || !independent_stores(view) ? 1:detail::active_workers(rows*view.shape[2]);
    const auto run=[&](auto begin,auto end){
        if(operation==RankOperation::equalize) {
            if(view.uint16)process<std::uint16_t,true>(view,operation,prepared,begin,end);else process<std::uint8_t,true>(view,operation,prepared,begin,end);
        } else {
            if(view.uint16)process<std::uint16_t>(view,operation,prepared,begin,end);else process<std::uint8_t>(view,operation,prepared,begin,end);
        }
    };
    if(workers==1){run(std::size_t{0},rows);return;}
    // C++ exceptions must leave an OpenMP team through the calling thread.
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{run(begin,end);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
// Statistics/vectors are a separate admission surface. Validate every read
// value and every accessed output byte before any worker can write.
std::vector<double> statistics_validate(const RankView& v,RankOperation op,std::size_t depth) {
    validate(v,RankOperation::minimum);
    const auto item=pixel_size(v.output_type),bound=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(depth && (v.shape[0]!=1 || depth>v.bins || depth>bound/item))throw std::invalid_argument("invalid rank vector depth or dimensions");
    const auto width=depth ? item*depth:item;
    std::size_t span=width;
    for(int axis=0;axis<3;++axis){const auto extent=v.shape[axis]-1,step=std::size_t(v.output_strides[axis]);if(extent && step>(bound-span)/extent)throw std::overflow_error("rank vector output strides overflow");span+=extent*step;}
    if(!independent_stores(v,width))throw std::invalid_argument("rank statistics output stores overlap");
    const auto pixels=v.shape[0]*v.shape[1]*v.shape[2],fp=v.footprint_shape[0]*v.footprint_shape[1]*v.footprint_shape[2];
    if(fp>std::size_t(std::numeric_limits<std::int64_t>::max())/(v.uint16 ? 65535:255))throw std::overflow_error("rank statistics population sum overflow");
    if(v.serial_alias || overlaps(v,v.image,pixels*(v.uint16 ? 2:1),width) || overlaps(v,v.mask,pixels,width) || overlaps(v,v.footprint,fp,width))throw std::invalid_argument("overlapping buffers are outside rank-statistics-v1 parity");
    std::vector<double> logarithms(op==RankOperation::geometric_mean ? v.bins:0,0.);
    // Cache log only for globally occupied values; each output still accumulates
    // in ascending-bin order. No reassociation/incremental floating reduction.
    std::vector<std::uint8_t> seen(logarithms.empty() ? 0:v.bins,0);
    const auto* image=static_cast<const unsigned char*>(v.image);
    for(std::size_t i=0;i<pixels;++i){std::uint16_t value;if(v.uint16)std::memcpy(&value,image+2*i,2);else value=image[i];if(value>=v.bins)throw std::invalid_argument("rank image exceeded histogram bounds");if(!seen.empty())seen[value]=1;}
    for(std::size_t i=0;i<seen.size();++i)if(seen[i])logarithms[i]=std::log(double(i+1));
    return logarithms;
}
void statistics_run(const RankView& view,RankOperation operation,std::size_t depth) {
    const auto logarithms=statistics_validate(view,operation,depth);const auto prepared=plan(view);
    const auto rows=view.shape[0]*view.shape[1];
    const auto invoke=[&](std::size_t begin,std::size_t end,bool check){if(view.uint16)process<std::uint16_t>(view,operation,prepared,begin,end,logarithms,depth,check);else process<std::uint8_t>(view,operation,prepared,begin,end,logarithms,depth,check);};
    // C floating-to-unsigned conversion outside the target range is undefined
    // upstream. Certify all values first, preserving output on rejected calls.
    const bool integral=view.output_type==PixelType::boolean || view.output_type==PixelType::uint8 || view.output_type==PixelType::uint16;
    // An even capacity gives extrema in [-.5, bins-1.5], so truncation
    // certifies the entire target domain without another raster traversal.
    const bool bounded_integral=!(view.bins&1) && (view.output_type==PixelType::uint16 || view.bins<=256);
    if(operation==RankOperation::subtract_mean && integral && !bounded_integral)invoke(0,rows,true);
    const auto workers=detail::active_workers(rows*view.shape[2]);
    if(workers==1){invoke(0,rows,false);return;}
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{invoke(begin,end,false);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
}
std::uint32_t rank_statistics_capabilities() noexcept {return 0xe000u;}
std::uint32_t rank_vector_capabilities() noexcept {return 0x10000u;}
void rank_statistics_into(const RankView& view,RankOperation op) {
    if(int(op)<13 || int(op)>15)throw std::invalid_argument("invalid rank statistics operation");
    statistics_run(view,op,0);
}
void rank_windowed_histogram_into(const RankView& view,std::size_t depth) {
    if(!depth)throw std::invalid_argument("rank output pixel depth must be positive");
    statistics_run(view,RankOperation::windowed_histogram,depth);
}
RankAliases rank_aliases(const RankView& view) {
    validate(view,RankOperation::minimum);return aliases(view);
}
void rank_filter_into(const RankView& view,RankOperation operation) {
    // Preserve the old admission contract for existing SDK callers.
    if(int(operation)>=5 && view.serial_alias)throw std::invalid_argument("overlapping buffers are outside rank-histogram-v2 parity");
    validate(view,operation);run(view,operation);
}
void rank_filter_into(const RankView& view,RankOperation operation,RankAliases requested) {
    validate(view,operation);
    const auto actual=aliases(view);
    if(requested.image_output!=actual.image_output || requested.mask_output!=actual.mask_output)throw std::invalid_argument("rank alias descriptor does not match accessed buffers");
    const bool feedback=actual.image_output || actual.mask_output;
    if(feedback && (int(operation)<5 || int(operation)>8))throw std::invalid_argument("overlapping buffers are outside rank alias capability");
    const auto footprint_pixels=view.footprint_shape[0]*view.footprint_shape[1]*view.footprint_shape[2];
    if(overlaps(view,view.footprint,footprint_pixels))throw std::invalid_argument("rank footprint/output alias is outside defined coverage");
    if(feedback) {
        bool zero=true;
        for(std::size_t i=0;i<footprint_pixels && zero;++i)zero=view.footprint[i]==0;
        if(!zero && view.mask) {
            zero=true;
            const auto pixels=view.shape[0]*view.shape[1]*view.shape[2];
            for(std::size_t i=0;i<pixels && zero;++i)zero=view.mask[i]==0;
        }
        const bool integral=view.output_type==PixelType::uint8 || view.output_type==PixelType::boolean || view.output_type==PixelType::uint16;
        if(operation==RankOperation::entropy && integral && !zero)throw std::invalid_argument("integral entropy feedback is outside defined coverage");
        if(actual.image_output && view.uint16) {
            if(!integral)throw std::invalid_argument("uint16 image/float output feedback is outside defined coverage");
            bool whole_words=view.output_type==PixelType::uint16 &&
                (reinterpret_cast<std::uintptr_t>(view.output)%2==reinterpret_cast<std::uintptr_t>(view.image)%2);
            for(int axis=0;axis<3;++axis)if(view.shape[axis]>1 && view.output_strides[axis]%2)whole_words=false;
            if(!whole_words && view.bins!=65536 && !zero)throw std::invalid_argument("uint16 mixed-byte feedback requires full histogram bounds");
        }
    }
    auto prepared=view;prepared.serial_alias=feedback;
    run(prepared,operation);
}
std::uint32_t rank_alias_capabilities() noexcept {return 0x1e0u;}
std::uint32_t rank_histogram_capabilities() noexcept {return 0x1fffu;}
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
