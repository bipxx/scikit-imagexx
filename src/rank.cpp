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
    // Percentile mean/sum need whole-bin weighted prefixes. Existing
    // operations allocate/update none of these additional blocks.
    std::vector<std::int64_t> weighted_blocks;
    std::uint64_t root=0;
    std::int64_t sum=0,pop=0;
    explicit Histogram(std::size_t bins,bool extrema,bool prefix=false,bool weighted=false,bool counts_only=false):limit(bins),counts(extrema ? bins:0),bits(extrema && !counts_only ? (bins+63)/64:0),summary(extrema && !counts_only ? (bins+4095)/4096:0),prefix_blocks(prefix ? (bins+63)/64:0),weighted_blocks(weighted ? (bins+63)/64:0){}
    template<bool Prefix,bool CountsOnly=false> void add(std::uint32_t value,std::int64_t count) {
        pop+=count;
        // Opcode16 queries only counts/pop. Keep its full-bin counters and
        // existing bounds check, without maintaining unused extrema or sums.
        if constexpr(CountsOnly) {
            static_assert(!Prefix);
            if(value>=limit)throw std::invalid_argument("aliased rank image exceeded histogram bounds");
            counts[value]+=count;
            return;
        }
        sum+=std::int64_t(value)*count;
        if(value>=limit)throw std::invalid_argument("aliased rank image exceeded histogram bounds");
        if(counts.empty())return;
        const auto before=counts[value];counts[value]+=count;
        if constexpr(Prefix){prefix_blocks[value/64]+=count;if(!weighted_blocks.empty())weighted_blocks[value/64]+=std::int64_t(value)*count;}
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
    std::int64_t four_blocks(const std::vector<std::int64_t>& blocks,std::size_t start)const {
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())return simd::rank_count_prefix(blocks.data()+start,4);
#endif
        return blocks[start]+blocks[start+1]+blocks[start+2]+blocks[start+3];
    }
    std::size_t crossing(double target,bool strict,bool reverse=false)const {
        if(!pop)return 0;
        if(target<=0)return strict ? std::size_t(integer(reverse ? RankOperation::maximum:RankOperation::minimum)):(reverse ? limit-1:0);
        // The original strict scans retain their initialized0 on no crossing.
        if(target>=double(pop))return strict ? 0:std::size_t(integer(reverse ? RankOperation::minimum:RankOperation::maximum));
        const auto hit=[&](std::int64_t value){return strict ? double(value)>target:double(value)>=target;};
        std::int64_t cumulative=0;
        if(!reverse) {
            std::size_t block=0;
            while(block<prefix_blocks.size()) {
                if(block+4<=prefix_blocks.size()) {
                    const auto chunk=four_blocks(prefix_blocks,block);
                    if(!hit(cumulative+chunk)){cumulative+=chunk;block+=4;continue;}
                }
                const auto next=cumulative+prefix_blocks[block];
                if(hit(next))for(auto i=block*64;i<std::min(limit,(block+1)*64);++i){cumulative+=counts[i];if(hit(cumulative))return i;}
                cumulative=next;++block;
            }
        } else {
            auto block=prefix_blocks.size();
            while(block) {
                if(block>=4) {
                    const auto chunk=four_blocks(prefix_blocks,block-4);
                    if(!hit(cumulative+chunk)){cumulative+=chunk;block-=4;continue;}
                }
                --block;const auto next=cumulative+prefix_blocks[block];
                if(hit(next))for(auto i=std::min(limit,(block+1)*64);i>block*64;){--i;cumulative+=counts[i];if(hit(cumulative))return i;}
                cumulative=next;
            }
        }
        return 0;
    }
    struct Aggregate {std::int64_t count=0,weighted=0;};
    Aggregate range(std::size_t lo,std::size_t hi)const {
        Aggregate result;
        const auto edge=[&](std::size_t begin,std::size_t end) {
#ifdef SKIMAGEXX_AVX2
            if(has_avx2()) {
                simd::rank_weighted_prefix(counts.data()+begin,end-begin,begin,result.count,result.weighted);
                return;
            }
#endif
            for(auto i=begin;i<end;++i){result.count+=counts[i];result.weighted+=counts[i]*std::int64_t(i);}
        };
        const auto first=std::min(hi,((lo+63)/64)*64);
        edge(lo,first);lo=first;
        const auto blocks=(hi-lo)/64;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2()) {
            result.count+=simd::rank_count_prefix(prefix_blocks.data()+lo/64,blocks);
            result.weighted+=simd::rank_count_prefix(weighted_blocks.data()+lo/64,blocks);
        } else
#endif
        for(std::size_t b=lo/64;b<lo/64+blocks;++b){result.count+=prefix_blocks[b];result.weighted+=weighted_blocks[b];}
        lo+=blocks*64;edge(lo,hi);return result;
    }
    std::int64_t bilateral(RankOperation op,std::int64_t center,std::ptrdiff_t s0,std::ptrdiff_t s1)const {
        // Upstream returns before evaluating either possibly extreme radius.
        if(!pop)return 0;
        // Clip the strict predicate without forming center +/- signed radius.
        // Every subtraction below is first bounded to the histogram domain.
        const auto last=std::int64_t(limit-1);
        const auto lo=s1>center ? std::size_t{0} : s1<=center-last ? limit:std::size_t(center-s1+1);
        const auto hi=s0<=-center ? std::size_t{0} : s0>=std::int64_t(limit)-center ? limit:std::size_t(center+s0);
        if(lo>=hi)return 0;
        const auto values=range(lo,hi);
        if(op==RankOperation::pop_bilateral)return values.count;
        // Feedback histograms can contain a zero selected count and a nonzero
        // signed weighted sum. Cython's guard still writes zero in that case.
        return values.count ? (op==RankOperation::mean_bilateral ? values.weighted/values.count:values.weighted):0;
    }
    Aggregate membership_prefix(double target,bool inclusive)const {
        if(target<0 || (!inclusive && target==0))return {};
        if(inclusive && target>=double(pop))return {pop,weighted_blocks.empty() ? 0:sum};
        const auto take=[&](std::int64_t value){return inclusive ? double(value)<=target:double(value)<target;};
        Aggregate result;std::size_t block=0;
        while(block<prefix_blocks.size()) {
            if(block+4<=prefix_blocks.size()) {
                const auto chunk=four_blocks(prefix_blocks,block);
                if(take(result.count+chunk)){result.count+=chunk;if(!weighted_blocks.empty())result.weighted+=four_blocks(weighted_blocks,block);block+=4;continue;}
            }
            if(take(result.count+prefix_blocks[block])) {
                result.count+=prefix_blocks[block];if(!weighted_blocks.empty())result.weighted+=weighted_blocks[block];++block;continue;
            }
            for(auto i=block*64;i<std::min(limit,(block+1)*64);++i) {
                if(!take(result.count+counts[i]))return result;
                result.count+=counts[i];if(!weighted_blocks.empty())result.weighted+=counts[i]*std::int64_t(i);
            }
            ++block;
        }
        return result;
    }
    Aggregate selected(double p0,double p1)const {
        if(!pop)return {};
        if(p0==0 && p1==1)return {pop,sum};
        const auto lower=membership_prefix(p0*double(pop),false),upper=membership_prefix(p1*double(pop),true);
        return {upper.count-lower.count,upper.weighted-lower.weighted};
    }
    std::int64_t percentile_integer(RankOperation op,std::int64_t center,double p0,double p1)const {
        if(!pop)return 0;
        if(op==RankOperation::percentile)return p0==1 ? integer(RankOperation::maximum):std::int64_t(crossing(p0*double(pop),true));
        if(op==RankOperation::threshold_percentile)return std::int64_t(limit-1)*(center>=std::int64_t(crossing(p0*double(pop),false)));
        if(op==RankOperation::mean_percentile || op==RankOperation::pop_percentile || op==RankOperation::sum_percentile) {
            const auto values=selected(p0,p1);
            if(op==RankOperation::pop_percentile)return values.count;
            return values.count ? (op==RankOperation::mean_percentile ? values.weighted/values.count:values.weighted):0;
        }
        const bool strict=op!=RankOperation::gradient_percentile;
        const auto low=std::int64_t(crossing(p0*double(pop),strict)),high=std::int64_t(crossing((1.-p1)*double(pop),strict,true));
        if(op==RankOperation::gradient_percentile)return high-low;
        if(op==RankOperation::autolevel_percentile)return high>low ? std::int64_t(limit-1)*(std::min(std::max(low,center),high)-low)/(high-low):high-low;
        return high-center<center-low ? high:low;
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
    std::int64_t majority()const {
        if(!pop)return 0;
        std::int64_t votes=counts[0],candidate=0;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2()) {
            std::uint32_t indices[4];std::size_t n=0;
            occupied([&](std::size_t index,std::int64_t){
                if(!index)return true;
                indices[n++]=std::uint32_t(index);
                if(n==4){simd::rank_majority4(counts.data(),indices,votes,candidate);n=0;}
                return true;
            });
            for(std::size_t i=0;i<n;++i)if(counts[indices[i]]>votes){votes=counts[indices[i]];candidate=indices[i];}
            return candidate;
        }
#endif
        // Independent source decision; do not alias modal's signed-feedback
        // admission or its initialization/tie semantics into this operation.
        occupied([&](std::size_t index,std::int64_t count){if(index && count>votes){votes=count;candidate=std::int64_t(index);}return true;});
        return candidate;
    }
    std::int64_t otsu()const {
        if(!pop)return 0;
        std::int64_t q1=counts[0],mu1=0,candidate=0;double best=0.;
        const auto score=[&](std::size_t index,std::int64_t q,std::int64_t moment){
            if(q==pop)return false; // source breaks before updating mu1
            q1=q;mu1=moment;
            const auto mu2=sum-mu1;
            const double remaining=double(pop)-double(q1);
            // In pinned Cython mu2*q1 is an INT64 product BEFORE promotion.
            // Preserve this expression tree and ascending FP evaluation order.
            const double t=remaining*double(mu1)-double(mu2*q1);
            const double sigma=(t*t)/(double(q1)*remaining);
            if(sigma>best){best=sigma;candidate=std::int64_t(index);}
            return true;
        };
#ifdef SKIMAGEXX_AVX2
        if(has_avx2()) {
            std::uint32_t indices[4];std::size_t n=0;bool stopped=false;
            occupied([&](std::size_t index,std::int64_t){
                if(!index)return true;
                indices[n++]=std::uint32_t(index);
                if(n==4) {
                    std::int64_t q[4],moment[4];
                    simd::rank_decision_prefix4(counts.data(),indices,q1,mu1,q,moment);
                    for(std::size_t lane=0;lane<4;++lane)if(!score(indices[lane],q[lane],moment[lane])){stopped=true;return false;}
                    n=0;
                }
                return true;
            });
            if(!stopped)for(std::size_t i=0;i<n;++i){const auto index=indices[i];const auto count=counts[index];if(!score(index,q1+count,mu1+std::int64_t(index)*count))break;}
            return candidate;
        }
#endif
        occupied([&](std::size_t index,std::int64_t count){return !index || score(index,q1+count,mu1+std::int64_t(index)*count);});
        return candidate;
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
template<class T,bool Prefix,bool CountsOnly=false> void update(Histogram& hist,const RankView& v,const std::vector<Run>& runs,std::ptrdiff_t p,std::ptrdiff_t r,std::ptrdiff_t c,int sign,int row_adjust=0,int col_adjust=0) {
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
                if(simd::rank_uniform(data+i*sizeof(T),sizeof(T)==2,value))hist.add<Prefix,CountsOnly>(value,std::int64_t(width)*sign);
                else for(std::size_t j=0;j<width;++j)hist.add<Prefix,CountsOnly>(read(data+(i+j)*sizeof(T)),sign);
            }
        }
#endif
        for(;i<n;++i)if(!v.mask || v.mask[index+i])hist.add<Prefix,CountsOnly>(read(data+i*sizeof(T)),sign);
    }
}
template<class T,class Value> void store(unsigned char* out,Value value) {
    T result;
    if constexpr(std::is_integral_v<T> && std::is_floating_point_v<Value>)result=T(std::int64_t(value));else result=T(value);
    std::memcpy(out,&result,sizeof(result));
}
template<class T,bool Prefix=false,bool CountsOnly=false> void process(const RankView& v,RankOperation op,const Plan& prepared,std::size_t begin,std::size_t end,const std::vector<double>& logarithms={},std::size_t depth=0,bool domain_only=false,double p0=0.,double p1=1.,std::ptrdiff_t s0=0,std::ptrdiff_t s1=0) {
    while(begin<end) {
        const auto p=begin/v.shape[1],start=begin%v.shape[1],finish=std::min(v.shape[1],start+end-begin);
        const bool weighted=op==RankOperation::mean_percentile || op==RankOperation::subtract_mean_percentile || op==RankOperation::sum_percentile || (int(op)>=26 && int(op)<=28);
        Histogram hist(v.bins,op!=RankOperation::mean && op!=RankOperation::sum && op!=RankOperation::pop && op!=RankOperation::threshold && op!=RankOperation::subtract_mean,Prefix,weighted,CountsOnly);
        auto r=std::ptrdiff_t(start),c=std::ptrdiff_t{0};
        update<T,Prefix,CountsOnly>(hist,v,prepared.initial,p,r,c,1);
        const auto write=[&](){
            auto* out=static_cast<unsigned char*>(v.output)+std::ptrdiff_t(p)*v.output_strides[0]+r*v.output_strides[1]+c*v.output_strides[2];
            const auto emit=[&](auto value){
                if(domain_only){const double truncated=std::trunc(double(value));const double maximum=v.output_type==PixelType::uint16 ? 65535.:255.;if(!std::isfinite(truncated) || truncated<0 || truncated>maximum)throw std::invalid_argument(op==RankOperation::subtract_mean_percentile ? "subtract_mean_percentile unsigned conversion outside defined coverage":"subtract_mean unsigned conversion outside defined coverage");return;}
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
            else if(op==RankOperation::otsu)emit(hist.otsu());
            else if(op==RankOperation::majority)emit(hist.majority());
            else if(int(op)>=9) {
                // Radius hooks may change uint16 pixels after bins/buffers
                // were captured. A zero bilateral population never indexes
                // those bins upstream, so emit before the center-bin guard.
                if(int(op)>=26 && int(op)<=28 && !hist.pop){emit(std::int64_t{0});return;}
                T pixel;const auto index=(std::size_t(p)*v.shape[1]+std::size_t(r))*v.shape[2]+std::size_t(c);
                std::memcpy(&pixel,static_cast<const unsigned char*>(v.image)+index*sizeof(T),sizeof(T));
                // Bilateral uses g only in a clipped interval comparison,
                // never as hist[g]. A masked center may exceed captured bins.
                if(pixel>=v.bins && (int(op)<26 || int(op)>28))throw std::invalid_argument("rank center exceeded histogram bounds");
                if(int(op)>=26 && int(op)<=28)emit(hist.bilateral(op,pixel,s0,s1));
                else if(int(op)>=17) {
                    if(op==RankOperation::subtract_mean_percentile) {
                        const auto values=hist.selected(p0,p1);
                        emit(values.count ? double(std::int64_t(pixel)-values.weighted/values.count)*.5+double(v.bins/2):0.);
                    } else emit(hist.percentile_integer(op,pixel,p0,p1));
                }
                else if(op==RankOperation::threshold)emit(hist.pop && double(pixel)>double(hist.sum)/double(hist.pop) ? std::int64_t{1}:std::int64_t{0});
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
                update<T,Prefix,CountsOnly>(hist,v,prepared.enter[2],p,r,c,1);update<T,Prefix,CountsOnly>(hist,v,prepared.leave[2],p,r,c,-1,0,-1);write();
            }
            else for(c=std::ptrdiff_t(v.shape[2])-2;c>=0;--c) {
                update<T,Prefix,CountsOnly>(hist,v,prepared.leave[2],p,r,c,1);update<T,Prefix,CountsOnly>(hist,v,prepared.enter[2],p,r,c,-1,0,1);write();
            }
            c=east ? std::ptrdiff_t(v.shape[2])-1:0;
            if(++r>=std::ptrdiff_t(finish))break;
            update<T,Prefix,CountsOnly>(hist,v,prepared.enter[1],p,r,c,1);update<T,Prefix,CountsOnly>(hist,v,prepared.leave[1],p,r,c,-1,-1,0);write();east=!east;
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
// A zero-based order statistic and the count/weight strictly below its bin.
// Updates at the current bin are reflected in counts; only earlier bins
// change the prefix. Seeking starts from the previous crossing.
struct QuantileCursor {
    unsigned bin=0;
    std::uint32_t below=0;
    std::int64_t weight=0;

    template<bool Weighted> void seek(const std::uint32_t* counts,std::uint32_t rank) {
        while(below>rank) {
            --bin;
            below-=counts[bin];
            if constexpr(Weighted)weight-=std::int64_t(bin)*counts[bin];
        }
        while(below+counts[bin]<=rank) {
            below+=counts[bin];
            if constexpr(Weighted)weight+=std::int64_t(bin)*counts[bin];
            ++bin;
        }
    }
};

template<RankOperation Op> struct ByteHistogram {
    using R=RankOperation;
    static constexpr bool selected=Op==R::mean_percentile || Op==R::pop_percentile ||
        Op==R::sum_percentile || Op==R::subtract_mean_percentile;
    static constexpr bool bilateral=Op==R::mean_bilateral || Op==R::sum_bilateral || Op==R::pop_bilateral;
    static constexpr bool weighted=(selected && Op!=R::pop_percentile) || Op==R::mean_bilateral || Op==R::sum_bilateral;
    static constexpr bool two_cursors=selected || Op==R::gradient_percentile ||
        Op==R::autolevel_percentile || Op==R::enhance_contrast_percentile;
    static constexpr bool quantile=two_cursors || Op==R::median || Op==R::percentile || Op==R::threshold_percentile;
    static constexpr bool sum_needed=(selected && weighted) || Op==R::sum || Op==R::mean ||
        Op==R::threshold || Op==R::subtract_mean || Op==R::otsu;
    static constexpr bool occupied=!quantile && !bilateral && Op!=R::equalize && !(
        Op==R::mean || Op==R::sum || Op==R::pop || Op==R::threshold || Op==R::subtract_mean);
    static constexpr bool blocks=Op==R::equalize || bilateral;
    static constexpr bool bins_needed=occupied || blocks || quantile;
    std::array<std::uint32_t,bins_needed ? 256:0> counts{};
    std::array<std::uint64_t,occupied ? 4:0> bits{};
    std::array<std::uint32_t,blocks ? 4:0> populations{};
    std::array<std::int64_t,bilateral && weighted ? 4:0> weights{};
    std::array<QuantileCursor,two_cursors ? 2:quantile ? 1:0> cursors{};
    std::uint32_t pop=0;
    std::int64_t sum=0;

    template<bool Add> void update(unsigned value) {
        constexpr int delta=Add ? 1:-1;
        if constexpr(sum_needed)sum+=std::int64_t(value)*delta;
        if constexpr(bins_needed)counts[value]+=delta;
        if constexpr(occupied) {
            const auto bit=std::uint64_t{1}<<(value%64);
            if constexpr(Add)bits[value/64]|=bit;
            else if(!counts[value])bits[value/64]&=~bit;
        }
        if constexpr(blocks)populations[value/64]+=delta;
        if constexpr(bilateral && weighted)weights[value/64]+=std::int64_t(value)*delta;
        for(auto& cursor:cursors) {
            const auto change=delta*int(value<cursor.bin);
            cursor.below+=change;
            if constexpr(selected && weighted)cursor.weight+=std::int64_t(value)*change;
        }
    }

    template<class F> void visit(F use) const {
        for(unsigned block=0;block<4;++block) {
            auto word=bits[block];
            while(word) {
                const auto bin=block*64+first(word);
                if(!use(bin,counts[bin]))return;
                word&=word-1;
            }
        }
    }
    unsigned minimum() const {
        for(unsigned block=0;block<4;++block)if(bits[block])return block*64+first(bits[block]);
        return 0;
    }
    unsigned maximum() const {
        for(unsigned block=4;block--;)if(bits[block])return block*64+last(bits[block]);
        return 0;
    }
    std::int64_t noise(unsigned g) const {
        if(counts[g])return 0;
        const unsigned block=g/64;
        const auto split=~std::uint64_t{0}<<(g%64);
        unsigned lower=0,upper=255;
        if(const auto word=bits[block]&~split)lower=block*64+last(word);
        else for(auto b=block;b--;)if(bits[b]){lower=b*64+last(bits[b]);break;}
        if(const auto word=bits[block]&split)upper=block*64+first(word);
        else for(auto b=block+1;b<4;++b)if(bits[b]){upper=b*64+first(bits[b]);break;}
        return std::min(g-lower,upper-g);
    }
    Histogram::Aggregate prefix(unsigned end) const {
        Histogram::Aggregate result;
        const auto blocks_end=end/64;
        for(unsigned block=0;block<blocks_end;++block) {
            result.count+=populations[block];
            if constexpr(weighted)result.weighted+=weights[block];
        }
        for(unsigned bin=blocks_end*64;bin<end;++bin) {
            result.count+=counts[bin];
            if constexpr(weighted)result.weighted+=std::int64_t(bin)*counts[bin];
        }
        return result;
    }
    // Negative ranks encode the constant zero result of a scan with no
    // crossing. Inclusive membership takes all bins through a CDF tie.
    static std::array<std::int64_t,2> ranks(std::uint32_t pop,double p0,double p1) {
        if constexpr(Op==R::median)return {pop/2,0};
        else if constexpr(Op==R::percentile)return {p0==1 ? std::int64_t(pop)-1:std::int64_t(p0*pop),0};
        else if constexpr(Op==R::threshold_percentile)return {std::int64_t(std::ceil(p0*pop))-1,0};
        else if constexpr(selected)return {std::int64_t(std::ceil(p0*pop))-1,std::int64_t(p1*pop)};
        else if constexpr(Op==R::gradient_percentile)
            return {std::int64_t(std::ceil(p0*pop))-1,pop-std::int64_t(std::ceil((1.-p1)*pop))};
        else if constexpr(two_cursors)
            return {std::int64_t(p0*pop),std::int64_t(pop)-1-std::int64_t((1.-p1)*pop)};
        else return {};
    }
    std::int64_t crossing(unsigned cursor,std::int64_t rank,std::uint32_t population) {
        if(rank<0)return 0;
        if(rank>=population)return Op==R::gradient_percentile && cursor==1 ? 255:0;
        cursors[cursor].template seek<false>(counts.data(),std::uint32_t(rank));
        return cursors[cursor].bin;
    }
    Histogram::Aggregate membership(unsigned cursor,std::int64_t rank,std::uint32_t population) {
        if(rank<0)return {};
        if(rank>=population)return {population,sum};
        auto& current=cursors[cursor];
        current.template seek<weighted>(counts.data(),std::uint32_t(rank));
        return {current.below,current.weight};
    }

    template<bool Interior>
    auto value(std::uint32_t population,std::int64_t g,const std::array<std::int64_t,2>& rank,
               const std::vector<double>& logarithms,std::ptrdiff_t s0,std::ptrdiff_t s1) {
        if constexpr(Op==R::mean || Op==R::subtract_mean) {
            if constexpr(!Interior)if(!population)return 0.;
            const double mean=double(sum)/double(population);
            if constexpr(Op==R::mean)return mean;
            else return ((double(g)-mean)/2.+128.)-1.;
        } else if constexpr(Op==R::equalize) {
            if constexpr(!Interior)if(!population)return 0.;
            return double(255*prefix(unsigned(g)+1).count)/double(population);
        } else if constexpr(Op==R::entropy || Op==R::geometric_mean) {
            double result=0.;
            if constexpr(!Interior)if(!population)return result;
            visit([&](unsigned bin,std::uint32_t count) {
                if constexpr(Op==R::entropy) {
                    const double p=double(count)/double(population);
                    result-=p*std::log(p)/0.6931471805599453;
                } else result+=double(count)*logarithms[bin];
                return true;
            });
            if constexpr(Op==R::geometric_mean) {
                const double mean=std::exp(result/double(population))-1.;
                return double(std::int64_t(mean>0 ? mean+.5:mean-.5));
            } else return result;
        } else if constexpr(selected || bilateral) {
            Histogram::Aggregate lo,hi;
            if(Interior || population) {
                if constexpr(selected) {
                    lo=membership(0,rank[0],population);
                    hi=membership(1,rank[1],population);
                } else {
                    const auto left=s1>g ? 0:s1<=g-255 ? 256:unsigned(g-s1+1);
                    const auto right=s0<=-g ? 0:s0>=256-g ? 256:unsigned(g+s0);
                    if(left<right){lo=prefix(left);hi=prefix(right);}
                }
            }
            const auto count=hi.count-lo.count,weight=hi.weighted-lo.weighted;
            if constexpr(Op==R::pop_percentile || Op==R::pop_bilateral)return count;
            else if constexpr(Op==R::subtract_mean_percentile)return count ? double(g-weight/count)*.5+128.:0.;
            else if constexpr(Op==R::mean_percentile || Op==R::mean_bilateral)return count ? weight/count:0;
            else return count ? weight:0;
        } else if constexpr(Op==R::noise_filter) {
            return noise(unsigned(g));
        } else {
            if constexpr(!Interior)if(!population)return std::int64_t{0};
            if constexpr(Op==R::pop)return std::int64_t(population);
            else if constexpr(Op==R::sum)return sum;
            else if constexpr(Op==R::threshold)return std::int64_t(double(g)>double(sum)/double(population));
            else if constexpr(Op==R::threshold_percentile)return std::int64_t(255)*(g>=crossing(0,rank[0],population));
            else if constexpr(Op==R::median || Op==R::percentile)return crossing(0,rank[0],population);
            else if constexpr(Op==R::minimum)return std::int64_t(minimum());
            else if constexpr(Op==R::maximum)return std::int64_t(maximum());
            else if constexpr(Op==R::modal || Op==R::majority) {
                std::int64_t mode=0;std::uint32_t votes=0;
                visit([&](unsigned bin,std::uint32_t count){if(count>votes){votes=count;mode=bin;}return true;});
                return mode;
            } else if constexpr(Op==R::otsu) {
                std::int64_t q=counts[0],moment=0,result=0;double best=0.;
                visit([&](unsigned bin,std::uint32_t count) {
                    if(!bin)return true;
                    q+=count;
                    if(q==population)return false;
                    moment+=std::int64_t(bin)*count;
                    const double remaining=double(population)-double(q);
                    const double t=remaining*double(moment)-double((sum-moment)*q);
                    const double score=(t*t)/(double(q)*remaining);
                    if(score>best){best=score;result=bin;}
                    return true;
                });
                return result;
            } else {
                std::int64_t lo,hi;
                if constexpr(quantile){lo=crossing(0,rank[0],population);hi=crossing(1,rank[1],population);}
                else {lo=minimum();hi=maximum();}
                if constexpr(Op==R::gradient || Op==R::gradient_percentile)return hi-lo;
                else if constexpr(Op==R::autolevel)return hi>lo ? 255*(g-lo)/(hi-lo):0;
                else if constexpr(Op==R::autolevel_percentile)
                    return hi>lo ? 255*(std::min(std::max(lo,g),hi)-lo)/(hi-lo):hi-lo;
                else return hi-g<g-lo ? hi:lo;
            }
        }
    }
};

struct BytePlan {
    std::vector<std::ptrdiff_t> enter,leave;
    std::uint32_t population=0;
};
BytePlan byte_plan(const RankView& v,const Plan& prepared,RankOperation op) {
    BytePlan result;
    if(op==RankOperation::windowed_histogram || v.uint16 || v.mask || v.serial_alias || v.bins!=256 ||
       v.shape[0]!=1 || v.footprint_shape[0]!=1 ||
       v.footprint_shape[1]>v.shape[1] || v.footprint_shape[2]>v.shape[2] ||
       v.footprint_shape[1]*v.footprint_shape[2]>std::size_t(std::numeric_limits<std::int32_t>::max()) ||
       !independent_stores(v))return result;
    const auto pixels=v.shape[1]*v.shape[2],footprint=v.footprint_shape[1]*v.footprint_shape[2];
    if(overlaps(v,v.image,pixels) || overlaps(v,v.footprint,footprint))return result;
    for(const auto& run:prepared.initial)result.population+=std::uint32_t(run.count);
    if(!result.population)return result;
    const auto stride=std::ptrdiff_t(v.shape[2]);
    // Each horizontal run contributes one entering and one leaving endpoint.
    // Express both relative to the new center for an eastward step.
    for(const auto& run:prepared.enter[2])result.enter.push_back(run.row*stride+run.col);
    for(const auto& run:prepared.leave[2])result.leave.push_back(run.row*stride+run.col-1);
    return result;
}

template<RankOperation Op>
void process_bytes(const RankView& v,const Plan& prepared,const BytePlan& fast,std::size_t begin,std::size_t end,
                   const std::vector<double>& logarithms,double p0,double p1,std::ptrdiff_t s0,std::ptrdiff_t s1) {
    if(begin==end)return;
    ByteHistogram<Op> hist;
    const auto* image=static_cast<const std::uint8_t*>(v.image);
    const auto rows=std::ptrdiff_t(v.shape[1]),cols=std::ptrdiff_t(v.shape[2]);
    const auto top=v.center[1],bottom=rows-std::ptrdiff_t(v.footprint_shape[1])+top+1;
    const auto left=v.center[2],right=cols-std::ptrdiff_t(v.footprint_shape[2])+left;
    const auto fixed=hist.ranks(fast.population,p0,p1);
    auto r=std::ptrdiff_t(begin),c=std::ptrdiff_t{0};
    const auto clipped=[&](auto add,const std::vector<Run>& runs,int dy,int dx) {
        for(const auto& run:runs) {
            const auto row=r+run.row+dy,col=c+run.col+dx;
            if(row<0 || row>=rows)continue;
            const auto lo=std::max(std::ptrdiff_t{0},-col);
            const auto hi=std::min(std::ptrdiff_t(run.count),cols-col);
            if(lo>=hi)continue;
            hist.pop+=add ? std::uint32_t(hi-lo):-std::uint32_t(hi-lo);
            if constexpr(Op!=RankOperation::pop)
                for(auto i=lo;i<hi;++i)hist.template update<add.value>(image[row*cols+(col+i)]);
        }
    };
    constexpr std::true_type add{};
    constexpr std::false_type remove{};
    const auto write=[&](auto interior,auto population,const auto& ranks) {
        auto* out=static_cast<unsigned char*>(v.output)+r*v.output_strides[1]+c*v.output_strides[2];
        const auto value=hist.template value<interior.value>(population,image[r*cols+c],ranks,logarithms,s0,s1);
        switch(v.output_type) {
            case PixelType::boolean:case PixelType::uint8:store<std::uint8_t>(out,value);break;
            case PixelType::uint16:store<std::uint16_t>(out,value);break;
            case PixelType::float32:store<float>(out,value);break;
            case PixelType::float64:store<double>(out,value);break;
            default:break;
        }
    };
    const auto edge=[&](){write(remove,hist.pop,hist.ranks(hist.pop,p0,p1));};
    const auto sweep=[&](auto east) {
        constexpr int step=east.value ? 1:-1;
        const auto stop=east ? cols:-1;
        const auto first=east ? left+1:right-1,last=east ? right:left;
        const auto move=[&]() {
            clipped(add,east ? prepared.enter[2]:prepared.leave[2],0,0);
            clipped(remove,east ? prepared.leave[2]:prepared.enter[2],0,-step);
            edge();
        };
        c+=step;
        if(r>=top && r<bottom && right>left) {
            for(;c!=first;c+=step)move();
            for(;east ? c<=last:c>=last;c+=step) {
                if constexpr(Op!=RankOperation::pop) {
                    const auto* center=image+r*cols+c+(east ? 0:1);
                    for(std::size_t i=0;i<fast.enter.size();++i) {
                        hist.template update<true>(center[(east ? fast.enter:fast.leave)[i]]);
                        hist.template update<false>(center[(east ? fast.leave:fast.enter)[i]]);
                    }
                }
                write(add,fast.population,fixed);
            }
        }
        for(;c!=stop;c+=step)move();
        c-=step;
    };
    clipped(add,prepared.initial,0,0);
    for(bool east=true;;east=!east) {
        edge();
        if(east)sweep(add);else sweep(remove);
        if(++r>=std::ptrdiff_t(end))break;
        clipped(add,prepared.enter[1],0,0);
        clipped(remove,prepared.leave[1],-1,0);
    }
}

void byte_dispatch(const RankView& v,RankOperation op,const Plan& prepared,const BytePlan& fast,
                   std::size_t begin,std::size_t end,const std::vector<double>& logs={},
                   double p0=0.,double p1=1.,std::ptrdiff_t s0=0,std::ptrdiff_t s1=0) {
    switch(op) {
#define BYTE_CASE(name) case RankOperation::name: return process_bytes<RankOperation::name>(v,prepared,fast,begin,end,logs,p0,p1,s0,s1);
        BYTE_CASE(minimum) BYTE_CASE(maximum) BYTE_CASE(mean) BYTE_CASE(sum) BYTE_CASE(pop)
        BYTE_CASE(gradient) BYTE_CASE(median) BYTE_CASE(modal) BYTE_CASE(entropy) BYTE_CASE(autolevel)
        BYTE_CASE(equalize) BYTE_CASE(enhance_contrast) BYTE_CASE(threshold) BYTE_CASE(geometric_mean)
        BYTE_CASE(noise_filter) BYTE_CASE(subtract_mean) BYTE_CASE(autolevel_percentile)
        BYTE_CASE(enhance_contrast_percentile) BYTE_CASE(gradient_percentile) BYTE_CASE(mean_percentile)
        BYTE_CASE(pop_percentile) BYTE_CASE(subtract_mean_percentile) BYTE_CASE(threshold_percentile)
        BYTE_CASE(sum_percentile) BYTE_CASE(percentile) BYTE_CASE(mean_bilateral)
        BYTE_CASE(pop_bilateral) BYTE_CASE(sum_bilateral) BYTE_CASE(otsu) BYTE_CASE(majority)
#undef BYTE_CASE
        case RankOperation::windowed_histogram:break;
    }
}

void run(const RankView& view,RankOperation operation) {
    const auto prepared=plan(view);const auto rows=view.shape[0]*view.shape[1];
    const auto fast=byte_plan(view,prepared,operation);
    const auto workers=view.serial_alias || !independent_stores(view) ? 1:detail::active_workers(rows*view.shape[2]);
    const auto run=[&](auto begin,auto end){
        if(fast.population)return byte_dispatch(view,operation,prepared,fast,begin,end);
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
    // New percentile entries can certify a full typed histogram range without
    // warming the entire input a second time. Existing R4 validation/log order
    // stays unchanged; narrower u16 capacities still require a prewrite scan.
    if(int(op)>=17 && v.bins>=(v.uint16 ? 65536u:256u))return logarithms;
    const auto* image=static_cast<const unsigned char*>(v.image);
    for(std::size_t i=0;i<pixels;++i){std::uint16_t value;if(v.uint16)std::memcpy(&value,image+2*i,2);else value=image[i];if(value>=v.bins)throw std::invalid_argument("rank image exceeded histogram bounds");if(!seen.empty())seen[value]=1;}
    for(std::size_t i=0;i<seen.size();++i)if(seen[i])logarithms[i]=std::log(double(i+1));
    return logarithms;
}
void statistics_run(const RankView& view,RankOperation operation,std::size_t depth) {
    const auto logarithms=statistics_validate(view,operation,depth);const auto prepared=plan(view);
    const auto fast=byte_plan(view,prepared,operation);
    const auto rows=view.shape[0]*view.shape[1];
    const auto invoke=[&](std::size_t begin,std::size_t end,bool check){
        if(!check && fast.population)return byte_dispatch(view,operation,prepared,fast,begin,end,logarithms);
        if(operation==RankOperation::windowed_histogram) {
            if(view.uint16)process<std::uint16_t,false,true>(view,operation,prepared,begin,end,logarithms,depth,check);
            else process<std::uint8_t,false,true>(view,operation,prepared,begin,end,logarithms,depth,check);
        } else {
            if(view.uint16)process<std::uint16_t>(view,operation,prepared,begin,end,logarithms,depth,check);
            else process<std::uint8_t>(view,operation,prepared,begin,end,logarithms,depth,check);
        }
    };
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
void percentile_run(const RankView& view,RankOperation operation,double p0,double p1) {
    if(view.shape[0]!=1 || view.footprint_shape[0]!=1 || view.center[0]!=0)throw std::invalid_argument("rank percentile requires 2D input");
    statistics_validate(view,operation,0);
    const auto fp=view.footprint_shape[1]*view.footprint_shape[2];
    // Original C population is double and integer weighted sums must also
    // remain defined during enter-before-leave rolling updates.
    if(fp>(std::uint64_t{1}<<52) || fp>std::size_t(std::numeric_limits<std::int64_t>::max())/(2*(view.uint16 ? 65535:255)))throw std::overflow_error("rank percentile population or sum outside defined coverage");
    const auto prepared=plan(view);const auto rows=view.shape[1];
    const auto fast=byte_plan(view,prepared,operation);
    const auto invoke=[&](std::size_t begin,std::size_t end,bool check){
        if(!check && fast.population)return byte_dispatch(view,operation,prepared,fast,begin,end,{},p0,p1);
        if(view.uint16)process<std::uint16_t,true>(view,operation,prepared,begin,end,{},0,check,p0,p1);
        else process<std::uint8_t,true>(view,operation,prepared,begin,end,{},0,check,p0,p1);
    };
    // This family has no ordinary subtract_mean -1 offset. Its full u16
    // result domain is defined; narrow destinations need exact certification.
    if(operation==RankOperation::subtract_mean_percentile && view.bins>256 && (view.output_type==PixelType::boolean || view.output_type==PixelType::uint8))invoke(0,rows,true);
    const auto workers=detail::active_workers(rows*view.shape[2]);
    if(workers==1){invoke(0,rows,false);return;}
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{invoke(begin,end,false);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
}
std::uint32_t rank_percentile_capabilities() noexcept {return 0x3fe0000u;}
std::uint32_t rank_bilateral_capabilities() noexcept {return 0x1c000000u;}
std::uint32_t rank_decisions_capabilities() noexcept {return 0x60000000u;}
void rank_decisions_into(const RankView& view,RankOperation op) {
    if(op!=RankOperation::otsu && op!=RankOperation::majority)throw std::invalid_argument("invalid rank decisions operation");
    validate(view,RankOperation::minimum);
    const auto pixels=view.shape[0]*view.shape[1]*view.shape[2];
    const auto footprint_pixels=view.footprint_shape[0]*view.footprint_shape[1]*view.footprint_shape[2];
    if(overlaps(view,view.image,pixels*(view.uint16 ? 2:1)) || overlaps(view,view.mask,pixels) || overlaps(view,view.footprint,footprint_pixels))throw std::invalid_argument("overlapping buffers are outside rank-decisions-v1 parity");
    const auto prepared=plan(view);std::uint64_t population=0;
    for(const auto& run:prepared.initial)population+=run.count;
    const auto maximum=std::uint64_t(std::numeric_limits<std::int64_t>::max());
    const auto largest=std::uint64_t(std::min(view.bins-1,view.uint16 ? std::size_t{65535}:std::size_t{255}));
    const auto population_bound=std::min(population,std::uint64_t(pixels));
    // Source moment sums and mu2*q1 are signed integers; population itself
    // must also remain exact in the source float64 counter. The quadratic
    // bound is conservative and prewrite, never invented numerical wrapping.
    if(population_bound>(std::uint64_t{1}<<52) || (population_bound && largest && population_bound>maximum/largest/population_bound))throw std::overflow_error("rank decisions moment products outside defined coverage");
    if(population && view.bins<(view.uint16 ? 65536u:256u)) {
        const auto* data=static_cast<const unsigned char*>(view.image);
        // Masked-out centers are not histogram indices. In particular a
        // zero footprint/mask remains valid after safe shift-hook mutations.
        for(std::size_t i=0;i<pixels;++i)if(!view.mask || view.mask[i]) {
            std::uint16_t value;if(view.uint16)std::memcpy(&value,data+2*i,2);else value=data[i];
            if(value>=view.bins)throw std::invalid_argument("rank decisions image exceeded histogram bounds");
        }
    }
    const auto rows=view.shape[0]*view.shape[1];
    const auto fast=byte_plan(view,prepared,op);
    const auto invoke=[&](std::size_t begin,std::size_t end){
        if(fast.population)return byte_dispatch(view,op,prepared,fast,begin,end);
        if(view.uint16)process<std::uint16_t>(view,op,prepared,begin,end);else process<std::uint8_t>(view,op,prepared,begin,end);
    };
    const auto workers=independent_stores(view) ? detail::active_workers(pixels):1;
    if(workers==1){invoke(0,rows);return;}
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{invoke(begin,end);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
void rank_bilateral_into(const RankView& view,RankOperation op,std::ptrdiff_t s0,std::ptrdiff_t s1) {
    if(int(op)<26 || int(op)>28)throw std::invalid_argument("invalid rank bilateral operation");
    validate(view,RankOperation::minimum);
    if(view.shape[0]!=1 || view.footprint_shape[0]!=1 || view.center[0]!=0)throw std::invalid_argument("rank bilateral requires 2D input");
    const auto pixels=view.shape[1]*view.shape[2],fp=view.footprint_shape[1]*view.footprint_shape[2];
    if(overlaps(view,view.footprint,fp))throw std::invalid_argument("rank bilateral footprint/output alias is outside defined coverage");
    const auto feedback=aliases(view);
    bool zero=true;for(std::size_t i=0;i<fp && zero;++i)zero=view.footprint[i]==0;
    if(!zero && view.mask){zero=true;for(std::size_t i=0;i<pixels && zero;++i)zero=view.mask[i]==0;}
    const auto maximum=std::numeric_limits<std::ptrdiff_t>::max();
    if(!zero && (s0<std::ptrdiff_t(view.bins-1)-maximum || s1>maximum-std::ptrdiff_t(view.bins-1)))throw std::invalid_argument("rank bilateral radius arithmetic outside defined coverage");
    if(feedback.image_output) {
        const bool whole=view.output_type==(view.uint16 ? PixelType::uint16:PixelType::uint8) || (!view.uint16 && view.output_type==PixelType::boolean);
        if(!whole || (view.uint16 && (reinterpret_cast<std::uintptr_t>(view.output)%2!=reinterpret_cast<std::uintptr_t>(view.image)%2)))throw std::invalid_argument("rank bilateral mixed-type image feedback is outside defined coverage");
        if(view.uint16)for(int axis=0;axis<3;++axis)if(view.shape[axis]>1 && view.output_strides[axis]%2)throw std::invalid_argument("rank bilateral mixed-byte image feedback is outside defined coverage");
        if(!zero && view.bins!=(view.uint16 ? 65536u:256u))throw std::invalid_argument("rank bilateral image feedback requires full histogram bounds");
    }
    if(feedback.mask_output && view.output_type!=PixelType::uint8 && view.output_type!=PixelType::boolean)throw std::invalid_argument("rank bilateral mixed-type mask feedback is outside defined coverage");
    // L1 counts can grow under enter-before-leave feedback. Bound all update
    // events, including partial SIMD lane sums, before modifying any output.
    const auto magnitude=std::uint64_t(view.uint16 ? 65535:255);
    const auto limit=std::min(std::uint64_t{1}<<50,std::uint64_t(std::numeric_limits<std::int64_t>::max())/(4*magnitude));
    if(fp>limit || pixels>limit/fp)throw std::overflow_error("rank bilateral signed histogram outside defined coverage");
    if(!zero && view.bins<(view.uint16 ? 65536u:256u)) {
        const auto* data=static_cast<const unsigned char*>(view.image);
        // A disjoint mask permanently excludes its zero pixels from every
        // histogram read. Mask/output feedback can re-enable them, so retain
        // the full-value certificate for that existing alias admission.
        for(std::size_t i=0;i<pixels;++i)if(!view.mask || view.mask[i] || feedback.mask_output){std::uint16_t value;if(view.uint16)std::memcpy(&value,data+2*i,2);else value=data[i];if(value>=view.bins)throw std::invalid_argument("rank image exceeded histogram bounds");}
    }
    const auto prepared=plan(view);const auto rows=view.shape[1];
    const auto fast=byte_plan(view,prepared,op);
    const auto invoke=[&](std::size_t begin,std::size_t end){
        if(fast.population)return byte_dispatch(view,op,prepared,fast,begin,end,{},0.,1.,s0,s1);
        if(view.uint16)process<std::uint16_t,true>(view,op,prepared,begin,end,{},0,false,0.,1.,s0,s1);
        else process<std::uint8_t,true>(view,op,prepared,begin,end,{},0,false,0.,1.,s0,s1);
    };
    const auto workers=feedback.image_output || feedback.mask_output || !independent_stores(view) ? 1:detail::active_workers(pixels);
    if(workers==1){invoke(0,rows);return;}
    std::vector<std::exception_ptr> failures(std::size_t(workers),std::exception_ptr{});
    detail::partition(rows,workers,[&](auto begin,auto end,auto id){try{invoke(begin,end);}catch(...){failures[id]=std::current_exception();}});
    for(const auto& failure:failures)if(failure)std::rethrow_exception(failure);
}
void rank_percentile_into(const RankView& view,RankOperation op,double p0,double p1) {
    if(int(op)<17 || int(op)>25)throw std::invalid_argument("invalid rank percentile operation");
    const bool single=op==RankOperation::percentile || op==RankOperation::threshold_percentile;
    if(!std::isfinite(p0) || !std::isfinite(p1) || p0<0 || p0>1 || p1<0 || p1>1 || (!single && p0>p1))throw std::invalid_argument("rank percentile parameters outside defined coverage");
    percentile_run(view,op,p0,p1);
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
