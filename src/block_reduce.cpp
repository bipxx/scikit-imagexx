#include "block_reduce_internal.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <atomic>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx::detail {
namespace {
class Capture {
    std::fenv_t saved_;bool active_,valid_;
public:
    explicit Capture(bool active):active_(active),valid_(!active||std::feholdexcept(&saved_)==0){}
    ~Capture(){if(active_&&valid_)std::fesetenv(&saved_);}
    bool valid()const{return valid_;}
    unsigned flags()const{
        const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
        return ((f&FE_DIVBYZERO)?1u:0u)|((f&FE_OVERFLOW)?2u:0u)|((f&FE_UNDERFLOW)?4u:0u)|((f&FE_INVALID)?8u:0u);
    }
};
std::size_t multiply(std::size_t a,std::size_t b){
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(b&&a>limit/b)throw std::overflow_error("block sum extent overflow");return a*b;
}
struct Plan {std::vector<std::size_t> blocks,strides;std::size_t count=1,volume=1,input_count=1,output_bytes;};
Plan plan(const BlockReduceInput& in,const std::vector<std::size_t>& factors,void* out,const std::vector<std::size_t>& strides){
    const auto rank=in.shape.size();
    if(factors.size()!=rank||strides.size()!=rank)throw std::invalid_argument("one block sum factor and output stride per axis required");
    if(in.half&&in.type!=PixelType::uint16)throw std::invalid_argument("binary16 block sum requires uint16 storage");
    const auto input_bytes=pixel_size(in.type);Plan p;p.output_bytes=in.half?2:in.type==PixelType::float32?4:8;
    p.blocks.resize(rank);p.strides.resize(rank);
    for(std::size_t d=rank;d--;){
        if(!factors[d]||in.shape[d]%factors[d])throw std::invalid_argument("positive block factors must divide padded shape");
        p.blocks[d]=in.shape[d]/factors[d];p.strides[d]=p.input_count;
        p.input_count=multiply(p.input_count,in.shape[d]);p.volume=multiply(p.volume,factors[d]);p.count=multiply(p.count,p.blocks[d]);
    }
    multiply(p.input_count,input_bytes);multiply(p.count,p.output_bytes);
    if(!p.count)return p;
    if(!in.data||!out)throw std::invalid_argument("nonnull block sum buffers required");
    std::vector<std::pair<std::size_t,std::size_t>> order;
    for(std::size_t d=0;d<rank;++d)if(p.blocks[d]>1)order.emplace_back(strides[d],p.blocks[d]);
    std::sort(order.begin(),order.end());auto extent=p.output_bytes;
    for(const auto& axis:order){if(axis.first!=extent)throw std::invalid_argument("dense block sum output strides required");extent=multiply(extent,axis.second);}
    return p;
}
// NumPy's eight-lane/128-element pairwise pattern; BSD notice is retained in
// third_party/numpy-LICENSE.txt. Each block uses logical C traversal, unchanged
// by worker count or SIMD packing across independent output cells.
template<class T,class Get>T pairwise(std::size_t begin,std::size_t count,const Get& get){
    if(count<8){T sum=-T(0);for(std::size_t i=0;i<count;++i)sum=T(sum+get(begin+i));return sum;}
    if(count<=128){
        T lanes[8];for(std::size_t j=0;j<8;++j)lanes[j]=get(begin+j);
        std::size_t i=8;for(;i+8<=count;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=T(lanes[j]+get(begin+i+j));
        T sum=T(T(T(lanes[0]+lanes[1])+T(lanes[2]+lanes[3]))+T(T(lanes[4]+lanes[5])+T(lanes[6]+lanes[7])));
        for(;i<count;++i)sum=T(sum+get(begin+i));return sum;
    }
    auto left=count/2;left-=left%8;const T a=pairwise<T>(begin,left,get),b=pairwise<T>(begin+left,count-left,get);return T(a+b);
}
std::uint16_t half_store(float value){
    std::uint32_t word;std::memcpy(&word,&value,4);const auto encoded=encode_half(value);const auto exponent=word&0x7f800000u;
    if(exponent!=0x7f800000u){
        if((encoded&0x7fffu)==0x7c00u)std::feraiseexcept(FE_OVERFLOW);
        else if(exponent<=0x38000000u&&decode_half(encoded)!=value)std::feraiseexcept(FE_UNDERFLOW);
    }
    return encoded;
}
template<class Input,class Acc,bool Boolean=false,bool Half=false>
unsigned run(const BlockReduceInput& in,const std::vector<std::size_t>& factors,void* out,const std::vector<std::size_t>& strides,const Plan& p){
    if(!p.count)return 0;
    const auto rank=in.shape.size();
    const auto* data=static_cast<const unsigned char*>(in.data);auto* output=static_cast<unsigned char*>(out);
    std::size_t half_segment=1;
    if constexpr(Half)for(std::size_t d=rank;d--;){
        half_segment*=factors[d];if(p.blocks[d]>1)break;
    }
    const bool vectorized=has_avx2();(void)vectorized;std::atomic<unsigned> flags{0};
    partition(p.count,active_workers(p.input_count),[&](auto begin,auto end,auto){
        Capture capture(std::is_floating_point_v<Acc>);if(!capture.valid()){flags.fetch_or(16u,std::memory_order_relaxed);return;}
        for(auto index=begin;index<end;){
            std::size_t rem=index,base=0,target=0;
            for(std::size_t d=rank;d--;){const auto coordinate=rem%p.blocks[d];rem/=p.blocks[d];base+=coordinate*factors[d]*p.strides[d];target+=coordinate*strides[d];}
#ifdef SKIMAGEXX_AVX2
            if constexpr(std::is_same_v<Input,std::uint8_t> && !Boolean){
                if(vectorized&&rank==2&&factors[0]==2&&factors[1]==2&&strides[1]==sizeof(Acc)){
                    const auto remaining=std::min(end-index,p.blocks[1]-index%p.blocks[1]);
                    const auto done=simd::block_reduce_row(data+base,reinterpret_cast<std::uint64_t*>(output+target),in.shape[1],remaining);
                    if(done){index+=done;continue;}
                }
            }
            if constexpr(std::is_floating_point_v<Input>){
                if(vectorized&&rank==2&&strides[1]==sizeof(Acc)){
                    const auto remaining=std::min(end-index,p.blocks[1]-index%p.blocks[1]);
                    const auto done=simd::block_reduce_row(reinterpret_cast<const Input*>(data)+base,reinterpret_cast<Acc*>(output+target),in.shape[1],factors[0],factors[1],remaining);
                    if(done){index+=done;continue;}
                }
            }
#endif
            const auto get=[&](std::size_t position)->Acc{
                auto offset=base;
                for(std::size_t d=rank;d--;){offset+=(position%factors[d])*p.strides[d];position/=factors[d];}
                Input value;std::memcpy(&value,data+offset*sizeof(Input),sizeof(Input));
                if constexpr(Half)return decode_half(value);
                else if constexpr(Boolean)return value!=0;
                else return Acc(value);
            };
            if constexpr(Half){
                // HALF_add stores after each contiguous reduction segment.
                // A nontrivial output axis separates adjacent C-order segments.
                std::uint16_t word=0;
                for(std::size_t begin=0;begin<p.volume;begin+=half_segment)
                    word=half_store(decode_half(word)+pairwise<float>(begin,half_segment,get));
                std::memcpy(output+target,&word,2);
            }else{
                Acc sum=0;
                if constexpr(std::is_floating_point_v<Acc>)sum=Acc(Acc(0)+pairwise<Acc>(0,p.volume,get));
                else for(std::size_t i=0;i<p.volume;++i)sum+=get(i); // unsigned modular arithmetic, including signed input bits.
                std::memcpy(output+target,&sum,sizeof(sum));
            }
            ++index;
        }
        if constexpr(std::is_floating_point_v<Acc>)flags.fetch_or(capture.flags(),std::memory_order_relaxed);
    });
    const auto result=flags.load(std::memory_order_relaxed);
    if(result&16u)throw std::runtime_error("cannot capture block sum floating environment");return result;
}
}
unsigned block_reduce_sum(const BlockReduceInput& in,const std::vector<std::size_t>& factors,void* out,const std::vector<std::size_t>& strides){
    const auto p=plan(in,factors,out,strides);
    if(in.half)return run<std::uint16_t,float,false,true>(in,factors,out,strides,p);
    switch(in.type){
    case PixelType::boolean:return run<std::uint8_t,std::uint64_t,true>(in,factors,out,strides,p);
#define BLOCK(tag,Input) case PixelType::tag:return run<Input,std::uint64_t>(in,factors,out,strides,p);
    BLOCK(uint8,std::uint8_t) BLOCK(int8,std::int8_t) BLOCK(uint16,std::uint16_t) BLOCK(int16,std::int16_t)
    BLOCK(uint32,std::uint32_t) BLOCK(int32,std::int32_t) BLOCK(uint64,std::uint64_t) BLOCK(int64,std::int64_t)
#undef BLOCK
    case PixelType::float32:return run<float,float>(in,factors,out,strides,p);
    case PixelType::float64:return run<double,double>(in,factors,out,strides,p);
    }
    throw std::invalid_argument("unsupported block sum input type");
}
}
