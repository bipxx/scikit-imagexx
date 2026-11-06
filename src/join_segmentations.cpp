#include "skimagexx/join_segmentations.hpp"
#include "skimagexx/component_consumers.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#if defined(__SSE2__) && !defined(SKIMAGEXX_JOIN_SCALAR)
#include <emmintrin.h>
#endif

namespace skimagexx { namespace {
std::uint64_t mask(JoinType t) { return t.bytes==8 ? ~std::uint64_t(0) : (std::uint64_t(1)<<(8*t.bytes))-1; }
void check(JoinType t) {
    if((t.bytes!=1 && t.bytes!=2 && t.bytes!=4 && t.bytes!=8) || (t.floating && t.bytes!=8))
        throw std::invalid_argument("join stage dtype unsupported");
}
std::uint64_t read(const void* p, std::size_t i, JoinType t) {
    std::uint64_t x=0;std::memcpy(&x,static_cast<const char*>(p)+i*t.bytes,t.bytes);
    if(t.is_signed && t.bytes<8 && (x & (std::uint64_t(1)<<(t.bytes*8-1)))) x|=~mask(t);
    return x;
}
void write(void* p,std::size_t i,JoinType t,std::uint64_t x) {
    std::memcpy(static_cast<char*>(p)+i*t.bytes,&x,t.bytes);
}
double number(std::uint64_t x,JoinType t) { return t.is_signed ? double(std::int64_t(x)) : double(x); }
std::uint64_t extend(std::uint64_t x,JoinType t) {
    x&=mask(t);if(t.is_signed && t.bytes<8 && (x&(std::uint64_t(1)<<(t.bytes*8-1))))x|=~mask(t);return x;
}
} // namespace
void join_encode(const void* left,JoinType lt,const void* right,JoinType rt,std::size_t n,
                 std::uint64_t factor,JoinType mt,JoinType et,void* output) {
    check(lt);check(rt);check(mt);check(et);
    if(lt.floating || rt.floating || (mt.floating && !et.floating))throw std::invalid_argument("join stage dtype unsupported");
    const auto workers=detail::active_workers(n);
    detail::partition(n,workers,[&](std::size_t first,std::size_t last,std::size_t){
        auto i=first;
#if defined(__SSE2__) && !defined(SKIMAGEXX_JOIN_SCALAR)
        // Eight independent 16-bit multiply/add lanes; the two wraps match
        // scalar modulo arithmetic even when intermediate values are negative.
        if(lt.bytes==2 && rt.bytes==2 && mt.bytes==2 && et.bytes==2 && !mt.floating && !et.floating && !std::getenv("SKIMAGEXX_DISABLE_SIMD")) {
            const auto multiplier=_mm_set1_epi16(static_cast<short>(factor));
            for(;last-i>=8;i+=8) {
                const auto a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const char*>(left)+i*2));
                const auto b=_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const char*>(right)+i*2));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(static_cast<char*>(output)+i*2),_mm_add_epi16(_mm_mullo_epi16(a,multiplier),b));
            }
        }
#endif
        for(;i<last;++i) {
            const auto a=read(left,i,lt),b=read(right,i,rt);
            if(mt.floating) {
                const double product=double(factor)*number(a,lt);
                static_cast<double*>(output)[i]=product+number(b,rt);
            } else {
                const auto product=extend(factor*a,mt);
                if(et.floating)static_cast<double*>(output)[i]=number(product,mt)+number(b,rt);
                else write(output,i,et,product+b);
            }
        }
    });
}
void join_divmod(const void* labels,JoinType lt,std::size_t n,std::uint64_t factor,
                 JoinType dt,void* quotient,void* remainder) {
    check(lt);check(dt);
    if(!factor || lt.floating || dt.floating)throw std::invalid_argument("join integer divmod requires positive factor");
    const auto workers=detail::active_workers(n);
    detail::partition(n,workers,[&](std::size_t first,std::size_t last,std::size_t){
        for(auto i=first;i<last;++i) {
            const auto raw=read(labels,i,lt);
            // join's relabel gate rejects negative encoded integers first.
            write(quotient,i,dt,raw/factor);write(remainder,i,dt,raw%factor);
        }
    });
}
namespace {
JoinType source_type(PixelType type) {
    switch(type) {
        case PixelType::int8:return {1,true};case PixelType::uint8:return {1,false};
        case PixelType::int16:return {2,true};case PixelType::uint16:return {2,false};
        case PixelType::int32:return {4,true};case PixelType::uint32:return {4,false};
        case PixelType::int64:return {8,true};case PixelType::uint64:return {8,false};
        default:throw std::invalid_argument("join source must have an integer dtype");
    }
}
std::size_t count(const ComponentsView& view) {
    if(view.shape.size()!=view.strides.size() || view.swap || view.half)throw std::invalid_argument("join source requires native integer view");
    std::size_t n=1;for(auto extent:view.shape) {
        if(!extent)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
        if(extent>std::size_t(PTRDIFF_MAX)/8/n)throw std::overflow_error("join dimensions overflow");n*=extent;
    }
    if(!view.data)throw std::invalid_argument("join source data is null");return n;
}
std::vector<std::uint64_t> words(const ComponentsView& view,JoinType type,std::size_t n) {
    std::vector<std::uint64_t> out(n);const auto workers=detail::active_workers(n);
    detail::partition(n,workers,[&](auto first,auto last,auto){
        for(auto i=first;i<last;++i) {
            auto index=i;std::ptrdiff_t offset=0;
            for(auto axis=view.shape.size();axis-->0;){offset+=std::ptrdiff_t(index%view.shape[axis])*view.strides[axis];index/=view.shape[axis];}
            out[i]=read(static_cast<const char*>(view.data)+offset,0,type);
        }
    });
    if(type.is_signed && std::any_of(out.begin(),out.end(),[](auto x){return std::int64_t(x)<0;}))throw std::invalid_argument("Cannot relabel array that contains negative values.");
    return out;
}
JoinArray array(JoinType type,std::vector<std::size_t> shape,std::size_t n) {return {type,std::move(shape),std::vector<std::uint8_t>(n*type.bytes)};}
struct Ranked {JoinArray array;std::vector<std::uint64_t> unique,ids;};
Ranked rank(const std::vector<std::uint64_t>& input,JoinType type,const std::vector<std::size_t>& shape) {
    auto unique=consumer_sorted_unique(input.data(),input.size());const bool zero=unique.front()==0;
    const auto last=unique.size()-std::size_t(zero);unsigned required=1;
    while(required<8 && last>(required==8 ? ~std::uint64_t(0) : (std::uint64_t(1)<<(required*8))-1))required*=2;
    const auto limit=type.is_signed ? mask(type)>>1 : mask(type);
    if(type.bytes<required || last>=limit)type={required,false};
    auto values=array(type,{unique.size()},unique.size());std::vector<std::uint64_t> ids(unique.size());
    for(std::size_t i=0;i<ids.size();++i){ids[i]=i+std::size_t(!zero);write(values.data.data(),i,type,ids[i]);}
    auto out=array(type,shape,input.size());
    consumer_map_into(input.data(),input.size(),unique.data(),values.data.data(),unique.size(),type.bytes,64,out.data.data());
    return {std::move(out),std::move(unique),std::move(ids)};
}
JoinArray mapped(const JoinArray& indices,const Ranked& backward,JoinType original_type) {
    const auto n=indices.data.size()/indices.type.bytes;std::vector<std::uint64_t> keys(n);
    for(std::size_t i=0;i<n;++i)keys[i]=read(indices.data.data(),i,indices.type);
    auto values=array(original_type,{backward.unique.size()},backward.unique.size());
    for(std::size_t i=0;i<backward.unique.size();++i)write(values.data.data(),i,original_type,backward.unique[i]);
    auto out=array(original_type,indices.shape,n);
    consumer_map_into(keys.data(),n,backward.ids.data(),values.data.data(),backward.ids.size(),original_type.bytes,indices.type.bytes*8,out.data.data());return out;
}
} // namespace
JoinResult join_segmentations(const ComponentsView& left,const ComponentsView& right,const JoinParameters& p) {
    if(left.shape!=right.shape)throw std::invalid_argument("Cannot join segmentations of different shape.");
    auto lt=source_type(left.type),rt=source_type(right.type);const auto n=count(left);count(right);
    check(p.product_type);check(p.sum_type);check(p.division_type);
    if(p.product_type.floating || p.sum_type.floating || p.division_type.floating)throw std::invalid_argument("join SDK floating promotion is outside integer coverage");
    const auto a=words(left,lt,n),b=words(right,rt,n);auto first=rank(a,lt,left.shape),second=rank(b,rt,right.shape);
    auto factor_type=rt;if(rt.bytes==1 && rt.is_signed)factor_type={2,true};
    const auto factor=extend(*std::max_element(b.begin(),b.end())+1,factor_type);
    if(!factor || (factor_type.is_signed && std::int64_t(factor)<=0))throw std::invalid_argument("join SDK nonpositive/overflowed original factor unsupported");
    auto encoded=array(p.sum_type,left.shape,n);
    join_encode(first.array.data.data(),first.array.type,second.array.data.data(),second.array.type,n,factor,p.product_type,p.sum_type,encoded.data.data());
    std::vector<std::uint64_t> encoded_words(n);
    for(std::size_t i=0;i<n;++i)encoded_words[i]=read(encoded.data.data(),i,encoded.type);
    if(encoded.type.is_signed && std::any_of(encoded_words.begin(),encoded_words.end(),[](auto x){return std::int64_t(x)<0;}))throw std::invalid_argument("Cannot relabel array that contains negative values.");
    auto result=rank(encoded_words,encoded.type,left.shape);JoinResult out{};out.joined=std::move(result.array);
    if(!p.return_mapping)return out;
    if(p.division_type.bytes<p.sum_type.bytes || (p.sum_type.bytes==p.division_type.bytes && !p.sum_type.is_signed && p.division_type.is_signed))throw std::invalid_argument("join division promotion cannot narrow encoded dtype");
    out.has_mapping=true;const auto size=result.unique.size();out.keys=array(out.joined.type,{size},size);
    auto labels=array(encoded.type,{size},size),q=array(p.division_type,{size},size),r=array(p.division_type,{size},size);
    for(std::size_t i=0;i<size;++i){write(out.keys.data.data(),i,out.keys.type,result.ids[i]);write(labels.data.data(),i,encoded.type,result.unique[i]);}
    join_divmod(labels.data.data(),labels.type,size,factor,p.division_type,q.data.data(),r.data.data());
    out.first_values=mapped(q,first,lt);out.second_values=mapped(r,second,rt);return out;
}
}
