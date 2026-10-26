#include "skimagexx/colocalization.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "half.hpp"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#pragma STDC FENV_ACCESS ON

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::uint64_t coloc_bool_count(const unsigned char*,const unsigned char*,std::size_t);
}
#endif

namespace skimagexx {
namespace {
using Type=ColocType;
constexpr auto address_limit=std::numeric_limits<std::uintptr_t>::max();
constexpr auto offset_limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
std::size_t multiply(std::size_t a,std::size_t b) {
    if(b && a>offset_limit/b)throw std::overflow_error("colocalization size overflow");
    return a*b;
}
std::size_t add(std::size_t a,std::size_t b) {
    if(b>offset_limit || a>offset_limit-b)throw std::overflow_error("colocalization extent overflow");
    return a+b;
}
std::size_t magnitude(std::ptrdiff_t value) {
    return value<0 ? std::size_t(-(value+1))+1 : std::size_t(value);
}
struct Range {std::uintptr_t begin,end;};
Range range(const void* data,std::size_t bytes) {
    const auto begin=reinterpret_cast<std::uintptr_t>(data);
    if(bytes && !data)throw std::invalid_argument("colocalization null span");
    if(bytes>address_limit-begin)throw std::overflow_error("colocalization span overflow");
    return {begin,begin+bytes};
}
bool overlaps(Range a,Range b) {return a.begin<a.end && b.begin<b.end && a.begin<b.end && b.begin<a.end;}
void disjoint(Range output,const void* data,std::size_t bytes) {
    if(overlaps(output,range(data,bytes)))throw std::invalid_argument("colocalization output overlaps read-only storage");
}
void output_check(ColocBuffer output,std::size_t bytes,const ColocView& a,
                  const ColocView* b=nullptr,const ColocOrder* order=nullptr,const unsigned* flags=nullptr) {
    if(output.bytes!=bytes)throw std::invalid_argument("colocalization exact output byte size required");
    const auto out=range(output.data,output.bytes);
    auto one=[&](const ColocView& view) {
        disjoint(out,view.allocation,view.allocation_bytes);
        disjoint(out,&view,sizeof(view));
        disjoint(out,view.shape.data,multiply(view.shape.size,sizeof(std::size_t)));
        disjoint(out,view.strides.data,multiply(view.strides.size,sizeof(std::ptrdiff_t)));
    };
    one(a);if(b)one(*b);if(order)disjoint(out,order,sizeof(*order));if(flags)disjoint(out,flags,sizeof(*flags));
}
void shapes(const ColocView& a,const ColocView& b) {
    if(a.shape.size!=b.shape.size)throw std::invalid_argument("colocalization shapes differ");
    for(std::size_t i=0;i<a.shape.size;++i)if(a.shape.data[i]!=b.shape.data[i])throw std::invalid_argument("colocalization shapes differ");
}
void order_check(const ColocView& a,const ColocOrder& order) {
    if(order.rank!=a.shape.size)throw std::invalid_argument("colocalization order rank differs");
    std::array<bool,colocalization_max_rank> seen{};
    for(std::size_t i=0;i<order.rank;++i) {
        const auto axis=order.axes[i];
        if(axis>=order.rank || seen[axis])throw std::invalid_argument("colocalization order is not a permutation");
        seen[axis]=true;
    }
}
ColocOrder c_order(const ColocView& view) {
    ColocOrder result;result.rank=view.shape.size;
    for(std::size_t i=0;i<result.rank;++i)result.axes[i]=result.rank-1-i;
    return result;
}
std::ptrdiff_t offset(const ColocView& view,const ColocOrder& order,std::size_t index) {
    std::ptrdiff_t result=0;
    for(std::size_t i=0;i<order.rank;++i) {
        const auto axis=order.axes[i],length=view.shape.data[axis];
        result+=std::ptrdiff_t(index%length)*view.strides.data[axis];index/=length;
    }
    return result;
}
const unsigned char* pointer(const ColocView& v,const ColocOrder& order,std::size_t i) {
    return static_cast<const unsigned char*>(v.data)+offset(v,order,i);
}
template<class T> T read(const unsigned char* data,bool swapped) {
    T value;
    if(swapped && sizeof(T)>1) {
        unsigned char bytes[sizeof(T)];for(std::size_t i=0;i<sizeof(T);++i)bytes[i]=data[sizeof(T)-1-i];
        std::memcpy(&value,bytes,sizeof(T));
    } else std::memcpy(&value,data,sizeof(T));
    return value;
}
template<class T> void write(void* output,std::size_t index,T value) {
    std::memcpy(static_cast<unsigned char*>(output)+index*sizeof(T),&value,sizeof(T));
}
std::uint64_t bits(const ColocView& v,const unsigned char* data,std::size_t item) {
    // Callers cache the byte width after validating the view's dtype.
    switch(item) {
    case 1:return *data;
    case 2:return read<std::uint16_t>(data,v.swapped);
    case 4:return read<std::uint32_t>(data,v.swapped);
    default:return read<std::uint64_t>(data,v.swapped);
    }
}
bool floating(Type type) {return type==Type::float16 || type==Type::float32 || type==Type::float64;}
bool signed_integer(Type type) {return type==Type::int8 || type==Type::int16 || type==Type::int32 || type==Type::int64;}
std::uint64_t low_bits_mask(std::size_t item) {return item==8 ? ~std::uint64_t(0):(std::uint64_t(1)<<(item*8))-1;}
bool nan(Type type,std::uint64_t value) {
    if(type==Type::float16)return (value&0x7c00)==0x7c00 && (value&0x03ff);
    if(type==Type::float32)return (value&0x7f800000)==0x7f800000 && (value&0x007fffff);
    if(type==Type::float64)return (value&0x7ff0000000000000ULL)==0x7ff0000000000000ULL && (value&0x000fffffffffffffULL);
    return false;
}
bool zero(Type type,std::uint64_t value) {
    if(type==Type::float16)return !(value&0x7fff);
    if(type==Type::float32)return !(value&0x7fffffff);
    if(type==Type::float64)return !(value&0x7fffffffffffffffULL);
    return value==0;
}
bool binary(Type type,std::uint64_t value) {
    if(type==Type::boolean)return true;
    if(zero(type,value))return true;
    if(type==Type::float16)return value==0x3c00;
    if(type==Type::float32)return value==0x3f800000;
    if(type==Type::float64)return value==0x3ff0000000000000ULL;
    return value==1;
}
class Capture {
    std::fenv_t saved_;
public:
    Capture() {if(std::feholdexcept(&saved_))throw std::runtime_error("cannot capture colocalization floating environment");}
    ~Capture(){std::fesetenv(&saved_);}
    unsigned flags()const {
        const auto f=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
        return ((f&FE_DIVBYZERO)?coloc_divide:0u)|((f&FE_OVERFLOW)?coloc_overflow:0u)|
            ((f&FE_UNDERFLOW)?coloc_underflow:0u)|((f&FE_INVALID)?coloc_invalid:0u);
    }
};
void fp_check() {if(!coloc_fp_supported())throw ColocProviderRequired("colocalization requires nearest rounding and gradual underflow");}
// Half product/reduction storage conversion. Preserve raw sign/payload; flag
// overflow/underflow as NumPy's scalar software conversion does, without F16C.
std::uint16_t half(float value) {
    std::uint32_t word;std::memcpy(&word,&value,4);
    const auto encoded=detail::encode_half(value);
    const auto exponent=word&0x7f800000u;
    if(exponent!=0x7f800000u) {
        if((encoded&0x7fffu)==0x7c00u)std::feraiseexcept(FE_OVERFLOW);
        else if(exponent<=0x38000000u && detail::decode_half(encoded)!=value)std::feraiseexcept(FE_UNDERFLOW);
    }
    return encoded;
}
template<class T> T real(const ColocView& v,const unsigned char* data) {
    if(v.type==Type::float16)return T(detail::decode_half(read<std::uint16_t>(data,v.swapped)));
    if(v.type==Type::float32)return T(read<float>(data,v.swapped));
    return T(read<double>(data,v.swapped));
}
template<class T,class Get> T pairwise(std::size_t start,std::size_t n,const Get& get) {
    // Exact primary NumPy FLOAT/DOUBLE/HALF_pairwise_sum; BSD notice is in
    // third_party/numpy-LICENSE.txt. Do not reassociate or contract.
    if(n<8) {T sum=-T(0);for(std::size_t i=0;i<n;++i)sum=T(sum+get(start+i));return sum;}
    if(n<=128) {
        T lanes[8];for(std::size_t j=0;j<8;++j)lanes[j]=get(start+j);
        std::size_t i=8;for(;i<n-n%8;i+=8)for(std::size_t j=0;j<8;++j)lanes[j]=T(lanes[j]+get(start+i+j));
        T sum=T(T(T(lanes[0]+lanes[1])+T(lanes[2]+lanes[3]))+T(T(lanes[4]+lanes[5])+T(lanes[6]+lanes[7])));
        for(;i<n;++i)sum=T(sum+get(start+i));return sum;
    }
    auto left=n/2;left-=left%8;
    const T a=pairwise<T>(start,left,get),b=pairwise<T>(start+left,n-left,get);
    return T(a+b);
}
bool single_stride(const ColocView& v,const ColocOrder& order,std::ptrdiff_t& step) {
    bool found=false;std::size_t previous=1;
    for(std::size_t i=0;i<order.rank;++i) {
        const auto axis=order.axes[i],length=v.shape.data[axis];if(length<=1)continue;
        const auto stride=v.strides.data[axis];
        if(!found){step=stride;found=true;previous=length;}
        else {
            if(magnitude(step) && previous>offset_limit/magnitude(step))return false;
            if(std::ptrdiff_t(previous)*step!=stride)return false;
            previous=multiply(previous,length);
        }
    }
    if(!found)step=std::ptrdiff_t(coloc_itemsize(v.type));return true;
}
struct ReadAccess {
    const ColocView& view;
    const ColocOrder& order;
    std::ptrdiff_t step=0;
    bool affine;
    ReadAccess(const ColocView& v,const ColocOrder& o,std::size_t n)
        :view(v),order(o),affine(!n || single_stride(v,o,step)){}
    const unsigned char* operator[](std::size_t i)const {
        // The validated view bounds every signed offset; single_stride proves
        // that this traversal is affine, including negative/broadcast strides.
        return affine ? static_cast<const unsigned char*>(view.data)+std::ptrdiff_t(i)*step
                      :pointer(view,order,i);
    }
};
// Inputs to this internal ordering helper have already been validated.
ColocOrder keep_order(const ColocView& v,const ColocView* other);
template<class Body> ColocExecution partition(std::size_t n,const Body& body) {
    const int workers=detail::active_workers(n);
    detail::partition(n,workers,[&](auto begin,auto end,auto){body(begin,end);});
    return {workers,false};
}
void boolean_required(const ColocView& v) {if(v.type!=Type::boolean)throw std::invalid_argument("colocalization boolean stage requires bool storage");}
std::uint64_t count_impl(const ColocView& a,const ColocView* b) {
    const auto n=coloc_validate(a);boolean_required(a);
    if(b){coloc_validate(*b);boolean_required(*b);shapes(a,*b);}
    if(!n)return 0;
    // Integer counts are independent of traversal order. A common KEEPORDER
    // also exposes dense F-order operands to the existing byte-count leaf.
    const auto order=keep_order(a,b);const int workers=detail::active_workers(n);
    std::vector<std::uint64_t> counts(std::size_t(workers),0);
    const ReadAccess first(a,order,n),second(b ? *b:a,order,n);
    const bool dense=first.affine && first.step==1 && (!b || (second.affine && second.step==1));
    detail::partition(n,workers,[&](auto begin,auto end,auto id) {
        std::uint64_t value=0;
#ifdef SKIMAGEXX_AVX2
        if(dense && has_avx2())value=simd::coloc_bool_count(static_cast<const unsigned char*>(a.data)+begin,
            b ? static_cast<const unsigned char*>(b->data)+begin:nullptr,end-begin);
        else
#endif
        for(auto i=begin;i<end;++i)value+=(*first[i]!=0) && (!b || *second[i]!=0);
        counts[id]=value;
    });
    std::uint64_t result=0;for(auto value:counts)result+=value;return result;
}
}

std::size_t coloc_itemsize(ColocType type) {
    switch(type) {
    case Type::boolean:case Type::uint8:case Type::int8:return 1;
    case Type::uint16:case Type::int16:case Type::float16:return 2;
    case Type::uint32:case Type::int32:case Type::float32:return 4;
    case Type::uint64:case Type::int64:case Type::float64:return 8;
    }throw std::invalid_argument("unknown colocalization dtype");
}
std::size_t coloc_validate(const ColocView& v) {
    const auto item=coloc_itemsize(v.type);
    if(v.shape.size!=v.strides.size || v.shape.size>colocalization_max_rank)throw std::invalid_argument("colocalization rank/strides mismatch");
    range(v.shape.data,multiply(v.shape.size,sizeof(std::size_t)));range(v.strides.data,multiply(v.strides.size,sizeof(std::ptrdiff_t)));
    bool empty=false;for(std::size_t i=0;i<v.shape.size;++i)empty|=v.shape.data[i]==0;
    if(empty)return 0;
    std::size_t n=1,negative=0,positive=0;
    for(std::size_t i=0;i<v.shape.size;++i) {
        n=multiply(n,v.shape.data[i]);const auto stride=v.strides.data[i];
        const auto span=multiply(v.shape.data[i]-1,magnitude(stride));
        if(stride<0)negative=add(negative,span);else positive=add(positive,span);
    }
    multiply(n,item);positive=add(positive,item);
    const auto data=reinterpret_cast<std::uintptr_t>(v.data),base=reinterpret_cast<std::uintptr_t>(v.allocation);
    range(v.allocation,v.allocation_bytes);
    if(!data || !base || negative>data || positive>address_limit-data || data-negative<base ||
            data+positive>base+v.allocation_bytes)throw std::invalid_argument("colocalization view escapes owning allocation");
    return n;
}
bool coloc_fp_supported()noexcept {
    if(std::fegetround()!=FE_TONEAREST)return false;
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if(_mm_getcsr()&((1u<<15)|(1u<<6)))return false;
#endif
    return true;
}
namespace {
ColocOrder keep_order(const ColocView& v,const ColocView* other) {
    auto order=c_order(v);
    // npyiter_find_best_axis_ordering's stable insertion sort. Zero/broadcast
    // strides are ambiguous; conflicting operands retain C-axis priority.
    // Length-one axes have iterator stride0 and do not constrain this order.
    for(std::size_t i=1;i<order.rank;++i) {
        const auto axis=order.axes[i];auto place=i;
        for(auto j=i;j-->0;) {
            bool ambiguous=true,swap=false;
            auto compare=[&](const ColocView& input) {
                const auto before=order.axes[j];
                const auto a=input.shape.data[axis]<=1 ? 0:input.strides.data[axis];
                const auto b=input.shape.data[before]<=1 ? 0:input.strides.data[before];
                if(a && b) {
                    if(magnitude(b)<=magnitude(a))swap=false;
                    else if(ambiguous)swap=true;
                    ambiguous=false;
                }
            };
            compare(v);if(other)compare(*other);
            if(!ambiguous){if(swap)place=j;else break;}
        }
        for(auto j=i;j>place;--j)order.axes[j]=order.axes[j-1];
        order.axes[place]=axis;
    }
    return order;
}
}
ColocOrder coloc_keep_order(const ColocView& v) {coloc_validate(v);return keep_order(v,nullptr);}
ColocOrder coloc_keep_order(const ColocView& a,const ColocView& b) {
    coloc_validate(a);coloc_validate(b);shapes(a,b);return keep_order(a,&b);
}
bool coloc_sum_supported(const ColocView& v,ColocReductionPlan plan) {
    const auto n=coloc_validate(v);
    if(plan.profile!=ColocProfile::numpy_1244 && plan.profile!=ColocProfile::numpy_254)return false;
    if(!plan.buffer_elements)return false;
    if(!floating(v.type))return true;
    std::ptrdiff_t stride=0;
    return coloc_fp_supported() && (!n || single_stride(v,coloc_keep_order(v),stride));
}
bool coloc_all_finite(const ColocView& v) {
    const auto n=coloc_validate(v);if(!floating(v.type))return true;
    const auto order=keep_order(v,nullptr);const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    const auto exponent=v.type==Type::float16 ? 0x7c00ULL:v.type==Type::float32 ? 0x7f800000ULL:0x7ff0000000000000ULL;
    for(std::size_t i=0;i<n;++i)if((bits(v,input[i],item)&exponent)==exponent)return false;
    return true;
}
ColocType coloc_square_type(Type t) {coloc_itemsize(t);return t==Type::boolean ? Type::int8:t;}
ColocType coloc_sum_type(Type t) {
    coloc_itemsize(t);if(floating(t))return t;
    return t==Type::boolean || signed_integer(t) ? Type::int64:Type::uint64;
}
ColocPredicate coloc_binary_valid(const ColocView& v) {
    const auto n=coloc_validate(v);const auto order=keep_order(v,nullptr);const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    for(std::size_t i=0;i<n;++i)if(!binary(v.type,bits(v,input[i],item)))return {false,0};
    return {true,0};
}
ColocExecution coloc_binary_into(const ColocView& v,const ColocOrder& order,ColocBuffer out) {
    const auto n=coloc_validate(v);order_check(v,order);output_check(out,n,v,nullptr,&order);
    if(!coloc_binary_valid(v).value)throw std::domain_error("colocalization nonbinary input");
    const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    return partition(n,[&](auto begin,auto end){for(auto i=begin;i<end;++i)write<std::uint8_t>(out.data,i,!zero(v.type,bits(v,input[i],item)));});
}
std::uint64_t coloc_count_nonzero(const ColocView& v) {return count_impl(v,nullptr);}
std::uint64_t coloc_joint_count(const ColocView& a,const ColocView& b) {return count_impl(a,&b);}
ColocPredicate coloc_min_negative(const ColocView& v) {
    const auto n=coloc_validate(v);if(!n)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    bool negative=false,has_nan=false;const auto order=keep_order(v,nullptr);const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    const auto sign=std::uint64_t(1)<<(item*8-1);
    const bool can_negative=floating(v.type) || signed_integer(v.type);
    for(std::size_t i=0;i<n;++i) {
        const auto value=bits(v,input[i],item);has_nan|=nan(v.type,value);
        if(can_negative)negative|=(value&sign) && !zero(v.type,value);
    }
    return {negative && !has_nan,0};
}
ColocExecution coloc_gather_into(const ColocView& v,const ColocView& mask_view,ColocBuffer out,std::size_t selected) {
    const auto n=coloc_validate(v);coloc_validate(mask_view);boolean_required(mask_view);shapes(v,mask_view);
    output_check(out,multiply(selected,coloc_itemsize(v.type)),v,&mask_view);
    if(coloc_count_nonzero(mask_view)!=selected)throw std::invalid_argument("colocalization gather count changed");
    const auto order=c_order(v);const int workers=detail::active_workers(n);
    const ReadAccess input(v,order,n),selection(mask_view,order,n);
    std::vector<std::size_t> counts(std::size_t(workers),0),starts(std::size_t(workers),0);
    auto bounds=[&](std::size_t task) {
        const auto block=n/std::size_t(workers),extra=n%std::size_t(workers);
        const auto begin=task*block+std::min(task,extra);
        return std::pair<std::size_t,std::size_t>{begin,begin+block+(task<extra)};
    };
    detail::partition(std::size_t(workers),workers,[&](auto first,auto last,auto){
        for(auto task=first;task<last;++task){const auto span=bounds(task);for(auto i=span.first;i<span.second;++i)counts[task]+=*selection[i]!=0;}
    });
    for(std::size_t i=1;i<counts.size();++i)starts[i]=starts[i-1]+counts[i-1];
    const auto item=coloc_itemsize(v.type);
    detail::partition(std::size_t(workers),workers,[&](auto first,auto last,auto){
      for(auto task=first;task<last;++task) {
        const auto span=bounds(task);auto next=starts[task];for(auto i=span.first;i<span.second;++i)if(*selection[i]) {
            auto* target=static_cast<unsigned char*>(out.data)+next++*item;
            std::memcpy(target,input[i],item);
        }
      }
    });return {workers,false};
}
ColocScalar coloc_sum(const ColocView& v,ColocReductionPlan plan) {
    const auto n=coloc_validate(v);const auto order=coloc_keep_order(v);
    if(plan.profile!=ColocProfile::numpy_1244 && plan.profile!=ColocProfile::numpy_254)throw std::invalid_argument("unknown colocalization NumPy profile");
    if(!plan.buffer_elements)throw std::invalid_argument("colocalization reduction buffer must be positive");
    ColocScalar result;result.type=coloc_sum_type(v.type);
    const auto item=coloc_itemsize(v.type);
    if(!floating(v.type)) {
        const ReadAccess input(v,order,n);
        const bool extend=signed_integer(v.type) && item<8;
        const auto sign=std::uint64_t(1)<<(item*8-1),extension=~low_bits_mask(item);
        std::uint64_t sum=0;for(std::size_t i=0;i<n;++i) {
            auto value=bits(v,input[i],item);
            if(v.type==Type::boolean)value=value!=0;
            else if(extend && (value&sign))value|=extension;
            sum+=value;
        }result.bits=sum;return result;
    }
    fp_check();if(!coloc_all_finite(v))throw ColocProviderRequired("colocalization nonfinite sum is a provider frontier");
    std::ptrdiff_t step=0;
    if(n && !single_stride(v,order,step))throw ColocProviderRequired("colocalization irregular floating reduction iterator is a provider frontier");
    // NpyIter reduction: single-stride aligned/native input needs no cast
    // buffer in2.5 and GROWINNER exposes the entire inner loop. In1.24 the
    // buffered reduction loop stops at the supplied iterator buffer size.
    const bool needs_buffer=v.swapped || (reinterpret_cast<std::uintptr_t>(v.data)%item) || (magnitude(step)%item);
    const auto block=(plan.profile==ColocProfile::numpy_1244 || needs_buffer) ? plan.buffer_elements:std::max(n,std::size_t(1));
    Capture capture;
    if(v.type==Type::float64) {
        double sum=0;auto get=[&](auto i){return real<double>(v,static_cast<const unsigned char*>(v.data)+std::ptrdiff_t(i)*step);};
        for(std::size_t begin=0;begin<n;) {const auto length=std::min(block,n-begin);sum=double(sum+pairwise<double>(begin,length,get));begin+=length;}
        std::memcpy(&result.bits,&sum,8);
    } else {
        float sum=0;std::uint16_t half_sum=0;auto get=[&](auto i){return real<float>(v,static_cast<const unsigned char*>(v.data)+std::ptrdiff_t(i)*step);};
        for(std::size_t begin=0;begin<n;) {
            const auto length=std::min(block,n-begin);
            sum=float(sum+pairwise<float>(begin,length,get));
            if(v.type==Type::float16){half_sum=half(sum);sum=detail::decode_half(half_sum);}
            begin+=length;
        }
        if(v.type==Type::float16)result.bits=half_sum;
        else {std::uint32_t word;std::memcpy(&word,&sum,4);result.bits=word;}
    }
    result.fp_status=capture.flags();return result;
}
namespace {
ColocExecution product(const ColocView& a,const ColocView* b,const ColocOrder& order,ColocBuffer out,unsigned& flags) {
    const auto n=coloc_validate(a);order_check(a,order);
    if(b){coloc_validate(*b);shapes(a,*b);if(b->type!=a.type && b->type!=Type::boolean)throw ColocProviderRequired("colocalization mixed dtype product is a provider frontier");}
    const auto output_type=b ? a.type:coloc_square_type(a.type);const auto item=coloc_itemsize(output_type);
    output_check(out,multiply(n,item),a,b,&order,&flags);
    output_check({&flags,sizeof(flags)},sizeof(flags),a,b,&order);
    if(floating(a.type)) {
        fp_check();
        if(!coloc_all_finite(a) || (b && !coloc_all_finite(*b)))
            throw ColocProviderRequired("colocalization nonfinite product is a provider frontier");
    }
    const ReadAccess first_input(a,order,n),second_input(b ? *b:a,order,n);
    // Square preserves input width (including bool -> int8); multiply admits
    // only the same dtype or a one-byte boolean second operand.
    const auto second_item=b && b->type==Type::boolean ? std::size_t(1):item;
    const auto product_mask=low_bits_mask(item);
    Capture capture;
    auto body=[&](std::size_t begin,std::size_t end) {
        for(auto i=begin;i<end;++i) {
            const auto* first=first_input[i];const auto* second=b ? second_input[i]:first;
            if(!floating(a.type)) {
                auto x=bits(a,first,item);if(a.type==Type::boolean)x=x!=0;
                auto y=b ? bits(*b,second,second_item):x;if(b && b->type==Type::boolean)y=y!=0;
                const auto value=(x*y)&product_mask;
                if(item==1)write<std::uint8_t>(out.data,i,std::uint8_t(value));
                else if(item==2)write<std::uint16_t>(out.data,i,std::uint16_t(value));
                else if(item==4)write<std::uint32_t>(out.data,i,std::uint32_t(value));
                else write<std::uint64_t>(out.data,i,value);
            } else if(a.type==Type::float64) {
                const double x=real<double>(a,first),y=b ? (b->type==Type::boolean ? double(*second!=0):real<double>(*b,second)):x;
                write<double>(out.data,i,double(x*y));
            } else {
                const float x=real<float>(a,first),y=b ? (b->type==Type::boolean ? float(*second!=0):real<float>(*b,second)):x;
                const float value=float(x*y);
                if(a.type==Type::float16)write<std::uint16_t>(out.data,i,half(value));else write<float>(out.data,i,value);
            }
        }
    };
    ColocExecution execution;
    if(floating(a.type))body(0,n);else execution=partition(n,body);
    flags=capture.flags();return execution;
}
}
ColocExecution coloc_square_into(const ColocView& a,const ColocOrder& order,ColocBuffer out,unsigned& flags) {return product(a,nullptr,order,out,flags);}
ColocExecution coloc_multiply_into(const ColocView& a,const ColocView& b,const ColocOrder& order,ColocBuffer out,unsigned& flags) {return product(a,&b,order,out,flags);}
} // namespace skimagexx
