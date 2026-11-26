#include "skimagexx/colocalization.hpp"
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
#include <utility>
#include <vector>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif
#if defined(_MSC_VER)
#pragma fenv_access(on)
#elif defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

#ifdef SKIMAGEXX_AVX2
namespace skimagexx::simd {
std::size_t coloc_intersection_avx2(const unsigned char*,const unsigned char*,
    const unsigned char*,std::size_t,unsigned,std::array<std::uint64_t,3>&);
void coloc_intersection_reverse_avx2(const unsigned char*,const unsigned char*,
    std::size_t,std::size_t,std::array<std::uint64_t,3>&);
std::size_t coloc_intersection_wide_avx2(const unsigned char*,const unsigned char*,
    std::size_t,std::size_t,std::uint64_t,std::uint64_t,std::array<std::uint64_t,3>&);
std::uint64_t coloc_bool_count(const unsigned char*,const unsigned char*,std::size_t);
void coloc_manders_leaf(const unsigned char*,const unsigned char*,std::size_t,bool,double*,unsigned&);
void coloc_manders_leaf(const unsigned char*,const unsigned char*,std::size_t,bool,float*,unsigned&);
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
        for(;i<n;++i)sum=T(sum+get(start+i));
        return sum;
    }
    auto left=n/2;left-=left%8;
    const T a=pairwise<T>(start,left,get),b=pairwise<T>(start+left,n-left,get);
    return T(a+b);
}
template<class T,class Get> T reduce_blocks(std::size_t n,std::size_t block,const Get& get) {
    T sum=0;
    for(std::size_t begin=0;begin<n;) {
        const auto length=std::min(block,n-begin);
        sum+=pairwise<T>(begin,length,get);
        begin+=length;
    }
    return sum;
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
    if(!found)step=std::ptrdiff_t(coloc_itemsize(v.type));
    return true;
}
template<class U> int dense_finite_min(const void* data,std::size_t n,U infinity) {
    constexpr U sign=U(1)<<(sizeof(U)*8-1);
    U negative=0,nonfinite=0;
    const auto* bytes=static_cast<const unsigned char*>(data);
    for(std::size_t i=0;i<n;++i) {
        const auto value=read<U>(bytes+i*sizeof(U),false);
        negative|=U(value>sign);
        nonfinite|=U((value&(sign-1))>=infinity);
    }
    return nonfinite ? -1:int(negative!=0);
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
bool inner_contiguous(const ColocView& view,const ColocOrder& order) {
    return !order.rank || view.strides.data[order.axes[0]]==std::ptrdiff_t(coloc_itemsize(view.type));
}
// The initial seek is per partition. Carrying to the next row uses only
// outer-axis strides and never changes the order of the logical elements.
template<std::size_t N> class Rows {
    const ColocView& shape_;
    const ColocOrder& order_;
    const std::array<const ColocView*,N> views_;
    std::array<std::size_t,colocalization_max_rank> coordinates_{};
    std::array<const unsigned char*,N> row_{};
    std::size_t width_, column_, remaining_;
public:
    Rows(const ColocView& shape,const ColocOrder& order,
         std::array<const ColocView*,N> views,std::size_t begin,std::size_t end,bool dense=false)
        :shape_(shape),order_(order),views_(views),
         width_(dense ? end : order.rank ? shape.shape.data[order.axes[0]] : 1),
         column_(begin ? begin%width_ : 0),remaining_(end-begin) {
        auto index=begin-column_;
        if(index)for(std::size_t k=0;k<order.rank;++k) {
            const auto length=shape.shape.data[order.axes[k]];
            coordinates_[k]=index%length;index/=length;
        }
        for(std::size_t j=0;j<N;++j)if(views_[j]) {
            row_[j]=static_cast<const unsigned char*>(views_[j]->data);
            if(begin!=column_)row_[j]+=offset(*views_[j],order,begin-column_);
        }
    }
    std::size_t available()const {return std::min(width_-column_,remaining_);}
    const unsigned char* data(std::size_t j,std::size_t item)const {
        return row_[j] ? row_[j]+column_*item:nullptr;
    }
    void advance(std::size_t count) {
        remaining_-=count;
        column_+=count;
        if(column_!=width_ || !remaining_)return;
        column_=0;
        for(std::size_t k=1;k<order_.rank;++k) {
            const auto axis=order_.axes[k],length=shape_.shape.data[axis];
            if(length==1)continue;
            const bool carry=++coordinates_[k]==length;
            const auto delta=carry ? -std::ptrdiff_t(length-1):1;
            for(std::size_t j=0;j<N;++j)if(views_[j])
                row_[j]+=delta*views_[j]->strides.data[axis];
            if(!carry)return;
            coordinates_[k]=0;
        }
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
    std::uint64_t single=0;
    std::vector<std::uint64_t> counts(workers>1 ? std::size_t(workers):0,0);
    const ReadAccess first(a,order,n),second(b ? *b:a,order,n);
    const bool dense=first.affine && first.step==1 && (!b || (second.affine && second.step==1));
    const bool rows=inner_contiguous(a,order) && (!b || inner_contiguous(*b,order));
#ifdef SKIMAGEXX_AVX2
    const bool avx2=has_avx2();
#endif
    detail::partition(n,workers,[&](auto begin,auto end,auto id) {
        std::uint64_t value=0;
        const auto scan=[&](const unsigned char* x,const unsigned char* y,std::size_t length) {
#ifdef SKIMAGEXX_AVX2
            if(avx2){value+=simd::coloc_bool_count(x,y,length);return;}
#endif
            if(y)for(std::size_t i=0;i<length;++i)value+=(x[i]!=0) && (y[i]!=0);
            else for(std::size_t i=0;i<length;++i)value+=x[i]!=0;
        };
        if(dense && begin<end)scan(first[begin],b ? second[begin]:nullptr,end-begin);
        else if(rows && begin<end) {
            Rows<2> row(a,order,{&a,b},begin,end);
            while(const auto length=row.available()) {
                scan(row.data(0,1),row.data(1,1),length);row.advance(length);
            }
        } else for(auto i=begin;i<end;++i)value+=(*first[i]!=0) && (!b || *second[i]!=0);
        if(workers==1)single=value;else counts[id]=value;
    });
    for(auto value:counts)single+=value;
    return single;
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
    if(!n)return true;
    const auto order=keep_order(v,nullptr);const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    const auto exponent=v.type==Type::float16 ? 0x7c00ULL:v.type==Type::float32 ? 0x7f800000ULL:0x7ff0000000000000ULL;
    const auto scan=[&](const unsigned char* data,std::size_t length) {
        if(!v.swapped && v.type==Type::float32)return dense_finite_min<std::uint32_t>(data,length,0x7f800000u)>=0;
        if(!v.swapped && v.type==Type::float64)return dense_finite_min<std::uint64_t>(data,length,0x7ff0000000000000ULL)>=0;
        for(std::size_t i=0;i<length;++i)if((bits(v,data+i*item,item)&exponent)==exponent)return false;
        return true;
    };
    if(input.affine && input.step==std::ptrdiff_t(item))return scan(input[0],n);
    if(inner_contiguous(v,order)) {
        Rows<1> row(v,order,{&v},0,n);
        while(const auto length=row.available()) {
            if(!scan(row.data(0,item),length))return false;
            row.advance(length);
        }
        return true;
    }
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
    if((v.type==Type::uint8 || v.type==Type::int8) && input.affine && input.step==1) {
        // Only integer bytes 0/1 are valid; bool deliberately permits other
        // nonzero representations. This read-only reduction can vectorize.
        const auto* data=static_cast<const unsigned char*>(v.data);std::uint8_t invalid=0;
        for(std::size_t i=0;i<n;++i)invalid=std::uint8_t(invalid|(data[i]&0xfeu));
        return {invalid==0,0};
    }
    const auto item=coloc_itemsize(v.type);
    for(std::size_t i=0;i<n;++i)if(!binary(v.type,bits(v,input[i],item)))return {false,0};
    return {true,0};
}
ColocExecution coloc_binary_into(const ColocView& v,const ColocOrder& order,ColocBuffer out) {
    const auto n=coloc_validate(v);order_check(v,order);output_check(out,n,v,nullptr,&order);
    if(!coloc_binary_valid(v).value)throw std::domain_error("colocalization nonbinary input");
    const ReadAccess input(v,order,n);
    if((v.type==Type::uint8 || v.type==Type::int8) && input.affine && input.step==1) {
        const auto* source=static_cast<const unsigned char*>(v.data);
        auto* target=static_cast<unsigned char*>(out.data);
        return partition(n,[&](auto begin,auto end){
            if(begin<end)std::memcpy(target+begin,source+begin,end-begin);
        });
    }
    const auto item=coloc_itemsize(v.type);
    return partition(n,[&](auto begin,auto end){for(auto i=begin;i<end;++i)write<std::uint8_t>(out.data,i,!zero(v.type,bits(v,input[i],item)));});
}
std::uint64_t coloc_count_nonzero(const ColocView& v) {return count_impl(v,nullptr);}
std::uint64_t coloc_joint_count(const ColocView& a,const ColocView& b) {return count_impl(a,&b);}
namespace {
using IntersectionCounts=std::array<std::uint64_t,3>;
// All byte spans and common traversal order have been checked by the
// binding. The third result flags invalid integer binary controls.
IntersectionCounts coloc_intersection_bytes(const unsigned char* first,
    const unsigned char* second,const unsigned char* mask,std::size_t n,unsigned integer_inputs) {
    constexpr std::uint64_t low=0x0101010101010101ULL;
    const auto truth=[](std::uint64_t value) {
        constexpr std::uint64_t lower=0x7f7f7f7f7f7f7f7fULL;
        constexpr std::uint64_t high=0x8080808080808080ULL;
        return (((value&lower)+lower)|value)&high;
    };
    const auto workers=detail::active_workers(n);
    using Counts=std::array<std::uint64_t,3>;
    Counts single{};
    std::vector<Counts> partial(workers>1 ? std::size_t(workers):0);
    detail::partition(n,workers,[&](auto i,auto end,auto worker) {
        auto& counts=workers==1 ? single:partial[worker];
        std::uint64_t invalid=0;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())i+=simd::coloc_intersection_avx2(first+i,second+i,mask ? mask+i:nullptr,end-i,integer_inputs,counts);
#endif
        while(end-i>=8) {
            // Each byte lane can accumulate up to 255 independent words.
            const auto stop=i+8*std::min<std::size_t>((end-i)/8,255);
            std::uint64_t total=0,joint=0;
            for(;i<stop;i+=8) {
                const auto a=read<std::uint64_t>(first+i,false);
                const auto b=read<std::uint64_t>(second+i,false);
                const auto selection=mask ? read<std::uint64_t>(mask+i,false):low;
                if(integer_inputs&1)invalid|=a&~low;
                if(integer_inputs&2)invalid|=b&~low;
                if(integer_inputs&4)invalid|=selection&~low;
                const auto selected=truth(a)&truth(selection);
                total+=selected>>7;
                joint+=(selected&truth(b))>>7;
            }
            // Widen before summing the byte lanes, whose total can exceed255.
            for(unsigned shift=0;shift<64;shift+=8) {
                counts[0]+=(total>>shift)&0xff;
                counts[1]+=(joint>>shift)&0xff;
            }
        }
        for(;i<end;++i) {
            const auto a=first[i],b=second[i],selection=mask ? mask[i]:static_cast<unsigned char>(1);
            invalid|=((integer_inputs&1) && a>1) || ((integer_inputs&2) && b>1) ||
                ((integer_inputs&4) && selection>1);
            counts[0]+=(a!=0 && selection!=0);
            counts[1]+=(a!=0 && b!=0 && selection!=0);
        }
        counts[2]|=invalid!=0;
    });
    for(const auto& counts:partial) {
        single[0]+=counts[0];single[1]+=counts[1];single[2]|=counts[2];
    }
    return single;
}
}
// Dense spans have been authenticated by the binding. Each operand is
// classified independently, so different builtin dtypes can share the loop.
IntersectionCounts coloc_intersection_dense(const void* first,const void* second,
    std::size_t n,Type a,Type b) {
    const auto* x=static_cast<const unsigned char*>(first);
    const auto* y=static_cast<const unsigned char*>(second);
    const auto item0=coloc_itemsize(a),item1=coloc_itemsize(b);
    if(item0==1 && item1==1)
        return coloc_intersection_bytes(x,y,nullptr,n,unsigned(a!=Type::boolean)|unsigned(b!=Type::boolean)*2);

    // Float binary controls use integer operations throughout. In particular,
    // signed zero is valid, and a signaling NaN reaches the staged path unread
    // by any floating-point comparison here.
    const auto one=[](Type type) -> std::uint64_t {
        if(type==Type::float32)return 0x3f800000;
        if(type==Type::float64)return 0x3ff0000000000000ULL;
        return 1;
    };
    const auto one0=one(a),one1=one(b);
    const auto flags0=low_bits_mask(item0) >> unsigned(floating(a));
    const auto flags1=low_bits_mask(item1) >> unsigned(floating(b));
    const int workers=detail::active_workers(n);
    IntersectionCounts single{};
    std::vector<IntersectionCounts> partial(workers>1 ? std::size_t(workers):0);
    detail::partition(n,workers,[&](auto begin,auto end,auto worker) {
        auto& counts=workers==1 ? single:partial[worker];
#ifdef SKIMAGEXX_AVX2
        if(a==b && has_avx2())
            begin+=simd::coloc_intersection_wide_avx2(x+begin*item0,y+begin*item1,
                end-begin,item0,one0,flags0,counts);
#endif
        const auto scan=[&](const auto& load0,const auto& load1) {
            for(auto i=begin;i<end;++i) {
                const auto v=load0(i)&flags0,w=load1(i)&flags1;
                const bool hit0=a==Type::boolean ? v!=0:v==one0;
                const bool hit1=b==Type::boolean ? w!=0:w==one1;
                counts[2]|=(a!=Type::boolean && v && !hit0) || (b!=Type::boolean && w && !hit1);
                counts[0]+=hit0;
                counts[1]+=hit0 && hit1;
            }
        };
        const auto load=[](const unsigned char* data,std::size_t item,std::size_t i) -> std::uint64_t {
            switch(item) {
            case 1:return data[i];
            case 2:return read<std::uint16_t>(data+i*2,false);
            case 4:return read<std::uint32_t>(data+i*4,false);
            default:return read<std::uint64_t>(data+i*8,false);
            }
        };
        const auto scan_fixed=[&](auto width) {
            scan([&](auto i){return load(x,width,i);},[&](auto i){return load(y,width,i);});
        };
        if(item0==8 && item1==8)scan_fixed(std::integral_constant<std::size_t,8>{});
        else if(item0==4 && item1==4)scan_fixed(std::integral_constant<std::size_t,4>{});
        else if(item0==2 && item1==2)scan_fixed(std::integral_constant<std::size_t,2>{});
        else scan([&](auto i){return load(x,item0,i);},[&](auto i){return load(y,item1,i);});
    });
    for(const auto& counts:partial) {
        single[0]+=counts[0];single[1]+=counts[1];single[2]|=counts[2];
    }
    return single;
}

// A dense bool image against the same layout with its leading axis reversed.
IntersectionCounts coloc_intersection_reverse_rows(const void* first,const void* second,
    std::size_t n,std::size_t width) {
    const auto* x=static_cast<const unsigned char*>(first);
    const auto* y=static_cast<const unsigned char*>(second);
    const auto workers=detail::active_workers(n);
    IntersectionCounts single{};
    std::vector<IntersectionCounts> partial(workers>1 ? std::size_t(workers):0);
    detail::partition(n/width,workers,[&](auto begin,auto end,auto worker) {
        if(begin==end)return;
        auto& counts=workers==1 ? single:partial[worker];
#ifdef SKIMAGEXX_AVX2
        if(width%32==0 && has_avx2()) {
            simd::coloc_intersection_reverse_avx2(x+begin*width,y-begin*width,end-begin,width,counts);
            return;
        }
#endif
        for(auto row=begin;row<end;++row) {
            const auto* a=x+row*width;
            const auto* b=y-row*width;
            for(std::size_t i=0;i<width;++i) {
                counts[0]+=a[i]!=0;
                counts[1]+=a[i]!=0 && b[i]!=0;
            }
        }
    });
    for(const auto& counts:partial) {
        single[0]+=counts[0];single[1]+=counts[1];
    }
    return single;
}

IntersectionCounts coloc_intersection_strided(const ColocView& a,const ColocView& b,
    const ColocView* mask,unsigned integer_inputs) {
    const auto n=coloc_validate(a);coloc_validate(b);shapes(a,b);
    if(mask){coloc_validate(*mask);shapes(a,*mask);}
    if(!n)return {};
    const auto order=keep_order(a,&b);
    std::ptrdiff_t step0,step1,step_mask=0;
    if(!single_stride(a,order,step0) || !single_stride(b,order,step1) ||
        (mask && !single_stride(*mask,order,step_mask))) {
        if(!inner_contiguous(a,order) || !inner_contiguous(b,order) ||
            (mask && !inner_contiguous(*mask,order)))return {0,0,1};
        IntersectionCounts counts{};
        Rows<3> row(a,order,{&a,&b,mask},0,n);
        while(const auto length=row.available()) {
            const auto part=coloc_intersection_bytes(row.data(0,1),row.data(1,1),row.data(2,1),length,integer_inputs);
            counts[0]+=part[0];counts[1]+=part[1];
            if(part[2])return {0,0,1};
            row.advance(length);
        }
        return counts;
    }
    const auto* first=static_cast<const unsigned char*>(a.data);
    const auto* second=static_cast<const unsigned char*>(b.data);
    const auto* roi=mask ? static_cast<const unsigned char*>(mask->data):nullptr;
    if(step0==1 && step1==1 && (!mask || step_mask==1))
        return coloc_intersection_bytes(first,second,roi,n,integer_inputs);
    IntersectionCounts counts{};
    const auto scan=[&](auto stride0,auto stride1) {
        unsigned invalid=0;
        for(std::size_t i=0;i<n;++i) {
            const auto index=std::ptrdiff_t(i);
            const auto x=first[index*stride0],y=second[index*stride1];
            const auto selected=roi ? roi[index*step_mask]:static_cast<unsigned char>(1);
            if(integer_inputs&1)invalid|=x;
            if(integer_inputs&2)invalid|=y;
            if(integer_inputs&4)invalid|=selected;
            counts[0]+=(x!=0 && selected!=0);
            counts[1]+=(x!=0 && y!=0 && selected!=0);
        }
        counts[2]=invalid>1;
    };
    if(step0==2 && step1==2)scan(std::integral_constant<int,2>{},std::integral_constant<int,2>{});
    else scan(step0,step1);
    return counts;
}
int coloc_finite_min_negative(const ColocView& v) {
    const auto n=coloc_validate(v);
    if(!n)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    if(!v.swapped && (v.type==Type::float32 || v.type==Type::float64)) {
        const auto order=keep_order(v,nullptr);
        const auto item=coloc_itemsize(v.type);
        std::ptrdiff_t step;
        const bool dense=single_stride(v,order,step) && step==std::ptrdiff_t(item);
        if(dense || inner_contiguous(v,order)) {
            Rows<1> row(v,order,{&v},0,n,dense);
            int negative=0;
            while(const auto length=row.available()) {
                const auto* data=row.data(0,item);
                const auto status=v.type==Type::float32 ? dense_finite_min<std::uint32_t>(data,length,0x7f800000u)
                    :dense_finite_min<std::uint64_t>(data,length,0x7ff0000000000000ULL);
                if(status<0)return -1;
                negative|=status;row.advance(length);
            }
            return negative;
        }
    }
    if(!coloc_all_finite(v))return -1;
    return int(coloc_min_negative(v).value);
}
ColocPredicate coloc_min_negative(const ColocView& v) {
    const auto n=coloc_validate(v);if(!n)throw std::invalid_argument("zero-size array to reduction operation minimum which has no identity");
    bool negative=false,has_nan=false;const auto order=keep_order(v,nullptr);const ReadAccess input(v,order,n);
    const auto item=coloc_itemsize(v.type);
    const auto sign=std::uint64_t(1)<<(item*8-1);
    if(!floating(v.type) && !signed_integer(v.type))return {false,0};
    const auto scan=[&](const unsigned char* data) {
        const auto value=bits(v,data,item);has_nan|=nan(v.type,value);
        negative|=(value&sign) && !zero(v.type,value);
    };
    if(inner_contiguous(v,order) && !input.affine) {
        Rows<1> row(v,order,{&v},0,n);
        while(const auto length=row.available()) {
            const auto* data=row.data(0,item);
            for(std::size_t i=0;i<length;++i)scan(data+i*item);
            row.advance(length);
        }
    } else for(std::size_t i=0;i<n;++i)scan(input[i]);
    return {negative && !has_nan,0};
}
ColocExecution coloc_gather_into(const ColocView& v,const ColocView& mask_view,ColocBuffer out,std::size_t selected) {
    const auto n=coloc_validate(v);coloc_validate(mask_view);boolean_required(mask_view);shapes(v,mask_view);
    output_check(out,multiply(selected,coloc_itemsize(v.type)),v,&mask_view);
    if(coloc_count_nonzero(mask_view)!=selected)throw std::invalid_argument("colocalization gather count changed");
    const auto order=c_order(v);const int workers=detail::active_workers(n);
    const ReadAccess input(v,order,n),selection(mask_view,order,n);
    const auto item=coloc_itemsize(v.type);
    if(workers==1) {
        auto* target=static_cast<unsigned char*>(out.data);
        auto gather=[&](auto copy) {
            for(std::size_t i=0;i<n;++i)if(*selection[i]) {
                copy(target,input[i]);target+=item;
            }
        };
        switch(item) {
            case 1:gather([](auto* to,const auto* from){*to=*from;});break;
            case 2:gather([](auto* to,const auto* from){std::memcpy(to,from,2);});break;
            case 4:gather([](auto* to,const auto* from){std::memcpy(to,from,4);});break;
            case 8:gather([](auto* to,const auto* from){std::memcpy(to,from,8);});break;
        }
        return {1,false};
    }
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
    fp_check();
    std::ptrdiff_t step=0;
    if(n && !single_stride(v,order,step)) {
        if(!coloc_all_finite(v))throw ColocProviderRequired("colocalization nonfinite sum is a provider frontier");
        throw ColocProviderRequired("colocalization irregular floating reduction iterator is a provider frontier");
    }
    // NpyIter reduction: single-stride aligned/native input needs no cast
    // buffer in2.5 and GROWINNER exposes the entire inner loop. In1.24 the
    // buffered reduction loop stops at the supplied iterator buffer size.
    const bool needs_buffer=v.swapped || (reinterpret_cast<std::uintptr_t>(v.data)%item) || (magnitude(step)%item);
    const auto block=(plan.profile==ColocProfile::numpy_1244 || needs_buffer) ? plan.buffer_elements:std::max(n,std::size_t(1));
    Capture capture;
    if(v.type==Type::float64) {
        const auto* data=static_cast<const unsigned char*>(v.data);
        const auto sum=!v.swapped && step==8
            ? reduce_blocks<double>(n,block,[&](auto i){return read<double>(data+8*i,false);})
            : reduce_blocks<double>(n,block,[&](auto i){return read<double>(data+std::ptrdiff_t(i)*step,v.swapped);});
        std::memcpy(&result.bits,&sum,8);
    } else if(v.type==Type::float32) {
        const auto* data=static_cast<const unsigned char*>(v.data);
        const auto sum=!v.swapped && step==4
            ? reduce_blocks<float>(n,block,[&](auto i){return read<float>(data+4*i,false);})
            : reduce_blocks<float>(n,block,[&](auto i){return read<float>(data+std::ptrdiff_t(i)*step,v.swapped);});
        std::uint32_t word;std::memcpy(&word,&sum,4);result.bits=word;
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
    const auto exponent=v.type==Type::float16 ? 0x7c00ULL:v.type==Type::float32 ? 0x7f800000ULL:0x7ff0000000000000ULL;
    if((result.bits&exponent)==exponent && !coloc_all_finite(v))
        throw ColocProviderRequired("colocalization nonfinite sum is a provider frontier");
    result.fp_status=capture.flags();return result;
}
namespace {
template<class T> ColocScalar packed_scalar(Type type,T value) {
    ColocScalar result;result.type=type;
    if constexpr(sizeof(T)==4) {
        std::uint32_t raw;std::memcpy(&raw,&value,4);result.bits=raw;
    } else std::memcpy(&result.bits,&value,8);
    return result;
}
template<class T,class Step0,class Step1> ColocScalar product_sum(const ColocView& a,const ColocView* b,
    std::size_t n,std::size_t block,Step0 step0,Step1 step1) {
    const auto* first=static_cast<const unsigned char*>(a.data);
    const auto* second=b ? static_cast<const unsigned char*>(b->data):first;
    const auto load=[](const unsigned char* data,std::ptrdiff_t offset){return read<T>(data+offset,false);};
    Capture capture;
    T sum;
    if(!b)sum=reduce_blocks<T>(n,block,[&](auto i){const auto x=load(first,std::ptrdiff_t(i)*step0);return T(x*x);});
    else if(b->type==Type::boolean)
        sum=reduce_blocks<T>(n,block,[&](auto i){const auto j=std::ptrdiff_t(i);return T(load(first,j*step0)*T(second[j*step1]!=0));});
    else sum=reduce_blocks<T>(n,block,[&](auto i){const auto j=std::ptrdiff_t(i);return T(load(first,j*step0)*load(second,j*step1));});
    // If either original stage can signal, replay the two separate stages
    // so callbacks, warnings and nonfinite scalar payloads keep their order.
    if(capture.flags() || !std::isfinite(sum))throw ColocProviderRequired("colocalization product or sum requires separate stages");
    return packed_scalar(a.type,sum);
}
template<class T> ColocScalar product_sum(const ColocView& a,const ColocView* b,
    std::size_t n,std::size_t block,std::ptrdiff_t step0,std::ptrdiff_t step1) {
    using Dense=std::integral_constant<std::ptrdiff_t,sizeof(T)>;
    if(step0==sizeof(T) && step1==sizeof(T))return product_sum<T,Dense,Dense>(a,b,n,block,Dense{},Dense{});
    if(step0==sizeof(T) && step1==1)
        return product_sum<T,Dense,std::integral_constant<std::ptrdiff_t,1>>(a,b,n,block,Dense{},{});
    return product_sum<T,std::ptrdiff_t,std::ptrdiff_t>(a,b,n,block,step0,step1);
}
template<class T,bool Overlap>
int manders(const ColocView&,const ColocView&,const ColocView*,const ColocOrder&,
            std::size_t,std::size_t,std::size_t,bool,std::array<ColocScalar,3>&,
            bool product_only=false);

template<class T,bool Boolean> ColocScalar row_product_sum(const ColocView& a,const ColocView* b,
    const ColocOrder& order,std::size_t n,std::size_t block) {
#ifdef SKIMAGEXX_AVX2
    if constexpr(Boolean) {
        if(has_avx2()) {
            std::array<ColocScalar,3> result;
            if(manders<T,false>(a,*b,nullptr,order,n,n,block,false,result,true)==3)return result[1];
        }
    }
#endif
    constexpr auto second_item=Boolean ? 1:sizeof(T);
    Rows<2> row(a,order,{&a,b},0,n);
    const unsigned char* first=nullptr;
    const unsigned char* second=nullptr;
    std::size_t remaining=0;
    Capture capture;
    const auto sum=reduce_blocks<T>(n,block,[&](auto) {
        if(!remaining) {
            remaining=row.available();
            first=row.data(0,sizeof(T));second=row.data(1,second_item);
            row.advance(remaining);
        }
        const auto x=read<T>(first,false);
        auto y=x;
        if constexpr(Boolean)y=T(*second!=0);
        else if(b)y=read<T>(second,false);
        first+=sizeof(T);if(b)second+=second_item;--remaining;
        return T(x*y);
    });
    if(capture.flags() || !std::isfinite(sum))
        throw ColocProviderRequired("colocalization product or sum requires separate stages");
    return packed_scalar(a.type,sum);
}
}
// Private binding route. Both inputs follow the materialized product's
// memory order, so the reduction retains NumPy's original grouping.
ColocScalar coloc_product_sum(const ColocView& a,const ColocView* b,ColocReductionPlan plan) {
    const auto n=coloc_validate(a);
    if(b){coloc_validate(*b);shapes(a,*b);}
    if(!plan.buffer_elements)throw ColocProviderRequired("colocalization product sum requires a nonzero reduction buffer");
    if(!n || a.swapped || (a.type!=Type::float32 && a.type!=Type::float64) ||
        (b && (b->swapped || (b->type!=Type::boolean && b->type!=a.type))))
        throw ColocProviderRequired("colocalization product requires a native real dtype");
    const auto order=keep_order(a,b);
    std::ptrdiff_t step0,step1=std::ptrdiff_t(coloc_itemsize(a.type));
    fp_check();
    const auto block=plan.profile==ColocProfile::numpy_1244 ? plan.buffer_elements:n;
    if(!single_stride(a,order,step0) || (b && !single_stride(*b,order,step1))) {
        if(!inner_contiguous(a,order) || (b && !inner_contiguous(*b,order)))
            throw ColocProviderRequired("colocalization product requires a shared affine or inner-contiguous order");
        const bool boolean=b && b->type==Type::boolean;
        if(a.type==Type::float32)
            return boolean ? row_product_sum<float,true>(a,b,order,n,block):row_product_sum<float,false>(a,b,order,n,block);
        return boolean ? row_product_sum<double,true>(a,b,order,n,block):row_product_sum<double,false>(a,b,order,n,block);
    }
    if(a.type==Type::float32)return product_sum<float>(a,b,n,block,step0,step1);
    return product_sum<double>(a,b,n,block,step0,step1);
}
namespace {
template<class T,bool Overlap>
auto manders_leaf(const unsigned char* first,const unsigned char* second,
                  std::size_t length,unsigned& status,[[maybe_unused]] bool avx2)
    -> std::array<T,Overlap ? 3:2> {
    using Sums=std::array<T,Overlap ? 3:2>;
    const auto read_terms=[&](std::size_t i) -> Sums {
        const T x=read<T>(first+i*sizeof(T),false);
        status|=!std::isfinite(x) ? 1u:0u;
        status|=x<T(0) ? 2u:0u;
        if constexpr(Overlap) {
            const T y=read<T>(second+i*sizeof(T),false);
            status|=!std::isfinite(y) ? 1u:0u;
            status|=y<T(0) ? 4u:0u;
            return {T(x*x),T(y*y),T(x*y)};
        } else return {x,T(x*T(second[i]!=0))};
    };
    const auto add_terms=[](Sums& sums,const Sums& terms) {
        for(std::size_t j=0;j<sums.size();++j)sums[j]+=terms[j];
    };
    Sums sums;
    std::size_t i=0;
    if(length<8)sums.fill(-T(0));
    else {
        const auto full=length&~std::size_t(7);
#ifdef SKIMAGEXX_AVX2
        if(avx2)simd::coloc_manders_leaf(first,second,full,Overlap,sums.data(),status);
        else
#endif
        {
            Sums lanes[8];
            for(std::size_t k=0;k<8;++k)lanes[k]=read_terms(k);
            for(std::size_t j=8;j<full;j+=8)
                for(std::size_t k=0;k<8;++k)add_terms(lanes[k],read_terms(j+k));
            for(std::size_t j=0;j<sums.size();++j)
                sums[j]=((lanes[0][j]+lanes[1][j])+(lanes[2][j]+lanes[3][j]))+
                        ((lanes[4][j]+lanes[5][j])+(lanes[6][j]+lanes[7][j]));
        }
        i=full;
    }
    for(;i<length;++i)add_terms(sums,read_terms(i));
    return sums;
}

template<class T,bool Overlap,class Leaf>
auto manders_pairwise(std::size_t n,const Leaf& leaf) -> std::array<T,Overlap ? 3:2> {
    if(n<=128)return leaf(n);
    const auto left=(n/2)&~std::size_t(7);
    auto a=manders_pairwise<T,Overlap>(left,leaf);
    const auto b=manders_pairwise<T,Overlap>(n-left,leaf);
    for(std::size_t j=0;j<a.size();++j)a[j]+=b[j];
    return a;
}
template<class T,bool Overlap,class Leaf>
int manders_reduce(std::size_t selected,std::size_t block,const Leaf& leaf,
                   const unsigned& status,const Capture& capture,Type type,
                   std::array<ColocScalar,3>& result,bool product_only) {
    using Sums=std::array<T,Overlap ? 3:2>;
    Sums sums{};
    while(selected) {
        const auto take=std::min(block,selected);
        const auto partial=manders_pairwise<T,Overlap>(take,leaf);
        for(std::size_t j=0;j<sums.size();++j)sums[j]+=partial[j];
        selected-=take;
    }
    if((status&1) || capture.flags())return 0;
    if(product_only) {
        if(!std::isfinite(sums[1]))return 0;
        result[1]=packed_scalar(type,sums[1]);
        return 3;
    }
    if(status&2)return 1;
    if(status&4)return 2;
    for(const auto sum:sums)if(!std::isfinite(sum))return 0;
    // Scalar arithmetic stays in Python with its exact NumPy promotion and
    // pow behavior. Only admit results for which it cannot call an error
    // handler before the original method would have read the dot operands.
    if constexpr(Overlap) {
        const T product=sums[0]*sums[1];
        if(!std::isnormal(product))return 0;
        if(sums[2]!=T(0)) {
            const T quotient=sums[2]/std::sqrt(product);
            if(!(quotient>=2*std::numeric_limits<T>::min()) || !std::isfinite(quotient))return 0;
        }
    } else if(sums[0]!=T(0)) {
        const T quotient=sums[1]/sums[0];
        if(!std::isfinite(quotient))return 0;
    }
    if(capture.flags())return 0;
    for(std::size_t j=0;j<sums.size();++j)result[j]=packed_scalar(type,sums[j]);
    return 3;
}

template<class T,bool Overlap>
int manders_contiguous(const unsigned char* first,const unsigned char* second,
                       std::size_t n,std::size_t block,Type type,
                       std::array<ColocScalar,3>& result) {
    Capture capture;
    unsigned status=0;
    bool avx2=false;
#ifdef SKIMAGEXX_AVX2
    avx2=has_avx2();
#endif
    const auto leaf=[&](std::size_t length) {
        const auto sums=manders_leaf<T,Overlap>(first,second,length,status,avx2);
        first+=length*sizeof(T);
        second+=length*(Overlap ? sizeof(T):1);
        return sums;
    };
    return manders_reduce<T,Overlap>(n,block,leaf,status,capture,type,result,false);
}

template<class T,bool Overlap>
int manders(const ColocView& a,const ColocView& b,const ColocView* mask,
            const ColocOrder& order,std::size_t n,std::size_t selected,
            std::size_t block,bool dense,std::array<ColocScalar,3>& result,bool product_only) {
    constexpr auto second_item=Overlap ? sizeof(T):1;
    Rows<3> row(a,order,{&a,&b,mask},0,n,dense);
    unsigned status=0;
    Capture capture;
    bool avx2=false;
#ifdef SKIMAGEXX_AVX2
    avx2=has_avx2();
#endif
    const auto leaf=[&](std::size_t length) {
        std::array<unsigned char,128*sizeof(T)> first_buffer,second_buffer;
        const auto* first=row.data(0,sizeof(T));
        const auto* second=row.data(1,second_item);
        if(!mask && row.available()>=length)row.advance(length);
        else {
            // Pack only leaves that cross a row boundary or need boolean
            // selection. Excluded samples must never enter the reduction tree.
            std::size_t copied=0;
            while(copied<length) {
                const auto run=row.available();
                if(!run)throw ColocProviderRequired("colocalization selection count changed");
                const auto* x=row.data(0,sizeof(T));
                const auto* y=row.data(1,second_item);
                const auto* roi=row.data(2,1);
                if(roi) {
                    std::size_t i=0;
                    for(;i<run && copied<length;++i)if(roi[i]) {
                        std::memcpy(first_buffer.data()+copied*sizeof(T),x+i*sizeof(T),sizeof(T));
                        std::memcpy(second_buffer.data()+copied*second_item,y+i*second_item,second_item);
                        ++copied;
                    }
                    row.advance(i);
                } else {
                    const auto take=std::min(run,length-copied);
                    std::memcpy(first_buffer.data()+copied*sizeof(T),x,take*sizeof(T));
                    std::memcpy(second_buffer.data()+copied*second_item,y,take*second_item);
                    row.advance(take);copied+=take;
                }
            }
            first=first_buffer.data();second=second_buffer.data();
        }
        return manders_leaf<T,Overlap>(first,second,length,status,avx2);
    };
    return manders_reduce<T,Overlap>(selected,block,leaf,status,capture,a.type,result,product_only);
}
}

int coloc_contiguous_manders(const void* a,const void* b,std::size_t n,Type type,
                             ColocReductionPlan plan,bool overlap,std::array<ColocScalar,3>& result) {
    if(!n || !plan.buffer_elements || !coloc_fp_supported())return 0;
    const auto block=plan.profile==ColocProfile::numpy_1244 ? plan.buffer_elements:n;
    const auto* first=static_cast<const unsigned char*>(a);
    const auto* second=static_cast<const unsigned char*>(b);
    if(type==Type::float32)
        return overlap ? manders_contiguous<float,true>(first,second,n,block,type,result)
                       : manders_contiguous<float,false>(first,second,n,block,type,result);
    return overlap ? manders_contiguous<double,true>(first,second,n,block,type,result)
                   : manders_contiguous<double,false>(first,second,n,block,type,result);
}

// Private fused route. Zero requests staged replay, including exceptional
// arithmetic, empty selections, and incompatible NumPy traversal plans.
int coloc_dense_manders(const ColocView& a,const ColocView& b,const ColocView* mask,
                       ColocReductionPlan plan,bool overlap,std::array<ColocScalar,3>& result) {
    const auto n=coloc_validate(a);
    coloc_validate(b);shapes(a,b);
    if(mask){coloc_validate(*mask);boolean_required(*mask);shapes(a,*mask);}
    if(!n || !plan.buffer_elements || a.swapped || b.swapped ||
        (a.type!=Type::float32 && a.type!=Type::float64) ||
        b.type!=(overlap ? a.type:Type::boolean) || !coloc_fp_supported())return 0;
    const auto order=mask ? c_order(a):keep_order(a,&b);
    const auto same_order=[&](const ColocView& view) {
        const auto own=keep_order(view,nullptr);
        return std::equal(order.axes.begin(),order.axes.begin()+order.rank,own.axes.begin());
    };
    if(!mask && (!same_order(a) || (overlap && !same_order(b))))return 0;
    const auto dense_in_order=[&](const ColocView& view) {
        std::ptrdiff_t step;
        return single_stride(view,order,step) && step==std::ptrdiff_t(coloc_itemsize(view.type));
    };
    const bool dense_a=dense_in_order(a);
    const bool dense=dense_a && dense_in_order(b) && (!mask || dense_in_order(*mask));
    if(!dense && (!inner_contiguous(a,order) || !inner_contiguous(b,order) ||
                  (mask && !inner_contiguous(*mask,order))))return 0;
    const auto item=coloc_itemsize(a.type);
    if(!mask && !overlap) {
        // NumPy can split an irregular input reduction at different
        // boundaries from its contiguous, materialized product.
        if(!dense_a && n>plan.buffer_elements)return 0;
        if(reinterpret_cast<std::uintptr_t>(a.data)%item ||
            std::any_of(a.strides.data,a.strides.data+a.strides.size,
                        [&](auto step){return step%std::ptrdiff_t(item)!=0;}))return 0;
    }
    const auto selected=mask ? std::size_t(count_impl(*mask,nullptr)):n;
    if(!selected)return 0;
    if(selected==n)mask=nullptr;
    const auto block=plan.profile==ColocProfile::numpy_1244 ? plan.buffer_elements:selected;
    if(a.type==Type::float32)
        return overlap ? manders<float,true>(a,b,mask,order,n,selected,block,dense,result)
                       : manders<float,false>(a,b,mask,order,n,selected,block,dense,result);
    return overlap ? manders<double,true>(a,b,mask,order,n,selected,block,dense,result)
                   : manders<double,false>(a,b,mask,order,n,selected,block,dense,result);
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
    auto store=[&](std::size_t i,const unsigned char* first,const unsigned char* second) {
        if(!floating(a.type)) {
            auto x=bits(a,first,item);if(a.type==Type::boolean)x=x!=0;
            auto y=b ? bits(*b,second,second_item):x;if(b && b->type==Type::boolean)y=y!=0;
            const auto value=(x*y)&product_mask;
            if(item==1)write<std::uint8_t>(out.data,i,std::uint8_t(value));
            else if(item==2)write<std::uint16_t>(out.data,i,std::uint16_t(value));
            else if(item==4)write<std::uint32_t>(out.data,i,std::uint32_t(value));
            else write<std::uint64_t>(out.data,i,value);
        } else if(a.type==Type::float64) {
            const double x=read<double>(first,a.swapped),y=b ? (b->type==Type::boolean ? double(*second!=0):read<double>(second,b->swapped)):x;
            write<double>(out.data,i,double(x*y));
        } else {
            const float x=a.type==Type::float32 ? read<float>(first,a.swapped):real<float>(a,first);
            const float y=b ? (b->type==Type::boolean ? float(*second!=0):(b->type==Type::float32 ? read<float>(second,b->swapped):real<float>(*b,second))):x;
            const float value=float(x*y);
            if(a.type==Type::float16)write<std::uint16_t>(out.data,i,half(value));else write<float>(out.data,i,value);
        }
    };
    const bool rows=(!first_input.affine || (b && !second_input.affine)) &&
        inner_contiguous(a,order) && (!b || inner_contiguous(*b,order));
    auto body=[&](std::size_t begin,std::size_t end) {
        if(rows && begin<end) {
            Rows<2> row(a,order,{&a,b},begin,end);
            while(const auto length=row.available()) {
                const auto* first=row.data(0,item);
                const auto* second=row.data(1,second_item);
                for(std::size_t i=0;i<length;++i)
                    store(begin+i,first+i*item,b ? second+i*second_item:first+i*item);
                begin+=length;row.advance(length);
            }
        } else for(auto i=begin;i<end;++i) {
            const auto* first=first_input[i];
            store(i,first,b ? second_input[i]:first);
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
