#include "skimagexx/metrics.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
namespace skimagexx::simd {
namespace {
// The Windows CRT's INFINITY macro can raise overflow under /fp:strict.
constexpr double metric_infinity=std::numeric_limits<double>::infinity();

__m128i low32(__m256i x){x=_mm256_shuffle_epi32(x,_MM_SHUFFLE(2,0,2,0));return _mm_unpacklo_epi64(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));}
__m256d unsigned32(__m128i value){return _mm256_add_pd(_mm256_cvtepi32_pd(value),_mm256_mul_pd(_mm256_cvtepi32_pd(_mm_srli_epi32(value,31)),_mm256_set1_pd(4294967296.)));}
__m256d wide(__m256i value,bool is_signed) {
    const auto high=low32(_mm256_srli_epi64(value,32));
    return _mm256_add_pd(_mm256_mul_pd(is_signed ? _mm256_cvtepi32_pd(high):unsigned32(high),_mm256_set1_pd(4294967296.)),unsigned32(low32(value)));
}
__m128 half4(__m128i packed,bool swap) {
    if(swap)packed=_mm_shuffle_epi8(packed,_mm_setr_epi8(1,0,3,2,5,4,7,6,-1,-1,-1,-1,-1,-1,-1,-1));
    const auto value=_mm_cvtepu16_epi32(packed),fraction=_mm_and_si128(value,_mm_set1_epi32(1023)),
               exponent=_mm_srli_epi32(_mm_and_si128(value,_mm_set1_epi32(0x7c00)),10),
               sign=_mm_slli_epi32(_mm_and_si128(value,_mm_set1_epi32(0x8000)),16);
    const auto normal=_mm_or_si128(sign,_mm_or_si128(_mm_slli_epi32(_mm_add_epi32(exponent,_mm_set1_epi32(112)),23),_mm_slli_epi32(fraction,13)));
    const auto small=_mm_or_si128(sign,_mm_castps_si128(_mm_mul_ps(_mm_cvtepi32_ps(fraction),_mm_set1_ps(0x1p-24f))));
    const auto special=_mm_or_si128(sign,_mm_or_si128(_mm_set1_epi32(0x7f800000),_mm_slli_epi32(fraction,13)));
    auto result=_mm_blendv_epi8(normal,small,_mm_cmpeq_epi32(exponent,_mm_setzero_si128()));
    result=_mm_blendv_epi8(result,special,_mm_cmpeq_epi32(exponent,_mm_set1_epi32(31)));
    return _mm_castsi128_ps(result);
}
__m128i swap16(__m128i value) {
    return _mm_shuffle_epi8(value,_mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14));
}
__m128i swap32(__m128i value) {
    return _mm_shuffle_epi8(value,_mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12));
}
__m256i swap64(__m256i value) {
    return _mm256_shuffle_epi8(value,_mm256_setr_epi8(7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8,
                                                        7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8));
}
template<class T> __m256d raw(const unsigned char* p,bool swap=false) {
    if constexpr(sizeof(T)==1) {std::uint32_t word;std::memcpy(&word,p,4);const auto v=_mm_cvtsi32_si128(int(word));return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi8_epi32(v):_mm_cvtepu8_epi32(v));}
    else if constexpr(sizeof(T)==2){auto v=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));if(swap)v=swap16(v);return _mm256_cvtepi32_pd(std::is_signed_v<T> ? _mm_cvtepi16_epi32(v):_mm_cvtepu16_epi32(v));}
    else if constexpr(sizeof(T)==4){auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));if(swap)v=swap32(v);return std::is_signed_v<T> ? _mm256_cvtepi32_pd(v):unsigned32(v);}
    else {auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));if(swap)v=swap64(v);return wide(v,std::is_signed_v<T>);}
}
template<class T,bool Boolean=false> __m256d narrow(const unsigned char* p,std::ptrdiff_t step,bool swap) {
    // Four independent byte-safe reads: no widened gather can cross a row,
    // allocation or guard page for 8/16-bit elements or negative strides.
    const auto lane=[&](int i) {
        T value;
        if constexpr(sizeof(T)==2) {
            const auto* source=p+std::ptrdiff_t(i)*step;
            if(swap) {const unsigned char bytes[2]{source[1],source[0]};std::memcpy(&value,bytes,2);}
            else std::memcpy(&value,source,2);
        } else std::memcpy(&value,p+std::ptrdiff_t(i)*step,1);
        if constexpr(Boolean)return int(value!=0);
        else return int(value);
    };
    return _mm256_cvtepi32_pd(_mm_setr_epi32(lane(0),lane(1),lane(2),lane(3)));
}
__m256d load(const MetricImageView& view,std::ptrdiff_t offset,std::ptrdiff_t step) {
    const auto* p=static_cast<const unsigned char*>(view.data)+offset;
    // A four-lane caller has already bounded this logical run to its row.
    // Read exactly those four elements in ascending physical order, then
    // restore logical order. Avoid recursive generic dtype dispatch for the
    // common native floating reverse; keep conversion before permutation as
    // in the generic path (including signaling-NaN payloads/status).
    if(!view.half && !view.swap) {
        if(view.type==PixelType::float64 && step==-8)
            return _mm256_permute4x64_pd(_mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p-24))),_MM_SHUFFLE(0,1,2,3));
        if(view.type==PixelType::float32 && step==-4)
            return _mm256_permute4x64_pd(_mm256_cvtps_pd(_mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p-12)))),_MM_SHUFFLE(0,1,2,3));
    }
    if(step==0)return _mm256_set1_pd(detail::metric_read(view,offset));
    if(view.half && (step==2 || step==-2)) {
        auto value=half4(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(step==2 ? p:p-6)),view.swap);
        if(step<0)value=_mm_shuffle_ps(value,value,_MM_SHUFFLE(0,1,2,3));
        return _mm256_cvtps_pd(value);
    }
    if(view.half) {
        // AVX2 has no 16-bit gather, and widened gathers may cross the final
        // element/allocation. Pack four exact byte-safe reads, then share the
        // existing integer/float binary16 decoder (no F16C requirement).
        std::uint16_t words[4];
        for(int i=0;i<4;++i)std::memcpy(words+i,p+std::ptrdiff_t(i)*step,2);
        const auto packed=_mm_setr_epi16(short(words[0]),short(words[1]),short(words[2]),short(words[3]),0,0,0,0);
        return _mm256_cvtps_pd(half4(packed,view.swap));
    }
    const auto bytes=std::ptrdiff_t(pixel_size(view.type));
    if(!view.half && step==-bytes)
        return _mm256_permute4x64_pd(load(view,offset-3*bytes,bytes),_MM_SHUFFLE(0,1,2,3));
    if(!view.half && step!=bytes && bytes<=2) {
        switch(view.type) {
            case PixelType::boolean:return narrow<std::uint8_t,true>(p,step,false);
            case PixelType::uint8:return narrow<std::uint8_t>(p,step,false);
            case PixelType::int8:return narrow<std::int8_t>(p,step,false);
            case PixelType::uint16:return narrow<std::uint16_t>(p,step,view.swap);
            case PixelType::int16:return narrow<std::int16_t>(p,step,view.swap);
            default:break;
        }
    }
    if(!view.swap && !view.half && step!=bytes && bytes>=4) {
        const auto indices=_mm256_set_epi64x(3*step,2*step,step,0);
        switch(view.type) {
            case PixelType::float32:return _mm256_cvtps_pd(_mm256_i64gather_ps(reinterpret_cast<const float*>(p),indices,1));
            case PixelType::float64:return _mm256_i64gather_pd(reinterpret_cast<const double*>(p),indices,1);
            case PixelType::int32:return _mm256_cvtepi32_pd(_mm256_i64gather_epi32(reinterpret_cast<const int*>(p),indices,1));
            case PixelType::uint32:return unsigned32(_mm256_i64gather_epi32(reinterpret_cast<const int*>(p),indices,1));
            case PixelType::int64:return wide(_mm256_i64gather_epi64(reinterpret_cast<const long long*>(p),indices,1),true);
            case PixelType::uint64:return wide(_mm256_i64gather_epi64(reinterpret_cast<const long long*>(p),indices,1),false);
            default:break;
        }
    }
    if(step!=bytes) {
        alignas(32) double values[4];for(int i=0;i<4;++i)values[i]=detail::metric_read(view,offset+std::ptrdiff_t(i)*step);return _mm256_load_pd(values);
    }
    switch(view.type) {
        case PixelType::boolean:{std::uint32_t word;std::memcpy(&word,p,4);const auto v=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(word)));return _mm256_cvtepi32_pd(_mm_andnot_si128(_mm_cmpeq_epi32(v,_mm_setzero_si128()),_mm_set1_epi32(1)));}
        #define LOAD(tag,T) case PixelType::tag:return raw<T>(p,view.swap);
        LOAD(uint8,std::uint8_t) LOAD(int8,std::int8_t) LOAD(uint16,std::uint16_t) LOAD(int16,std::int16_t)
        LOAD(uint32,std::uint32_t) LOAD(int32,std::int32_t) LOAD(uint64,std::uint64_t) LOAD(int64,std::int64_t)
        #undef LOAD
        case PixelType::float32:{auto v=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p));if(view.swap)v=swap32(v);return _mm256_cvtps_pd(_mm_castsi128_ps(v));}
        case PixelType::float64:{auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));if(view.swap)v=swap64(v);return _mm256_castsi256_pd(v);}
        default:return _mm256_setzero_pd();
    }
}
}
bool metric_swapped_f32_range(const unsigned char* data,std::size_t count,MetricRawRange& result) {
    if(count<8)return false;
    const auto reverse=_mm256_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,
                                        3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
    const auto exponent=_mm256_set1_epi32(0x7f800000),sign=_mm256_set1_epi32(-2147483647-1);
    auto low=_mm256_set1_epi32(-1),high=_mm256_setzero_si256();
    std::size_t i=0;
    for(;i+8<=count;i+=8) {
        const auto bits=_mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data+4*i)),reverse);
        const auto exp=_mm256_and_si256(bits,exponent);
        const auto rejected=_mm256_or_si256(_mm256_cmpeq_epi32(exp,_mm256_setzero_si256()),_mm256_cmpeq_epi32(exp,exponent));
        if(_mm256_movemask_epi8(rejected))return false;
        // Integer ordering is exact for finite normal IEEE754 values. Reject
        // zeros/subnormals/NaNs/Inf so DAZ, zero ties and FP exception behavior
        // remain entirely in the unchanged scalar path. No FP instruction is
        // executed even speculatively, and false never modifies the output.
        const auto key=_mm256_xor_si256(bits,_mm256_or_si256(_mm256_srai_epi32(bits,31),sign));
        low=_mm256_min_epu32(low,key);high=_mm256_max_epu32(high,key);
    }
    alignas(32) std::uint32_t lows[8],highs[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lows),low);_mm256_store_si256(reinterpret_cast<__m256i*>(highs),high);
    std::uint32_t minimum=lows[0],maximum=highs[0];
    for(int lane=1;lane<8;++lane){if(lows[lane]<minimum)minimum=lows[lane];if(highs[lane]>maximum)maximum=highs[lane];}
    for(;i<count;++i) {
        std::uint32_t bits;std::memcpy(&bits,data+4*i,4);
        bits=((bits&0x000000ffu)<<24)|((bits&0x0000ff00u)<<8)|((bits&0x00ff0000u)>>8)|((bits&0xff000000u)>>24);
        const auto exp=bits&0x7f800000u;
        if(exp==0 || exp==0x7f800000u)return false;
        const auto key=bits^((bits&0x80000000u) ? 0xffffffffu:0x80000000u);
        if(key<minimum)minimum=key;
        if(key>maximum)maximum=key;
    }
    result.low=minimum^((minimum&0x80000000u) ? 0x80000000u:0xffffffffu);
    result.high=maximum^((maximum&0x80000000u) ? 0x80000000u:0xffffffffu);
    result.has_nan=false;
    return true;
}
bool metric_leaf_classify(const double* error,const double* energy,const double* values,std::size_t count,
                         MetricNormalization mode,MetricSums& result) {
    // Inspect exponent bits before any floating range comparison. Exceptional
    // values retain the original scalar classification and flagged replay.
    const auto exponent=_mm256_set1_epi64x(0x7ff0000000000000LL);
    const auto magnitude=_mm256_set1_epi64x(0x7fffffffffffffffLL);
    const bool range=mode==MetricNormalization::min_max,euclidean=mode==MetricNormalization::euclidean;
    auto invalid=_mm256_setzero_si256(),zeros=_mm256_setzero_si256();
    auto low=_mm256_set1_pd(metric_infinity),high=_mm256_set1_pd(-metric_infinity);
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto value=_mm256_loadu_pd(values+i),err=_mm256_loadu_pd(error+i);
        const auto exceptional=[&](const __m256d v){return _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(v),exponent),exponent);};
        invalid=_mm256_or_si256(invalid,_mm256_or_si256(exceptional(value),exceptional(err)));
        if(euclidean)invalid=_mm256_or_si256(invalid,exceptional(_mm256_loadu_pd(energy+i)));
        if(range) {
            zeros=_mm256_or_si256(zeros,_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(value),magnitude),_mm256_setzero_si256()));
            // No externally visible flags are consumed on this speculative
            // path. NaNs below still force the untouched scalar replay.
            low=_mm256_min_pd(low,value);high=_mm256_max_pd(high,value);
        }
    }
    if(_mm256_movemask_pd(_mm256_castsi256_pd(invalid)) || (range && _mm256_movemask_pd(_mm256_castsi256_pd(zeros))))return false;
    double minimum=metric_infinity,maximum=-metric_infinity;
    if(range) {
        alignas(32) double lows[4],highs[4];_mm256_store_pd(lows,low);_mm256_store_pd(highs,high);
        for(int j=0;j<4;++j){if(lows[j]<minimum)minimum=lows[j];if(highs[j]>maximum)maximum=highs[j];}
    }
    for(;i<count;++i) {
        if(!std::isfinite(values[i]) || !std::isfinite(error[i]) || (euclidean && !std::isfinite(energy[i])))return false;
        if(range){if(values[i]==0)return false;if(values[i]<minimum)minimum=values[i];if(values[i]>maximum)maximum=values[i];}
    }
    if(range && count){result.low=minimum;result.high=maximum;result.valid=true;}
    return true;
}
std::size_t metric_stage_values(const MetricImageView& a,const MetricImageView& b,std::ptrdiff_t first,std::ptrdiff_t second,
    std::ptrdiff_t step0,std::ptrdiff_t step1,std::size_t count,bool single,int operation,void* output) {
    // Half conversion preserves signaling payloads without raising invalid;
    // arithmetic may promote through double and correctly reports invalid.
    if(operation==0 && a.half)return 0;
    std::size_t i=0;
    for(;i+4<=count;i+=4) {
        const auto x=load(a,first+std::ptrdiff_t(i)*step0,step0);
        if(single) {
            const auto value=_mm256_cvtpd_ps(x);
            const auto result=operation==0 ? value:operation==1 ? _mm_sub_ps(value,_mm256_cvtpd_ps(load(b,second+std::ptrdiff_t(i)*step1,step1))):_mm_mul_ps(value,value);
            _mm_storeu_ps(static_cast<float*>(output)+i,result);
        } else {
            const auto result=operation==0 ? x:operation==1 ? _mm256_sub_pd(x,load(b,second+std::ptrdiff_t(i)*step1,step1)):_mm256_mul_pd(x,x);
            _mm256_storeu_pd(static_cast<double*>(output)+i,result);
        }
    }
    return i;
}
std::size_t metric_values(const MetricImageView& a,const MetricImageView& b,std::ptrdiff_t first,std::ptrdiff_t second,
    std::ptrdiff_t step0,std::ptrdiff_t step1,std::size_t count,bool single,MetricNormalization mode,double* error,double* energy,double* values) {
    std::size_t i=0;
    if(single && !a.half && !b.half && !a.swap && !b.swap &&
       a.type==PixelType::float32 && b.type==PixelType::float32 && step0==4 && step1==4 &&
       !(_mm_getcsr() & (_MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK))) {
        const auto* left=static_cast<const unsigned char*>(a.data)+first;
        const auto* right=static_cast<const unsigned char*>(b.data)+second;
        for(;i+4<=count;i+=4) {
            const auto x=_mm_loadu_ps(reinterpret_cast<const float*>(left+4*i));
            const auto y=_mm_loadu_ps(reinterpret_cast<const float*>(right+4*i));
            const auto delta=_mm_sub_ps(x,y);
            _mm256_storeu_pd(error+i,_mm256_cvtps_pd(_mm_mul_ps(delta,delta)));
            _mm256_storeu_pd(values+i,_mm256_cvtps_pd(x));
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_cvtps_pd(_mm_mul_ps(x,x)));
        }
        return i;
    }
    if(!single && !a.half && !b.half && !a.swap && !b.swap &&
       a.type==PixelType::float64 && b.type==PixelType::float64 && step0==8 && step1==8) {
        const auto* left=static_cast<const unsigned char*>(a.data)+first;
        const auto* right=static_cast<const unsigned char*>(b.data)+second;
        for(;i+4<=count;i+=4) {
            const auto x=_mm256_loadu_pd(reinterpret_cast<const double*>(left+8*i));
            const auto y=_mm256_loadu_pd(reinterpret_cast<const double*>(right+8*i));
            const auto delta=_mm256_sub_pd(x,y);
            _mm256_storeu_pd(error+i,_mm256_mul_pd(delta,delta));
            _mm256_storeu_pd(values+i,x);
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_mul_pd(x,x));
        }
        return i;
    }
    for(;i+4<=count;i+=4) {
        const auto x=load(a,first+std::ptrdiff_t(i)*step0,step0),y=load(b,second+std::ptrdiff_t(i)*step1,step1);
        if(single) {
            const auto xf=_mm256_cvtpd_ps(x),yf=_mm256_cvtpd_ps(y),delta=_mm_sub_ps(xf,yf);
            _mm256_storeu_pd(error+i,_mm256_cvtps_pd(_mm_mul_ps(delta,delta)));
            _mm256_storeu_pd(values+i,_mm256_cvtps_pd(xf));
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_cvtps_pd(_mm_mul_ps(xf,xf)));
        } else {
            const auto delta=_mm256_sub_pd(x,y);_mm256_storeu_pd(error+i,_mm256_mul_pd(delta,delta));_mm256_storeu_pd(values+i,x);
            if(mode==MetricNormalization::euclidean)_mm256_storeu_pd(energy+i,_mm256_mul_pd(x,x));
        }
    }
    return i;
}

namespace {
struct DoubleLanes {
    __m256d lo,hi;
    DoubleLanes operator+(DoubleLanes rhs) const {
        return {_mm256_add_pd(lo,rhs.lo),_mm256_add_pd(hi,rhs.hi)};
    }
    DoubleLanes operator-(DoubleLanes rhs) const {
        return {_mm256_sub_pd(lo,rhs.lo),_mm256_sub_pd(hi,rhs.hi)};
    }
    DoubleLanes square() const {return {_mm256_mul_pd(lo,lo),_mm256_mul_pd(hi,hi)};}
};

template<class T> struct MetricLanes {using type=DoubleLanes;};
template<> struct MetricLanes<float> {using type=__m256;};
template<class T> using WorkLanes=typename MetricLanes<T>::type;

inline DoubleLanes widened(DoubleLanes x) {return x;}
inline DoubleLanes widened(__m256 x) {
    return {_mm256_cvtps_pd(_mm256_castps256_ps128(x)),_mm256_cvtps_pd(_mm256_extractf128_ps(x,1))};
}

template<class T> T horizontal_sum(const T* x) {
    return ((x[0]+x[1])+(x[2]+x[3]))+((x[4]+x[5])+(x[6]+x[7]));
}
double horizontal_sum(DoubleLanes x) {
    alignas(32) double lanes[8];
    _mm256_store_pd(lanes,x.lo);
    _mm256_store_pd(lanes+4,x.hi);
    return horizontal_sum(lanes);
}
float horizontal_sum(__m256 x) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes,x);
    return horizontal_sum(lanes);
}

template<class T,bool Dense> class MetricCursor {
public:
    MetricCursor(const MetricImageView& view,std::size_t begin)
        :view_(&view),data_(static_cast<const unsigned char*>(view.data)),index_(begin) {
        if constexpr(!Dense) row();
    }

    explicit MetricCursor(const unsigned char* data):data_(data),index_(0) {
        static_assert(Dense);
    }

    const unsigned char* position() {return take(0);}

    T scalar() {
        T value;
        std::memcpy(&value,take(1),sizeof(T));
        return value;
    }

    WorkLanes<T> vector() {
        if constexpr(!Dense) {
            if(!remaining_)row();
            if(remaining_<8) {
                // A leaf can straddle any number of rows. Pack only the lanes
                // crossing a row boundary without reading beyond either row.
                T values[8];
                for(auto& value:values)value=scalar();
                return load8(reinterpret_cast<const unsigned char*>(values));
            }
        }
        return load8(take(8));
    }

private:
    static WorkLanes<T> load8(const unsigned char* ptr) {
        if constexpr(std::is_same_v<T,float>)return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
        else return {_mm256_loadu_pd(reinterpret_cast<const double*>(ptr)),
                     _mm256_loadu_pd(reinterpret_cast<const double*>(ptr+4*sizeof(T)))};
    }

    const unsigned char* take(std::size_t count) {
        if constexpr(Dense) {
            const auto* ptr=data_+index_*sizeof(T);
            index_+=count;
            return ptr;
        } else {
            if(!remaining_)row();
            const auto* ptr=data_+offset_;
            remaining_-=count;
            index_+=count;
            offset_+=std::ptrdiff_t(count*sizeof(T));
            return ptr;
        }
    }

    void row() {
        const auto width=view_->shape.back(), column=index_%width;
        auto outer=index_/width;
        remaining_=width-column;
        offset_=std::ptrdiff_t(column*sizeof(T));
        for(std::size_t axis=view_->shape.size()-1;axis>0;) {
            --axis;
            offset_+=std::ptrdiff_t(outer%view_->shape[axis])*view_->strides[axis];
            outer/=view_->shape[axis];
        }
    }

    const MetricImageView* view_=nullptr;
    const unsigned char* data_;
    std::size_t index_,remaining_=0;
    std::ptrdiff_t offset_=0;
};

template<class T,MetricNormalization Mode,bool Dense> class FastMetricTree {
    static constexpr bool energy=Mode==MetricNormalization::euclidean;
    static constexpr bool mean=Mode==MetricNormalization::mean;
    static constexpr bool range=Mode==MetricNormalization::min_max;
    using Aux=std::conditional_t<mean,T,double>;
    static constexpr std::size_t leaf_width=128;
    struct Sums {double error=0;Aux aux=0;};
    struct LeafLanes {
        DoubleLanes error,energy;
        WorkLanes<T> mean;
    };

public:
    FastMetricTree(const MetricImageView& a,const MetricImageView& b):a_(a),b_(b) {}

    std::tuple<double,double,double,double,bool> run(std::size_t n,std::size_t block) {
        Sums total;
        if(!block || n<=block)total=tree(0,n);
        else {
            // Legacy NumPy starts with zero and merges each buffer in order.
            for(std::size_t begin=0;begin<n;) {
                const auto count=std::min(block,n-begin);
                total=merge(total,tree(begin,count));
                begin+=count;
            }
        }
        if constexpr(range) {
            double lo[8],hi[8];
            _mm256_storeu_pd(lo,low_.lo);_mm256_storeu_pd(lo+4,low_.hi);
            _mm256_storeu_pd(hi,high_.lo);_mm256_storeu_pd(hi+4,high_.hi);
            for(int i=0;i<8;++i)bounds(lo[i],hi[i]);
            // The pinned range retains the first zero on a tie across leaves.
            // Replay signed-zero endpoints instead of choosing a SIMD lane.
            if((low_scalar_==0 || high_scalar_==0) &&
               (negative_zero_ || _mm256_movemask_epi8(negative_zero_lanes_)!=0))return {};
        }
        // NaN or infinity in either operand also makes its squared error
        // nonfinite, and nonnegative errors cannot cancel in the tree.
        return {total.error,total.aux==0 ? 0.0 : double(total.aux),low_scalar_,high_scalar_,
                std::isfinite(total.error) && std::isfinite(total.aux)};
    }

private:
    static Sums merge(Sums a,Sums b) {return {a.error+b.error,Aux(a.aux+b.aux)};}

    Sums tree(std::size_t begin,std::size_t n) {
        if(n<=leaf_width)return leaf(begin,n);
        if(n==4*leaf_width) {
            if constexpr(Dense)return four_leaves(begin);
            // With aligned leaf starts and row widths, each leaf can walk
            // contiguous columns even if the outer strides run backwards.
            else if(begin%leaf_width==0 && a_.shape.back()%leaf_width==0)return four_leaves(begin);
        }
        const auto half=(n/2)&~std::size_t{7};
        const auto left=tree(begin,half);
        return merge(left,tree(begin+half,n-half));
    }

    void bounds(double lo,double hi) {
        if(lo<low_scalar_)low_scalar_=lo;
        if(hi>high_scalar_)high_scalar_=hi;
    }

    void bounds(DoubleLanes x) {
        low_={_mm256_min_pd(low_.lo,x.lo),_mm256_min_pd(low_.hi,x.hi)};
        high_={_mm256_max_pd(high_.lo,x.lo),_mm256_max_pd(high_.hi,x.hi)};
        const auto minus_zero=_mm256_set1_epi64x(INT64_MIN);
        const auto is_minus_zero=_mm256_or_si256(
            _mm256_cmpeq_epi64(_mm256_castpd_si256(x.lo),minus_zero),
            _mm256_cmpeq_epi64(_mm256_castpd_si256(x.hi),minus_zero));
        negative_zero_lanes_=_mm256_or_si256(negative_zero_lanes_,is_minus_zero);
    }

    template<bool Seed,bool Flat>
    void group(LeafLanes& lanes,MetricCursor<T,Flat>& a,MetricCursor<T,Flat>& b) {
        const auto x=a.vector(),y=b.vector();
        DoubleLanes err,en;
        if constexpr(std::is_same_v<T,float>) {
            const auto delta=_mm256_sub_ps(x,y);
            err=widened(_mm256_mul_ps(delta,delta));
            if constexpr(energy)en=widened(_mm256_mul_ps(x,x));
        } else {
            err=(x-y).square();
            if constexpr(energy)en=x.square();
        }
        lanes.error=Seed ? err : lanes.error+err;
        if constexpr(energy)lanes.energy=Seed ? en : lanes.energy+en;
        if constexpr(mean) {
            if constexpr(std::is_same_v<T,float>)lanes.mean=Seed ? x : _mm256_add_ps(lanes.mean,x);
            else lanes.mean=Seed ? x : lanes.mean+x;
        }
        if constexpr(range)bounds(widened(x));
    }

    static Sums finish(const LeafLanes& lanes) {
        Sums sum{horizontal_sum(lanes.error),-Aux(0)};
        if constexpr(energy)sum.aux=horizontal_sum(lanes.energy);
        if constexpr(mean)sum.aux=horizontal_sum(lanes.mean);
        return sum;
    }

    Sums four_leaves(std::size_t begin) {
        const auto cursor=[](const MetricImageView& view,std::size_t index) {
            return MetricCursor<T,true>(MetricCursor<T,Dense>(view,index).position());
        };
        auto a0=cursor(a_,begin),a1=cursor(a_,begin+leaf_width);
        auto a2=cursor(a_,begin+2*leaf_width),a3=cursor(a_,begin+3*leaf_width);
        auto b0=cursor(b_,begin),b1=cursor(b_,begin+leaf_width);
        auto b2=cursor(b_,begin+2*leaf_width),b3=cursor(b_,begin+3*leaf_width);
        LeafLanes v0,v1,v2,v3;
        // Interleave four independent leaves without changing the order of
        // additions inside each leaf or the tree that merges their results.
        group<true>(v0,a0,b0);
        group<true>(v1,a1,b1);
        group<true>(v2,a2,b2);
        group<true>(v3,a3,b3);
        for(std::size_t i=8;i<leaf_width;i+=8) {
            group<false>(v0,a0,b0);
            group<false>(v1,a1,b1);
            group<false>(v2,a2,b2);
            group<false>(v3,a3,b3);
        }
        return merge(merge(finish(v0),finish(v1)),merge(finish(v2),finish(v3)));
    }

    Sums leaf(std::size_t begin,std::size_t n) {
        MetricCursor<T,Dense> a(a_,begin),b(b_,begin);
        Sums sum{-0.0,-Aux(0)};
        std::size_t i=0;
        if(n>=8) {
            LeafLanes lanes;
            // Seed the lanes with the first eight values, matching the
            // pinned tree even for signed zeros. Products round before add.
            group<true>(lanes,a,b);
            for(i=8;i+8<=n;i+=8)group<false>(lanes,a,b);
            sum=finish(lanes);
        }
        for(;i<n;++i) {
            const T x=a.scalar(),delta=x-b.scalar();
            sum.error+=double(T(delta*delta));
            if constexpr(energy)sum.aux+=double(T(x*x));
            if constexpr(mean)sum.aux+=x;
            if constexpr(range) {
                bounds(x,x);
                negative_zero_|=x==0 && std::signbit(x);
            }
        }
        return sum;
    }

    const MetricImageView &a_,&b_;
    double low_scalar_=metric_infinity,high_scalar_=-metric_infinity;
    DoubleLanes low_{_mm256_set1_pd(metric_infinity),_mm256_set1_pd(metric_infinity)};
    DoubleLanes high_{_mm256_set1_pd(-metric_infinity),_mm256_set1_pd(-metric_infinity)};
    __m256i negative_zero_lanes_=_mm256_setzero_si256();
    bool negative_zero_=false;
};

template<class T,bool Dense>
std::tuple<double,double,double,double,bool> fast_tree(
    const MetricImageView& a,const MetricImageView& b,std::size_t n,MetricNormalization mode,std::size_t block) {
    switch(mode) {
        case MetricNormalization::euclidean:return FastMetricTree<T,MetricNormalization::euclidean,Dense>(a,b).run(n,block);
        case MetricNormalization::mean:return FastMetricTree<T,MetricNormalization::mean,Dense>(a,b).run(n,block);
        case MetricNormalization::min_max:return FastMetricTree<T,MetricNormalization::min_max,Dense>(a,b).run(n,block);
        default:return FastMetricTree<T,MetricNormalization::none,Dense>(a,b).run(n,block);
    }
}
}

std::tuple<double,double,double,double,bool> metric_fast_reduce(
    const MetricImageView& a,const MetricImageView& b,std::size_t n,MetricNormalization mode,
    bool raw_range,std::size_t block) {
    if(_mm_getcsr() & (_MM_FLUSH_ZERO_MASK | _MM_DENORMALS_ZERO_MASK))return {};
    if(raw_range)mode=MetricNormalization::min_max;
    const bool dense=a.shape.empty() && b.shape.empty();
    if(a.type==PixelType::float32)
        return dense ? fast_tree<float,true>(a,b,n,mode,block) : fast_tree<float,false>(a,b,n,mode,block);
    return dense ? fast_tree<double,true>(a,b,n,mode,block) : fast_tree<double,false>(a,b,n,mode,block);
}
}
