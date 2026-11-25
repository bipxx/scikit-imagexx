// Independent output lanes preserve ndimage's axis/corner multiplication and
// summation order. BSD reference attribution: scipy-ndimage-LICENSE.txt.
#include "resample_detail.hpp"
#include <immintrin.h>
#include "resize_cast.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>

namespace skimagexx::simd {
namespace {
__m256d unsigned32(__m128i value) {
    const auto signed_value=_mm256_cvtepi32_pd(value);
    const auto negative=_mm256_cmp_pd(signed_value,_mm256_setzero_pd(),_CMP_LT_OQ);
    return _mm256_add_pd(signed_value,_mm256_and_pd(negative,_mm256_set1_pd(0x1p32)));
}
__m128i words32(__m256i value,bool high) {
    const auto lo=_mm256_castsi256_si128(value),hi=_mm256_extracti128_si256(value,1);
    return high ? _mm_unpacklo_epi64(_mm_shuffle_epi32(lo,_MM_SHUFFLE(3,1,3,1)),_mm_shuffle_epi32(hi,_MM_SHUFFLE(3,1,3,1)))
                : _mm_unpacklo_epi64(_mm_shuffle_epi32(lo,_MM_SHUFFLE(2,0,2,0)),_mm_shuffle_epi32(hi,_MM_SHUFFLE(2,0,2,0)));
}
__m128 mask32(__m256i valid) {
    const int mask=_mm256_movemask_pd(_mm256_castsi256_pd(valid));
    return _mm_castsi128_ps(_mm_set_epi32(mask&8 ? -1 : 0,mask&4 ? -1 : 0,mask&2 ? -1 : 0,mask&1 ? -1 : 0));
}
template<class T> __m256d gather(const T* input,__m256i index,__m256i valid,double constant) {
    const auto mask=_mm256_castsi256_pd(valid),cval=_mm256_set1_pd(constant);
    if constexpr(std::is_same_v<T,double>)return _mm256_mask_i64gather_pd(cval,input,index,mask,8);
    else if constexpr(std::is_same_v<T,float>) {
        const auto values=_mm256_mask_i64gather_ps(_mm_setzero_ps(),input,index,mask32(valid),4);
        return _mm256_blendv_pd(cval,_mm256_cvtps_pd(values),mask);
    } else if constexpr(sizeof(T)==4) {
        const auto values=_mm256_mask_i64gather_epi32(_mm_setzero_si128(),reinterpret_cast<const int*>(input),index,_mm_castps_si128(mask32(valid)),4);
        const auto converted=std::is_signed_v<T> ? _mm256_cvtepi32_pd(values) : unsigned32(values);
        return _mm256_blendv_pd(cval,converted,mask);
    } else if constexpr(sizeof(T)==8) {
        const auto values=_mm256_mask_i64gather_epi64(_mm256_setzero_si256(),reinterpret_cast<const long long*>(input),index,valid,8);
        const auto high=words32(values,true),low=words32(values,false);
        const auto high_value=std::is_signed_v<T> ? _mm256_cvtepi32_pd(high) : unsigned32(high);
        const auto converted=_mm256_add_pd(_mm256_mul_pd(high_value,_mm256_set1_pd(0x1p32)),unsigned32(low));
        return _mm256_blendv_pd(cval,converted,mask);
    } else {
        // AVX2 has no byte/word gather. Read exactly the selected elements;
        // widening a read here would overrun the final source element.
        alignas(32) std::int64_t offsets[4];alignas(32) double values[4];
        _mm256_store_si256(reinterpret_cast<__m256i*>(offsets),index);
        const auto bits=_mm256_movemask_pd(mask);
        for(int lane=0;lane<4;++lane)values[lane]=(bits&(1<<lane)) ? double(input[offsets[lane]]) : constant;
        return _mm256_load_pd(values);
    }
}
template<class T,bool Boolean> void store(T* output,__m256d value) {
    if constexpr(std::is_same_v<T,double>)_mm256_storeu_pd(output,value);
    else if constexpr(std::is_same_v<T,float>)_mm_storeu_ps(output,_mm256_cvtpd_ps(value));
    else if constexpr(sizeof(T)==8) {
        alignas(32) double values[4];_mm256_store_pd(values,value);
        for(int lane=0;lane<4;++lane)output[lane]=detail::resize_sample_cast<T,Boolean>(values[lane]);
    } else {
        if constexpr(!Boolean) {
            const auto positive=_mm256_cmp_pd(value,_mm256_setzero_pd(),_CMP_GT_OQ);
            if constexpr(std::is_unsigned_v<T>)value=_mm256_blendv_pd(_mm256_setzero_pd(),_mm256_add_pd(value,_mm256_set1_pd(.5)),positive);
            else value=_mm256_blendv_pd(_mm256_sub_pd(value,_mm256_set1_pd(.5)),_mm256_add_pd(value,_mm256_set1_pd(.5)),positive);
            value=_mm256_min_pd(_mm256_max_pd(value,_mm256_set1_pd(double(std::numeric_limits<T>::min()))),_mm256_set1_pd(double(std::numeric_limits<T>::max())));
        }
        auto high_bit=_mm_setzero_si128();
        if constexpr(std::is_same_v<T,std::uint32_t>) {
            const auto upper=_mm256_cmp_pd(value,_mm256_set1_pd(0x1p31),_CMP_GE_OQ);
            // Keep the adjusted operand nonnegative: subtracting 2**32
            // would change truncation of the sampler's fractional +.5.
            value=_mm256_sub_pd(value,_mm256_and_pd(upper,_mm256_set1_pd(0x1p31)));
            high_bit=_mm_and_si128(_mm_castps_si128(mask32(_mm256_castpd_si256(upper))),_mm_set1_epi32(std::numeric_limits<std::int32_t>::min()));
        }
        auto integers=_mm_xor_si128(_mm256_cvttpd_epi32(value),high_bit);
        if constexpr(sizeof(T)==4)_mm_storeu_si128(reinterpret_cast<__m128i*>(output),integers);
        else if constexpr(sizeof(T)==2) {
            integers=_mm_and_si128(integers,_mm_set1_epi32(65535));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(output),_mm_packus_epi32(integers,_mm_setzero_si128()));
        } else {
            integers=_mm_and_si128(integers,_mm_set1_epi32(255));
            const auto words=_mm_packus_epi32(integers,_mm_setzero_si128());
            const auto bytes=_mm_packus_epi16(words,_mm_setzero_si128());
            const auto packed=_mm_cvtsi128_si32(bytes);std::memcpy(output,&packed,4);
        }
    }
}
struct ColumnTap {
    __m256i offset,valid;
    __m256d weight;
};

ColumnTap column_tap(const detail::ZoomPoint* points,std::size_t lanes,int tap) {
    // A short row fragment repeats its final valid lane. Only the requested
    // results are stored, and no point from the next row is read.
    const auto& a=points[0];
    const auto& b=points[lanes>=2 ? 1 : 0];
    const auto& c=points[lanes>=3 ? 2 : lanes-1];
    const auto& d=points[lanes-1];
    const auto offset=_mm256_setr_epi64x(a.offsets[tap],b.offsets[tap],c.offsets[tap],d.offsets[tap]);
    return {offset,_mm256_cmpgt_epi64(offset,_mm256_set1_epi64x(-1)),
        _mm256_setr_pd(a.weights[tap],b.weights[tap],c.weights[tap],d.weights[tap])};
}

template<class Output,bool Boolean,class Block>
void sample_columns(const detail::ZoomPoint* points,Output* output,std::size_t width,Block block) {
    for(;width>=4;width-=4,points+=4,output+=4)store<Output,Boolean>(output,block(points,4));
    if(width) {
        Output tail[4];
        store<Output,Boolean>(tail,block(points,width));
        std::memcpy(output,tail,width*sizeof(Output));
    }
}

template<class Input,class Output,bool Boolean>
void bilinear_row(const Input* input,Output* output,const detail::ZoomPoint& row,
                  const detail::ZoomPoint* columns,std::size_t width,double cval) {
    const auto* top=row.offsets[0]>=0 ? input+row.offsets[0] : nullptr;
    const auto* bottom=row.offsets[1]>=0 ? input+row.offsets[1] : nullptr;
    const auto top_weight=_mm256_set1_pd(row.weights[0]);
    const auto bottom_weight=_mm256_set1_pd(row.weights[1]);
    const auto fill=_mm256_set1_pd(cval);
    sample_columns<Output,Boolean>(columns,output,width,[&](const auto* points,std::size_t lanes) {
        const auto left=column_tap(points,lanes,0),right=column_tap(points,lanes,1);
        const auto weighted=[&](const Input* source,const ColumnTap& column,__m256d row_weight) {
            const auto value=source ? gather(source,column.offset,column.valid,cval) : fill;
            return _mm256_mul_pd(_mm256_mul_pd(value,row_weight),column.weight);
        };
        // Keep the axis multiplication and corner addition in ndimage order.
        auto total=_mm256_add_pd(_mm256_setzero_pd(),weighted(top,left,top_weight));
        total=_mm256_add_pd(total,weighted(top,right,top_weight));
        total=_mm256_add_pd(total,weighted(bottom,left,bottom_weight));
        return _mm256_add_pd(total,weighted(bottom,right,bottom_weight));
    });
}

template<class Input,class Output,bool Boolean>
void sample_row(const Input* input,Output* output,const detail::ZoomPlan& plan,
                const detail::ZoomPoint* const* row,const detail::ZoomPoint* columns,
                std::size_t width,std::ptrdiff_t* row_offsets) {
    const auto outer_rank=plan.axes.size()-1;
    const auto taps_per_axis=std::size_t(plan.order+1);
    const auto outer_corners=plan.corners/taps_per_axis;
    __m256d row_weights[64][6];
    if(plan.order)for(std::size_t axis=0;axis<outer_rank;++axis)
        for(std::size_t tap=0;tap<taps_per_axis;++tap)
            row_weights[axis][tap]=_mm256_set1_pd(row[axis]->weights[tap]);
    const auto footprint_stride=(outer_rank+1)*taps_per_axis;
    const auto* row_taps=plan.footprint.data();
    for(std::size_t corner=0;corner<outer_corners;++corner,row_taps+=footprint_stride) {
        auto& index=row_offsets[corner];
        index=0;
        for(std::size_t axis=0;axis<outer_rank;++axis) {
            const auto offset=row[axis]->offsets[row_taps[axis]];
            if(offset<0) {index=-1;break;}
            index+=offset;
        }
    }
    const auto fill=_mm256_set1_pd(plan.cval);
    sample_columns<Output,Boolean>(columns,output,width,[&](const auto* points,std::size_t lanes) {
        ColumnTap column[6];
        for(std::size_t tap=0;tap<taps_per_axis;++tap)column[tap]=column_tap(points,lanes,int(tap));
        auto total=_mm256_setzero_pd();
        const auto* taps=plan.footprint.data();
        for(std::size_t corner=0;corner<outer_corners;++corner,taps+=footprint_stride) {
            const auto offset=row_offsets[corner];
            const auto* source=offset>=0 ? input+offset : nullptr;
            for(std::size_t tap=0;tap<taps_per_axis;++tap) {
                auto value=source ? gather(source,column[tap].offset,column[tap].valid,plan.cval) : fill;
                if(plan.order) {
                    for(std::size_t axis=0;axis<outer_rank;++axis)
                        value=_mm256_mul_pd(value,row_weights[axis][taps[axis]]);
                    value=_mm256_mul_pd(value,column[tap].weight);
                }
                total=_mm256_add_pd(total,value);
            }
        }
        return total;
    });
}

template<class Input,class Output,bool Boolean>
std::size_t sample_pairs(const Input* input,Output* output,const detail::ZoomPlan& plan,
                         std::size_t begin,std::size_t end) {
    // Profile sampling pairs one Y coordinate with one X coordinate per lane.
    // Its private plan supplies exactly four pairs with unit strides.
    if(end-begin<4)return begin;
    auto total=_mm256_setzero_pd();
    for(std::size_t corner=0;corner<plan.corners;++corner) {
        const auto y=column_tap(plan.axes[0].data()+begin,4,plan.footprint[2*corner]);
        const auto x=column_tap(plan.axes[1].data()+begin,4,plan.footprint[2*corner+1]);
        auto value=gather(input,_mm256_add_epi64(y.offset,x.offset),
                          _mm256_and_si256(y.valid,x.valid),plan.cval);
        if(plan.order)value=_mm256_mul_pd(_mm256_mul_pd(value,y.weight),x.weight);
        total=_mm256_add_pd(total,value);
    }
    store<Output,Boolean>(output+begin,total);
    return begin+4;
}

template<class Input,class Output,bool Boolean=false>
std::size_t sample(const Input* input,Output* output,const detail::ZoomPlan& plan,
                   std::size_t begin,std::size_t end,std::ptrdiff_t* row_offsets) {
    // NumPy has at most 64 image axes. Keep row coordinates on the stack.
    const auto rank=plan.axes.size();
    if(rank==0 || rank>64)return begin;
    const auto width=plan.axes.back().size();
    if(width<2)return begin;
    if(rank==2 && plan.output_strides.front()==1)
        return sample_pairs<Input,Output,Boolean>(input,output,plan,begin,end);
    const bool bilinear=plan.order==1 && rank==2;
    const detail::ZoomPoint* row[64];
    for(auto column=begin%width;begin<end;column=0) {
        const auto span=std::min(width-column,end-begin);
        const auto* columns=plan.axes.back().data()+column;
        if(bilinear) {
            const auto& point=plan.axes.front()[begin/width];
            bilinear_row<Input,Output,Boolean>(input,output+begin,point,columns,span,plan.cval);
        } else {
            for(std::size_t axis=0;axis+1<rank;++axis)
                row[axis]=&plan.axes[axis][(begin/plan.output_strides[axis])%plan.axes[axis].size()];
            sample_row<Input,Output,Boolean>(input,output+begin,plan,row,columns,span,row_offsets);
        }
        begin+=span;
    }
    return begin;
}
} // namespace
std::size_t zoom_grid_sample_typed(const void* input,void* output,const detail::ZoomPlan& plan,
                                  std::size_t begin,std::size_t end,PixelType source,PixelType target,
                                  std::ptrdiff_t* row_offsets) {
    if(source==PixelType::float64 && target==PixelType::float32)
        return sample(static_cast<const double*>(input),static_cast<float*>(output),plan,begin,end,row_offsets);
    if(source!=target)return begin;
    switch(source) {
        case PixelType::boolean:return sample<std::uint8_t,std::uint8_t,true>(static_cast<const std::uint8_t*>(input),static_cast<std::uint8_t*>(output),plan,begin,end,row_offsets);
        #define SAMPLE(tag,T) case PixelType::tag:return sample(static_cast<const T*>(input),static_cast<T*>(output),plan,begin,end,row_offsets);
        SAMPLE(uint8,std::uint8_t) SAMPLE(int8,std::int8_t) SAMPLE(uint16,std::uint16_t) SAMPLE(int16,std::int16_t)
        SAMPLE(uint32,std::uint32_t) SAMPLE(int32,std::int32_t) SAMPLE(uint64,std::uint64_t) SAMPLE(int64,std::int64_t)
        SAMPLE(float32,float) SAMPLE(float64,double)
        #undef SAMPLE
    }
    return begin;
}
std::size_t zoom_sample_typed(const void* input,void* output,const detail::ZoomPlan& plan,
                             std::size_t begin,std::size_t end,PixelType source,PixelType target) {
    std::array<std::ptrdiff_t,128> local;
    const auto outer_corners=plan.corners/std::size_t(plan.order+1);
    std::vector<std::ptrdiff_t> scratch;
    if(outer_corners>local.size())scratch.resize(outer_corners);
    return zoom_grid_sample_typed(input,output,plan,begin,end,source,target,
                                 scratch.empty() ? local.data() : scratch.data());
}
std::size_t zoom_sample(const double* input,double* output,const detail::ZoomPlan& plan,
                        std::size_t begin,std::size_t end) {
    return zoom_sample_typed(input,output,plan,begin,end,PixelType::float64,PixelType::float64);
}
std::size_t profile_coordinate_row(double start,double delta,double step,
                                   std::size_t width,double* output){
    std::size_t j=0;const auto first=_mm256_set1_pd(start);
    const auto scale=_mm256_set1_pd(step==0.?delta:step),div=_mm256_set1_pd(double(width-1));
    for(;width-j>=4;j+=4){
        auto values=_mm256_setr_pd(double(j),double(j+1),double(j+2),double(j+3));
        if(step==0.)values=_mm256_div_pd(values,div);
        values=_mm256_mul_pd(values,scale);
        _mm256_storeu_pd(output+j,_mm256_add_pd(values,first));
    }
    return j;
}
std::size_t profile_coordinate_single(const double* centers,double radius,
                                      std::size_t count,double* output){
    std::size_t i=0;const auto extent=_mm256_set1_pd(radius),zero=_mm256_setzero_pd();
    for(;count-i>=4;i+=4){
        const auto center=_mm256_loadu_pd(centers+i);
        const auto start=_mm256_sub_pd(center,extent),stop=_mm256_add_pd(center,extent);
        const auto delta=_mm256_sub_pd(stop,start);
        const auto values=_mm256_mul_pd(zero,delta);
        _mm256_storeu_pd(output+i,_mm256_add_pd(values,start));
    }
    return i;
}
} // namespace skimagexx::simd
