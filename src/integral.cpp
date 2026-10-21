#include "skimagexx/integral.hpp"
#include "skimagexx/image.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
namespace skimagexx {
namespace {
std::size_t elements(const std::vector<std::size_t>& shape,std::size_t bytes) {
    if(std::find(shape.begin(),shape.end(),0)!=shape.end())return 0;
    std::size_t count=1;
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    for(auto n:shape) {if(n>limit/count)throw std::overflow_error("integral dimensions overflow");count*=n;}
    return count;
}
template<class Input,class Output,bool Boolean=false> void cast(const Input* input,Output* output,std::size_t count,int workers,PixelType source) {
    (void)source;
    detail::partition(count,workers,[&](auto begin,auto end,auto) {
#ifdef SKIMAGEXX_AVX2
        if constexpr(std::is_same_v<Output,std::uint64_t>) {
            if(has_avx2())begin+=simd::integral_cast_u64(input+begin,output+begin,end-begin,source);
        }
#endif
        for(auto i=begin;i<end;++i)output[i]=Boolean ? Output(input[i]!=0) : Output(input[i]);
    });
}
template<class Output> void prepare(const void* input,Output* output,std::size_t count,PixelType source,int workers) {
#define RAW(tag,type) case PixelType::tag:cast(static_cast<const type*>(input),output,count,workers,source);return;
    switch(source) {
    case PixelType::boolean:cast<std::uint8_t,Output,true>(static_cast<const std::uint8_t*>(input),output,count,workers,source);return;
    RAW(uint8,std::uint8_t) RAW(int8,std::int8_t) RAW(uint16,std::uint16_t) RAW(int16,std::int16_t)
    RAW(uint32,std::uint32_t) RAW(int32,std::int32_t) RAW(uint64,std::uint64_t) RAW(int64,std::int64_t)
    default:throw std::invalid_argument("integer integral input required");
    }
#undef RAW
}
template<class T> void accumulate(T* output,const std::vector<std::size_t>& shape,const std::vector<std::size_t>& physical,
    std::size_t count,int workers,PixelType type) {
    std::size_t logical_stride=count;
    for(std::size_t axis=0;axis<shape.size();++axis) {
        const auto length=shape[axis];logical_stride/=length;
        const auto stride=physical.empty() ? logical_stride : physical[axis];
        if(length<=1)continue;
        if(stride==1) {
            const auto rows=count/length;
            detail::partition(rows,workers,[&](auto begin,auto end,auto) {
                for(auto row=begin;row<end;++row) {
                    auto* values=output+row*length;std::size_t position=1;
#ifdef SKIMAGEXX_AVX2
                    if constexpr(std::is_same_v<T,std::uint64_t>) {
                        if(has_avx2())position=std::max(position,simd::integral_prefix_u64(values,values,length));
                    }
#endif
                    // Starting from the first value, rather than 0 + value,
                    // preserves negative zero in floating cumulative sums.
                    T sum=values[position-1];
                    for(;position<length;++position) {sum=T(sum+values[position]);values[position]=sum;}
                }
            });
        } else {
            // Tiles expose independent column chains to SIMD and to workers.
            const auto desired=(stride+std::size_t(workers)-1)/std::size_t(workers);
            const auto tile=std::min(std::size_t(256),std::max(std::size_t(16),(desired+15)/16*16));
            const auto tiles=(stride+tile-1)/tile,slabs=count/(length*stride);
            detail::partition(slabs*tiles,workers,[&](auto begin,auto end,auto) {
                for(auto job=begin;job<end;++job) {
                    const auto slab=job/tiles,inner=job%tiles*tile,span=std::min(tile,stride-inner);
                    for(std::size_t x=1;x<length;++x) {
                        auto* current=output+slab*length*stride+x*stride+inner;
                        const auto* previous=current-stride;std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
                        if(has_avx2())done=simd::integral_add(previous,current,span,type);
#endif
                        for(auto i=done;i<span;++i)current[i]=T(previous[i]+current[i]);
                    }
                }
            });
        }
    }
}
}
void integral_line_u64_into(const void* input,std::uint64_t* output,std::size_t count,std::ptrdiff_t stride,bool swap) {
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if(count>limit/sizeof(std::uint64_t))throw std::overflow_error("integral line output span overflow");
    if(!count)return;
    if(!input || !output)throw std::invalid_argument("nonnull integral line storage required");
    if(count>1) {
        if(stride==std::numeric_limits<std::ptrdiff_t>::min())throw std::overflow_error("integral line stride overflow");
        const auto magnitude=std::size_t(stride<0 ? -stride : stride);
        if(magnitude>(limit-7)/(count-1))throw std::overflow_error("integral line input span overflow");
    }
    const auto workers=detail::active_workers(count);
    struct Segment {std::size_t begin=0,end=0;std::uint64_t offset=0;};
    std::vector<Segment> segments(workers>1 ? std::size_t(workers) : 0,Segment{});
    const auto* bytes=static_cast<const unsigned char*>(input);
    const bool vectorized=has_avx2();(void)vectorized;
    detail::partition(count,workers,[&](auto begin,auto end,auto worker) {
        if(workers>1) {segments[worker].begin=begin;segments[worker].end=end;}
        if(begin==end)return;
        std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
        if(vectorized)done=simd::integral_strided_prefix_u64(bytes+std::ptrdiff_t(begin)*stride,output+begin,end-begin,stride,swap);
#endif
        std::uint64_t sum=done ? output[begin+done-1] : 0;
        for(auto i=begin+done;i<end;++i) {
            std::uint64_t value;std::memcpy(&value,bytes+std::ptrdiff_t(i)*stride,8);
            if(swap) {
                std::uint64_t reversed=0;
                for(unsigned b=0;b<8;++b)reversed|=((value>>(8*b))&255)<<(8*(7-b));
                value=reversed;
            }
            sum+=value;output[i]=sum;
        }
    });
    if(workers==1)return;
    std::uint64_t carry=0;
    for(auto& segment:segments) {
        segment.offset=carry;
        if(segment.end>segment.begin)carry+=output[segment.end-1];
    }
    // Schedule stored segments rather than assuming both OpenMP teams match:
    // dynamic teams or thread limits can change actual team sizes per region.
    detail::partition(segments.size(),workers,[&](auto first,auto last,auto) {
        for(auto s=first;s<last;++s) {
            const auto& segment=segments[s];if(!segment.offset || segment.begin==segment.end)continue;
            auto i=segment.begin;
#ifdef SKIMAGEXX_AVX2
            if(vectorized)i+=simd::integral_offset_u64(output+i,segment.end-i,segment.offset);
#endif
            for(;i<segment.end;++i)output[i]+=segment.offset;
        }
    });
}
void integral_image_into(const void* input,void* output,PixelType source,PixelType target,const std::vector<std::size_t>& shape,
    const std::vector<std::size_t>& strides) {
    if(target!=PixelType::int64 && target!=PixelType::uint64 && target!=PixelType::float32 && target!=PixelType::float64)
        throw std::invalid_argument("unsupported integral output precision");
    const auto floating=source==PixelType::float32 || source==PixelType::float64;
    if((target==PixelType::int64 || target==PixelType::uint64) && floating)throw std::invalid_argument("integer integral input required");
    if(target==PixelType::float32 && source!=PixelType::float32)throw std::invalid_argument("float32 integral requires float32 input");
    const auto count=elements(shape,std::max(pixel_size(source),pixel_size(target)));
    if(!strides.empty() && strides.size()!=shape.size())throw std::invalid_argument("one integral stride per axis required");
    if(!count)return;
    if(!strides.empty()) {
        std::vector<std::pair<std::size_t,std::size_t>> axes;
        for(std::size_t axis=0;axis<shape.size();++axis)if(shape[axis]>1)axes.emplace_back(strides[axis],shape[axis]);
        std::sort(axes.begin(),axes.end());std::size_t extent=1;
        for(const auto& axis:axes) {if(axis.first!=extent)throw std::invalid_argument("dense positive integral strides required");extent*=axis.second;}
    }
    if(!input || !output)throw std::invalid_argument("nonnull integral storage required");
    const auto workers=detail::active_workers(count);
    if(shape.size()==1 && workers==1 && (source==PixelType::int64 || source==PixelType::uint64)
            && (target==PixelType::int64 || target==PixelType::uint64)) {
        const auto* values=static_cast<const std::uint64_t*>(input);
        auto* result=static_cast<std::uint64_t*>(output);std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
        if(has_avx2())done=simd::integral_prefix_u64(values,result,count);
#endif
        std::uint64_t sum=done ? result[done-1] : 0;
        for(auto i=done;i<count;++i) {sum+=values[i];result[i]=sum;}
        return;
    }
    if(target==PixelType::float64) {
        auto* values=static_cast<double*>(output);
        if(floating)convert_into(input,output,count,source,target);
        else {
            detail::partition(count,workers,[&](auto begin,auto end,auto) {
                const auto* bytes=static_cast<const unsigned char*>(input)+begin*pixel_size(source);
                std::size_t prepared=0;
#ifdef SKIMAGEXX_AVX2
                if(has_avx2())prepared=simd::gaussian_prepare(bytes,values+begin,end-begin,source,true);
#endif
                prepare(bytes+prepared*pixel_size(source),values+begin+prepared,end-begin-prepared,source,1);
            });
        }
        accumulate(values,shape,strides,count,workers,target);
    } else if(target==PixelType::float32) {
        convert_into(input,output,count,source,target);
        accumulate(static_cast<float*>(output),shape,strides,count,workers,target);
    } else {
        // Corresponding signed and unsigned types may alias. Unsigned addition
        // defines wraparound without signed-overflow undefined behavior.
        auto* values=static_cast<std::uint64_t*>(output);
        prepare(input,values,count,source,workers);
        accumulate(values,shape,strides,count,workers,target);
    }
}
}
