#include "skimagexx/stain.hpp"
#include "skimagexx/image.hpp"
#include "stain_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <stdexcept>

namespace skimagexx::detail {
const std::array<double,256>& stain_log_table() {
    static const auto table=[] {
        std::array<double,256> result{};
        for (std::size_t i=0;i<256;++i) result[i]=std::log(std::max(double(i)*(1.0/255.0),1e-6));
        return result;
    }();
    return table;
}
}
namespace skimagexx {
namespace {
template<class Input,class Output> void execute(const void* source,void* destination,std::size_t n,PixelType type,
    const detail::StainSpec& spec,std::ptrdiff_t pixels,std::ptrdiff_t channels) {
    const auto* input=static_cast<const Input*>(source); auto* output=static_cast<Output*>(destination);
    detail::partition(n,detail::active_workers(n),[&](auto begin,auto end,auto) {
        std::size_t done=0;
#ifdef SKIMAGEXX_AVX2
        if constexpr (std::is_same_v<Output,double>) if (has_avx2())
            done=simd::stain_transform(input+std::ptrdiff_t(begin)*pixels,output+begin*spec.channels,end-begin,type,spec,pixels,channels);
#else
        (void)type;
#endif
        for (auto i=begin+done;i<end;++i) detail::stain_pixel(input+std::ptrdiff_t(i)*pixels,output+i*spec.channels,spec,channels);
    });
}
}
void stain_transform_into(const void* input,void* output,std::size_t n,PixelType input_type,StainTransform transform,
    const double* weights,std::size_t output_channels,PixelType intermediate,PixelType output_type,
    std::ptrdiff_t pixels,std::ptrdiff_t channels,double adjust) {
    if (transform!=StainTransform::separate && transform!=StainTransform::combine) throw std::invalid_argument("unknown stain transform");
    if ((intermediate!=PixelType::float32 && intermediate!=PixelType::float64) ||
        (output_type!=PixelType::float32 && output_type!=PixelType::float64)) throw std::invalid_argument("stain precision must be float32 or float64");
    if (!(adjust>0) || !std::isfinite(adjust)) throw std::invalid_argument("log adjustment must be finite and positive");
    const auto bytes=std::max(sizeof(double),std::max(pixel_size(input_type),pixel_size(output_type)));
    const auto limit=std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    const auto magnitude=[](std::ptrdiff_t v){return v<0 ? std::size_t(-(v+1))+1 : std::size_t(v);};
    const auto step=magnitude(pixels),offset=magnitude(channels);
    if (output_channels>limit/3 || n>limit/std::max(std::size_t(3),output_channels) || offset>limit/2 ||
        (n>1 && step>(limit-2*offset)/(n-1))) throw std::overflow_error("stain array or strides overflow");
    if (output_channels && !weights) throw std::invalid_argument("matrix must be nonnull");
    if (n && (!input || (output_channels && !output))) throw std::invalid_argument("image buffers must be nonnull");
    if (!n || !output_channels) return;
    const detail::StainSpec spec{transform,weights,output_channels,intermediate==PixelType::float32,adjust};
#define STAIN_CASE(KIND,TYPE) case PixelType::KIND: if (output_type==PixelType::float32) return execute<TYPE,float>(input,output,n,input_type,spec,pixels,channels); else return execute<TYPE,double>(input,output,n,input_type,spec,pixels,channels)
    switch(input_type) {
        STAIN_CASE(boolean,bool); STAIN_CASE(uint8,std::uint8_t); STAIN_CASE(int8,std::int8_t);
        STAIN_CASE(uint16,std::uint16_t); STAIN_CASE(int16,std::int16_t); STAIN_CASE(uint32,std::uint32_t);
        STAIN_CASE(int32,std::int32_t); STAIN_CASE(uint64,std::uint64_t); STAIN_CASE(int64,std::int64_t);
        STAIN_CASE(float32,float); STAIN_CASE(float64,double);
    }
#undef STAIN_CASE
}
}
