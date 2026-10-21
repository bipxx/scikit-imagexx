#include "skimagexx/color.hpp"
#include "skimagexx/image.hpp"
#include "color_detail.hpp"
#include "color_matrices.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <stdexcept>

namespace skimagexx {
namespace detail {
const ColorSpec& color_spec(ColorTransform transform) {
    const int id = static_cast<int>(transform);
    if (id < 0 || id >= int(color_specs.size())) throw std::invalid_argument("unknown color transform");
    return color_specs[static_cast<std::size_t>(id)];
}
const std::array<double,256>& srgb_decode_table() {
    static const auto table = [] {
        std::array<double,256> result{};
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = decode_srgb(double(i)*(1.0/255.0));
        return result;
    }();
    return table;
}
}
namespace {
template<class Input> void color_execute(const void* source, void* destination, std::size_t n, const detail::ColorSpec& spec, PixelType input_type,
                                        std::ptrdiff_t pixel_stride = 3, std::ptrdiff_t channel_stride = 1) {
    const auto* p = static_cast<const Input*>(source);
    auto* out = static_cast<detail::ColorOutput<Input>*>(destination);
    const bool vectorized = has_avx2();
    detail::partition(n,detail::active_workers(n),[&](auto begin, auto end, auto) {
        std::size_t consumed = 0;
#ifdef SKIMAGEXX_AVX2
        if (vectorized) consumed = simd::color_transform(p+std::ptrdiff_t(begin)*pixel_stride,out+3*begin,end-begin,input_type,spec,pixel_stride,channel_stride);
#else
        (void)vectorized; (void)input_type;
#endif
        for (auto i = begin+consumed; i < end; ++i) detail::color_pixel(p+std::ptrdiff_t(i)*pixel_stride,out+3*i,spec,channel_stride);
    });
}
}
void color_transform_into(const void* source, void* out, std::size_t n, PixelType input_type, ColorTransform transform) {
    const auto& spec = detail::color_spec(transform);
    if (n && (!source || !out)) throw std::invalid_argument("input and output must be nonnull");
    const auto bytes = std::max(pixel_size(input_type),input_type == PixelType::float32 ? std::size_t{4} : std::size_t{8});
    if (n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/(3*bytes)) throw std::overflow_error("array size overflows");
    if (input_type == PixelType::boolean && spec.input_offset) throw std::invalid_argument("boolean YCbCr subtraction is unsupported");
    if (!n) return;
    switch (input_type) {
        case PixelType::boolean: return color_execute<bool>(source,out,n,spec,input_type);
        case PixelType::uint8: return color_execute<std::uint8_t>(source,out,n,spec,input_type);
        case PixelType::int8: return color_execute<std::int8_t>(source,out,n,spec,input_type);
        case PixelType::uint16: return color_execute<std::uint16_t>(source,out,n,spec,input_type);
        case PixelType::int16: return color_execute<std::int16_t>(source,out,n,spec,input_type);
        case PixelType::uint32: return color_execute<std::uint32_t>(source,out,n,spec,input_type);
        case PixelType::int32: return color_execute<std::int32_t>(source,out,n,spec,input_type);
        case PixelType::uint64: return color_execute<std::uint64_t>(source,out,n,spec,input_type);
        case PixelType::int64: return color_execute<std::int64_t>(source,out,n,spec,input_type);
        case PixelType::float32: return color_execute<float>(source,out,n,spec,input_type);
        case PixelType::float64: return color_execute<double>(source,out,n,spec,input_type);
        default: throw std::invalid_argument("unsupported color input dtype");
    }
}
template<class T> void color_transform_strided_into(const T* source, T* out, std::size_t n,
    std::ptrdiff_t pixel_stride, std::ptrdiff_t channel_stride, ColorTransform transform) {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>);
    const auto& spec = detail::color_spec(transform);
    if (n && (!source || !out)) throw std::invalid_argument("input and output must be nonnull");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T);
    const auto magnitude = [](std::ptrdiff_t stride) {
        return stride < 0 ? std::size_t(-(stride+1))+1 : std::size_t(stride);
    };
    const auto pixels = magnitude(pixel_stride), channels = magnitude(channel_stride);
    if (n > limit/3 || channels > limit/2 || (n > 1 && pixels > (limit-2*channels)/(n-1)))
        throw std::overflow_error("color strides overflow");
    if (!n) return;
    color_execute<T>(source,out,n,spec,std::is_same_v<T,float> ? PixelType::float32 : PixelType::float64,pixel_stride,channel_stride);
}
template void color_transform_strided_into(const float*,float*,std::size_t,std::ptrdiff_t,std::ptrdiff_t,ColorTransform);
template void color_transform_strided_into(const double*,double*,std::size_t,std::ptrdiff_t,std::ptrdiff_t,ColorTransform);
} // namespace skimagexx
