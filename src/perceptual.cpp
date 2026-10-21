#include "skimagexx/perceptual.hpp"
#include "skimagexx/image.hpp"
#include "perceptual_detail.hpp"
#include "execution.hpp"
#include "simd.hpp"
#include <vector>
#include <stdexcept>

namespace skimagexx {
void lab_scalar_negative_z_into(const float* input, double* output, PerceptualTransform transform,
    std::array<double,3> white, std::ptrdiff_t channels) {
    if (!input || !output) throw std::invalid_argument("input and output must be nonnull");
    if (transform != PerceptualTransform::lab2xyz && transform != PerceptualTransform::lab2rgb)
        throw std::invalid_argument("expected inverse Lab transform");
    const auto stride_limit = std::numeric_limits<std::ptrdiff_t>::max()/std::ptrdiff_t(2*sizeof(float));
    if (channels > stride_limit || channels < -stride_limit)
        throw std::overflow_error("color strides overflow");
    for (auto value : white) if (!(value > 0) || !std::isfinite(value)) throw std::invalid_argument("white point must be finite and positive");
    const float y = (input[0]+16.0f)/116.0f, x = input[channels]/500.0f+y;
    if (!(y-input[2*channels]/200.0f < 0)) throw std::invalid_argument("expected negative Z");
    double values[] = {double(x),double(y),0};
    for (int j = 0; j < 3; ++j)
        values[j] = (values[j] > 0.2068966 ? std::pow(values[j],3.0) : (values[j]-16.0/116.0)/7.787)*white[j];
    if (transform == PerceptualTransform::lab2rgb) detail::color_pixel(values,output,detail::color_spec(ColorTransform::xyz2rgb));
    else std::copy(values,values+3,output);
}
namespace {
template<class Input> PerceptualInfo execute_perceptual(const void* input, void* output, std::size_t n,
    PixelType type, const detail::PerceptualSpec& spec, std::ptrdiff_t pixels, std::ptrdiff_t channels) {
    const auto* p = static_cast<const Input*>(input);
    auto* out = static_cast<detail::ColorOutput<Input>*>(output);
    const auto workers = detail::active_workers(n);
    PerceptualInfo serial;
    std::vector<PerceptualInfo> parallel(workers > 1 ? static_cast<std::size_t>(workers) : 0);
    auto* partials = workers > 1 ? parallel.data() : &serial;
    const bool vectorized = has_avx2();
    detail::partition(n,workers,[&](auto begin, auto end, auto id) {
        PerceptualInfo status; // Keep hot warning counters off shared cache lines.
        std::size_t consumed = 0;
#ifdef SKIMAGEXX_AVX2
        if (vectorized) consumed = simd::perceptual_transform(p+std::ptrdiff_t(begin)*pixels,out+begin*spec.channels,
            end-begin,type,spec,pixels,channels,status);
#else
        (void)type; (void)vectorized;
#endif
        for (auto i = begin+consumed; i < end; ++i)
            detail::perceptual_pixel(p+std::ptrdiff_t(i)*pixels,out+i*spec.channels,spec,channels,status);
        partials[id] = status;
    });
    PerceptualInfo result;
    for (int j = 0; j < workers; ++j) {
        result.negative_z += partials[j].negative_z; result.unsupported = result.unsupported || partials[j].unsupported;
    }
    return result;
}
}
PerceptualInfo perceptual_into(const void* input, void* output, std::size_t n, PixelType type,
    PerceptualTransform transform, std::array<double,3> white, std::size_t channels,
    std::ptrdiff_t pixels, std::ptrdiff_t channel_stride,std::optional<std::array<double,2>> uv_reference) {
    const int op = static_cast<int>(transform);
    if (op < 0 || op > 11) throw std::invalid_argument("unknown perceptual transform");
    if (channels < 3 || ((op!=6 && op!=7) && channels != 3)) throw std::invalid_argument("invalid color channel count");
    for (auto value : white) if (!(value > 0) || !std::isfinite(value)) throw std::invalid_argument("white point must be finite and positive");
    if (n && (!input || !output)) throw std::invalid_argument("input and output must be nonnull");
    const auto bytes = std::max(pixel_size(type),type == PixelType::float32 ? std::size_t{4} : std::size_t{8});
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/bytes;
    const auto magnitude = [](std::ptrdiff_t stride) { return stride < 0 ? std::size_t(-(stride+1))+1 : std::size_t(stride); };
    const auto step = magnitude(pixels), offset = magnitude(channel_stride);
    if (channels > limit || n > limit/channels || offset > limit/(channels-1) ||
        (n > 1 && step > (limit-(channels-1)*offset)/(n-1))) throw std::overflow_error("color strides overflow");
    detail::PerceptualSpec spec{transform,white,channels};
    if (op>=8) spec.uv=type==PixelType::float32 ? detail::luv_reference<float>(white) : detail::luv_reference<double>(white);
    if (uv_reference) {
        for (auto value:*uv_reference) if (!(value>0) || !std::isfinite(value)) throw std::invalid_argument("u/v reference must be finite and positive");
        spec.uv=*uv_reference;
    }
    if (!n) return {};
    switch (type) {
        case PixelType::boolean: return execute_perceptual<bool>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::uint8: return execute_perceptual<std::uint8_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::int8: return execute_perceptual<std::int8_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::uint16: return execute_perceptual<std::uint16_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::int16: return execute_perceptual<std::int16_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::uint32: return execute_perceptual<std::uint32_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::int32: return execute_perceptual<std::int32_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::uint64: return execute_perceptual<std::uint64_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::int64: return execute_perceptual<std::int64_t>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::float32: return execute_perceptual<float>(input,output,n,type,spec,pixels,channel_stride);
        case PixelType::float64: return execute_perceptual<double>(input,output,n,type,spec,pixels,channel_stride);
        default: throw std::invalid_argument("unsupported pixel type");
    }
}
} // namespace skimagexx
