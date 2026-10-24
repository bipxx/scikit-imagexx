#include "skimagexx/tv_reductions.hpp"

#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON

namespace skimagexx {
namespace {
std::size_t itemsize(TvReductionType type) {
    switch (type) {
    case TvReductionType::float32: return sizeof(float);
    case TvReductionType::float64: return sizeof(double);
    }
    throw std::invalid_argument("unknown TV reduction type");
}
std::size_t validate(TvReductionInput input) {
    const auto size = itemsize(input.type);
    if (input.count > std::numeric_limits<std::size_t>::max() / size)
        throw std::overflow_error("TV reduction byte size overflow");
    const auto bytes = input.count * size;
    const auto address = reinterpret_cast<std::uintptr_t>(input.data);
    if (bytes && (!input.data || address % size))
        throw std::invalid_argument("TV reduction requires aligned nonnull input");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
        throw std::overflow_error("TV reduction address overflow");
    if (std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("TV reduction requires nearest rounding");
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    if (_mm_getcsr() & ((1u << 15) | (1u << 6)))
        throw std::invalid_argument("TV reduction requires gradual underflow");
#endif
    return bytes;
}
class Capture {
    std::fenv_t original_;
public:
    Capture() {
        if (std::feholdexcept(&original_))
            throw std::runtime_error("cannot capture TV floating environment");
    }
    ~Capture() { std::fesetenv(&original_); }
    unsigned flags() const {
        const int flags = std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW);
        return ((flags & FE_INVALID) ? tv_fp_invalid : 0u) |
               ((flags & FE_DIVBYZERO) ? tv_fp_divide : 0u) |
               ((flags & FE_OVERFLOW) ? tv_fp_overflow : 0u) |
               ((flags & FE_UNDERFLOW) ? tv_fp_underflow : 0u);
    }
};

// Association adapted from NumPy v1.24.4 and v2.5.4 loops_utils.h.src;
// NumPy's BSD notice is retained in third_party/numpy-LICENSE.txt.
// FLOAT/DOUBLE_pairwise_sum, PW_BLOCKSIZE=128. Do not reassociate or use FMA.
template<class T> T pairwise(const T* a, std::size_t n) {
    if (n < 8) {
        T result = -T(0);
        for (std::size_t i = 0; i < n; ++i) result = T(result + a[i]);
        return result;
    }
    if (n <= 128) {
        T lanes[8];
        for (std::size_t j = 0; j < 8; ++j) lanes[j] = a[j];
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (std::size_t j = 0; j < 8; ++j) lanes[j] = T(lanes[j] + a[i+j]);
        T result = T(T(T(lanes[0]+lanes[1]) + T(lanes[2]+lanes[3])) +
                     T(T(lanes[4]+lanes[5]) + T(lanes[6]+lanes[7])));
        for (; i < n; ++i) result = T(result + a[i]);
        return result;
    }
    std::size_t half = n / 2;
    half -= half % 8;
    const T left = pairwise(a, half);
    const T right = pairwise(a+half, n-half);
    return T(left + right);
}
template<class T> TvReductionResult sum(TvReductionInput input, TvReductionPlan plan) {
    Capture capture;
    T value = T(0); // NumPy's ADD reduction identity, distinct from leaf -0.
    const auto* data = static_cast<const T*>(input.data);
    const std::size_t block = plan.profile == TvReductionProfile::numpy_1244_buffered
        ? plan.buffer_elements : input.count;
    for (std::size_t i = 0; i < input.count;) {
        const auto count = input.count-i < block ? input.count-i : block;
        value = T(value + pairwise(data+i, count));
        i += count;
    }
    TvReductionResult result{0, capture.flags()};
    std::memcpy(&result.bits, &value, sizeof(T));
    return result;
}
template<class T> unsigned square(TvReductionInput input, void* output) {
    Capture capture;
    const auto* source = static_cast<const T*>(input.data);
    auto* target = static_cast<T*>(output);
    for (std::size_t i = 0; i < input.count; ++i) target[i] = T(source[i] * source[i]);
    return capture.flags();
}
} // namespace

TvReductionResult tv_sum_contiguous(TvReductionInput input, TvReductionPlan plan) {
    validate(input);
    if (plan.profile != TvReductionProfile::numpy_1244_buffered &&
        plan.profile != TvReductionProfile::numpy_254_aligned_contiguous)
        throw std::invalid_argument("unknown audited TV reduction profile");
    if (!plan.buffer_elements)
        throw std::invalid_argument("TV reduction buffer must be positive");
    return input.type == TvReductionType::float32 ? sum<float>(input, plan) : sum<double>(input, plan);
}
unsigned tv_square_contiguous_into(TvReductionInput input, void* output) {
    const auto bytes = validate(input);
    validate({output, input.count, input.type});
    const auto a = reinterpret_cast<std::uintptr_t>(input.data);
    const auto b = reinterpret_cast<std::uintptr_t>(output);
    if (bytes && a < b+bytes && b < a+bytes)
        throw std::invalid_argument("TV square requires disjoint output");
    return input.type == TvReductionType::float32 ? square<float>(input, output) : square<double>(input, output);
}
} // namespace skimagexx
