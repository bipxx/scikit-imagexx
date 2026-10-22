#include "simd.hpp"
#include <immintrin.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <type_traits>

namespace skimagexx::simd {
namespace {
__m128i low_words64(__m256i value) {
    const auto lanes = _mm256_shuffle_epi32(value,_MM_SHUFFLE(2,0,2,0));
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(lanes),_mm256_extracti128_si256(lanes,1));
}
template<class T> T maximum_wide(const T* p, std::size_t n) {
    constexpr bool wide = sizeof(T) == 8, sign = std::is_signed_v<T>;
    constexpr std::size_t lanes = 32/sizeof(T);
    auto best = wide ? _mm256_set1_epi64x(static_cast<long long>(std::numeric_limits<T>::lowest())) :
                       _mm256_set1_epi32(static_cast<int>(std::numeric_limits<T>::lowest()));
    const auto flip = _mm256_set1_epi64x(std::numeric_limits<long long>::min());
    std::size_t i = 0;
    for (; i+lanes <= n; i += lanes) {
        const auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
        if constexpr (!wide && sign) best = _mm256_max_epi32(best,value);
        else if constexpr (!wide) best = _mm256_max_epu32(best,value);
        else {
            const auto greater = sign ? _mm256_cmpgt_epi64(value,best) :
                _mm256_cmpgt_epi64(_mm256_xor_si256(value,flip),_mm256_xor_si256(best,flip));
            best = _mm256_blendv_epi8(best,value,greater);
        }
    }
    T values[lanes];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(values),best);
    T result = *std::max_element(values,values+lanes);
    for (; i < n; ++i) result = std::max(result,p[i]);
    return result;
}
template<class T> void bounds16(const T* p, std::size_t n, double& low, double& high) {
    constexpr bool signed_input = std::is_signed_v<T>;
    auto minimum = _mm256_set1_epi16(static_cast<short>(std::numeric_limits<T>::max()));
    auto maximum = _mm256_set1_epi16(static_cast<short>(std::numeric_limits<T>::lowest()));
    std::size_t i = 0;
    for (; i+16 <= n; i += 16) {
        const auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
        if constexpr (signed_input) {
            minimum = _mm256_min_epi16(minimum,value); maximum = _mm256_max_epi16(maximum,value);
        } else {
            minimum = _mm256_min_epu16(minimum,value); maximum = _mm256_max_epu16(maximum,value);
        }
    }
    T minima[16], maxima[16];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(minima),minimum);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(maxima),maximum);
    low = double(*std::min_element(minima,minima+16)); high = double(*std::max_element(maxima,maxima+16));
    for (; i < n; ++i) { low = std::min(low,double(p[i])); high = std::max(high,double(p[i])); }
}
__m128i compare_double(const double* p) {
    const auto compare = _mm256_castpd_si256(_mm256_cmp_pd(_mm256_loadu_pd(p),_mm256_set1_pd(0.5),_CMP_GT_OQ));
    const auto lanes = _mm256_shuffle_epi32(compare,_MM_SHUFFLE(2,0,2,0));
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(lanes),_mm256_extracti128_si256(lanes,1));
}
void store_ints(void* out, __m256i values, PixelType target) {
    const auto low = _mm256_castsi256_si128(values), high = _mm256_extracti128_si256(values,1);
    if (target == PixelType::int16) _mm_storeu_si128(static_cast<__m128i*>(out),_mm_packs_epi32(low,high));
    else {
        const auto words = _mm_packus_epi32(low,high);
        if (target == PixelType::uint16) _mm_storeu_si128(static_cast<__m128i*>(out),words);
        else _mm_storel_epi64(static_cast<__m128i*>(out),_mm_packus_epi16(words,_mm_setzero_si128()));
    }
}
void store_four(void* out, __m128i values, PixelType target) {
    if (target == PixelType::int16) _mm_storel_epi64(static_cast<__m128i*>(out),_mm_packs_epi32(values,_mm_setzero_si128()));
    else {
        const auto words = _mm_packus_epi32(values,_mm_setzero_si128());
        if (target == PixelType::uint16) _mm_storel_epi64(static_cast<__m128i*>(out),words);
        else {
            const auto bytes = _mm_packus_epi16(words,_mm_setzero_si128());
            const auto value = _mm_cvtsi128_si32(bytes);
            std::memcpy(out,&value,sizeof(value));
        }
    }
}
}
void bounds(const std::uint16_t* p, std::size_t n, double& low, double& high) { bounds16(p,n,low,high); }
void bounds(const std::int16_t* p, std::size_t n, double& low, double& high) { bounds16(p,n,low,high); }
std::uint32_t maximum(const std::uint32_t* p, std::size_t n) { return maximum_wide(p,n); }
std::int32_t maximum(const std::int32_t* p, std::size_t n) { return maximum_wide(p,n); }
std::uint64_t maximum(const std::uint64_t* p, std::size_t n) { return maximum_wide(p,n); }
std::int64_t maximum(const std::int64_t* p, std::size_t n) { return maximum_wide(p,n); }
std::size_t convert(const void* source, void* destination, std::size_t n,
                    PixelType input, PixelType output, bool unscaled, ConversionInfo& info) {
    std::size_t i = 0;
    const auto outsize = pixel_size(output);
    if(input==PixelType::uint16 && output==PixelType::int16) {
        const auto* p=static_cast<const std::uint16_t*>(source);
        auto* out=static_cast<std::int16_t*>(destination);
        for(;i+16<=n;i+=16) {
            auto values=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
            if(!unscaled)values=_mm256_srli_epi16(values,1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),values);
        }
        if(i+8<=n) {
            auto values=_mm_loadu_si128(reinterpret_cast<const __m128i*>(p+i));
            if(!unscaled)values=_mm_srli_epi16(values,1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out+i),values);i+=8;
        }
        return i;
    }
    if (output == PixelType::boolean && input != PixelType::boolean && input != PixelType::uint8 &&
        input != PixelType::float32 && input != PixelType::float64) {
        const auto* p = static_cast<const unsigned char*>(source);
        auto* out = static_cast<unsigned char*>(destination);
        const auto width = pixel_size(input);
        const bool sign = input == PixelType::int8 || input == PixelType::int16 || input == PixelType::int32 || input == PixelType::int64;
        if (width == 1) {
            for (; i+32 <= n; i += 32) {
                const auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm256_and_si256(
                    _mm256_cmpgt_epi8(value,_mm256_set1_epi8(63)),_mm256_set1_epi8(1)));
            }
        } else if (width == 2) {
            const auto limit = _mm256_set1_epi16(sign ? 16383 : 0);
            const auto flip = _mm256_set1_epi16(static_cast<short>(0x8000));
            for (; i+32 <= n; i += 32) {
                auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+2*i));
                auto b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+2*i+32));
                if (!sign) { a = _mm256_xor_si256(a,flip); b = _mm256_xor_si256(b,flip); }
                // Unsigned threshold is 32767, equivalent to signed value >= 0 after flipping.
                const auto threshold = sign ? limit : _mm256_set1_epi16(-1);
                const auto packed = _mm256_packs_epi16(_mm256_cmpgt_epi16(a,threshold),_mm256_cmpgt_epi16(b,threshold));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm256_and_si256(
                    _mm256_permute4x64_epi64(packed,_MM_SHUFFLE(3,1,2,0)),_mm256_set1_epi8(1)));
            }
        } else if (width == 4) {
            for (; i+32 <= n; i += 32) {
                __m256i comparisons[4];
                for (int j = 0; j < 4; ++j) {
                    const auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+4*i+32*j));
                    comparisons[j] = sign ? _mm256_cmpgt_epi32(value,_mm256_set1_epi32(1073741823)) : _mm256_srai_epi32(value,31);
                }
                const auto packed = _mm256_packs_epi16(_mm256_packs_epi32(comparisons[0],comparisons[1]),
                                                      _mm256_packs_epi32(comparisons[2],comparisons[3]));
                const auto ordered = _mm256_permutevar8x32_epi32(packed,_mm256_setr_epi32(0,4,1,5,2,6,3,7));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm256_and_si256(ordered,_mm256_set1_epi8(1)));
            }
        } else {
            const auto flip = _mm256_set1_epi64x(std::numeric_limits<long long>::min());
            for (; i+16 <= n; i += 16) {
                __m128i comparisons[4];
                for (int j = 0; j < 4; ++j) {
                    const auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+8*i+32*j));
                    const auto comparison = sign ? _mm256_cmpgt_epi64(value,_mm256_set1_epi64x(4611686018427387904LL)) :
                        _mm256_cmpgt_epi64(_mm256_xor_si256(value,flip),_mm256_setzero_si256());
                    comparisons[j] = low_words64(comparison);
                }
                const auto packed = _mm_packs_epi16(_mm_packs_epi32(comparisons[0],comparisons[1]),_mm_packs_epi32(comparisons[2],comparisons[3]));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out+i),_mm_and_si128(packed,_mm_set1_epi8(1)));
            }
        }
        return i;
    }
    if ((input == PixelType::uint32 || input == PixelType::int32 || input == PixelType::uint64 || input == PixelType::int64) &&
        (output == PixelType::uint8 || output == PixelType::uint16 || output == PixelType::int16)) {
        const bool wide = pixel_size(input) == 8;
        const bool sign = input == PixelType::int32 || input == PixelType::int64;
        const int target_bits = output == PixelType::uint8 ? 8 : output == PixelType::uint16 ? 16 : 15;
        if (!wide) {
            const auto* p = static_cast<const std::uint32_t*>(source);
            for (; i+8 <= n; i += 8) {
                auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
                if (sign && output == PixelType::int16) {
                    value = unscaled ? _mm256_srai_epi32(_mm256_slli_epi32(value,16),16) : _mm256_srai_epi32(value,16);
                } else if (sign) {
                    if (unscaled) {
                        if (output == PixelType::uint8) value = _mm256_srai_epi32(_mm256_slli_epi32(value,16),16);
                        value = _mm256_and_si256(_mm256_max_epi32(value,_mm256_setzero_si256()),_mm256_set1_epi32((1<<target_bits)-1));
                    } else value = _mm256_srl_epi32(_mm256_max_epi32(value,_mm256_setzero_si256()),_mm_cvtsi32_si128(31-target_bits));
                } else if (!unscaled) value = _mm256_srl_epi32(value,_mm_cvtsi32_si128(32-target_bits));
                store_ints(static_cast<unsigned char*>(destination)+i*outsize,value,output);
            }
        } else {
            const auto* p = static_cast<const std::uint64_t*>(source);
            for (; i+4 <= n; i += 4) {
                auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
                if (!unscaled) {
                    if (sign && output != PixelType::int16)
                        value = _mm256_and_si256(value,_mm256_cmpgt_epi64(value,_mm256_setzero_si256()));
                    const int shift = sign && output == PixelType::int16 ? 48 : 64-int(sign)-target_bits;
                    value = _mm256_srl_epi64(value,_mm_cvtsi32_si128(shift));
                }
                auto narrowed = low_words64(value);
                if (sign && output == PixelType::int16)
                    narrowed = _mm_srai_epi32(_mm_slli_epi32(narrowed,16),16);
                else if (sign && unscaled) {
                    if (output == PixelType::uint8) narrowed = _mm_srai_epi32(_mm_slli_epi32(narrowed,16),16);
                    narrowed = _mm_and_si128(_mm_max_epi32(narrowed,_mm_setzero_si128()),_mm_set1_epi32((1<<target_bits)-1));
                }
                store_four(static_cast<unsigned char*>(destination)+i*outsize,narrowed,output);
            }
        }
        return i;
    }
    if (input == PixelType::float64 && output == PixelType::float32) {
        const auto* p = static_cast<const double*>(source);
        auto* out = static_cast<float*>(destination);
        const unsigned saved = _mm_getcsr();
        _mm_setcsr(saved & ~0x3fU);
        for (; i+16 <= n; i += 16) {
            _mm_storeu_ps(out+i,_mm256_cvtpd_ps(_mm256_loadu_pd(p+i)));
            _mm_storeu_ps(out+i+4,_mm256_cvtpd_ps(_mm256_loadu_pd(p+i+4)));
            _mm_storeu_ps(out+i+8,_mm256_cvtpd_ps(_mm256_loadu_pd(p+i+8)));
            _mm_storeu_ps(out+i+12,_mm256_cvtpd_ps(_mm256_loadu_pd(p+i+12)));
        }
        for (; i+4 <= n; i += 4) _mm_storeu_ps(out+i,_mm256_cvtpd_ps(_mm256_loadu_pd(p+i)));
        const unsigned raised = _mm_getcsr();
        info.floating_overflow = (raised & 8U) != 0;
        info.floating_underflow = (raised & 16U) != 0;
        info.invalid_float_cast = (raised & 1U) != 0;
        _mm_setcsr(saved | (raised & 0x3fU));
        return i;
    }
    if (input == PixelType::float32 && output == PixelType::float64) {
        const auto* p = static_cast<const float*>(source);
        auto* out = static_cast<double*>(destination);
        const unsigned saved = _mm_getcsr();
        _mm_setcsr(saved & ~0x3fU);
        for (; i+16 <= n; i += 16) {
            for (int offset = 0; offset < 16; offset += 4)
                _mm256_storeu_pd(out+i+offset,_mm256_cvtps_pd(_mm_loadu_ps(p+i+offset)));
        }
        for (; i+4 <= n; i += 4) _mm256_storeu_pd(out+i,_mm256_cvtps_pd(_mm_loadu_ps(p+i)));
        const unsigned raised = _mm_getcsr();
        info.invalid_float_cast = (raised & 1U) != 0;
        _mm_setcsr(saved | (raised & 0x3fU));
        return i;
    }
    if (output == PixelType::boolean && (input == PixelType::float32 || input == PixelType::float64)) {
        auto* out = static_cast<unsigned char*>(destination);
        if (input == PixelType::float32) {
            const auto* p = static_cast<const float*>(source);
            const auto half = _mm256_set1_ps(0.5f);
            for (; i+32 <= n; i += 32) {
                const auto a = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(p+i),half,_CMP_GT_OQ));
                const auto b = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(p+i+8),half,_CMP_GT_OQ));
                const auto c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(p+i+16),half,_CMP_GT_OQ));
                const auto d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_loadu_ps(p+i+24),half,_CMP_GT_OQ));
                const auto packed = _mm256_packs_epi16(_mm256_packs_epi32(a,b),_mm256_packs_epi32(c,d));
                const auto ordered = _mm256_permutevar8x32_epi32(packed,_mm256_setr_epi32(0,4,1,5,2,6,3,7));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),_mm256_and_si256(ordered,_mm256_set1_epi8(1)));
            }
        } else {
            const auto* p = static_cast<const double*>(source);
            for (; i+16 <= n; i += 16) {
                const auto a = _mm_packs_epi32(compare_double(p+i),compare_double(p+i+4));
                const auto b = _mm_packs_epi32(compare_double(p+i+8),compare_double(p+i+12));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out+i),_mm_and_si128(_mm_packs_epi16(a,b),_mm_set1_epi8(1)));
            }
        }
        return i;
    }
    if (input == PixelType::uint8 || input == PixelType::int8 || input == PixelType::boolean) {
        const auto* p = static_cast<const std::uint8_t*>(source);
        const bool binary = input == PixelType::boolean;
        const bool signed_byte = input == PixelType::int8;
        if (output == PixelType::boolean || (binary && output == PixelType::uint8)) {
            auto* out = static_cast<std::uint8_t*>(destination);
            for (; i+32 <= n; i += 32) {
                const auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p+i));
                auto value = binary ? _mm256_xor_si256(_mm256_cmpeq_epi8(raw,_mm256_setzero_si256()),_mm256_set1_epi8(-1)) :
                                       _mm256_and_si256(_mm256_srli_epi16(raw,7),_mm256_set1_epi8(1));
                if (output == PixelType::boolean && binary) value = _mm256_and_si256(value,_mm256_set1_epi8(1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out+i),value);
            }
            return i;
        }
        for (; i+8 <= n; i += 8) {
            auto raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+i));
            if (binary) raw = _mm_andnot_si128(_mm_cmpeq_epi8(raw,_mm_setzero_si128()),_mm_set1_epi8(1));
            const auto values = signed_byte ? _mm256_cvtepi8_epi32(raw) : _mm256_cvtepu8_epi32(raw);
            if (output == PixelType::float32) {
                auto value = _mm256_mul_ps(_mm256_cvtepi32_ps(values),_mm256_set1_ps(binary ? 1.0f : signed_byte ? float(1.0/127.0) : float(1.0/255.0)));
                if (signed_byte) value = _mm256_max_ps(value,_mm256_set1_ps(-1));
                _mm256_storeu_ps(static_cast<float*>(destination)+i,value);
            } else if (output == PixelType::float64) {
                auto* out = static_cast<double*>(destination)+i;
                const auto scale = _mm256_set1_pd(binary ? 1.0 : signed_byte ? 1.0/127.0 : 1.0/255.0);
                auto low = _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(values)),scale);
                auto high = _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(values,1)),scale);
                if (signed_byte) { low = _mm256_max_pd(low,_mm256_set1_pd(-1)); high = _mm256_max_pd(high,_mm256_set1_pd(-1)); }
                _mm256_storeu_pd(out,low); _mm256_storeu_pd(out+4,high);
            } else if (signed_byte && output == PixelType::uint8) {
                const auto value = _mm256_srli_epi32(_mm256_mullo_epi32(_mm256_max_epi32(values,_mm256_setzero_si256()),_mm256_set1_epi32(129)),6);
                store_ints(static_cast<unsigned char*>(destination)+i,value,output);
            } else if (output == PixelType::uint16 || output == PixelType::int16) {
                if (signed_byte) {
                    const auto value = output == PixelType::int16 ? _mm256_add_epi32(_mm256_mullo_epi32(values,_mm256_set1_epi32(257)),_mm256_set1_epi32(128)) :
                        _mm256_srli_epi32(_mm256_mullo_epi32(_mm256_max_epi32(values,_mm256_setzero_si256()),_mm256_set1_epi32(16513)),5);
                    store_ints(static_cast<unsigned char*>(destination)+i*outsize,value,output);
                    continue;
                }
                const auto scaled = binary ? _mm256_mullo_epi32(values,_mm256_set1_epi32(output == PixelType::uint16 ? 65535 : 32767)) :
                    _mm256_mullo_epi32(values,_mm256_set1_epi32(257));
                store_ints(static_cast<unsigned char*>(destination)+i*outsize,
                    output == PixelType::int16 && !binary ? _mm256_srli_epi32(scaled,1) : scaled,output);
            } else return i;
        }
        return i;
    }
    if (input == PixelType::uint16 || input == PixelType::int16) {
        const bool signed_input = input == PixelType::int16;
        const auto* p = static_cast<const std::uint16_t*>(source);
        for (; i+8 <= n; i += 8) {
            const auto words = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p+i));
            const auto values = signed_input ? _mm256_cvtepi16_epi32(words) : _mm256_cvtepu16_epi32(words);
            if (output == PixelType::float32) {
                auto value = _mm256_mul_ps(_mm256_cvtepi32_ps(values),_mm256_set1_ps(float(signed_input ? 1.0/32767.0 : 1.0/65535.0)));
                if (signed_input) value = _mm256_max_ps(value,_mm256_set1_ps(-1));
                _mm256_storeu_ps(static_cast<float*>(destination)+i,value);
            } else if (output == PixelType::float64) {
                const auto factor = _mm256_set1_pd(signed_input ? 1.0/32767.0 : 1.0/65535.0);
                auto low = _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(values)),factor);
                auto high = _mm256_mul_pd(_mm256_cvtepi32_pd(_mm256_extracti128_si256(values,1)),factor);
                if (signed_input) { low = _mm256_max_pd(low,_mm256_set1_pd(-1)); high = _mm256_max_pd(high,_mm256_set1_pd(-1)); }
                _mm256_storeu_pd(static_cast<double*>(destination)+i,low);
                _mm256_storeu_pd(static_cast<double*>(destination)+i+4,high);
            } else if (output == PixelType::uint8) {
                auto value = unscaled ? values : signed_input ? _mm256_srai_epi32(values,7) : _mm256_srli_epi32(values,8);
                if (signed_input) value = _mm256_max_epi32(value,_mm256_setzero_si256());
                store_ints(static_cast<unsigned char*>(destination)+i,value,output);
            } else if (output == PixelType::uint16) {
                const auto positive = _mm256_max_epi32(values,_mm256_setzero_si256());
                const auto value = signed_input ? _mm256_srli_epi32(_mm256_mullo_epi32(positive,_mm256_set1_epi32(32769)),14) : values;
                store_ints(static_cast<unsigned char*>(destination)+2*i,value,output);
            } else if (output == PixelType::int16) {
                const auto value = !signed_input && !unscaled ? _mm256_srli_epi32(values,1) : values;
                store_ints(static_cast<unsigned char*>(destination)+2*i,value,output);
            } else if (output == PixelType::boolean) {
                const auto value = _mm256_and_si256(_mm256_cmpgt_epi32(values,_mm256_set1_epi32(signed_input ? 16383 : 32767)),_mm256_set1_epi32(1));
                store_ints(static_cast<unsigned char*>(destination)+i,value,output);
            } else return i;
        }
        return i;
    }
    if (input == PixelType::float32) {
        const auto* p = static_cast<const float*>(source);
        if (output != PixelType::uint8 && output != PixelType::uint16 && output != PixelType::int16) return 0;
        for (; i+8 <= n; i += 8) {
            const auto value = _mm256_loadu_ps(p+i);
            {
                const auto scale = _mm256_set1_ps(output == PixelType::uint8 ? 255.0f : output == PixelType::uint16 ? 65535.0f : 32767.5f);
                auto adjusted = _mm256_mul_ps(value,scale);
                if (output == PixelType::int16) adjusted = _mm256_sub_ps(adjusted,_mm256_set1_ps(0.5f));
                adjusted = _mm256_round_ps(adjusted,_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                adjusted = _mm256_min_ps(_mm256_max_ps(adjusted,_mm256_set1_ps(output == PixelType::int16 ? -32768.0f : 0.0f)),
                    _mm256_set1_ps(output == PixelType::uint8 ? 255.0f : output == PixelType::uint16 ? 65535.0f : 32767.0f));
                adjusted = _mm256_blendv_ps(adjusted,_mm256_setzero_ps(),_mm256_cmp_ps(value,value,_CMP_UNORD_Q));
                store_ints(static_cast<unsigned char*>(destination)+i*outsize,_mm256_cvttps_epi32(adjusted),output);
            }
        }
        return i;
    }
    if (input == PixelType::float64) {
        const auto* p = static_cast<const double*>(source);
        if (output != PixelType::uint8 && output != PixelType::uint16 && output != PixelType::int16) return 0;
        for (; i+4 <= n; i += 4) {
            const auto value = _mm256_loadu_pd(p+i);
            {
                const auto scale = _mm256_set1_pd(output == PixelType::uint8 ? 255.0 : output == PixelType::uint16 ? 65535.0 : 32767.5);
                auto adjusted = _mm256_mul_pd(value,scale);
                if (output == PixelType::int16) adjusted = _mm256_sub_pd(adjusted,_mm256_set1_pd(0.5));
                adjusted = _mm256_round_pd(adjusted,_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                adjusted = _mm256_min_pd(_mm256_max_pd(adjusted,_mm256_set1_pd(output == PixelType::int16 ? -32768.0 : 0.0)),
                    _mm256_set1_pd(output == PixelType::uint8 ? 255.0 : output == PixelType::uint16 ? 65535.0 : 32767.0));
                adjusted = _mm256_blendv_pd(adjusted,_mm256_setzero_pd(),_mm256_cmp_pd(value,value,_CMP_UNORD_Q));
                store_four(static_cast<unsigned char*>(destination)+i*outsize,_mm256_cvttpd_epi32(adjusted),output);
            }
        }
        return i;
    }
    return 0;
}
} // namespace skimagexx::simd
