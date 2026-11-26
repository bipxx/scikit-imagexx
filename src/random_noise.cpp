#include "skimagexx/random_noise.hpp"

#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace skimagexx {
namespace {

struct Span {
    std::uintptr_t begin, end;
};

Span span(const double* data, std::size_t count) {
    if (count == 0)
        return {0, 0};
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    if (!data || address % alignof(double) ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) / sizeof(double))
        throw std::invalid_argument("invalid random_noise buffer");
    return {address, address + count * sizeof(double)};
}

void disjoint(Span target, Span source) {
    if (target.begin < source.end && source.begin < target.end)
        throw std::invalid_argument("random_noise output overlaps input");
}

void check_count(std::size_t count) {
    if (count == 0 || count > 524288 || std::fegetround() != FE_TONEAREST)
        throw std::invalid_argument("random_noise count or rounding mode is outside the bounded domain");
}

Span draw_span(RandomNoiseDraw draw, std::size_t count) {
    if (!(0 <= draw.probability && draw.probability <= 1))
        throw std::invalid_argument("random_noise probability must be in [0, 1]");
    const bool endpoint = draw.probability == 0 || draw.probability == 1;
    if (draw.count != (endpoint ? 0 : count) || (endpoint && draw.values))
        throw std::invalid_argument("random_noise draw count does not match the probability");
    return span(draw.values, draw.count);
}

}  // namespace

void random_noise_normal_into(const double* image, const double* noise, double* output,
    std::size_t count, bool speckle) {
    check_count(count);
    const auto target = span(output, count);
    disjoint(target, span(image, count));
    disjoint(target, span(noise, count));
    if (speckle) {
        for (std::size_t i = 0; i < count; ++i) {
            const double product = image[i] * noise[i];
            output[i] = image[i] + product;
        }
    } else {
        for (std::size_t i = 0; i < count; ++i)
            output[i] = image[i] + noise[i];
    }
}

void random_noise_impulse_into(const double* image, double* output, std::size_t count,
    RandomNoiseDraw flipped, RandomNoiseDraw salted, double low) {
    check_count(count);
    if (low != -1.0 && low != 0.0)
        throw std::invalid_argument("invalid random_noise impulse low value");
    const auto target = span(output, count);
    disjoint(target, span(image, count));
    disjoint(target, draw_span(flipped, count));
    disjoint(target, draw_span(salted, count));
    if (flipped.probability == 0) {
        std::copy_n(image, count, output);
    } else if (!salted.values) {
        const double value = salted.probability == 1 ? 1.0 : low;
        if (!flipped.values) {
            std::fill_n(output, count, value);
        } else {
            for (std::size_t i = 0; i < count; ++i)
                output[i] = flipped.values[i] <= flipped.probability ? value : image[i];
        }
    } else if (!flipped.values) {
        for (std::size_t i = 0; i < count; ++i)
            output[i] = salted.values[i] <= salted.probability ? 1.0 : low;
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            output[i] = flipped.values[i] <= flipped.probability ?
                (salted.values[i] <= salted.probability ? 1.0 : low) : image[i];
        }
    }
}

}  // namespace skimagexx
