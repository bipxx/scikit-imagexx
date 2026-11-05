// Source port: scikit-image0.26 draw_nd.py and NumPy linspace stages.
// BSD attribution: third_party/scikit-image-LICENSE.txt.
#include <skimagexx/line_nd.hpp>
#include "half.hpp"
#include "execution.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace skimagexx::detail {
namespace {
using Type = LineNDType;
struct FloatingScope {
    std::fenv_t saved;
    FloatingScope() {
        if (std::feholdexcept(&saved)) throw std::runtime_error("cannot capture line_nd floating environment");
    }
    ~FloatingScope() { std::fesetenv(&saved); }
};
double precision(double value, Type type) {
    if (type == Type::float16) {
        const auto single = float(value);
        const auto rounded = decode_half(encode_half(single));
        // Existing half.hpp performs software rounding, so report the same
        // inexact-tiny flag as a NumPy half arithmetic storage conversion.
        if (std::fabs(single) <= 0x1p-14f && rounded != single)
            std::feraiseexcept(FE_UNDERFLOW);
        return rounded;
    }
    if (type == Type::float32) return float(value);
    return value;
}
double divide(double value, std::size_t divisor, Type type) {
    if (type == Type::float16 || type == Type::float32)
        return precision(float(value) / float(divisor), type);
    return value / double(divisor);
}
template<class T> T load(const void* data) {
    T value; std::memcpy(&value, data, sizeof(value)); return value;
}
double value(LineNDVector input, std::size_t dimension) {
    const auto* pointer = static_cast<const unsigned char*>(input.data) +
        std::ptrdiff_t(input.size == 1 ? 0 : dimension) * input.stride;
    switch (input.type) {
        case Type::float16: return decode_half(load<std::uint16_t>(pointer));
        case Type::float32: return load<float>(pointer);
        case Type::float64: return load<double>(pointer);
        case Type::int8: return load<std::int8_t>(pointer);
        case Type::uint8: return load<std::uint8_t>(pointer);
        case Type::int64: return double(load<std::int64_t>(pointer));
        case Type::uint64: return double(load<std::uint64_t>(pointer));
    }
    return std::numeric_limits<double>::quiet_NaN();
}
double count_distance(double start, double stop, Type type) {
    if (type == Type::int8) {
        const auto wrapped = (int(stop) - int(start)) & 255;
        const auto delta = wrapped >= 128 ? wrapped - 256 : wrapped;
        // abs(INT8_MIN) wraps as an int8 NumPy ufunc, producing -128.
        return delta == -128 ? -128 : std::abs(delta);
    }
    if (type == Type::uint8) return (int(stop) - int(start)) & 255;
    return std::fabs(precision(stop - start, type));
}
double sample(const LineNDPlan& plan, std::size_t dimension, std::size_t point) {
    const auto p = plan.precision;
    double y = precision(double(point), p);
    const auto divisor = plan.endpoint ? plan.points - 1 : plan.points;
    if (divisor > 0) {
        if (plan.zero_step) {
            y = divide(y, divisor, p);
            y = precision(y * plan.delta[dimension], p);
        } else y = precision(y * plan.step[dimension], p);
    } else y = precision(y * plan.delta[dimension], p);
    y = precision(y + plan.start[dimension], p);
    if (plan.endpoint && plan.points > 1 && point == plan.points - 1)
        y = plan.stop[dimension];
    return y;
}
template<class T> void fill_dimension(const LineNDPlan& plan, std::size_t dimension,
                                     unsigned char* output, std::ptrdiff_t stride,
                                     bool floor_ties) {
    if constexpr (std::is_same_v<T, double>) {
        if (!plan.integer && stride == std::ptrdiff_t(sizeof(double))) {
            const auto count = plan.points;
            const auto first = plan.start[dimension], delta = plan.delta[dimension];
            const auto step = plan.step[dimension];
            const auto divisor = plan.endpoint && plan.points ? plan.points - 1 : plan.points;
            // Contiguous dimension: independent literal multiply then add,
            // preserving the global zero-step linspace branch for denormals.
            if (divisor && plan.zero_step) {
                for (std::size_t i = 0; i < count; ++i) {
                    const auto value = (double(i) / double(divisor)) * delta + first;
                    std::memcpy(output + i * sizeof(double), &value, sizeof(value));
                }
            } else {
                const auto factor = divisor ? step : delta;
                for (std::size_t i = 0; i < count; ++i) {
                    const auto value = double(i) * factor + first;
                    std::memcpy(output + i * sizeof(double), &value, sizeof(value));
                }
            }
            if (plan.endpoint && plan.points > 1)
                std::memcpy(output + (plan.points - 1) * sizeof(double), &plan.stop[dimension], sizeof(double));
            return;
        }
    }
    // Independent coordinates; no reassociation or contracted multiply-add.
    for (std::size_t i = 0; i < plan.points; ++i) {
        auto coordinate = sample(plan, dimension, i);
        if (plan.integer && dimension < plan.rounded_dimensions)
            coordinate = floor_ties ? std::floor(coordinate) : std::nearbyint(coordinate);
        T converted;
        if constexpr (std::is_same_v<T, std::uint16_t>) converted = encode_half(float(coordinate));
        else converted = T(coordinate);
        std::memcpy(output + std::ptrdiff_t(i) * stride, &converted, sizeof(converted));
    }
}
}

bool line_nd_prepare(LineNDVector start, LineNDVector stop, bool endpoint,
                     bool integer, LineNDPlan& output, LineNDScalarPromotion promotion) {
    FloatingScope environment;
    if (start.type != stop.type || !start.size || !stop.size ||
        std::max(start.size, stop.size) > 8 ||
        (start.size != stop.size && start.size != 1 && stop.size != 1)) return false;
    LineNDPlan plan;
    plan.dimensions = std::max(start.size, stop.size);
    plan.rounded_dimensions = start.size;
    plan.endpoint = endpoint; plan.integer = integer;
    plan.scalar_promotion = promotion;
    plan.precision = start.type == Type::float16 || start.type == Type::float32 ? start.type : Type::float64;
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t d = 0; d < plan.dimensions; ++d) {
        const auto a = value(start, d), b = value(stop, d);
        // Bounded exact integer conversion domain; reject wide unsigned wrap.
        if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(a) >= 0x1p52 || std::fabs(b) >= 0x1p52 ||
            (start.type == Type::uint64 && b < a)) return false;
        const auto distance = count_distance(a, b, start.type);
        if (!std::isfinite(distance)) return false;
        maximum = std::max(maximum, distance);
        plan.start[d] = a; plan.stop[d] = b;
        plan.delta[d] = precision(b - a, plan.precision);
        if (!std::isfinite(plan.delta[d])) return false;
    }
    const auto count = std::ceil(maximum) + (endpoint ? 1 : 0);
    if (count < 0 || count > double(std::numeric_limits<std::ptrdiff_t>::max() / 8 / plan.dimensions) ||
        (plan.precision == Type::float16 && count > 65504)) return false;
    plan.points = std::size_t(count);
    const auto divisor = endpoint && plan.points ? plan.points - 1 : plan.points;
    std::feclearexcept(FE_ALL_EXCEPT);
    if (divisor) {
        for (std::size_t d = 0; d < plan.dimensions; ++d) {
            plan.step[d] = divide(plan.delta[d], divisor, plan.precision);
            plan.zero_step = plan.zero_step || plan.step[d] == 0;
        }
    }
    plan.events.divide_underflow = bool(std::fetestexcept(FE_UNDERFLOW));
    output = plan;
    return true;
}

LineNDFPEvents line_nd_into(const LineNDPlan& plan, void* output,
                          std::ptrdiff_t point_step, std::ptrdiff_t dimension_step,
                          int workers) {
    FloatingScope environment;
    auto events = plan.events;
    if (!plan.points) return events;
    // Preserve FP stage facts per worker; aggregate only after independent work.
    const auto team = plan.points * plan.dimensions >= parallel_min ? workers : 1;
    unsigned underflow = 0;
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp parallel for num_threads(team) if(team > 1) reduction(|:underflow)
#endif
    for (std::ptrdiff_t d = 0; d < std::ptrdiff_t(plan.dimensions); ++d) {
        FloatingScope worker_environment;
        std::feclearexcept(FE_ALL_EXCEPT);
        const auto first = plan.points ? sample(plan, std::size_t(d), 0) : 0;
        const auto second = plan.points > 1 ? sample(plan, std::size_t(d), 1) : 0;
        const auto remainder = first - std::floor(first);
        const auto scalar_remainder = plan.scalar_promotion == LineNDScalarPromotion::weak_input
            ? precision(remainder, plan.precision) : remainder;
        const bool floor_ties = plan.points > 1 && scalar_remainder == .5 &&
            precision(second - first, plan.precision) == 1;
        auto* destination = static_cast<unsigned char*>(output) + d * dimension_step;
        if (plan.integer) fill_dimension<std::intptr_t>(plan, std::size_t(d), destination, point_step, floor_ties);
        else if (plan.precision == Type::float16) fill_dimension<std::uint16_t>(plan, std::size_t(d), destination, point_step, false);
        else if (plan.precision == Type::float32) fill_dimension<float>(plan, std::size_t(d), destination, point_step, false);
        else fill_dimension<double>(plan, std::size_t(d), destination, point_step, false);
        underflow |= unsigned(bool(std::fetestexcept(FE_UNDERFLOW)));
    }
    (void)team;
    events.multiply_underflow = bool(underflow);
    return events;
}
}
