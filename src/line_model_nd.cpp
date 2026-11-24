#include "skimagexx/line_model_nd.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace skimagexx {
namespace {

bool safe_prediction_coordinate(double value) noexcept {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t word;
    std::memcpy(&word, &value, sizeof(word));
    const auto magnitude = word & 0x7fffffffffffffffULL;
    // The binary64 finite magnitude interval [2^-128, 2^128], checked
    // without changing flags on unsupported NaN, infinity or denormal input.
    return magnitude == 0 ||
           (magnitude >= 0x37f0000000000000ULL && magnitude <= 0x47f0000000000000ULL);
}

} // namespace

bool line_model_nd_two_point(const double* input, std::size_t dimensions,
                            double* origin, double* direction) noexcept {
    if (!input || !origin || !direction || (dimensions != 2 && dimensions != 3) ||
        std::fegetround() != FE_TONEAREST)
        return false;
    constexpr auto bound = 1048576.0;
    for (std::size_t index = 0; index < 2 * dimensions; ++index) {
        const auto value = input[index];
        if (!std::isfinite(value) || std::abs(value) > bound || std::trunc(value) != value ||
            (value == 0 && std::signbit(value)))
            return false;
    }

    // These bounds make centered subtraction and the entire squared sum
    // exact in binary64, including the NumPy norm dot-product reduction.
    double squares = 0;
    for (std::size_t axis = 0; axis < dimensions; ++axis) {
        const auto first = input[axis];
        const auto second = input[dimensions + axis];
        const auto center = (first + second) / 2;
        const auto delta = (second - center) - (first - center);
        origin[axis] = center;
        direction[axis] = delta;
        squares += delta * delta;
    }
    if (squares != 0) {
        const auto norm = std::sqrt(squares);
        for (std::size_t axis = 0; axis < dimensions; ++axis)
            direction[axis] /= norm;
    }
    return true;
}

namespace {

template<std::size_t dimensions>
bool predict(const double* input, std::size_t count,
             const double* origin, const double* direction,
             std::size_t axis, double* output) noexcept {
    if (!input || !origin || !direction || !output || count < 128 || count > 65536 ||
        axis >= dimensions || std::fegetround() != FE_TONEAREST)
        return false;
    for (std::size_t i = 0; i < dimensions; ++i) {
        if (!safe_prediction_coordinate(origin[i]) || !safe_prediction_coordinate(direction[i]))
            return false;
    }
    if (direction[axis] == 0) return false;
    for (std::size_t i = 0; i < count; ++i) {
        if (!safe_prediction_coordinate(input[i])) return false;
    }
    // A nonzero subtraction is at least 2^-180. The quotient, products and
    // final additions stay finite and away from underflow. Compile this unit
    // with contraction and reciprocal substitution disabled.
    for (std::size_t i = 0; i < count; ++i) {
        const auto length = (input[i] - origin[axis]) / direction[axis];
        for (std::size_t column = 0; column < dimensions; ++column) {
            const auto product = length * direction[column];
            output[dimensions * i + column] = origin[column] + product;
        }
    }
    return true;
}

} // namespace

bool line_model_nd_predict_2d(const double* input, std::size_t count,
                             const double* origin, const double* direction,
                             std::size_t axis, double* output) noexcept {
    return predict<2>(input, count, origin, direction, axis, output);
}

bool line_model_nd_predict_3d(const double* input, std::size_t count,
                             const double* origin, const double* direction,
                             std::size_t axis, double* output) noexcept {
    return predict<3>(input, count, origin, direction, axis, output);
}

} // namespace skimagexx
