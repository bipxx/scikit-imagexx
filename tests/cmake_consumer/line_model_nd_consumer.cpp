#include "skimagexx/line_model_nd.hpp"

#include <array>
#include <cfenv>
#include <cmath>
#include <limits>

int main() {
    const double input[] = {1, 2, 3, 4, 6, 8};
    std::array<double, 3> center{}, direction{};
    if (!skimagexx::line_model_nd_two_point(input, 3, center.data(), direction.data()) ||
        center != std::array<double, 3>{2.5, 4.0, 5.5} ||
        direction != std::array<double, 3>{3/std::sqrt(50.0), 4/std::sqrt(50.0), 5/std::sqrt(50.0)})
        return 1;
    const double duplicates[] = {1, 2, 1, 2};
    if (!skimagexx::line_model_nd_two_point(duplicates, 2, center.data(), direction.data()) ||
        direction[0] != 0 || direction[1] != 0)
        return 2;
    const double invalid[] = {1, 2, 1, std::numeric_limits<double>::infinity()};
    if (skimagexx::line_model_nd_two_point(invalid, 2, center.data(), direction.data()))
        return 3;

    std::array<double, 128> values{};
    std::array<double, 256> prediction{};
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<double>(i) / 8;
    const double origin2[] = {1, 2}, direction2[] = {3, 4};
    if (!skimagexx::line_model_nd_predict_2d(values.data(), values.size(),
            origin2, direction2, 1, prediction.data()))
        return 4;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto t = (values[i] - origin2[1]) / direction2[1];
        if (prediction[2 * i] != origin2[0] + t * direction2[0] ||
            prediction[2 * i + 1] != origin2[1] + t * direction2[1])
            return 5;
    }
    const auto previous = prediction;
    values.back() = std::numeric_limits<double>::denorm_min();
    if (skimagexx::line_model_nd_predict_2d(values.data(), values.size(),
            origin2, direction2, 0, prediction.data()) || prediction != previous)
        return 6;
    values.back() = 0;
    const auto old_rounding = std::fegetround();
    if (std::fesetround(FE_DOWNWARD) != 0) return 7;
    const bool claimed = skimagexx::line_model_nd_predict_2d(values.data(), values.size(),
        origin2, direction2, 0, prediction.data());
    if (std::fesetround(old_rounding) != 0 || claimed || prediction != previous)
        return 8;

    std::array<double, 384> prediction3{};
    const double origin3[] = {1, 2, 3}, direction3[] = {2, -1, 0.5};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!skimagexx::line_model_nd_predict_3d(values.data(), values.size(),
                origin3, direction3, axis, prediction3.data()))
            return 9;
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto t = (values[i] - origin3[axis]) / direction3[axis];
            for (std::size_t col = 0; col < 3; ++col) {
                if (prediction3[3 * i + col] != origin3[col] + t * direction3[col])
                    return 10;
            }
        }
    }
    const auto previous3 = prediction3;
    if (skimagexx::line_model_nd_predict_3d(values.data(), values.size(),
            origin3, direction3, 3, prediction3.data()) || prediction3 != previous3)
        return 11;
    return 0;
}
