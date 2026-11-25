#include <skimagexx/polynomial.hpp>

#include <array>
#include <iostream>

int main() {
    const std::array<double, 4> points{2, 3, 1, -2};
    const std::array<double, 12> params{1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6};
    std::array<double, 4> result{};
    if (!skimagexx::polynomial_apply(points.data(), 2, params.data(), 6, result.data()) ||
        result != std::array<double, 4>{114, -114, 15, -15}) return 1;
    std::array<double, 28> equations{};
    const std::array<double, 2> weights{4, 1};
    if (!skimagexx::polynomial_constraints(points.data(), result.data(), weights.data(), 2, 1,
                                           equations.data()) ||
        equations != std::array<double, 28>{1, 2, 3, 0, 0, 0, 114,
                                            0.5, 0.5, -1, 0, 0, 0, 7.5,
                                            0, 0, 0, 1, 2, 3, -114,
                                            0, 0, 0, 0.5, 0.5, -1, -7.5}) return 2;
    std::cout << "polynomial SDK passed\n";
}
