#include <skimagexx/epipolar.hpp>

#include <array>
#include <cmath>
#include <iostream>

int main() {
    const std::array<double, 9> identity{1, 0, 0, 0, 1, 0, 0, 0, 1};
    const std::array<double, 4> src{2, 3, 1, -2}, dst{1, 2, -3, 5};
    std::array<double, 6> lines{};
    if (!skimagexx::epipolar_lines(src.data(), 2, identity.data(), lines.data()) ||
        lines != std::array<double, 6>{2, 3, 1, 1, -2, 1}) return 1;
    std::array<double, 2> errors{};
    if (!skimagexx::epipolar_residuals(src.data(), dst.data(), 2, identity.data(), errors.data()) ||
        std::abs(errors[0] - 9 / std::sqrt(18.0)) > 1e-13 ||
        std::abs(errors[1] - 12 / std::sqrt(39.0)) > 1e-13) return 2;

    std::array<double, 18> equations{};
    std::array<double, 9> left{}, right{}, matrix{};
    if (!skimagexx::epipolar_constraints(src.data(), dst.data(), 2, skimagexx::EpipolarScaling::raw,
                                        equations.data(), left.data(), right.data()) ||
        left != identity || right != identity ||
        equations != std::array<double, 18>{2, 3, 1, 4, 6, 2, 2, 3, 1,
                                            -3, 6, -3, 5, -10, 5, 1, -2, 1}) return 3;
    const std::array<double, 3> singular{4, 2, 1};
    skimagexx::epipolar_matrix(identity.data(), singular.data(), identity.data(),
                                left.data(), right.data(), false, matrix.data());
    if (matrix != std::array<double, 9>{4, 0, 0, 0, 2, 0, 0, 0, 0}) return 4;
    skimagexx::epipolar_matrix(identity.data(), singular.data(), identity.data(),
                                left.data(), right.data(), true, matrix.data());
    if (matrix != std::array<double, 9>{3, 0, 0, 0, 3, 0, 0, 0, 0}) return 5;
    std::cout << "epipolar SDK passed\n";
}
