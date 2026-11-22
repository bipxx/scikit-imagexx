#include <skimagexx/hausdorff.hpp>

#include <array>
#include <cmath>
#include <stdexcept>

namespace {
void require(bool value) {
    if (!value) throw std::runtime_error("Hausdorff SDK assertion failed");
}
}

int main() {
    using skimagexx::HausdorffMask;
    using skimagexx::HausdorffOperation;
    using skimagexx::HausdorffResult;
    std::array<std::uint8_t, 13> left{}, right{};
    left[6] = 2;
    right[2] = right[10] = 1;
    const HausdorffMask a{left.data(), {13}}, b{right.data(), {13}};
    HausdorffResult output;
    require(skimagexx::hausdorff(a, b, 1, HausdorffOperation::standard, output));
    require(output.distance == 4.0);
    require(skimagexx::hausdorff(a, b, 1, HausdorffOperation::modified, output));
    require(output.distance == 4.0);
    require(skimagexx::hausdorff(a, b, 1, HausdorffOperation::pair, output));
    require(output.points[0][output.attaining[0]][0] == 6 &&
            output.points[1][output.attaining[1]][0] == 2);
    left.fill(0);
    require(skimagexx::hausdorff(a, b, 1, HausdorffOperation::standard, output));
    require(std::isinf(output.distance));
    require(skimagexx::hausdorff(a, b, 1, HausdorffOperation::pair, output));
    require(output.count[0] == 0 && output.count[1] == 2);
}
