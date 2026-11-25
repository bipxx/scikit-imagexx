#include <skimagexx/brief.hpp>

#include <array>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) throw std::runtime_error("BRIEF SDK assertion failed");
}

template <typename Pixel>
void check_samples() {
    const std::array<Pixel, 12> image{
        9, 3, 2, 9,
        4, 3, std::numeric_limits<Pixel>::quiet_NaN(), 5,
        1, 3, 8, 1
    };
    const std::array<std::int64_t, 4> keypoints{1, 1, 1, 2};
    const std::array<std::int32_t, 6> left{0, -1, -1, 0, 1, 0};
    const std::array<std::int32_t, 6> right{0, 0, 0, 0, 0, 1};
    std::array<std::uint8_t, 6> descriptors{};
    skimagexx::brief_sample(image.data(), 3, 4, keypoints.data(), 2,
                            left.data(), right.data(), 3, descriptors.data());
    require(descriptors == std::array<std::uint8_t, 6>{0, 0, 0, 0, 0, 0});
    const std::array<std::int64_t, 4> border{1, 3, 0, 0};
    const std::array<std::int32_t, 2> zero{0, 0};
    const std::array<std::int32_t, 2> next_column{0, 1};
    skimagexx::brief_sample(image.data(), 3, 4, border.data(), 2,
                            next_column.data(), zero.data(), 1, descriptors.data());
    require(descriptors[0] == 1 && descriptors[1] == 1);

    const std::array<std::int64_t, 2> last{2, 3};
    try {
        skimagexx::brief_sample(image.data(), 3, 4, last.data(), 1,
                                next_column.data(), zero.data(), 1, descriptors.data());
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error("BRIEF accepted an out-of-storage sample");
}

} // namespace

int main() {
    check_samples<float>();
    check_samples<double>();
    skimagexx::brief_sample(static_cast<const float*>(nullptr), 0, 0,
                            nullptr, 0, nullptr, nullptr, 0, nullptr);
}
