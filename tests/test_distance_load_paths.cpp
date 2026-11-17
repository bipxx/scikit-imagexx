#include <skimagexx/distance.hpp>
#include <skimagexx/image.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

int main() {
    using namespace skimagexx;
    if (get_num_threads() != 1) throw std::runtime_error("SDK startup workers");
    struct RestoreWorkers { ~RestoreWorkers() { skimagexx::set_num_threads(1); } } restore_workers;
    for (int workers : {1, 4}) {
        set_num_threads(workers);
        for (std::size_t count : {7u, 8u, 9u, 15u, 16u, 17u}) {
            // Exact physical span, including only three channels of the last
            // four-stride pixel. ASan must observe no fourth-channel overread.
            std::vector<float> first((count - 1) * 4 + 3), second(count * 3), actual(count), expected(count);
            for (std::size_t i = 0; i < first.size(); ++i) first[i] = float(i);
            for (std::size_t i = 0; i < second.size(); ++i) second[i] = float(i) / 3.f;
            for (std::size_t i = 0; i < count; ++i) {
                const auto j = (count - 1 - i) * 3;
                const float x = second[j] - first[i * 4];
                const float y = second[j + 1] - first[i * 4 + 1];
                const float z = second[j + 2] - first[i * 4 + 2];
                expected[i] = std::sqrt(float(float(float(x*x) + float(y*y)) + float(z*z)));
            }
            delta_e76_into(first.data(), second.data() + (count - 1) * 3, actual.data(), count,
                PixelType::float32, PixelType::float32, 4, 1, -3, 1);
            if (std::memcmp(actual.data(), expected.data(), count * sizeof(float)))
                throw std::runtime_error("CIE76 exact-span deinterleave mismatch");
        }
    }
    set_num_threads(1);
    if (get_num_threads() != 1) throw std::runtime_error("SDK restored workers");
}
