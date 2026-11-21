#include <skimagexx/image.hpp>
#include <skimagexx/local_extrema.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr std::size_t side = 5;
constexpr std::size_t stride = 7;
using Mask = std::array<std::uint8_t, side * side>;

// Input and results of the saved scikit-image 0.26.0 default-plateau case.
constexpr Mask image{
    1, 0, 1, 0, 1,
    0, 4, 4, 1, 0,
    1, 4, 4, 0, 4,
    0, 1, 0, 4, 4,
    1, 0, 1, 0, 1,
};
constexpr Mask maxima{
    0, 0, 0, 0, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 0, 1,
    0, 0, 0, 1, 1,
    0, 0, 0, 0, 0,
};
constexpr Mask minima{
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1,
    0, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    0, 1, 0, 1, 0,
};

void check_mask(skimagexx::LocalExtremaKind kind, const Mask& expected) {
    constexpr std::uint8_t padding = 9;
    std::array<std::uint8_t, side * stride> flags;
    flags.fill(padding);
    skimagexx::local_extrema_2d_into(image.data(), image.size(), side, side, kind,
                                     skimagexx::LocalExtremaNeighbors::eight, true,
                                     flags.data(), flags.size(), stride);
    for (std::size_t row = 0; row < side; ++row) {
        for (std::size_t col = 0; col < stride; ++col) {
            const auto saved = col < side ? expected[row * side + col] : padding;
            if (flags[row * stride + col] != saved)
                throw std::runtime_error("local extrema mask or row padding differs from reference");
        }
    }
}

}  // namespace

int main() {
    if (skimagexx::get_num_threads() != 1)
        throw std::runtime_error("expected default worker count to be one");
#ifdef _OPENMP
    constexpr std::array budgets{1, 2};
#else
    constexpr std::array budgets{1};
#endif
    for (const int workers : budgets) {
        skimagexx::set_num_threads(workers);
        check_mask(skimagexx::LocalExtremaKind::maxima, maxima);
        check_mask(skimagexx::LocalExtremaKind::minima, minima);
        if (skimagexx::get_num_threads() != workers)
            throw std::runtime_error("local extrema changed the worker budget");
    }
    skimagexx::set_num_threads(1);
}
