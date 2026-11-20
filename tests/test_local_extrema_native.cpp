#include "skimagexx/local_extrema.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace skimagexx;

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("Incorrect local extrema mask or input validation");
}

template<class T>
void check() {
    const auto low = std::numeric_limits<T>::lowest();
    const auto high = std::numeric_limits<T>::max();
    const std::array<T, 9> image{high, high, low, high, high, T{1}, low, T{1}, T{1}};
    std::array<std::uint8_t, 15> flags;
    flags.fill(9);
    const auto run = [&](LocalExtremaKind kind, LocalExtremaNeighbors neighbors, bool borders) {
        local_extrema_2d_into(image.data(), image.size(), 3, 3, kind, neighbors, borders,
                             flags.data(), flags.size(), 5);
        for (std::size_t row = 0; row < 3; ++row)
            require(flags[row * 5 + 3] == 9 && flags[row * 5 + 4] == 9);
        return std::array<std::uint8_t, 9>{
            flags[0], flags[1], flags[2], flags[5], flags[6], flags[7],
            flags[10], flags[11], flags[12]};
    };
    require(run(LocalExtremaKind::maxima, LocalExtremaNeighbors::eight, true) ==
            std::array<std::uint8_t, 9>{1, 1, 0, 1, 1, 0, 0, 0, 0});
    require(run(LocalExtremaKind::minima, LocalExtremaNeighbors::four, true) ==
            std::array<std::uint8_t, 9>{0, 0, 1, 0, 0, 0, 1, 0, 0});
    require(run(LocalExtremaKind::maxima, LocalExtremaNeighbors::eight, false) ==
            std::array<std::uint8_t, 9>{});
    try {
        local_extrema_2d_into(image.data(), image.size(), 3, 3, LocalExtremaKind::maxima,
                             LocalExtremaNeighbors::eight, true, flags.data(), 12, 5);
        require(false);
    } catch (const std::invalid_argument&) {}
}

int main() {
    check<std::uint8_t>();
    check<std::int8_t>();
    check<std::uint16_t>();
    check<std::int16_t>();
    check<std::uint32_t>();
    check<std::int32_t>();
    check<std::uint64_t>();
    check<std::int64_t>();
}
