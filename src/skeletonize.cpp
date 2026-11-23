#include "skeletonize_internal.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace skimagexx {
namespace {

// scikit-image 0.26 Zhang class table, with neighbors NW,N,NE,E,SE,S,SW,W.
// A class is a mask of the phases in which this neighborhood is removed.
constexpr std::array<std::uint8_t, 256> classes{
    0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0
};

bool remove_phase(std::vector<std::uint8_t>& state, std::ptrdiff_t pitch, unsigned phase,
                  std::vector<std::uint32_t>& live, std::vector<std::uint32_t>& removed) {
    removed.clear();
    auto survivor = live.begin();
    // Keep the remaining foreground in row-major order. Defer changes to
    // state until every center has read the original phase neighborhood.
    for (auto index : live) {
        const auto* middle = state.data() + index;
        const auto* above = middle - pitch;
        const auto* below = middle + pitch;
        const auto code = above[-1] | (above[0] << 1) | (above[1] << 2)
            | (middle[1] << 3) | (below[1] << 4) | (below[0] << 5)
            | (below[-1] << 6) | (middle[-1] << 7);
        if (classes[code] & phase) {
            removed.push_back(index);
        } else {
            *survivor++ = index;
        }
    }
    live.erase(survivor, live.end());
    for (auto index : removed) {
        state[index] = 0;
    }
    return !removed.empty();
}

}  // namespace

bool detail::skeletonize_zhang_padded(
    const std::uint8_t* image, std::size_t rows, std::size_t columns,
    SkeletonizeInput kind, std::vector<std::uint8_t>& padded) {
    if (!image || rows < 1 || columns < 1 || rows > 512 || columns > 512
        || (kind != SkeletonizeInput::boolean && kind != SkeletonizeInput::uint8)) {
        return false;
    }
    const auto pitch = columns + 2;
    std::vector<std::uint8_t> state((rows + 2) * pitch);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* source = image + row * columns;
        auto* destination = state.data() + (row + 1) * pitch + 1;
        if (kind == SkeletonizeInput::boolean) {
            if (std::any_of(source, source + columns, [](auto value) { return value > 1; })) {
                return false;
            }
            std::copy_n(source, columns, destination);
        } else {
            std::transform(source, source + columns, destination, [](auto value) { return value != 0; });
        }
    }

    std::vector<std::uint32_t> live;
    live.reserve(rows * columns);
    for (std::size_t index = 0; index < state.size(); ++index) {
        if (state[index]) {
            live.push_back(static_cast<std::uint32_t>(index));
        }
    }
    std::vector<std::uint32_t> removed;
    while (true) {
        const bool first = remove_phase(state, std::ptrdiff_t(pitch), 1, live, removed);
        const bool second = remove_phase(state, std::ptrdiff_t(pitch), 2, live, removed);
        if (!first && !second) {
            break;
        }
    }
    padded = std::move(state);
    return true;
}

void detail::copy_skeletonize_zhang_rows(
    const std::vector<std::uint8_t>& padded, std::size_t rows, std::size_t columns,
    std::uint8_t* destination) {
    const auto pitch = columns + 2;
    for (std::size_t row = 0; row < rows; ++row) {
        std::copy_n(padded.data() + (row + 1) * pitch + 1, columns, destination + row * columns);
    }
}

bool skeletonize_zhang(
    const std::uint8_t* image, std::size_t rows, std::size_t columns,
    SkeletonizeInput kind, std::vector<std::uint8_t>& output) {
    std::vector<std::uint8_t> padded;
    if (!detail::skeletonize_zhang_padded(image, rows, columns, kind, padded)) {
        return false;
    }
    std::vector<std::uint8_t> result(rows * columns);
    detail::copy_skeletonize_zhang_rows(padded, rows, columns, result.data());
    output = std::move(result);
    return true;
}

}  // namespace skimagexx
