#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace skimagexx {

struct HausdorffMask {
    const std::uint8_t* bytes;
    // Row-major shape; ignore trailing dimensions past rank.
    std::array<std::size_t, 4> shape;
};

enum class HausdorffOperation { standard, modified, pair };

struct HausdorffResult {
    double distance = 0;
    std::array<std::size_t, 2> count{};
    std::array<std::size_t, 2> attaining{};
    // For the pair operation only, preserve each set in C-order enumeration.
    std::array<std::vector<std::array<int, 4>>, 2> points;
};

// Nonzero bytes represent points. Supports rank 1–4 and at most 262144 bytes
// and 256 points per mask, with extents at most 16384. Keep both buffers valid
// and unchanged for the whole call. No provider and no internal workers.
// Declines ambiguous multi-leaf pair ties and unsupported FENV or inputs.
bool hausdorff(const HausdorffMask& a, const HausdorffMask& b, std::size_t rank,
               HausdorffOperation operation, HausdorffResult& result);

} // namespace skimagexx
