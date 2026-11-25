#pragma once

#include "skimagexx/image.hpp"

namespace skimagexx {

struct MarchingCubesMesh {
    // Coordinates and normals are stored in x, y, z order.
    std::vector<std::array<float, 3>> vertices;
    std::vector<std::array<std::int32_t, 3>> faces;
    std::vector<std::array<float, 3>> normals;
    std::vector<float> values;
};

// Contiguous [z, y, x] volume and optional byte mask. A cube is visited if
// the mask at its maximal corner is nonzero. Level is in sample units.
MarchingCubesMesh marching_cubes(const float* volume, std::array<std::size_t, 3> shape,
                                double level, std::size_t step = 1, bool classic = false,
                                const std::uint8_t* mask = nullptr);

}  // namespace skimagexx
