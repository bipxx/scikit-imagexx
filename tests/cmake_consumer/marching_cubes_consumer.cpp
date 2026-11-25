#include "skimagexx/marching_cubes.hpp"

#include <cmath>

int main() {
    const std::array<float, 27> volume{
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    for (bool classic : {false, true}) {
        const auto mesh = skimagexx::marching_cubes(volume.data(), {3, 3, 3}, 1, 1, classic);
        if (mesh.vertices.size() != 6 || mesh.faces.size() != 8 ||
            mesh.normals.size() != 6 || mesh.values.size() != 6) return 1;
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            const auto& point = mesh.vertices[i];
            if (std::abs(point[0] - 1) + std::abs(point[1] - 1) + std::abs(point[2] - 1) != 0.5 ||
                mesh.values[i] != 2) return 2;
        }
    }
    const std::array<std::uint8_t, 27> mask{};
    return skimagexx::marching_cubes(volume.data(), {3, 3, 3}, 1, 1, false, mask.data()).faces.empty()
        ? 0 : 3;
}
