#include <skimagexx/watershed.hpp>

#include <array>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("installed watershed SDK result differs");
}

}  // namespace

int main() {
    const std::array<std::size_t, 1> line_shape{5};
    const std::array<double, 5> plateau{};
    const auto high = std::numeric_limits<std::uint64_t>::max();
    const std::array<std::uint64_t, 5> seeds{high, 0, 0, 0, 2};
    const std::array<std::uint8_t, 5> mask{1, 1, 0, 1, 1};
    std::array<std::uint64_t, 5> labels{};
    auto run = [&](const std::uint8_t* active, double compactness, bool line) {
        skimagexx::watershed_into(plateau.data(), seeds.data(), active, seeds.size(),
                                  line_shape.data(), line_shape.size(), compactness, line, labels.data());
    };
    run(nullptr, 0., false);
    require(labels == std::array<std::uint64_t, 5>{high, high, high, 2, 2});
    run(nullptr, 0., true);
    require(labels == std::array<std::uint64_t, 5>{high, high, 0, 2, 2});
    run(nullptr, 0.35, true);
    require(labels == std::array<std::uint64_t, 5>{high, high, high, 2, 2});
    run(mask.data(), 0., false);
    require(labels == std::array<std::uint64_t, 5>{high, high, 0, 2, 2});

    const std::array<std::size_t, 2> square_shape{3, 3};
    const skimagexx::WatershedNeighborhood diagonal{{1, 1}, {-6, -4, 4, 6}};
    const std::array<double, 9> square{};
    const std::array<std::int16_t, 9> diagonal_seeds{7, 0, 0, 0, 0, 0, 0, 0, -9};
    std::array<std::int16_t, 9> diagonal_labels{};
    auto run_diagonal = [&](const auto& neighborhood) {
        skimagexx::watershed_into(square.data(), diagonal_seeds.data(), nullptr, square.size(),
                                  square_shape.data(), square_shape.size(), 0., false,
                                  diagonal_labels.data(), &neighborhood);
    };
    run_diagonal(diagonal);
    require(diagonal_labels == std::array<std::int16_t, 9>{7, 0, 7, 0, 7, 0, 7, 0, -9});
    try {
        run_diagonal(skimagexx::WatershedNeighborhood{{1, 1}, {-7}});
        require(false);
    } catch (const std::invalid_argument&) {}
    require(diagonal_labels == std::array<std::int16_t, 9>{7, 0, 7, 0, 7, 0, 7, 0, -9});
}
