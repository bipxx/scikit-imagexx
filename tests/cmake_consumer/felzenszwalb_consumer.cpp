#include <skimagexx/felzenszwalb.hpp>

#include <array>

int main() {
    constexpr std::array<double, 4> input{0, 0, 1, 1};
    std::array<double, 6> costs{};
    constexpr std::array<std::ptrdiff_t, 6> order{0, 1, 2, 3, 4, 5};
    std::array<std::ptrdiff_t, 4> labels{};
    if (skimagexx::felzenszwalb_edge_count(2, 2) != 6) return 1;
    skimagexx::felzenszwalb_costs_into(input.data(), costs.data(), 2, 2, 1);
    if (costs != std::array<double, 6>{0, 0, 1, 1, 1, 1}) return 2;
    skimagexx::felzenszwalb_labels_into(costs.data(), order.data(), labels.data(), 2, 2, 1, 1);
    return labels == std::array<std::ptrdiff_t, 4>{0, 0, 1, 1} ? 0 : 3;
}
