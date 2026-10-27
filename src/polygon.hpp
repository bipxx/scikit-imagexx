#pragma once
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
// Stable native float64 contiguous columns, prepared in source order.
struct PolygonView {
    const double* x;
    const double* y;
    std::size_t vertices;
};
// Internal scalar classifier: caller has established vertices > 0.
std::uint8_t polygon_label(const PolygonView&, double x, double y) noexcept;
// Output is disjoint contiguous byte storage. Preserve labels 0/1/2/3 even
// when the Python caller allocates boolean storage; do not canonicalize.
void points_in_poly(const PolygonView&, const double* x, const double* y,
                    std::size_t count, std::uint8_t* output, int workers);
void grid_points_in_poly(const PolygonView&, std::size_t rows, std::size_t columns,
                         std::uint8_t* output, int workers);
}

namespace skimagexx::simd {
void polygon_points(const detail::PolygonView&, const double* x, const double* y,
                    std::uint8_t* output, std::size_t begin, std::size_t end);
void polygon_grid(const detail::PolygonView&, std::size_t columns,
                  std::uint8_t* output, std::size_t begin, std::size_t end);
}
