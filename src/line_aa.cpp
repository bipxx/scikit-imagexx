// Port of scikit-image0.26 _draw.pyx:_line_aa (BSD-3-Clause).
// Attribution: third_party/scikit-image-LICENSE.txt. Preserve literal ordered
// arithmetic rather than replacing sqrt with hypot or reassociating coverage.
#include <skimagexx/line_aa.hpp>
#include <cmath>
#include <limits>

namespace skimagexx::detail {
namespace {
template<bool Emit>
std::size_t visit(const DrawLineAAGeometry& geometry, DrawCoordinatePair output,
                  double* weights, std::ptrdiff_t weight_step) {
    auto row = geometry.r0, column = geometry.c0;
    double error = geometry.dc - geometry.dr;
    std::size_t count = 0;
    auto append = [&](std::int32_t r, std::int32_t c, double numerator) {
        if constexpr (Emit) {
            output.rows[std::ptrdiff_t(count) * output.row_step] = r;
            output.columns[std::ptrdiff_t(count) * output.column_step] = c;
            weights[std::ptrdiff_t(count) * weight_step] = std::fabs(numerator) / geometry.distance;
        }
        ++count;
    };
    while (true) {
        append(row, column, (error - geometry.dc) + geometry.dr);
        const double previous_error = error;
        const auto previous_column = column;
        if (2 * previous_error >= -geometry.dc) {
            if (column == geometry.c1) break;
            if (previous_error + geometry.dr < geometry.distance)
                append(row + geometry.sign_r, column, previous_error + geometry.dr);
            error -= geometry.dr;
            column += geometry.sign_c;
        }
        if (2 * previous_error <= geometry.dr) {
            if (row == geometry.r1) break;
            if (geometry.dc - previous_error < geometry.distance)
                append(row, previous_column + geometry.sign_c, geometry.dc - previous_error);
            error += geometry.dc;
            row += geometry.sign_r;
        }
    }
    return count;
}
}

bool draw_line_aa_prepare(std::intptr_t r0, std::intptr_t c0,
                          std::intptr_t r1, std::intptr_t c1,
                          DrawLineAAGeometry& output) {
    constexpr auto low = std::numeric_limits<std::int32_t>::min();
    constexpr auto high = std::numeric_limits<std::int32_t>::max();
    if (r0 <= low || r0 >= high || c0 <= low || c0 >= high ||
        r1 <= low || r1 >= high || c1 <= low || c1 >= high) return false;
    const auto row_delta = std::int64_t(r1) - std::int64_t(r0);
    const auto column_delta = std::int64_t(c1) - std::int64_t(c0);
    const auto dr = row_delta < 0 ? -row_delta : row_delta;
    const auto dc = column_delta < 0 ? -column_delta : column_delta;
    // Prove the square itself fits before evaluating even widened products.
    if (dr > 46340 || dc > 46340) return false;
    const auto squared = dc * dc + dr * dr;
    if (squared > high) return false;
    const double distance = dr + dc == 0 ? 1.0 : std::sqrt(double(squared));
    output = {std::int32_t(r0), std::int32_t(c0), std::int32_t(r1), std::int32_t(c1),
              std::int32_t(dr), std::int32_t(dc),
              r0 < r1 ? 1 : -1, c0 < c1 ? 1 : -1, distance};
    return true;
}

std::size_t draw_line_aa_count(const DrawLineAAGeometry& geometry, int workers) {
    (void)workers; // Exact dependent recurrence; no private pool or global read.
    return visit<false>(geometry, {}, nullptr, 0);
}

void draw_line_aa_into(const DrawLineAAGeometry& geometry, DrawCoordinatePair output,
                       double* weights, std::ptrdiff_t weight_step, int workers) {
    (void)workers;
    const auto count = visit<true>(geometry, output, weights, weight_step);
    // The original builds float64 distances then applies NumPy's1.-array.
    // This independent finish preserves the stored division rounding and lets
    // the compiler vectorize ordinary contiguous output without altering ties.
    for (std::size_t i = 0; i < count; ++i) {
        auto& value = weights[std::ptrdiff_t(i) * weight_step];
        value = 1.0 - value;
    }
}
}
