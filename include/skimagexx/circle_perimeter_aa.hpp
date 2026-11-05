#pragma once
#include <skimagexx/draw_coordinates.hpp>
#include <cstddef>
#include <cstdint>

namespace skimagexx::detail {
struct CirclePerimeterAARequest {
    std::intptr_t row=0, column=0, radius=0;
    bool clipped=false;
    std::intptr_t height=0, width=0;
};
// Prepared once at the typed boundary. Keep unchanged between count and fill.
// No Python objects, borrowed input buffers, callbacks or owned allocation.
struct CirclePerimeterAA {
    CirclePerimeterAARequest request;
    std::size_t max_points=0;
    std::intptr_t squared_radius=0;
};
// Refuses signed square/negation/origin-addition or address-size overflow before
// entering the recurrence. Refusal leaves output unchanged. Negative/zero radii
// still emit the source's initial eight points with unit weight. Integer shape
// bounds are literal: nonpositive bounds clip every point; clipping is stable.
bool prepare_circle_perimeter_aa(const CirclePerimeterAARequest&, CirclePerimeterAA&) noexcept;
// Exact ordered dependent recurrence; serial even when caller snapshot>1.
// No global worker reads or private pool. Ordinary rounding/strict FP required.
std::size_t circle_perimeter_aa_count(const CirclePerimeterAA&, int workers) noexcept;
// Caller establishes fresh disjoint aligned intp/intp/double storage, nonzero
// signed element strides, valid address spans and exact count capacity once.
// Empty output permits null pointers. No allocation or repeated boundary checks.
// Preserve ordered duplicates, zero weights and literal sqrt/ceil/subtraction.
void circle_perimeter_aa_fill(const CirclePerimeterAA&, DrawCoordinatePair,
                              double* weights, std::ptrdiff_t weight_step, int workers) noexcept;
}
